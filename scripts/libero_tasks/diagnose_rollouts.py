#!/usr/bin/env python
"""Diagnostic rollouts of a policy on LIBERO (libero_ext) tasks: what an evaluation does (same env, start states,
pre/post-processing and action chunking as lerobot-eval / eval_seeds_campaign.py), plus, per episode:
  - a video, agentview | wrist camera side by side (rotated as the dataset stores them), every second step;
  - per step: end-effector position, gripper opening, the action sent, and each goal predicate's truth value;
  - a summary: success, length, the step at which each goal predicate first held, the predicates still false.
For finding out why a policy fails (projects/dataset_gen.md §3c).

  DIAG_OUT=outputs/analysis/x DIAG_EPISODES=10 python scripts/libero_tasks/diagnose_rollouts.py \
      --policy.path=<checkpoint> [--policy.use_peft=true] --env.type=libero --env.task=libero_ext \
      --env.task_ids='[31]' --eval.batch_size=5 --policy.n_action_steps=10 --rename_map='{...}' --seed=1000
"""

import json
import os
from pathlib import Path

import imageio
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed

OUT = Path(os.environ.get("DIAG_OUT", "outputs/analysis/diag"))
N_EPISODES = int(os.environ.get("DIAG_EPISODES", "10"))


def goal_status(inner):
    """{predicate string: bool} for every goal predicate of the BDDL env."""
    env = inner.env
    return {" ".join(g): bool(env._eval_predicate(g)) for g in env.parsed_problem["goal_state"]}


@parser.wrap()
def main(cfg: EvalPipelineConfig):
    local_assets = "/home/josh/phddev/LIBERO/libero/libero/assets"
    if os.path.isdir(local_assets):  # desk PC: LIBERO's own assets, no hub download
        import libero.libero as libero_pkg

        libero_pkg._assets_path_cache = local_assets
    import lerobot.envs.libero  # noqa: F401  (registers the libero_ext suite)

    set_seed(cfg.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=False)
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    summaries = []
    for suite, by_task in envs.items():
        for task_id, venv in by_task.items():
            n_batches = (N_EPISODES + venv.num_envs - 1) // venv.num_envs
            for b in range(n_batches):
                seeds = [cfg.seed + b * venv.num_envs + i for i in range(venv.num_envs)]
                policy.reset()
                obs, _ = venv.reset(seed=seeds)
                inner = [e._env for e in venv.envs]
                lang = venv.envs[0].task_description
                n = venv.num_envs
                done = np.zeros(n, bool)
                success = np.zeros(n, bool)
                frames = [[] for _ in range(n)]
                trace = [[] for _ in range(n)]
                first_true = [{} for _ in range(n)]
                max_steps = venv.call("_max_episode_steps")[0]
                step = 0
                while not done.all() and step < max_steps:
                    for i in range(n):
                        if not done[i]:
                            if step % 2 == 0:
                                img = obs["pixels"]["image"][i][::-1, ::-1]
                                wr = obs["pixels"]["image2"][i][::-1, ::-1]
                                frames[i].append(np.concatenate([img, wr], axis=1))
                    o = preprocess_observation(obs)
                    o["task"] = list(venv.call("task_description"))  # the instruction (as lerobot-eval's rollout)
                    o = env_preprocessor(o)
                    o = preprocessor(o)
                    with torch.inference_mode():
                        act = policy.select_action(o)
                    act = postprocessor(act)
                    act = env_postprocessor({ACTION: act})[ACTION]
                    a = act.to("cpu").numpy()
                    rs = obs.get("robot_state", {})
                    eef = rs["eef"]["pos"] if rs else np.zeros((n, 3))
                    grip = rs["gripper"]["qpos"] if rs else np.zeros((n, 2))
                    obs, _, term, trunc, info = venv.step(a)
                    step += 1
                    for i in range(n):
                        if done[i]:
                            continue
                        rec = {"t": step, "eef": np.round(eef[i], 4).tolist(),
                               "grip": round(float(grip[i][0] - grip[i][1]), 4),
                               "action": np.round(a[i], 3).tolist()}
                        if term[i]:  # LiberoEnv has already reset this sub-env: success is all goals at once
                            success[i] = bool(info["is_success"][i]) if "is_success" in info else True
                            done[i] = True
                            rec["goals"] = "all (success)" if success[i] else "episode ended"
                        else:
                            st = goal_status(inner[i])
                            rec["goals"] = st
                            for g, v in st.items():
                                if v and g not in first_true[i]:
                                    first_true[i][g] = step
                        trace[i].append(rec)
                for i in range(n):
                    ep = b * n + i
                    if ep >= N_EPISODES:
                        break
                    tag = f"task{task_id:02d}_ep{ep:02d}_seed{seeds[i]}_{'OK' if success[i] else 'FAIL'}"
                    if frames[i]:
                        imageio.mimwrite(OUT / f"{tag}.mp4", frames[i], fps=10)
                    last = next((r["goals"] for r in reversed(trace[i]) if isinstance(r["goals"], dict)), {})
                    summ = {
                        "task_id": int(task_id), "language": lang, "episode": ep, "seed": seeds[i],
                        "success": bool(success[i]), "length": len(trace[i]),
                        "first_true": first_true[i],
                        "false_at_end": [] if success[i] else [g for g, v in last.items() if not v],
                        "video": f"{tag}.mp4",
                    }
                    summaries.append(summ)
                    (OUT / f"{tag}.trace.json").write_text(json.dumps(trace[i]))
                    print(f"{tag}: len {summ['length']} first_true {summ['first_true']} "
                          f"false_at_end {summ['false_at_end']}", flush=True)
    close_envs(envs)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main()
