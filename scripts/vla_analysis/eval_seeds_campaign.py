#!/usr/bin/env python3
"""E60 multi-seed eval campaign (Josh's spec, 7 Aug 26): 25 eps x 4 seeds per env,
final checkpoints only, vec-batched envs for throughput (no training resident).

One invocation = one policy (loaded once) x its envs (built once) x CAMP_SEEDS.
Env selection via --env.task_ids (memory runs: all five; specialists: their own).
Output: CAMP_OUT json {tag, seeds, results: {env_id: {seed: {"pc": float,
"successes": [bool]}}}} — per-episode successes retained for paired per-state
comparisons across configs (same seed => same init states).
"""
import json
import logging
import os
from contextlib import nullcontext
from pprint import pformat

import torch

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed

SEEDS = [int(s) for s in os.environ.get("CAMP_SEEDS", "1000,2000,3000,4000").split(",")]
TAG = os.environ.get("CAMP_TAG", "untagged")
OUT = os.environ.get("CAMP_OUT", f"/home/josh/lerobot/outputs/analysis/e60/seeds_{TAG}.json")


@parser.wrap()
def main(cfg: EvalPipelineConfig):
    logging.info(f"[campaign {TAG}] seeds={SEEDS}")
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        })
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    results: dict = {}
    for seed in SEEDS:
        with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=cfg.eval.n_episodes,
                max_episodes_rendered=0,
                videos_dir=None,
                start_seed=seed,
                max_parallel_tasks=cfg.env.max_parallel_tasks,
            )
        for entry in (info.get("per_task") or []):
            tid = entry.get("task_id")
            succ = [bool(s) for s in (entry.get("metrics") or {}).get("successes", [])][: cfg.eval.n_episodes]
            if tid is None or not succ:
                continue
            pc = 100.0 * sum(succ) / len(succ)
            results.setdefault(str(tid), {})[str(seed)] = {"pc": pc, "successes": succ}
            print(f"[campaign {TAG}] seed {seed} env {tid}: {pc:.1f} ({sum(succ)}/{len(succ)})", flush=True)

    close_envs(envs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"tag": TAG, "seeds": SEEDS, "n_episodes": cfg.eval.n_episodes, "results": results}, open(OUT, "w"), indent=1)
    print(f"[campaign {TAG}] wrote {OUT}")
    print(f"CAMPAIGN-{TAG}-DONE")


if __name__ == "__main__":
    main()
