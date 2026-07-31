#!/usr/bin/env python3
"""Off-trail probe, stage 1 (E56): harvest rollout-visited states + retrieval traces.

Closed-loop episodes with the SAME machinery as lerobot-eval (policy/processor/env
construction cloned from lerobot_eval.eval_main), but with an owned rollout loop so that
at every POLICY CALL (action-queue refill, every n_action_steps env steps) we save:
  - the RAW env observation (uint8 images + proprio, pre-preprocessing) so the scoring
    pass (probe_offtrail_score.py) can push the identical state through any model's own
    pipeline;
  - for memory policies: the retrieval trace of that call — per module, slot ids + softmax
    mixture mass aggregated over tokens/heads/firings. Uses the eval-mode recording that
    EVAL_MEMORY=True already provides (last_indices/last_scores set per forward, with
    route-once multiplicity/token-mask applied), captured by a forward hook on each
    HashingMemoryLite. Under frozen-route only the live pass B fires the module, so the
    trace is clean; VLM modules fire 1x/call (cached prefix), expert modules 1x/denoise
    step (10x/call).
  - executed actions, episode outcome, per-episode seed (cfg.seed + episode index — the
    same per-episode seeds as a serial lerobot-eval run, so outcomes are comparable).

Batch size is FORCED to 1: eval-mode trace recording flattens (env x token) rows, and
bs=1 keeps per-state attribution trivial. ~50 eps ~= 1h serial on the H200.

CLI = EvalPipelineConfig (same flags as lerobot-eval: --policy.path, --env.*, --seed,
--rename_map). Env knobs:
  HARVEST_OUT       output dir (required)
  HARVEST_EPISODES  episodes (default 50)
  HARVEST_TRACE     1/0 record retrieval traces (default 1; no-op for memoryless models)

Output in HARVEST_OUT:
  ep{e:03d}.npz        obs__px__<cam> (n_calls, H, W, 3 uint8) + obs__rs__* nested robot
                       state (flattened, see _flatten_raw_obs), states (n_calls, 9 raw
                       pos+quat+gripper), call_steps, actions (T, A), success, seed
  trace_ep{e:03d}.npz  c{i}__{module}__slots / __mass per call per module
  index.json           per-episode outcome/calls/steps, task string, obs keys, config
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed

try:
    from lerobot.policies.modules.memory_lite import HashingMemoryLite
except Exception:  # memoryless installs
    HashingMemoryLite = None


class TraceRecorder:
    """Forward hooks on every HashingMemoryLite; per call, aggregates slot->mass."""

    def __init__(self, policy):
        self.enabled = False
        self.entries = []  # (modkey, slots 1d np.int64, weights 1d np.float32)
        self.hooks = []
        self.modkeys = []
        if HashingMemoryLite is None:
            return
        for name, m in policy.named_modules():
            if isinstance(m, HashingMemoryLite):
                key = name.replace(".mlp.mem", "")
                self.modkeys.append(key)
                self.hooks.append(m.register_forward_hook(self._make_hook(key)))

    def _make_hook(self, key):
        def hook(module, args, output):
            if not self.enabled:
                return
            idx = getattr(module, "last_indices", None)
            sc = getattr(module, "last_scores", None)
            if idx is None or sc is None:
                return
            # last_scores: (rows, heads, knn) raw scores; mixture weight = per-head softmax
            w = torch.softmax(sc.float(), dim=-1)
            self.entries.append(
                (key, idx.reshape(-1).cpu().numpy().astype(np.int64), w.reshape(-1).cpu().numpy())
            )

        return hook

    def start_call(self):
        self.entries = []
        self.enabled = True

    def end_call(self):
        """Aggregate the call's firings into {modkey: (uniq_slots, mass)}."""
        self.enabled = False
        agg = {}
        for key in set(k for k, _, _ in self.entries):
            slots = np.concatenate([s for k, s, _ in self.entries if k == key])
            mass = np.concatenate([m for k, _, m in self.entries if k == key])
            uniq, inv = np.unique(slots, return_inverse=True)
            summed = np.zeros(uniq.shape[0], dtype=np.float64)
            np.add.at(summed, inv, mass)
            agg[key] = (uniq.astype(np.int64), summed.astype(np.float32))
        return agg


def _queue_len(policy):
    q = getattr(policy, "_action_queue", None)
    return None if q is None else len(q)


def _successes_from_info(info, n_envs):
    if "final_info" in info:
        fi = info["final_info"]
        if isinstance(fi, dict) and "is_success" in fi:
            return list(fi["is_success"])
    if "is_success" in info:
        s = info["is_success"]
        return list(s) if hasattr(s, "__len__") else [bool(s)] * n_envs
    return [False] * n_envs


def _flatten_raw_obs(raw):
    """Flatten one vec-env LIBERO observation (batch index 0) into {flat_key: np.ndarray}.

    obs_type=pixels_agent_pos yields {"pixels": {cam: (1,H,W,3) u8}, "robot_state": nested}.
    pixels.<cam> -> px__<cam>; robot_state nesting joins with "__" -> rs__eef__pos etc.
    probe_offtrail_score._rebuild_env_obs inverts this exactly.
    """
    out = {}
    for k in sorted(raw["pixels"].keys()):
        out[f"px__{k}"] = np.asarray(raw["pixels"][k][0])

    def rec(d, pre):
        for k in sorted(d.keys()):
            v = d[k]
            if isinstance(v, dict):
                rec(v, pre + (k,))
            elif v is not None:
                out["rs__" + "__".join(pre + (k,))] = np.asarray(v[0], dtype=np.float32)

    rec(raw["robot_state"], ())
    return out


@parser.wrap()
def main(cfg: EvalPipelineConfig):
    logging.basicConfig(level=logging.INFO)
    out_dir = Path(os.environ["HARVEST_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_eps = int(os.environ.get("HARVEST_EPISODES", "50"))
    want_trace = os.environ.get("HARVEST_TRACE", "1") == "1"

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    envs = make_env(cfg.env, n_envs=1, use_async_envs=False)
    (suite, task_envs), = envs.items()
    (task_id, env), = task_envs.items()
    print(f"[harvest] suite={suite} task_id={task_id}", flush=True)

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
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    rec = TraceRecorder(policy) if want_trace else None
    if rec is not None:
        print(f"[harvest] tracing {len(rec.modkeys)} memory modules: {rec.modkeys}", flush=True)
    n_action_steps = int(getattr(policy.config, "n_action_steps", 50))
    max_steps = int(env.call("_max_episode_steps")[0])

    index = {"episodes": [], "task": None, "obs_keys": None, "n_action_steps": n_action_steps,
             "policy_path": str(cfg.policy.pretrained_path), "seed0": cfg.seed,
             "suite": suite, "task_id": int(task_id), "trace": bool(rec and rec.modkeys)}

    for e in range(n_eps):
        seed = cfg.seed + e  # matches serial lerobot-eval per-episode seeding
        policy.reset()
        obs, info = env.reset(seed=[seed])
        task_str = list(env.call("task_description"))[0]
        if index["task"] is None:
            index["task"] = task_str
        obs_calls, states, call_steps, actions, call_traces = [], [], [], [], []
        success, done, step = False, False, 0

        while not done and step < max_steps:
            raw = obs
            o = preprocess_observation(obs)
            o["task"] = [task_str]
            o = env_pre(o)
            o = preprocessor(o)

            ql = _queue_len(policy)
            is_call = (ql == 0) if ql is not None else (step % n_action_steps == 0)
            if is_call:
                flat = _flatten_raw_obs(raw)
                if index["obs_keys"] is None:
                    index["obs_keys"] = sorted(flat.keys())
                obs_calls.append(flat)
                states.append(np.concatenate([flat["rs__eef__pos"], flat["rs__eef__quat"],
                                              flat["rs__gripper__qpos"]]).astype(np.float32))
                call_steps.append(step)
                if rec is not None:
                    rec.start_call()

            with torch.inference_mode():
                action = policy.select_action(o)
            if is_call and rec is not None:
                call_traces.append(rec.end_call())

            action = postprocessor(action)
            at = env_post({ACTION: action})
            a_np = at[ACTION].to("cpu").numpy()
            obs, reward, terminated, truncated, info = env.step(a_np)
            actions.append(a_np[0])
            succ = _successes_from_info(info, 1)
            success = success or bool(succ[0])
            done = bool(terminated[0] or truncated[0])
            if step + 1 == max_steps:
                done = True
            step += 1

        ep_arrays = {f"obs__{k}": np.stack([oc[k] for oc in obs_calls]) for k in obs_calls[0]}
        np.savez_compressed(
            out_dir / f"ep{e:03d}.npz",
            states=np.stack(states),
            call_steps=np.asarray(call_steps, dtype=np.int32),
            actions=np.stack(actions).astype(np.float32),
            success=np.asarray(success),
            seed=np.asarray(seed),
            **ep_arrays,
        )
        if rec is not None and rec.modkeys:
            tr = {}
            for ci, agg in enumerate(call_traces):
                for key, (slots, mass) in agg.items():
                    tr[f"c{ci:02d}__{key}__slots"] = slots
                    tr[f"c{ci:02d}__{key}__mass"] = mass
            np.savez_compressed(out_dir / f"trace_ep{e:03d}.npz", **tr)
        index["episodes"].append({"ep": e, "success": bool(success), "n_calls": len(call_steps),
                                  "n_steps": step, "seed": seed})
        n_succ = sum(x["success"] for x in index["episodes"])
        print(f"[harvest] ep{e:03d}: success={success} calls={len(call_steps)} steps={step} "
              f"(running {n_succ}/{e + 1})", flush=True)
        with open(out_dir / "index.json", "w") as f:
            json.dump(index, f, indent=1)

    env.close()
    print(f"[harvest] DONE: {n_eps} episodes -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
