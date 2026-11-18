#!/usr/bin/env python
"""Smoke test of the libero_ext suite through lerobot's own LIBERO env path (projects/dataset_gen.md).

1. lerobot.envs.libero registers the suite: it lists every task of suite.json, with its language.
2. The lerobot env config (``--env.type=libero --env.task=libero_ext``) builds vectorised envs for a few tasks;
   reset loads the stored start states; random actions step; observations have the expected keys and shapes;
   is_success stays false.
3. For every task: LiberoEnv resets from its first stored start state with is_success false; then the goal is
   forced on the underlying simulator (scene_lib.force_goal) and LiberoEnv.step reports is_success true.

  python scripts/libero_tasks/smoke_suite.py [--tasks 0,30,59]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=None, help="comma-separated task ids for step 3 (default: all)")
    ap.add_argument("--states", type=int, default=3, help="stored start states tried per task in step 3")
    a = ap.parse_args()

    import libero.libero as libero_pkg

    local = "/home/josh/phddev/LIBERO/libero/libero/assets"
    if os.path.isdir(local):  # desk PC: use the local LIBERO assets rather than hf-libero's hub download
        libero_pkg._assets_path_cache = local
    import scene_lib as sl
    from libero.libero import benchmark

    import lerobot.envs.libero as le
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

    # 1. registration
    suite = benchmark.get_benchmark_dict()["libero_ext"]()
    n = suite.get_num_tasks()
    print(f"[1] libero_ext registered: {n} tasks; first: '{suite.get_task(0).language}'")
    assert n > 0

    # 2. the lerobot config / vectorised env path, as lerobot-eval uses it
    cfg = LiberoEnvConfig(
        task="libero_ext", task_ids=[0, n // 2, n - 1], observation_height=256, observation_width=256
    )
    envs = cfg.create_envs(n_envs=2, use_async_envs=False)
    vec = envs["libero_ext"]

    def shapes(o, prefix=""):
        out = []
        for k, v in o.items():
            out += shapes(v, f"{prefix}{k}.") if isinstance(v, dict) else [f"{prefix}{k}{tuple(np.shape(v))}"]
        return out

    for tid, venv in vec.items():
        obs, info = venv.reset(seed=0)
        for _ in range(5):
            act = np.random.uniform(-0.3, 0.3, size=(venv.num_envs, 7))
            obs, rew, term, trunc, info = venv.step(act)
        print(
            f"[2] task {tid} '{suite.get_task(tid).language}': {', '.join(shapes(obs))}; "
            f"is_success after 5 random steps: {info.get('is_success')}"
        )
        venv.close()

    # 3. every task: start state, success false, forced goal, success true through LiberoEnv.step
    ids = [int(x) for x in a.tasks.split(",")] if a.tasks else list(range(n))
    bad = []
    n_runs = 0
    for tid in ids:
        for k in range(a.states):
            env = le.LiberoEnv(
                task_suite=suite,
                task_id=tid,
                task_suite_name="libero_ext",
                camera_name="agentview_image,robot0_eye_in_hand_image",
                init_states=True,
                observation_height=128,
                observation_width=128,
                episode_index=k * 17,
            )
            obs, info = env.reset()
            _, _, _, _, info = env.step(np.array(sl.NOOP, dtype=float))
            start_ok = not info["is_success"]
            forced = sl.force_goal(env._env)
            _, _, term, _, info2 = env.step(np.array(sl.NOOP, dtype=float))
            n_runs += 1
            if not (start_ok and bool(info2["is_success"])):
                bad.append((tid, k * 17, suite.get_task(tid).language, start_ok, forced, info2["is_success"]))
            env.close()
    print(
        f"[3] {n_runs - len(bad)} of {n_runs} runs ({len(ids)} tasks x {a.states} stored start states): success "
        f"false from the start state, true once the goal is forced"
    )
    for b in bad:
        print("    FAILED", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
