#!/usr/bin/env python
"""Pilot of the scripted expert (expert.py) on selected libero_ext tasks: success rate, episode lengths, and
optionally a video of one episode per task.

Episodes start from fresh resets (not the stored evaluation start states); half use widened starts and DART noise.

  python scripts/libero_tasks/pilot_expert.py --tasks "put the popcorn in the basket,lift the milk" --episodes 10
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expert as ex  # noqa: E402
import scene_lib as sl  # noqa: E402

EXT = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "envs" / "libero_ext"


def run_task(args):
    task, classes, n, seed, video_dir = args
    rng = np.random.default_rng(seed)
    env = sl.make_env(str(EXT / "files" / task["bddl_file"]))
    res = []
    for ep in range(n):
        wide = ep % 2 == 1
        sl.stable_reset(env)
        if wide and not ex.widen_start(env, rng):
            sl.stable_reset(env)
            wide = False
        exp = ex.Expert(env, task, classes, rng, dart_sigma=0.15 if wide else 0.0)
        try:
            ok, n_steps = exp.run()
        except Exception as e:  # noqa: BLE001
            ok, n_steps = False, len(exp.labels)
            print(f"    {task['language']}: episode {ep} error {type(e).__name__}: {e}")
        res.append({"ok": ok, "len": n_steps, "wide": wide, "why": "" if ok else exp.why, "eef": np.array(exp.eef).tolist() if ok else None,
                    "grip": [float(a[6]) for a in exp.labels] if ok else None})
    env.close()
    if video_dir:
        venv = sl.make_env(str(EXT / "files" / task["bddl_file"]), render=True, size=256)
        sl.stable_reset(venv)
        exp = ex.Expert(venv, task, classes, np.random.default_rng(seed + 1))
        frames = []
        orig = exp._step

        def step_and_grab(label):
            orig(label)
            frames.append(venv.env.sim.render(camera_name="agentview", height=256, width=256)[::-1, ::-1])

        exp._step = step_and_grab
        ok, _ = exp.run()
        import imageio

        name = task["name"][:60]
        imageio.mimwrite(os.path.join(video_dir, f"{name}{'' if ok else '_FAIL'}.mp4"), frames[::2], fps=10)
        venv.close()
    return task["language"], res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="comma-separated task instructions (or 'all')")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--out", default=None, help="save per-episode results (json)")
    a = ap.parse_args()
    suite = json.loads((EXT / "suite.json").read_text())
    builds = {r["scene"]: r for r in json.loads((EXT / "build_results.json").read_text())}
    want = None if a.tasks == "all" else {x.strip() for x in a.tasks.split(",")}
    jobs = []
    for i, t in enumerate(suite["tasks"]):
        if want is None or t["language"] in want:
            jobs.append((t, builds[t["scene"]]["classes"], a.episodes, 1000 + i, a.video_dir))
    if a.video_dir:
        os.makedirs(a.video_dir, exist_ok=True)
    with Pool(min(a.workers, len(jobs))) as pool:
        out = pool.map(run_task, jobs, chunksize=1)
    for lang, res in out:
        ok = [r["ok"] for r in res]
        lens = [r["len"] for r in res if r["ok"]]
        wide_ok = [r["ok"] for r in res if r["wide"]]
        print(f"{np.mean(ok) * 100:5.0f}%  ({sum(ok)}/{len(ok)}; widened starts {sum(wide_ok)}/{len(wide_ok)})  "
              f"len {np.mean(lens) if lens else 0:.0f}+-{np.std(lens) if lens else 0:.0f}  {lang}")
        for r in res:
            if not r["ok"]:
                print(f"         fail{' (widened)' if r['wide'] else ''}: {r['why']}")
    if a.out:
        Path(a.out).write_text(json.dumps({lang: res for lang, res in out}))


if __name__ == "__main__":
    main()
