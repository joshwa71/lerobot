#!/usr/bin/env python
"""Expert demonstrations for the libero_ext suite (projects/dataset_gen.md, §3), without images.

For each task: episodes from fresh resets (LIBERO's own start-state variation: objects anywhere in their init
regions, the arm at home; not the stored evaluation start states), two in three plain and one in three with DART
noise (the recorded action is the expert's label, the executed one carries the noise), kept only if they succeed, until --demos successes or --max-tries episodes. Episodes end at the first
step where the goal holds, as LIBERO's evaluation episodes do.

One mode per task (dataset_gen.md §3c-3d): demos that differ in a discrete choice (which way the hand is turned to
grasp or release) give the policy modes it cannot choose between. The task's grasp orientation is fixed before its
episodes (expert.grasp_reference), and each successful episode's mode signature (yaw_signature: the wrist yaw at
each change of hand orientation between gripper events) is compared with the others': only episodes in the most
common mode are kept. meta records the share of successes in that mode before filtering (mode_share; below ~0.9
means the expert still makes the choice inconsistently, worth a look).

One compressed npz per task in --out:

  qpos_k      T x nq float64   simulator positions before each action (the renderer sets these to draw frames)
  state_k     T x 8  float32   [eef pos, eef axis-angle, gripper qpos], as LeRobot's LIBERO processor builds it
  action_k    T x 7  float32   the expert's action (the training label)
  executed_k  T x 7  float32   what was sent (label plus DART noise)
  init_k      full flattened simulator state at t=0 (for replay checks)
  wide_k      1 if the episode had DART noise (the name is historical: it once also meant a widened start)
  meta        json: task index, name, language, bddl file, nq, joint names, success counts, grasp_ref, mode_share,
              mode (the kept signature, degrees)

Images are rendered on a cluster GPU node from qpos (scripts/datasets/build_libero_ext.py). Resumes: tasks whose
npz already holds --demos episodes are skipped.

  python scripts/libero_tasks/generate_demos.py --out outputs/libero_ext_demos [--workers 6] [--demos 50]
"""

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expert as ex  # noqa: E402
import scene_lib as sl  # noqa: E402

EXT = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "envs" / "libero_ext"


def state8(obs):
    from robosuite.utils import transform_utils as tu

    return np.concatenate(
        [obs["robot0_eef_pos"], tu.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


YAW_TOL = 20.0  # degrees: two episodes are in the same mode if every hand orientation differs by less than this
EVAL_WAIT_STEPS = 10  # lerobot's LiberoEnv: no-op steps after each reset (num_steps_wait)


def yaw_signature(state, action):
    """The episode's discrete hand-orientation choices: the wrist yaw (degrees from the start pose, about world z)
    at each gripper close or open, with consecutive events within 15 degrees merged (a retry or a deliberate
    handle miss at the same orientation is not a new choice)."""
    from robosuite.utils import transform_utils as tu

    r0 = tu.quat2mat(tu.axisangle2quat(state[0, 3:6]))
    g = action[:, 6] > 0
    events = [0] + [t for t in range(1, len(g)) if g[t] != g[t - 1]]
    sig = []
    for t in events[1:]:
        r = tu.quat2mat(tu.axisangle2quat(state[t, 3:6])) @ r0.T
        y = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
        if not sig or abs((y - sig[-1] + 180) % 360 - 180) >= 15.0:
            sig.append(y)
    return sig


def same_mode(a, b):
    return len(a) == len(b) and all(abs((x - y + 180) % 360 - 180) < YAW_TOL for x, y in zip(a, b, strict=True))


def dominant(sigs):
    """Indices of the episodes in the most common mode (the signature with the most episodes within YAW_TOL)."""
    best = []
    for s in sigs:
        members = [j for j, t in enumerate(sigs) if same_mode(s, t)]
        if len(members) > len(best):
            best = members
    return best


def run_task(job):
    idx, task, classes, out_dir, n_demos, max_tries, seed = job
    path = Path(out_dir) / f"{idx:02d}_{task['name'][:80]}.npz"
    if path.exists():
        with np.load(path) as z:
            if json.loads(str(z["meta"]))["n_demos"] >= n_demos:
                return idx, task["language"], "exists", None
    t0 = time.time()
    rng = np.random.default_rng(seed)
    env = sl.make_env(str(EXT / "files" / task["bddl_file"]))
    e = env.env
    m = e.sim.model._model
    grasp_ref = ex.grasp_reference(env, task, classes, np.random.default_rng(seed + 1))
    episodes, why = [], {}  # every success: (arrays, signature)
    tries = 0
    keep = []
    while len(keep) < n_demos and tries < max_tries:
        tries += 1
        # start states as LIBERO's: the suite's own reset (objects anywhere in their ~3 cm init regions, the arm at
        # its home pose); no widened starts. One episode in three carries DART noise on the executed actions.
        wide = tries % 3 == 0
        sl.stable_reset(env)
        init = e.sim.get_state().flatten().copy()
        # start exactly as an evaluation episode does (lerobot's LiberoEnv.reset): a fresh reset, the start state,
        # then 10 unrecorded no-op steps with the gripper commanded open. robosuite integrates the gripper command
        # from zero after a reset, so without this every demo's first frame showed the fingers at 0.040, where an
        # evaluation starts at 0.039, the opening they also have just after a release; the task 56 specialist took
        # the evaluation's start for "the first step is done" and went straight to the second object (§3d).
        env.reset()
        env.set_init_state(init)
        for _ in range(EVAL_WAIT_STEPS):
            env.step(np.array(sl.NOOP, dtype=float))
        exp = ex.Expert(env, task, classes, rng, dart_sigma=0.15 if wide else 0.0, grasp_ref=grasp_ref)
        qpos, states = [], []
        orig = exp._step

        def step(label, executed=None, orig=orig, qpos=qpos, states=states, exp=exp):
            if len(exp.labels) < exp.max_len:  # (else orig raises OutOfTime without stepping)
                qpos.append(e.sim.data.qpos.copy())
                states.append(state8(exp.obs()))
            orig(label, executed)

        exp._step = step
        try:
            ok, n = exp.run()
        except Exception as err:  # noqa: BLE001
            ok, n = False, 0
            exp.why = f"error {type(err).__name__}"
            if os.environ.get("GEN_DEBUG"):
                traceback.print_exc()
        if not ok:
            why[exp.why[:50]] = why.get(exp.why[:50], 0) + 1
            continue
        assert len(qpos) == len(exp.labels) == len(exp.executed)
        ep = {
            "qpos": np.array(qpos),
            "state": np.array(states, dtype=np.float32),
            "action": np.array(exp.labels, dtype=np.float32),
            "executed": np.array(exp.executed, dtype=np.float32),
            "init": init,
            "wide": np.array(int(wide)),
        }
        episodes.append((ep, yaw_signature(ep["state"], ep["action"])))
        keep = dominant([sig for _, sig in episodes])
    env.close()
    keep = keep[:n_demos]
    arrays = {}
    for k, j in enumerate(keep):
        for name, val in episodes[j][0].items():
            arrays[f"{name}_{k}"] = val
    k = len(keep)
    n_wide = sum(int(episodes[j][0]["wide"]) for j in keep)
    meta = {
        "task_index": idx,
        "name": task["name"],
        "language": task["language"],
        "bddl_file": task["bddl_file"],
        "nq": int(m.nq),
        "joints": [m.joint(j).name for j in range(m.njnt)],
        "n_demos": k,
        "tries": tries,
        "n_wide": n_wide,
        "n_success": len(episodes),
        "mode_share": round(len(dominant([sig for _, sig in episodes])) / max(1, len(episodes)), 3),
        "mode": [round(y) for y in episodes[keep[0]][1]] if keep else [],
        "grasp_ref": grasp_ref,
        "fail_reasons": why,
        "lengths": [int(len(arrays[f"action_{j}"])) for j in range(k)],
        "secs": round(time.time() - t0),
    }
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
    os.replace(tmp, path)
    return idx, task["language"], f"{k}/{tries}", meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--demos", type=int, default=50)
    ap.add_argument("--max-tries", type=int, default=200)
    ap.add_argument("--tasks", default=None, help="comma-separated task indices (default: all)")
    a = ap.parse_args()
    suite = json.loads((EXT / "suite.json").read_text())
    builds = {r["scene"]: r for r in json.loads((EXT / "build_results.json").read_text())}
    want = None if a.tasks is None else {int(x) for x in a.tasks.split(",")}
    os.makedirs(a.out, exist_ok=True)
    jobs = [
        (i, t, builds[t["scene"]]["classes"], a.out, a.demos, a.max_tries, 10_000 + i)
        for i, t in enumerate(suite["tasks"])
        if want is None or i in want
    ]
    # longest (two-step) tasks first, so the pool does not end on one long straggler
    jobs.sort(key=lambda j: -j[1]["kind"])
    with Pool(min(a.workers, len(jobs))) as pool:
        for idx, lang, res, meta in pool.imap_unordered(run_task, jobs, chunksize=1):
            extra = ""
            if meta:
                ln = meta["lengths"]
                extra = (f"  len {np.mean(ln):.0f} (max {max(ln) if ln else 0})  {meta['secs']}s  "
                         f"mode {meta['mode_share']:.2f} {meta['mode']}  {meta['fail_reasons'] or ''}")
            print(f"{idx:2d} {res:>8s}  {lang}{extra}", flush=True)


if __name__ == "__main__":
    main()
