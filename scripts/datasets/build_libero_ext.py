"""Build the libero_ext LeRobot dataset from the scripted expert's demos (scripts/libero_tasks/generate_demos.py).

The demos hold the simulator positions before every action. This sets each one and reads the environment's own
observation function (OffScreenRenderEnv at 256 x 256, as lerobot's LiberoEnv), so the frames are what an
evaluation feeds the policy in that state. The dataset copies libero_10's format (projects/dataset_gen.md, §3):
  observation.images.image / image2   agentview / wrist camera, stored rotated 180 degrees from the raw render
                                      (the HuggingFaceVLA/libero convention LeRobot's LIBERO processor undoes)
  observation.state                   8: eef pos, eef axis-angle, gripper qpos
  action                              7: the expert's action (OSC_POSE delta + gripper), the training label
  fps 10 (metadata, as libero_10; the simulator runs at 20 Hz), task_index = libero_ext suite order.

Resumable: every task is first written as its own dataset under --work (built in a temp dir and renamed into
place when complete); then all are aggregated into --out in suite order, and quantile stats (pi05) are added.

  python scripts/datasets/build_libero_ext.py --demos outputs/libero_ext_demos --work outputs/_build_libero_ext \
      --out outputs/libero_ext [--workers 4]
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "libero_tasks"))
EXT = REPO / "src" / "lerobot" / "envs" / "libero_ext"
REPO_ID = "libero_ext"
FEATURES = {
    "observation.images.image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}


def state8(obs):
    from robosuite.utils import transform_utils as tu

    return np.concatenate(
        [obs["robot0_eef_pos"], tu.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def build_task(job):
    npz, work = job
    import scene_lib as sl
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    z = np.load(npz)
    meta = json.loads(str(z["meta"]))
    idx = meta["task_index"]
    final = Path(work) / f"task_{idx:02d}"
    if (final / "meta" / "info.json").exists():
        return idx, "exists", 0.0
    tmp = Path(work) / f".task_{idx:02d}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    t0 = time.time()
    env = sl.make_env(str(EXT / "files" / meta["bddl_file"]), render=True, size=256)
    e = env.env
    m = e.sim.model._model
    assert m.nq == meta["nq"] and [m.joint(j).name for j in range(m.njnt)] == meta["joints"], (
        f"task {idx}: simulator model differs from the one the demos were generated with"
    )
    ds = LeRobotDataset.create(
        repo_id=f"{REPO_ID}_task{idx:02d}",
        fps=10,
        features=FEATURES,
        root=tmp,
        robot_type="panda",
        use_videos=False,
        image_writer_threads=4,
    )
    worst = 0.0
    for k in range(meta["n_demos"]):
        env.reset()
        qpos, state, action = z[f"qpos_{k}"], z[f"state_{k}"], z[f"action_{k}"]
        for t in range(len(action)):
            e.sim.data.qpos[:] = qpos[t]
            e.sim.forward()
            obs = e._get_observations(force_update=True)
            s = state8(obs)
            worst = max(worst, float(np.abs(s - state[t]).max()))
            ds.add_frame(
                {
                    "observation.images.image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                    "observation.images.image2": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
                    "observation.state": state[t],
                    "action": action[t],
                    "task": meta["language"],
                }
            )
        ds.save_episode()
    ds.finalize()
    env.close()
    os.replace(tmp, final)
    print(f"task {idx:2d}: {meta['n_demos']} episodes, {sum(meta['lengths'])} frames, "
          f"{time.time() - t0:.0f}s, max state difference {worst:.2e}", flush=True)
    return idx, "built", worst


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stats-workers", type=int, default=4)
    ap.add_argument("--partial", action="store_true", help="test builds: only the tasks whose demos are present")
    ap.add_argument("--stage", choices=["all", "tasks", "aggregate", "subset"], default="all",
                    help="tasks: per-task datasets only; aggregate: merge them (all built) and add stats; "
                    "subset: merge only --subset tasks into --out (a small dataset for quick checks)")
    ap.add_argument("--subset", default=None, help="comma-separated suite task indices, for --stage subset")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--drop-work", action="store_true", help="delete the per-task datasets once merged (scratch)")
    a = ap.parse_args()
    files = sorted(a.demos.glob("*.npz"))
    metas = [json.loads(str(np.load(f)["meta"])) for f in files]
    order = sorted(range(len(files)), key=lambda i: metas[i]["task_index"])
    suite = json.loads((EXT / "suite.json").read_text())
    idxs = [metas[i]["task_index"] for i in order]
    assert a.partial or idxs == list(range(len(suite["tasks"]))), "demos missing for some tasks"
    for i in order:
        assert metas[i]["language"] == suite["tasks"][metas[i]["task_index"]]["language"], "demos from another suite"
    a.work.mkdir(parents=True, exist_ok=True)
    if a.stage in ("all", "tasks"):
        jobs = [(str(files[i]), str(a.work)) for i in order]
        with Pool(a.workers) as pool:
            results = list(pool.imap_unordered(build_task, jobs, chunksize=1))
        worst = max(r[2] for r in results)
        print(f"all tasks built; max state difference (stored vs re-read) {worst:.2e}", flush=True)
    # The demos' states are robosuite's cached observations, taken a few physics substeps before the end of each
    # control step (as an evaluation sees them), so re-reading from the final positions differs by < 1 mm; a
    # model mismatch would show as centimetres.
        assert worst < 5e-3, "re-read states differ from the demos' own: model or state mismatch"
        if a.stage == "tasks":
            return
    if a.stage == "subset":
        idxs = [int(x) for x in a.subset.split(",")]
    missing = [i for i in idxs if not (a.work / f"task_{i:02d}" / "meta" / "info.json").exists()]
    assert not missing, f"per-task datasets missing: {missing}"

    if not (a.out / "meta" / "info.json").exists():
        from lerobot.datasets.aggregate import aggregate_datasets

        tmp_out = a.out.parent / f".{a.out.name}.tmp"
        shutil.rmtree(tmp_out, ignore_errors=True)
        aggregate_datasets(
            repo_ids=[f"{REPO_ID}_task{i:02d}" for i in idxs],
            aggr_repo_id=a.repo_id,
            roots=[a.work / f"task_{i:02d}" for i in idxs],
            aggr_root=tmp_out,
        )
        os.replace(tmp_out, a.out)
    if a.drop_work and a.stage != "subset":
        for i in idxs:
            shutil.rmtree(a.work / f"task_{i:02d}", ignore_errors=True)
    from libero_common import add_quantile_stats, write_build_info

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    add_quantile_stats(a.out, a.repo_id, workers=a.stats_workers)
    md = LeRobotDatasetMetadata(a.repo_id, root=a.out)
    tasks = list(md.tasks.index)
    assert tasks == [suite["tasks"][i]["language"] for i in idxs], "task_index is not the suite order"
    shutil.copy(EXT / "suite.json", a.out / "meta" / "libero_ext_suite.json")
    write_build_info(
        a.out,
        source="scripted expert demos (scripts/libero_tasks/generate_demos.py)",
        script="scripts/datasets/build_libero_ext.py",
        demos_dir=str(a.demos),
        episodes=md.total_episodes,
        frames=md.total_frames,
        tasks=len(tasks),
        suite_task_indices=idxs,
        demos_per_task=[next(m["n_demos"] for m in metas if m["task_index"] == i) for i in idxs],
    )
    print(f"done: {md.total_episodes} episodes, {md.total_frames} frames, {len(tasks)} tasks -> {a.out}")


if __name__ == "__main__":
    main()
