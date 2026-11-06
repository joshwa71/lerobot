"""Rebuild libero_90 (LeRobot v3.0) from jesbu1/libero_90_lerobot (LeRobot v2.0).

The source has 3959 episodes / 73 task strings (LIBERO-90 with no-op actions removed), images as PNG
in parquet, and NO normalisation stats. Steps, each into its own directory under --work:

  1. stage_v21  -- rename features to the HuggingFaceVLA/libero names our libero_10 uses
                   (image -> observation.images.image, wrist_image -> observation.images.image2,
                    state -> observation.state, actions -> action), compute per-episode stats with
                   lerobot's compute_episode_stats (incl. quantiles), write meta/episodes_stats.jsonl
                   and mark the dataset v2.1. Every episode is validated; problems go to
                   validation.json (candidates for the "dead episodes" stripped on Nebius).
  2. convert    -- lerobot's official convert_dataset_v21_to_v30 on a copy of the stage.
  3. finalise   -- copy to --out, dropping any --drop-episodes (task table kept, stats re-aggregated).

    python scripts/datasets/build_libero_90.py --src outputs/hf_src/jesbu1/libero_90_lerobot \
        --work outputs/_build_libero_90 --out outputs/libero_90 [--workers 16] [--drop-episodes 12 345]
"""

import argparse
import io
import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import serialize_dict
from libero_common import hub_revision, subset_keep_task_table, write_build_info

SRC_REPO = "jesbu1/libero_90_lerobot"
RENAME = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.image2",
    "state": "observation.state",
    "actions": "action",
}
FPS = 10


def _stage_episode(job):
    """Rename columns, validate, and compute per-episode stats for one source episode file."""
    src_path, dst_path, meta_ep, task_to_index, features = job
    t = pq.read_table(src_path).rename_columns([RENAME.get(c, c) for c in pq.read_schema(src_path).names])
    # drop the datasets-library schema metadata: it still names the old columns
    t = t.replace_schema_metadata(None)
    df = t.to_pandas()
    ep = meta_ep["episode_index"]
    problems = []

    n = len(df)
    if n == 0:
        return ep, None, ["empty"]
    if n != meta_ep["length"]:
        problems.append(f"rows {n} != episodes.jsonl length {meta_ep['length']}")
    if not (df["episode_index"] == ep).all():
        problems.append("episode_index column != file's episode")
    if not np.array_equal(df["frame_index"].to_numpy(), np.arange(n)):
        problems.append("frame_index not 0..n-1")
    if not np.allclose(df["timestamp"].to_numpy(), np.arange(n) / FPS, atol=1e-4):
        problems.append("timestamps not frame_index/fps")
    expected_task = task_to_index[meta_ep["tasks"][0]]
    if not (df["task_index"] == expected_task).all():
        problems.append(f"task_index {sorted(df['task_index'].unique())} != {expected_task}")

    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        problems.append("non-finite state/action")
    if np.abs(action).max() > 1.0 + 1e-6:
        problems.append(f"|action| max {np.abs(action).max():.3f} > 1")
    eef_travel = float(np.linalg.norm(state[:, :3].max(0) - state[:, :3].min(0)))
    if eef_travel < 0.02:
        problems.append(f"end-effector barely moves ({eef_travel * 100:.1f} cm)")
    notes = []  # informational, not grounds for dropping on their own
    if not (action[:, 6] > 0).any():
        notes.append("gripper never closes")  # normal for push/turn tasks

    images = {}
    for key in ("observation.images.image", "observation.images.image2"):
        cells = df[key].to_numpy()
        images[key] = [io.BytesIO(c["bytes"]) for c in cells]
        try:
            from PIL import Image

            im = Image.open(io.BytesIO(cells[-1]["bytes"]))
            if im.size != (256, 256):
                problems.append(f"{key} size {im.size}")
        except Exception as e:  # noqa: BLE001
            problems.append(f"{key} undecodable: {e!r}")

    numeric = {
        "observation.state": state,
        "action": action,
        "timestamp": df["timestamp"].to_numpy(),
        "frame_index": df["frame_index"].to_numpy(),
        "episode_index": df["episode_index"].to_numpy(),
        "index": df["index"].to_numpy(),
        "task_index": df["task_index"].to_numpy(),
    }
    stats = compute_episode_stats({**numeric, **images}, features)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, dst_path)
    summary = {"length": n, "eef_travel_m": round(eef_travel, 4), "notes": notes}
    return ep, {"episode_index": ep, "stats": serialize_dict(stats), **summary}, problems


def stage_v21(src: Path, stage: Path, workers: int) -> dict:
    if stage.exists():
        raise FileExistsError(f"{stage} exists; remove it to re-stage")
    info = json.loads((src / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.0", info["codebase_version"]
    info["features"] = {RENAME.get(k, k): v for k, v in info["features"].items()}
    info["codebase_version"] = "v2.1"

    (stage / "meta").mkdir(parents=True)
    (stage / "meta/info.json").write_text(json.dumps(info, indent=4))
    for name in ("tasks.jsonl", "episodes.jsonl"):
        shutil.copy(src / "meta" / name, stage / "meta" / name)
    tasks = [json.loads(line) for line in (src / "meta/tasks.jsonl").read_text().splitlines()]
    task_to_index = {t["task"]: t["task_index"] for t in tasks}
    episodes = [json.loads(line) for line in (src / "meta/episodes.jsonl").read_text().splitlines()]

    jobs = []
    for e in episodes:
        rel = info["data_path"].format(
            episode_chunk=e["episode_index"] // info["chunks_size"], episode_index=e["episode_index"]
        )
        jobs.append((src / rel, stage / rel, e, task_to_index, info["features"]))

    ep_stats, report = {}, {}
    with ProcessPoolExecutor(workers) as pool:
        for i, (ep, stats, problems) in enumerate(pool.map(_stage_episode, jobs, chunksize=8)):
            if stats is not None:
                ep_stats[ep] = stats
            if problems:
                report[ep] = problems
            if (i + 1) % 250 == 0:
                logging.info(f"staged {i + 1}/{len(jobs)} episodes ({len(report)} flagged)")

    with open(stage / "meta/episodes_stats.jsonl", "w") as f:
        for ep in sorted(ep_stats):
            f.write(json.dumps({"episode_index": ep, "stats": ep_stats[ep]["stats"]}) + "\n")
    summary = {
        "episodes": len(episodes),
        "flagged": report,
        "lengths": {ep: s["length"] for ep, s in ep_stats.items()},
        "eef_travel_m": {ep: s["eef_travel_m"] for ep, s in ep_stats.items()},
        "notes": {ep: s["notes"] for ep, s in ep_stats.items() if s["notes"]},
    }
    (stage.parent / "validation.json").write_text(json.dumps(summary, indent=1))
    logging.info(f"stage done: {len(ep_stats)} episodes, {len(report)} flagged -> validation.json")
    return report


def convert(stage: Path, converted: Path) -> None:
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

    if converted.exists():
        raise FileExistsError(f"{converted} exists; remove it to re-convert")
    shutil.copytree(stage, converted)  # the converter rearranges <root>, <root>_old, <root>_v30
    convert_dataset(repo_id="libero_90", root=converted, push_to_hub=False)
    shutil.rmtree(converted.parent / f"{converted.name}_old")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--steps", nargs="+", default=["stage", "convert", "finalise"],
                   choices=["stage", "convert", "finalise"])
    p.add_argument("--drop-episodes", type=int, nargs="*", default=[],
                   help="source episode indices to leave out of the final dataset")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    stage, converted = args.work / "stage_v21", args.work / "converted_v30"
    if "stage" in args.steps:
        stage_v21(args.src, stage, args.workers)
    if "convert" in args.steps:
        convert(stage, converted)
    if "finalise" in args.steps:
        ds = LeRobotDataset("libero_90", root=converted)
        keep = [e for e in range(ds.meta.total_episodes) if e not in set(args.drop_episodes)]
        logging.info(f"finalising {len(keep)}/{ds.meta.total_episodes} episodes -> {args.out}")
        out = subset_keep_task_table(ds, keep, args.out, repo_id="libero_90")
        assert out.meta.total_episodes == len(keep)
        write_build_info(
            args.out,
            source=SRC_REPO,
            source_revision=hub_revision(SRC_REPO),
            script="scripts/datasets/build_libero_90.py",
            renamed_features=RENAME,
            dropped_source_episodes=sorted(args.drop_episodes),
        )


if __name__ == "__main__":
    main()
