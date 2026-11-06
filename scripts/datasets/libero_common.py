"""Shared helpers for rebuilding the LIBERO datasets used by the memory experiments.

See scripts/datasets/build_libero_suites.py and build_libero_90.py.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from lerobot.datasets import dataset_tools as dt
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def subset_keep_task_table(
    src_meta: LeRobotDatasetMetadata, episodes: list[int], out_dir: Path, repo_id: str
) -> LeRobotDataset:
    """Copy `episodes` of a source dataset into a new one at `out_dir`, keeping the FULL task table.

    Same copy path as `dataset_tools.split_dataset` (data/videos copied, episodes renumbered from 0,
    normalisation stats aggregated from the kept episodes' per-episode stats), with one difference:
    the new task table is the source's, in source order, so every task keeps its task_index.
    `split_dataset` instead rebuilds the table from the tasks present, in `set()` order, which would
    silently reorder a sequential task sequence given as task indices (--online_task_ids).

    Only the source's metadata is loaded (the copy helpers read files by path), so this works on a
    partially downloaded source as long as the selected episodes' files are present.
    """
    src = SimpleNamespace(meta=src_meta, root=src_meta.root)
    meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=src.meta.fps,
        features=src.meta.features,
        robot_type=src.meta.robot_type,
        root=out_dir,
        use_videos=len(src.meta.video_keys) > 0,
        chunks_size=src.meta.chunks_size,
        data_files_size_in_mb=src.meta.data_files_size_in_mb,
        video_files_size_in_mb=src.meta.video_files_size_in_mb,
    )
    meta.save_episode_tasks(list(src.meta.tasks.sort_values("task_index").index))

    episode_mapping = {old: new for new, old in enumerate(sorted(episodes))}
    video_metadata = None
    if src.meta.video_keys:
        video_metadata = dt._copy_and_reindex_videos(src, meta, episode_mapping)
    data_metadata = dt._copy_and_reindex_data(src, meta, episode_mapping)
    dt._copy_and_reindex_episodes_metadata(src, meta, episode_mapping, data_metadata, video_metadata)
    return LeRobotDataset(repo_id=repo_id, root=out_dir)


def repaired_source(src_root: Path, work_root: Path) -> Path:
    """Return a view of `src_root` whose meta/episodes points at the data files episodes are really in.

    HuggingFaceVLA/libero's meta/episodes (re-uploaded 26 Sep 2025, same on every revision) records
    data/chunk_index + data/file_index for an older ~500 MB sharding, while the data is in ~100 MB
    files; its global dataset_from/to_index and lengths are still right. Normal loading goes by
    global index and doesn't notice, but dataset_tools reads episodes by file and fails. This makes
    `work_root` with a corrected copy of meta/ and symlinks to the source data/ (and videos/), after
    checking every episode's global range and length against the data. Episodes whose data file is
    not present (partial download) keep their stale entry and must not be selected.
    """
    import shutil

    import pandas as pd

    if work_root.exists():
        raise FileExistsError(f"{work_root} exists; remove it to rebuild the repaired source")
    work_root.mkdir(parents=True)
    shutil.copytree(src_root / "meta", work_root / "meta")
    for d in ("data", "videos"):
        if (src_root / d).exists():
            (work_root / d).symlink_to((src_root / d).resolve())

    actual = {}  # episode -> (chunk, file, from_index, to_index, n_frames)
    for p in sorted((src_root / "data").glob("chunk-*/file-*.parquet")):
        chunk, file = int(p.parent.name.split("-")[1]), int(p.stem.split("-")[1])
        df = pd.read_parquet(p, columns=["episode_index", "index"])
        for ep, g in df.groupby("episode_index"):
            assert ep not in actual, f"episode {ep} is split across data files"
            actual[int(ep)] = (chunk, file, int(g["index"].min()), int(g["index"].max()) + 1, len(g))

    changed = 0
    for f in sorted((work_root / "meta/episodes").glob("chunk-*/file-*.parquet")):
        m = pd.read_parquet(f)
        for i, ep in enumerate(m["episode_index"]):
            if ep not in actual:
                continue
            chunk, file, lo, hi, n = actual[ep]
            row = (m.at[i, "dataset_from_index"], m.at[i, "dataset_to_index"], m.at[i, "length"])
            assert row == (lo, hi, n), f"episode {ep}: meta range/length {row} != data {(lo, hi, n)}"
            changed += int((m.at[i, "data/chunk_index"], m.at[i, "data/file_index"]) != (chunk, file))
            m.at[i, "data/chunk_index"], m.at[i, "data/file_index"] = chunk, file
        m.to_parquet(f, index=False)
    print(f"repaired source: {len(actual)} episodes located in data files, {changed} file indices fixed")
    return work_root


def write_build_info(out_dir: Path, **fields) -> None:
    """Record how a dataset was built (source, revision, script, commit) next to its meta/."""
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    info = {"built_at": datetime.now(timezone.utc).isoformat(), "repo_commit": commit, **fields}
    (out_dir / "BUILD_INFO.json").write_text(json.dumps(info, indent=2) + "\n")


def hub_revision(repo_id: str, local_dir: Path | None = None) -> str | None:
    """Commit sha of a hub dataset, for provenance.

    Prefers the revision an `hf download --local-dir` copy actually came from (first line of its
    .cache/huggingface/download/**/*.metadata), else asks the hub; None if neither is available.
    """
    if local_dir is not None:
        for md in sorted((Path(local_dir) / ".cache/huggingface/download").rglob("*.metadata"))[:1]:
            return md.read_text().splitlines()[0].strip()
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo_id).sha
    except Exception:
        return None


def add_quantile_stats(root: Path, repo_id: str) -> None:
    """Recompute stats.json with quantiles (q01..q99), which pi05's quantile normalisation needs.

    Same computation as lerobot's scripts/augment_dataset_quantile_stats.py (per-episode stats over
    all frames, count-weighted aggregate), minus that script's unconditional push_to_hub + tagging.
    """
    from lerobot.datasets import write_stats
    from lerobot.scripts.augment_dataset_quantile_stats import compute_quantile_stats_for_dataset

    ds = LeRobotDataset(repo_id, root=root)
    write_stats(compute_quantile_stats_for_dataset(ds), root)
