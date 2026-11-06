"""Shared helpers for rebuilding the LIBERO datasets used by the memory experiments.

See scripts/datasets/build_libero_suites.py and build_libero_90.py.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from lerobot.datasets import dataset_tools as dt
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def subset_keep_task_table(
    src: LeRobotDataset, episodes: list[int], out_dir: Path, repo_id: str
) -> LeRobotDataset:
    """Copy `episodes` of `src` into a new dataset at `out_dir`, keeping the FULL source task table.

    Same copy path as `dataset_tools.split_dataset` (data/videos copied, episodes renumbered from 0,
    normalisation stats aggregated from the kept episodes' per-episode stats), with one difference:
    the new task table is the source's, in source order, so every task keeps its task_index.
    `split_dataset` instead rebuilds the table from the tasks present, in `set()` order, which would
    silently reorder a sequential task sequence given as task indices (--online_task_ids).
    """
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


def hub_revision(repo_id: str) -> str | None:
    """Current commit sha of a hub dataset, for provenance (None when offline)."""
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo_id).sha
    except Exception:
        return None
