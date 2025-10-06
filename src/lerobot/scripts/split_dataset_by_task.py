#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_FEATURES,
    DEFAULT_VIDEO_PATH,
    INFO_PATH,
    create_empty_dataset_info,
    get_hf_features_from_features,
    load_episodes,
    to_parquet_with_hf_images,
    write_info,
    write_stats,
    write_tasks,
    update_chunk_file_indices,
)


def _ensure_empty_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Destination already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _find_task_name(tasks_df: pd.DataFrame, task_index: int) -> str:
    # Map the provided numeric task_index to its task name using the dedicated column
    try:
        matches = tasks_df[tasks_df["task_index"].astype(int) == int(task_index)]
        if len(matches) != 1:
            raise ValueError
        return matches.index[0]
    except Exception as e:
        raise ValueError(f"Invalid task index {task_index}") from e


def _filter_episode_indices_for_task(src_meta: LeRobotDatasetMetadata, task_name: str) -> List[int]:
    # Load episodes (without stats) and select those that include the task_name in 'tasks' list
    eps = load_episodes(src_meta.root)
    df = eps.to_pandas()
    mask = df["tasks"].apply(lambda lst: task_name in lst)
    ep_indices = df.loc[mask, "episode_index"].astype(int).tolist()
    ep_indices.sort()
    return ep_indices


def _read_episode_stats_rows(src_root: Path, episodes_to_keep: List[int]) -> List[dict]:
    # Iterate over all meta/episodes parquet files and extract rows for selected episodes, including stats/* columns
    rows: List[dict] = []
    meta_dir = src_root / "meta/episodes"
    if not meta_dir.exists():
        return rows

    for chunk_dir in sorted(meta_dir.glob("chunk-*")):
        for file_path in sorted(chunk_dir.glob("file-*.parquet")):
            df = pd.read_parquet(file_path)
            if "episode_index" not in df.columns:
                continue
            sub = df[df["episode_index"].isin(episodes_to_keep)]
            if len(sub) == 0:
                continue
            rows.extend(sub.to_dict(orient="records"))
    return rows


def _aggregate_stats_from_episode_rows(rows: List[dict], features: dict) -> dict:
    # Build list[dict] of per-episode stats to feed into aggregate_stats
    stats_list: List[dict] = []
    for row in rows:
        ep_stats: Dict[str, Dict[str, np.ndarray]] = {}
        for key, value in row.items():
            if not isinstance(key, str) or not key.startswith("stats/"):
                continue
            # key format: stats/<feature_key>/<stat>
            parts = key.split("/")
            if len(parts) != 3:
                continue
            _, fkey, stat = parts
            if fkey not in ep_stats:
                ep_stats[fkey] = {}
            # Values are stored as scalars or lists; normalize shapes per compute_stats expectations
            if stat == "count":
                # ensure shape (1,)
                if isinstance(value, (int, float, np.integer, np.floating)):
                    arr = np.array([value])
                else:
                    arr = np.array(value)
                    if arr.ndim == 0:
                        arr = arr.reshape(1)
                ep_stats[fkey][stat] = arr
                continue

            # Non-count stats -> coerce to numeric ndarray
            arr = np.asarray(value)
            if arr.dtype == object:
                arr = np.array([np.asarray(x) for x in arr])
                if arr.dtype == object:
                    arr = np.array([np.asarray(x) for x in arr])
            if arr.dtype == object:
                arr = arr.astype(np.float32, copy=False)
            else:
                arr = arr.astype(np.float32, copy=False)
            if fkey in features and features[fkey]["dtype"] in ["image", "video"]:
                # expect (3,1,1)
                if arr.ndim == 1 and arr.shape[0] == 3:
                    arr = arr.reshape(3, 1, 1)
                elif arr.ndim == 2 and arr.shape == (3, 1):
                    arr = arr.reshape(3, 1, 1)
            ep_stats[fkey][stat] = arr

        if len(ep_stats) > 0:
            stats_list.append(ep_stats)

    if len(stats_list) == 0:
        return {}
    return aggregate_stats(stats_list)


def _copy_required_video_files(
    src_meta: LeRobotDatasetMetadata,
    dst_root: Path,
    episodes_to_keep: List[int],
) -> Dict[Tuple[str, int, int], Tuple[int, int]]:
    # Build mapping from (video_key, src_chunk, src_file) -> (dst_chunk, dst_file)
    mapping: Dict[Tuple[str, int, int], Tuple[int, int]] = {}
    video_keys = src_meta.video_keys
    if len(video_keys) == 0:
        return mapping

    # Determine which video files are referenced by selected episodes
    eps = load_episodes(src_meta.root)
    ep_df = eps.to_pandas().set_index("episode_index")

    # Assign contiguous dst chunk/file indices starting at 0 per encountered unique file
    next_chunk = 0
    next_file = 0

    for ep_idx in episodes_to_keep:
        row = ep_df.loc[ep_idx]
        for key in video_keys:
            src_chunk = int(row[f"videos/{key}/chunk_index"])
            src_file = int(row[f"videos/{key}/file_index"])
            mkey = (key, src_chunk, src_file)
            if mkey not in mapping:
                mapping[mkey] = (next_chunk, next_file)
                # copy file now
                src_path = src_meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=key, chunk_index=src_chunk, file_index=src_file
                )
                dst_path = dst_root / DEFAULT_VIDEO_PATH.format(
                    video_key=key, chunk_index=next_chunk, file_index=next_file
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))

                next_chunk, next_file = update_chunk_file_indices(
                    next_chunk, next_file, DEFAULT_CHUNK_SIZE
                )

    return mapping


def _build_episode_to_files_index(src_root: Path) -> Dict[int, List[Path]]:
    """Build an index mapping episode_idx -> list of parquet files containing that episode."""
    data_dir = src_root / "data"
    if not data_dir.exists():
        return {}
    
    episode_to_files: Dict[int, List[Path]] = {}
    
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for file_path in sorted(chunk_dir.glob("file-*.parquet")):
            # Read only episode_index column for speed
            df = pd.read_parquet(file_path, columns=["episode_index"])
            unique_episodes = df["episode_index"].unique()
            for ep_idx in unique_episodes:
                ep_idx = int(ep_idx)
                if ep_idx not in episode_to_files:
                    episode_to_files[ep_idx] = []
                episode_to_files[ep_idx].append(file_path)
    
    return episode_to_files


def _load_episode_data_from_files(file_paths: List[Path], episode_idx: int) -> pd.DataFrame:
    """Load data for a specific episode from known parquet files."""
    frames = []
    for file_path in file_paths:
        df = pd.read_parquet(file_path)
        ep_frames = df[df["episode_index"] == episode_idx]
        if len(ep_frames) > 0:
            frames.append(ep_frames)
    
    if not frames:
        return pd.DataFrame()
    
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("index", inplace=True)
    return combined


def _rebuild_data_and_meta(
    src_meta: LeRobotDatasetMetadata,
    dst_root: Path,
    episodes_to_keep: List[int],
    video_mapping: Dict[Tuple[str, int, int], Tuple[int, int]],
) -> Tuple[int, int]:
    # Write data parquet files (one per episode) and build episodes metadata DataFrame
    # Returns (total_episodes, total_frames)
    eps = load_episodes(src_meta.root)
    ep_df = eps.to_pandas().set_index("episode_index")

    # Prepare destination indices
    data_chunk = 0
    data_file = 0

    # Prepare episodes metadata rows
    meta_rows: List[dict] = []

    # Track running index offset for dataset index range
    current_dataset_index = 0

    # For stats columns, we will copy them from original episodes parquet rows
    stats_rows = _read_episode_stats_rows(src_meta.root, episodes_to_keep)
    stats_by_ep = {int(r["episode_index"]): r for r in stats_rows}

    # Build hf features to ensure consistent columns ordering when writing
    hf_features = get_hf_features_from_features(src_meta.features)

    # Build index once for all episodes (much faster than searching for each episode)
    logging.info("Building episode-to-files index...")
    episode_to_files = _build_episode_to_files_index(src_meta.root)

    for new_ep_idx, src_ep_idx in enumerate(episodes_to_keep):
        ep_meta = ep_df.loc[src_ep_idx]
        
        # Load episode data using pre-built index
        file_paths = episode_to_files.get(src_ep_idx, [])
        df = _load_episode_data_from_files(file_paths, src_ep_idx)

        # Reindex episode_index and global index
        length = len(df)
        if length == 0:
            logging.warning(f"Episode {src_ep_idx} has no data frames (task: {ep_meta['tasks']}). Skipping.")
            continue
        df["episode_index"] = new_ep_idx
        df["index"] = np.arange(current_dataset_index, current_dataset_index + length)

        # Write to a dedicated data file for this episode
        dst_data_path = dst_root / DEFAULT_DATA_PATH.format(
            chunk_index=data_chunk, file_index=data_file
        )
        dst_data_path.parent.mkdir(parents=True, exist_ok=True)
        if len(src_meta.image_keys) > 0:
            to_parquet_with_hf_images(df, dst_data_path)
        else:
            df.to_parquet(dst_data_path)

        # Build episode metadata row
        meta_row = {
            "episode_index": new_ep_idx,
            "tasks": ep_meta["tasks"],
            "length": int(length),
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "data/chunk_index": data_chunk,
            "data/file_index": data_file,
            "dataset_from_index": int(current_dataset_index),
            "dataset_to_index": int(current_dataset_index + length),
        }

        # Map videos
        for key in src_meta.video_keys:
            src_v_chunk = int(ep_meta[f"videos/{key}/chunk_index"])
            src_v_file = int(ep_meta[f"videos/{key}/file_index"])
            dst_v_chunk, dst_v_file = video_mapping[(key, src_v_chunk, src_v_file)]
            meta_row[f"videos/{key}/chunk_index"] = dst_v_chunk
            meta_row[f"videos/{key}/file_index"] = dst_v_file
            meta_row[f"videos/{key}/from_timestamp"] = float(ep_meta[f"videos/{key}/from_timestamp"])
            meta_row[f"videos/{key}/to_timestamp"] = float(ep_meta[f"videos/{key}/to_timestamp"])

        # Copy per-episode stats columns if present
        if src_ep_idx in stats_by_ep:
            for k, v in stats_by_ep[src_ep_idx].items():
                if isinstance(k, str) and k.startswith("stats/"):
                    meta_row[k] = v

        meta_rows.append(meta_row)

        # Advance counters
        current_dataset_index += length
        data_chunk, data_file = update_chunk_file_indices(data_chunk, data_file, DEFAULT_CHUNK_SIZE)

    # Write episodes metadata in a single file (small compared to data)
    if len(meta_rows) > 0:
        meta_df = pd.DataFrame(meta_rows)
        dst_meta_path = dst_root / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
        dst_meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_df.to_parquet(dst_meta_path, index=False)

    return len(episodes_to_keep), int(current_dataset_index)


def split_dataset_by_task(
    src_root: Path,
    dst_root_parent: Path,
) -> None:
    # Load source metadata
    src_meta = LeRobotDatasetMetadata(repo_id="local/src", root=src_root)

    # Prepare common immutable metadata pieces
    features = {**src_meta.features, **DEFAULT_FEATURES}
    use_videos = len(src_meta.video_keys) > 0

    tasks_df = src_meta.tasks.copy()

    # Iterate over actual task indices present in tasks_df (sorted)
    task_indices = tasks_df["task_index"].astype(int).sort_values().tolist()
    for task_index in task_indices:
        task_name = _find_task_name(tasks_df, task_index)
        episodes_to_keep = _filter_episode_indices_for_task(src_meta, task_name)
        if len(episodes_to_keep) == 0:
            logging.info(f"Skip task {task_index}: '{task_name}' (no episodes)")
            continue

        dst_root = dst_root_parent / f"libero_task_{task_index}"
        _ensure_empty_dir(dst_root)

        # Initialize destination metadata directory with empty dataset
        info = create_empty_dataset_info(
            codebase_version="v3.0",
            fps=src_meta.fps,
            features=features,
            use_videos=use_videos,
            robot_type=src_meta.robot_type,
            chunks_size=src_meta.chunks_size,
            data_files_size_in_mb=src_meta.data_files_size_in_mb,
            video_files_size_in_mb=src_meta.video_files_size_in_mb,
        )
        (dst_root / INFO_PATH).parent.mkdir(parents=True, exist_ok=True)
        write_info(info, dst_root)

        # Write tasks sorted by task_index to ensure iloc[task_index] matches the correct task
        write_tasks(tasks_df.sort_values("task_index"), dst_root)

        # Copy needed video files and build mapping from source -> destination indices
        video_mapping = _copy_required_video_files(src_meta, dst_root, episodes_to_keep)

        # Build data parquet and episodes metadata
        total_episodes, total_frames = _rebuild_data_and_meta(
            src_meta, dst_root, episodes_to_keep, video_mapping
        )

        # Aggregate stats from selected episodes and write
        ep_rows = _read_episode_stats_rows(src_meta.root, episodes_to_keep)
        stats = _aggregate_stats_from_episode_rows(ep_rows, src_meta.features)
        write_stats(stats, dst_root)

        # Finalize info.json
        info_path = dst_root / INFO_PATH
        info_dict = info_path.read_text()
        # Reload in-memory info via LeRobotDatasetMetadata for consistent write (avoid json parsing here)
        dst_meta = LeRobotDatasetMetadata(repo_id=f"local/libero_task_{task_index}", root=dst_root)
        dst_meta.info["total_episodes"] = total_episodes
        dst_meta.info["total_frames"] = total_frames
        dst_meta.info["total_tasks"] = len(tasks_df)
        dst_meta.info["splits"] = {"train": f"0:{total_episodes}"}
        write_info(dst_meta.info, dst_root)

        logging.info(
            f"Wrote task {task_index}: '{task_name}' with {total_episodes} episodes and {total_frames} frames -> {dst_root}"
        )


def main():
    parser = argparse.ArgumentParser(description="Split a LeRobot dataset into per-task datasets.")
    parser.add_argument(
        "--src_root",
        type=str,
        required=True,
        help="Path to the source dataset root (directory containing meta/, data/, videos/).",
    )
    parser.add_argument(
        "--dst_root_parent",
        type=str,
        required=True,
        help="Directory where per-task datasets will be created.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    split_dataset_by_task(Path(args.src_root), Path(args.dst_root_parent))


if __name__ == "__main__":
    main()


