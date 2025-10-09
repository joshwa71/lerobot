#!/usr/bin/env python

# Copyright 2025
# Licensed under the Apache License, Version 2.0

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_PATH,
    INFO_PATH,
    get_hf_features_from_features,
    load_episodes,
    load_info,
    load_stats,
    load_tasks,
    to_parquet_with_hf_images,
    update_chunk_file_indices,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
)


def _validate_all(sources: List[Path]) -> dict:
    infos = [load_info(src) for src in sources]
    base = infos[0]
    for info in infos[1:]:
        if info["codebase_version"] != base["codebase_version"]:
            raise ValueError("All source datasets must have the same codebase_version.")
        if info["fps"] != base["fps"]:
            raise ValueError("All source datasets must have identical fps.")
        if (info.get("video_path") is None) != (base.get("video_path") is None):
            raise ValueError("All datasets must either use videos or not use videos.")
        if info["features"] != base["features"]:
            raise ValueError("All source datasets must have identical features.")
    return base


def _build_unified_tasks(sources: List[Path], dst: Path) -> Tuple[pd.DataFrame, List[Dict[int, int]]]:
    tasks_list = [load_tasks(src) for src in sources]
    # Preserve order of first appearance across sources
    task_to_index: Dict[str, int] = {}
    for tasks_df in tasks_list:
        for task in tasks_df.index.tolist():
            if task not in task_to_index:
                task_to_index[task] = len(task_to_index)

    tasks_df = pd.DataFrame({"task_index": list(task_to_index.values())}, index=list(task_to_index.keys()))
    tasks_df = tasks_df.sort_values("task_index")
    write_tasks(tasks_df, dst)

    # For each source, build old numeric -> new numeric mapping
    old_to_new_per_source: List[Dict[int, int]] = []
    for tasks_df_src in tasks_list:
        mapping = {int(tasks_df_src.loc[t].task_index): int(task_to_index[t]) for t in tasks_df_src.index}
        old_to_new_per_source.append(mapping)

    return tasks_df, old_to_new_per_source


def _copy_and_map_videos(
    sources: List[Path],
    dst: Path,
    video_keys: List[str],
    chunks_size: int,
) -> List[Dict[Tuple[int, int, str, int], Tuple[int, int]]]:
    mappings: List[Dict[Tuple[int, int, str, int], Tuple[int, int]]] = [
        {} for _ in range(len(sources))
    ]

    current_chunk = 0
    current_file = 0

    def iter_source_videos(src_root: Path):
        for vkey in video_keys:
            vdir = src_root / f"videos/{vkey}"
            if not vdir.exists():
                continue
            for chunk_dir in sorted(vdir.glob("chunk-*/")):
                chunk_idx = int(chunk_dir.name.split("-")[-1])
                for fpath in sorted(chunk_dir.glob("file-*.mp4")):
                    file_idx = int(fpath.stem.split("-")[-1])
                    yield vkey, chunk_idx, file_idx, fpath

    for si, src in enumerate(sources):
        for vkey, old_chunk, old_file, fpath in iter_source_videos(src):
            new_chunk, new_file = current_chunk, current_file
            dst_path = dst / DEFAULT_VIDEO_PATH.format(video_key=vkey, chunk_index=new_chunk, file_index=new_file)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            # Copy video (metadata is remapped in episodes writing)
            dst_path.write_bytes(fpath.read_bytes())

            mappings[si][(old_chunk, old_file, vkey, si)] = (new_chunk, new_file)
            current_chunk, current_file = update_chunk_file_indices(current_chunk, current_file, chunks_size)

    return mappings


def _iter_episode_slices(episodes_ds: Dataset) -> List[Tuple[int, int, int]]:
    out = []
    for row in episodes_ds:
        ep_idx = int(row["episode_index"])  # pyarrow scalar -> int
        ep_from = int(row["dataset_from_index"])  # inclusive
        ep_to = int(row["dataset_to_index"])  # exclusive
        out.append((ep_idx, ep_from, ep_to))
    out.sort(key=lambda x: x[0])
    return out


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


def _rewrite_data_from_sources(
    sources: List[Path],
    dst: Path,
    task_maps: List[Dict[int, int]],
    data_files_size_in_mb: int,
    chunks_size: int,
    features: dict,
) -> Tuple[int, List[List[dict]]]:
    # One parquet file per episode (simpler and robust across libraries)
    current_chunk_idx = 0
    current_file_idx = 0
    frame_offset = 0

    contains_images = any(ft.get("dtype") == "image" for ft in features.values())

    all_base_records: List[List[dict]] = []  # per-source list of base episode records
    new_ep_counter = 0

    for si, src_root in enumerate(sources):
        logging.info(f"Processing source {si}: {src_root}")
        
        # Build index once per source (much faster than searching for each episode)
        logging.info(f"Building episode-to-files index for source {si}...")
        episode_to_files = _build_episode_to_files_index(src_root)
        
        src_episodes = load_episodes(src_root)
        src_records: List[dict] = []

        for row in src_episodes:
            old_ep_idx = int(row["episode_index"])  # scalar
            ep_from = int(row["dataset_from_index"])  # inclusive
            ep_to = int(row["dataset_to_index"])  # exclusive

            # Load episode data using pre-built index
            file_paths = episode_to_files.get(old_ep_idx, [])
            df = _load_episode_data_from_files(file_paths, old_ep_idx)

            if len(df) == 0:
                logging.warning(f"Source {si} episode {old_ep_idx} has no data frames. Skipping.")
                continue

            # Compute length from actual rows
            ep_length = int(len(df))

            # Assign new contiguous episode index
            new_ep_idx = new_ep_counter
            # Map columns: global index, episode_index, task_index
            df["index"] = (df["index"].astype(np.int64) - df["index"].min() + frame_offset).astype(np.int64)
            df["episode_index"] = new_ep_idx
            if "task_index" in df.columns:
                df["task_index"] = df["task_index"].astype(int).map(task_maps[si]).astype(int)

            # Write to destination in current (chunk,file)
            out_path = dst / DEFAULT_DATA_PATH.format(
                chunk_index=current_chunk_idx, file_index=current_file_idx
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if contains_images:
                to_parquet_with_hf_images(df, out_path)
            else:
                df.to_parquet(out_path)

            # Base episode record
            src_records.append(
                {
                    "episode_index": new_ep_idx,
                    "old_episode_index": old_ep_idx,
                    "length": ep_length,
                    "data/chunk_index": current_chunk_idx,
                    "data/file_index": current_file_idx,
                    "dataset_from_index": frame_offset,
                    "dataset_to_index": frame_offset + ep_length,
                }
            )

            # Advance counters
            frame_offset += ep_length
            current_chunk_idx, current_file_idx = update_chunk_file_indices(
                current_chunk_idx, current_file_idx, chunks_size
            )
            new_ep_counter += 1

        all_base_records.append(src_records)

    return frame_offset, all_base_records


def _write_episodes_metadata(
    sources: List[Path],
    dst: Path,
    video_maps: List[Dict[Tuple[int, int, str, int], Tuple[int, int]]],
    all_base_records: List[List[dict]],
) -> None:
    records: List[dict] = []

    for si, src_root in enumerate(sources):
        epi = load_episodes(src_root)
        epi_df = epi.to_pandas().set_index("episode_index")

        for base in all_base_records[si]:
            old_ep = int(base["old_episode_index"])
            new_ep = int(base["episode_index"])
            if old_ep not in epi_df.index:
                continue
            row = epi_df.loc[old_ep]
            tasks = list(row["tasks"]) if "tasks" in row else []
            length = int(base["length"])  # trusted from frames

            rec = {
                "episode_index": new_ep,
                "tasks": tasks,
                "length": length,
                "data/chunk_index": int(base["data/chunk_index"]),
                "data/file_index": int(base["data/file_index"]),
                "dataset_from_index": int(base["dataset_from_index"]),
                "dataset_to_index": int(base["dataset_to_index"]),
            }

            # Map videos if present
            for col in epi_df.columns:
                if not isinstance(col, str) or not col.startswith("videos/"):
                    continue
                if col.endswith("/chunk_index"):
                    vkey = col.split("/")[1]
                    old_chunk = row.get(f"videos/{vkey}/chunk_index")
                    old_file = row.get(f"videos/{vkey}/file_index")
                    if pd.isna(old_chunk) or pd.isna(old_file):
                        continue
                    old_chunk = int(old_chunk)
                    old_file = int(old_file)
                    new_chunk, new_file = video_maps[si][(old_chunk, old_file, vkey, si)]
                    rec[f"videos/{vkey}/chunk_index"] = int(new_chunk)
                    rec[f"videos/{vkey}/file_index"] = int(new_file)
                    rec[f"videos/{vkey}/from_timestamp"] = float(row.get(f"videos/{vkey}/from_timestamp", 0.0))
                    rec[f"videos/{vkey}/to_timestamp"] = float(row.get(f"videos/{vkey}/to_timestamp", 0.0))

            records.append(rec)

    # Sort and write
    records.sort(key=lambda r: r["episode_index"])
    ds = Dataset.from_pandas(pd.DataFrame(records))
    write_episodes(ds, dst)


def merge_datasets(sources: List[Path], target: Path) -> None:
    logging.info("Validating sources...")
    base_info = _validate_all(sources)
    fps = base_info["fps"]
    features = base_info["features"]
    chunks_size = int(base_info["chunks_size"])
    data_files_size_in_mb = int(base_info["data_files_size_in_mb"])
    video_files_size_in_mb = int(base_info["video_files_size_in_mb"]) if base_info.get("video_path") else 0

    logging.info("Initializing destination...")
    target.mkdir(parents=True, exist_ok=False)
    write_info(
        {
            **base_info,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "splits": {},
        },
        target,
    )

    logging.info("Unifying tasks...")
    tasks_df, task_maps = _build_unified_tasks(sources, target)

    logging.info("Aggregating stats (if present)...")
    stats_list = [s for s in (load_stats(src) for src in sources) if s is not None]
    if stats_list:
        merged_stats = aggregate_stats(stats_list) if len(stats_list) > 1 else stats_list[0]
        write_stats(merged_stats, target)

    video_keys = [k for k, ft in features.items() if ft["dtype"] == "video"]
    if len(video_keys) > 0:
        logging.info("Copying videos and building mappings...")
        video_maps = _copy_and_map_videos(sources, target, video_keys, chunks_size)
    else:
        video_maps = [{} for _ in sources]

    logging.info("Rewriting data and reindexing episodes...")
    total_frames, all_base_records = _rewrite_data_from_sources(
        sources,
        target,
        task_maps,
        data_files_size_in_mb,
        chunks_size,
        features,
    )

    logging.info("Writing episodes metadata...")
    _write_episodes_metadata(sources, target, video_maps, all_base_records)

    # Use actually written episodes count to be robust to skipped empties
    total_episodes = sum(len(recs) for recs in all_base_records)
    total_tasks = len(tasks_df)

    logging.info("Finalizing info.json...")
    write_info(
        {
            **base_info,
            "total_episodes": int(total_episodes),
            "total_frames": int(total_frames),
            "total_tasks": int(total_tasks),
            "splits": {"train": f"0:{total_episodes}"},
        },
        target,
    )

    logging.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Merge N LeRobot v3.0 datasets into a new dataset directory")
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        required=True,
        help="One or more source dataset roots (paths containing meta/data/videos).",
    )
    parser.add_argument("--target", type=str, required=True, help="Target dataset root (created)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    sources = [Path(p).resolve() for p in args.sources]
    target = Path(args.target).resolve()

    for src in sources:
        if not (src / INFO_PATH).exists():
            raise FileNotFoundError(f"Source dataset missing meta/info.json: {src}")
    if target.exists():
        raise FileExistsError(f"Target path already exists: {target}")

    merge_datasets(sources, target)


if __name__ == "__main__":
    main()


