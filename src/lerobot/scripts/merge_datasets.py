#!/usr/bin/env python

# Copyright 2025
# Licensed under the Apache License, Version 2.0

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_TASKS_PATH,
    DEFAULT_VIDEO_PATH,
    INFO_PATH,
    STATS_PATH,
    get_hf_dataset_size_in_mb,
    get_hf_features_from_features,
    load_episodes,
    load_info,
    load_nested_dataset,
    load_stats,
    load_tasks,
    to_parquet_with_hf_images,
    update_chunk_file_indices,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
)


def _validate_compatibility(src1: Path, src2: Path) -> dict:
    info1 = load_info(src1)
    info2 = load_info(src2)

    if info1["codebase_version"] != info2["codebase_version"]:
        raise ValueError("Source datasets must have the same codebase_version.")

    if info1["fps"] != info2["fps"]:
        raise ValueError("Source datasets must have identical fps.")

    if (info1.get("video_path") is None) != (info2.get("video_path") is None):
        raise ValueError("Both datasets must either use videos or not use videos.")

    if info1["features"] != info2["features"]:
        raise ValueError("Source datasets must have identical features.")

    return info1


def _merge_tasks(src1: Path, src2: Path, dst: Path) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    tasks1 = load_tasks(src1)
    tasks2 = load_tasks(src2)

    # Build string -> new_index preserving order: first tasks1 order, then new tasks from tasks2
    task_to_new_index: Dict[str, int] = {}
    for task in tasks1.index.tolist():
        if task not in task_to_new_index:
            task_to_new_index[task] = len(task_to_new_index)
    for task in tasks2.index.tolist():
        if task not in task_to_new_index:
            task_to_new_index[task] = len(task_to_new_index)

    # Create tasks dataframe with index as task strings and 'task_index' column
    tasks_df = pd.DataFrame({"task_index": list(task_to_new_index.values())}, index=list(task_to_new_index.keys()))
    tasks_df = tasks_df.sort_values("task_index")

    # Old numeric task_index -> new numeric task_index mappings
    tasks1_old_to_new = {int(tasks1.loc[t].task_index): int(task_to_new_index[t]) for t in tasks1.index}
    tasks2_old_to_new = {int(tasks2.loc[t].task_index): int(task_to_new_index[t]) for t in tasks2.index}

    write_tasks(tasks_df, dst)

    return tasks_df, tasks1_old_to_new, tasks2_old_to_new


def _copy_and_remap_videos(
    src1: Path,
    src2: Path,
    dst: Path,
    video_keys: List[str],
    chunks_size: int,
) -> Tuple[Dict[Tuple[int, int, str, int], Tuple[int, int]], Dict[Tuple[int, int, str, int], Tuple[int, int]]]:
    """
    Copy mp4 files from both sources into destination while allocating new sequential (chunk_idx, file_idx)
    indices. Returns mapping dicts to translate (old_chunk, old_file, video_key, which_source) -> (new_chunk, new_file).
    which_source is 0 for src1 and 1 for src2.
    """
    mapping_src1: Dict[Tuple[int, int, str, int], Tuple[int, int]] = {}
    mapping_src2: Dict[Tuple[int, int, str, int], Tuple[int, int]] = {}

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

    for which, src in enumerate([src1, src2]):
        for vkey, old_chunk, old_file, fpath in iter_source_videos(src):
            # Assign new indices
            new_chunk, new_file = current_chunk, current_file
            # Copy file
            dst_path = dst / DEFAULT_VIDEO_PATH.format(video_key=vkey, chunk_index=new_chunk, file_index=new_file)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fpath, dst_path)

            if which == 0:
                mapping_src1[(old_chunk, old_file, vkey, which)] = (new_chunk, new_file)
            else:
                mapping_src2[(old_chunk, old_file, vkey, which)] = (new_chunk, new_file)

            # Advance indices
            current_chunk, current_file = update_chunk_file_indices(current_chunk, current_file, chunks_size)

    return mapping_src1, mapping_src2


def _iter_episode_slices(episodes_ds: Dataset) -> List[Tuple[int, int, int]]:
    """
    Produce a list of (episode_index, from_idx, to_idx) for the given episodes metadata dataset.
    """
    out = []
    for row in episodes_ds:
        ep_idx = int(row["episode_index"])  # pyarrow scalar -> int
        ep_from = int(row["dataset_from_index"])  # inclusive
        ep_to = int(row["dataset_to_index"])  # exclusive
        out.append((ep_idx, ep_from, ep_to))
    # Ensure ordering by episode index
    out.sort(key=lambda x: x[0])
    return out


def _merge_data_and_build_episode_metadata(
    src_root: Path,
    dst_root: Path,
    tasks_old_to_new: Dict[int, int],
    episode_index_offset: int,
    start_frame_offset: int,
    data_files_size_in_mb: int,
    chunks_size: int,
    features: dict,
) -> Tuple[int, int, List[dict]]:
    """
    Merge per-episode frames from a source dataset into destination data files without splitting episodes.
    Returns (new_frame_offset, written_files_count, episode_metadata_records).
    """
    # Load source datasets
    hf_features = get_hf_features_from_features(features)
    src_hf_data = load_nested_dataset(src_root / "data", features=hf_features)
    src_episodes = load_episodes(src_root)

    # Prepare accumulation state
    current_chunk_idx = 0
    current_file_idx = 0
    if (dst_root / "data").exists():
        # Determine last written file indices if resuming for second source
        existing_files = sorted((dst_root / "data").glob("chunk-*/file-*.parquet"))
        if existing_files:
            last = existing_files[-1]
            current_chunk_idx = int(last.parent.name.split("-")[-1])
            current_file_idx = int(last.stem.split("-")[-1])
            # Move to next slot for appending
            current_chunk_idx, current_file_idx = update_chunk_file_indices(
                current_chunk_idx, current_file_idx, chunks_size
            )

    accumulated_size_mb = 0
    accumulation_df = None

    frame_offset = start_frame_offset
    episode_records: List[dict] = []

    # Iterate episodes in order
    ep_slices = _iter_episode_slices(src_episodes)
    for old_ep_idx, ep_from, ep_to in ep_slices:
        # Slice episode rows
        ep_ds = src_hf_data.select(range(ep_from, ep_to))

        # Update columns: index, episode_index, task_index
        def _map_columns(batch):
            n = len(batch["index"]) if "index" in batch else len(next(iter(batch.values())))
            if "index" in batch:
                batch["index"] = (np.array(batch["index"]).astype(np.int64) + frame_offset).tolist()
            if "episode_index" in batch:
                batch["episode_index"] = (
                    np.array(batch["episode_index"]).astype(np.int64) + episode_index_offset
                ).tolist()
            if "task_index" in batch:
                mapped = [tasks_old_to_new[int(x)] for x in batch["task_index"]]
                batch["task_index"] = mapped
            return batch

        ep_ds = ep_ds.map(_map_columns, batched=True)

        # Compute episode size in MB using arrow storage
        ep_size_mb = get_hf_dataset_size_in_mb(ep_ds)

        # Flush current file if adding this episode would exceed threshold
        if accumulation_df is not None and accumulated_size_mb + ep_size_mb >= data_files_size_in_mb:
            out_path = dst_root / DEFAULT_DATA_PATH.format(
                chunk_index=current_chunk_idx, file_index=current_file_idx
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            to_parquet_with_hf_images(accumulation_df, out_path)

            # Advance file indices and reset accumulation
            current_chunk_idx, current_file_idx = update_chunk_file_indices(
                current_chunk_idx, current_file_idx, chunks_size
            )
            accumulation_df = None
            accumulated_size_mb = 0

        # Convert episode dataset to pandas for accumulation
        ep_df = pd.DataFrame(ep_ds)

        # Register episode metadata for destination
        new_ep_idx = old_ep_idx + episode_index_offset
        ep_length = ep_to - ep_from
        if accumulation_df is None:
            assigned_chunk, assigned_file = current_chunk_idx, current_file_idx
        else:
            assigned_chunk, assigned_file = current_chunk_idx, current_file_idx

        episode_records.append(
            {
                "episode_index": new_ep_idx,
                "length": ep_length,
                "data/chunk_index": assigned_chunk,
                "data/file_index": assigned_file,
                "dataset_from_index": frame_offset,
                "dataset_to_index": frame_offset + ep_length,
            }
        )

        # Accumulate and update counters
        accumulation_df = ep_df if accumulation_df is None else pd.concat([accumulation_df, ep_df], ignore_index=True)
        accumulated_size_mb += ep_size_mb
        frame_offset += ep_length

    # Flush remaining accumulation
    if accumulation_df is not None and len(accumulation_df) > 0:
        out_path = dst_root / DEFAULT_DATA_PATH.format(chunk_index=current_chunk_idx, file_index=current_file_idx)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        to_parquet_with_hf_images(accumulation_df, out_path)

    return frame_offset, current_file_idx + 1, episode_records


def _merge_episodes_metadata(
    src1: Path,
    src2: Path,
    dst: Path,
    episode_index_offset_src2: int,
    video_key_to_mapping_src1: Dict[Tuple[int, int, str, int], Tuple[int, int]] | None,
    video_key_to_mapping_src2: Dict[Tuple[int, int, str, int], Tuple[int, int]] | None,
    tasks_df: pd.DataFrame,
    base_records_src1: List[dict],
    base_records_src2: List[dict],
) -> None:
    epi1 = load_episodes(src1)
    epi2 = load_episodes(src2)

    def _collect(epi_ds: Dataset, which: int, idx_offset: int, base_records: List[dict]) -> List[dict]:
        # Build lookup: new_ep_idx -> base record
        base_by_new_ep = {rec["episode_index"]: rec for rec in base_records}

        records: List[dict] = []
        for row in epi_ds:
            old_ep = int(row["episode_index"])  # scalar
            new_ep = old_ep + idx_offset
            length = int(row["length"]) if "length" in row else int(row["dataset_to_index"] - row["dataset_from_index"])  # fallback
            tasks = list(row["tasks"]) if "tasks" in row else []

            rec = {
                "episode_index": new_ep,
                "tasks": tasks,
                "length": length,
            }

            # Data file mapping from base records
            base = base_by_new_ep[new_ep]
            rec.update(
                {
                    "data/chunk_index": int(base["data/chunk_index"]),
                    "data/file_index": int(base["data/file_index"]),
                    "dataset_from_index": int(base["dataset_from_index"]),
                    "dataset_to_index": int(base["dataset_to_index"]),
                }
            )

            # Video mapping if any
            for key in row.keys():
                if not isinstance(key, str) or not key.startswith("videos/"):
                    continue
                if key.endswith("/chunk_index"):
                    vkey = key.split("/")[1]
                    old_chunk = int(row[f"videos/{vkey}/chunk_index"]) if row[f"videos/{vkey}/chunk_index"] is not None else None
                    old_file = int(row[f"videos/{vkey}/file_index"]) if row[f"videos/{vkey}/file_index"] is not None else None
                    if old_chunk is None or old_file is None:
                        continue
                    mapper = video_key_to_mapping_src1 if which == 0 else video_key_to_mapping_src2
                    new_chunk, new_file = mapper[(old_chunk, old_file, vkey, which)]
                    rec[f"videos/{vkey}/chunk_index"] = int(new_chunk)
                    rec[f"videos/{vkey}/file_index"] = int(new_file)
                    rec[f"videos/{vkey}/from_timestamp"] = float(row[f"videos/{vkey}/from_timestamp"]) if row[f"videos/{vkey}/from_timestamp"] is not None else None
                    rec[f"videos/{vkey}/to_timestamp"] = float(row[f"videos/{vkey}/to_timestamp"]) if row[f"videos/{vkey}/to_timestamp"] is not None else None

            records.append(rec)
        return records

    recs1 = _collect(epi1, which=0, idx_offset=0, base_records=base_records_src1)
    recs2 = _collect(epi2, which=1, idx_offset=episode_index_offset_src2, base_records=base_records_src2)

    # Combine and sort by episode_index
    all_recs = recs1 + recs2
    all_recs.sort(key=lambda r: r["episode_index"])

    # Write as a single parquet file
    ds = Dataset.from_pandas(pd.DataFrame(all_recs))
    write_episodes(ds, dst)


def _write_info(dst: Path, base_info: dict, total_episodes: int, total_frames: int, total_tasks: int) -> None:
    info = dict(base_info)
    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = int(total_tasks)
    info["splits"] = {"train": f"0:{total_episodes}"}
    write_info(info, dst)


def merge_two_datasets(src1_path: Path, src2_path: Path, dst_path: Path) -> None:
    logging.info("Validating compatibility and preparing destination...")
    dst_path.mkdir(parents=True, exist_ok=False)

    base_info = _validate_compatibility(src1_path, src2_path)
    fps = base_info["fps"]
    features = base_info["features"]
    chunks_size = int(base_info["chunks_size"])
    data_files_size_in_mb = int(base_info["data_files_size_in_mb"])
    video_files_size_in_mb = int(base_info["video_files_size_in_mb"]) if base_info.get("video_path") else 0

    # Initialize meta/info.json skeleton
    write_info(
        {
            **base_info,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "splits": {},
        },
        dst_path,
    )

    logging.info("Merging tasks...")
    tasks_df, tasks1_old_to_new, tasks2_old_to_new = _merge_tasks(src1_path, src2_path, dst_path)

    # Optionally aggregate stats
    logging.info("Aggregating stats (if present)...")
    stats_list = []
    stats1 = load_stats(src1_path)
    stats2 = load_stats(src2_path)
    if stats1 is not None:
        stats_list.append(stats1)
    if stats2 is not None:
        stats_list.append(stats2)
    if stats_list:
        merged_stats = aggregate_stats(stats_list) if len(stats_list) > 1 else stats_list[0]
        write_stats(merged_stats, dst_path)

    # Copy videos and make mapping from old (chunk,file) -> new (chunk,file)
    video_keys = [k for k, ft in features.items() if ft["dtype"] == "video"]
    if len(video_keys) > 0:
        logging.info("Copying and remapping video files...")
        vid_map1, vid_map2 = _copy_and_remap_videos(src1_path, src2_path, dst_path, video_keys, chunks_size)
    else:
        vid_map1, vid_map2 = {}, {}

    # Merge data and build episode metadata base records
    logging.info("Rewriting data files and reindexing episodes...")
    frame_offset = 0
    frame_offset, _, base_records_src1 = _merge_data_and_build_episode_metadata(
        src1_path,
        dst_path,
        tasks1_old_to_new,
        episode_index_offset=0,
        start_frame_offset=frame_offset,
        data_files_size_in_mb=data_files_size_in_mb,
        chunks_size=chunks_size,
        features=features,
    )

    epi1 = load_episodes(src1_path)
    num_eps_src1 = len(epi1)

    frame_offset, _, base_records_src2 = _merge_data_and_build_episode_metadata(
        src2_path,
        dst_path,
        tasks2_old_to_new,
        episode_index_offset=num_eps_src1,
        start_frame_offset=frame_offset,
        data_files_size_in_mb=data_files_size_in_mb,
        chunks_size=chunks_size,
        features=features,
    )

    # Write final episodes metadata
    logging.info("Writing episodes metadata...")
    _merge_episodes_metadata(
        src1_path,
        src2_path,
        dst_path,
        episode_index_offset_src2=num_eps_src1,
        video_key_to_mapping_src1=vid_map1 if video_keys else None,
        video_key_to_mapping_src2=vid_map2 if video_keys else None,
        tasks_df=tasks_df,
        base_records_src1=base_records_src1,
        base_records_src2=base_records_src2,
    )

    # Compute totals
    total_episodes = num_eps_src1 + len(load_episodes(src2_path))
    total_frames = frame_offset
    total_tasks = len(tasks_df)

    # Finalize info.json
    logging.info("Finalizing info.json...")
    _write_info(dst_path, base_info, total_episodes, total_frames, total_tasks)

    logging.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Merge two LeRobot v3.0 datasets into a new dataset directory")
    parser.add_argument("--source_dataset1", type=str, required=True, help="Path to first source dataset root")
    parser.add_argument("--source_dataset2", type=str, required=True, help="Path to second source dataset root")
    parser.add_argument("--target_dataset", type=str, required=True, help="Path to target dataset root (created)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    src1 = Path(args.source_dataset1).resolve()
    src2 = Path(args.source_dataset2).resolve()
    dst = Path(args.target_dataset).resolve()

    # Basic checks
    if not (src1 / INFO_PATH).exists() or not (src2 / INFO_PATH).exists():
        raise FileNotFoundError("Both source datasets must contain meta/info.json")
    if dst.exists():
        raise FileExistsError(f"Target path already exists: {dst}")

    merge_two_datasets(src1, src2, dst)


if __name__ == "__main__":
    main()


