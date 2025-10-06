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


def _build_episode_to_files_index(src_root: Path) -> Dict[int, List[Path]]:
    """Build an index mapping episode_idx -> list of parquet files containing that episode."""
    data_dir = src_root / "data"
    if not data_dir.exists():
        return {}
    
    episode_to_files: Dict[int, List[Path]] = {}
    
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for file_path in sorted(chunk_dir.glob("file-*.parquet")):
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


def _match_successful_to_failed_tasks(
    success_tasks: pd.DataFrame,
    failed_tasks: pd.DataFrame,
) -> Dict[int, int]:
    """Match successful task indices to failed task indices by name.
    
    Returns: {success_task_idx: failed_task_idx}
    """
    mapping = {}
    
    for success_name, success_row in success_tasks.iterrows():
        success_idx = int(success_row["task_index"])
        # Look for "Failed attempt to " + success_name in failed tasks
        expected_failed_name = f"Failed attempt to {success_name}"
        
        for failed_name, failed_row in failed_tasks.iterrows():
            if failed_name == expected_failed_name:
                failed_idx = int(failed_row["task_index"])
                mapping[success_idx] = failed_idx
                logging.info(f"Matched task {success_idx}: '{success_name}' -> {failed_idx}: '{failed_name}'")
                break
        else:
            logging.warning(f"No matching failed task found for '{success_name}'")
    
    return mapping


def _count_episodes_per_task(episodes_df: pd.DataFrame) -> Dict[str, int]:
    """Count episodes per task name."""
    counts = {}
    for _, row in episodes_df.iterrows():
        tasks = row.get("tasks", [])
        if tasks:
            for task_name in tasks:
                counts[task_name] = counts.get(task_name, 0) + 1
    return counts


def _get_episodes_for_task(episodes_df: pd.DataFrame, task_name: str) -> List[int]:
    """Get episode indices that contain a specific task."""
    episode_indices = []
    for _, row in episodes_df.iterrows():
        tasks = row.get("tasks", [])
        if task_name in tasks:
            episode_indices.append(int(row["episode_index"]))
    return episode_indices


def create_mixed_dataset(
    success_root: Path,
    failed_root: Path,
    output_root: Path,
) -> None:
    """Create a mixed dataset with equal successful and failed episodes per task."""
    
    # Load metadata from both datasets
    logging.info("Loading metadata from datasets...")
    success_info = load_info(success_root)
    failed_info = load_info(failed_root)
    
    # Validate compatibility
    if success_info["fps"] != failed_info["fps"]:
        raise ValueError("FPS mismatch between datasets")
    if success_info["features"] != failed_info["features"]:
        raise ValueError("Features mismatch between datasets")
    
    # Load tasks
    success_tasks = load_tasks(success_root)
    failed_tasks = load_tasks(failed_root)
    
    # Load episodes
    success_episodes = load_episodes(success_root).to_pandas()
    failed_episodes = load_episodes(failed_root).to_pandas()
    
    # Match tasks
    task_mapping = _match_successful_to_failed_tasks(success_tasks, failed_tasks)
    
    # Count episodes per task
    logging.info("Counting episodes per task...")
    success_counts = _count_episodes_per_task(success_episodes)
    
    # Build combined tasks dataframe with explicit ordering and indexing
    # Create list of (task_name, old_index, source) tuples
    task_entries = []
    
    # Add successful tasks first (these will be indices 0-9)
    for task_name, row in success_tasks.iterrows():
        task_entries.append((task_name, int(row["task_index"]), "success"))
    
    # Add failed tasks second (these will be indices 10-19)
    for task_name, row in failed_tasks.iterrows():
        task_entries.append((task_name, int(row["task_index"]), "failed"))
    
    # Build combined tasks dataframe with new contiguous indices
    combined_tasks_data = []
    task_name_to_new_index = {}
    
    for new_idx, (task_name, old_idx, source) in enumerate(task_entries):
        combined_tasks_data.append({"task_index": new_idx})
        task_name_to_new_index[task_name] = new_idx
        logging.info(f"Task {new_idx}: '{task_name}' (from {source})")
    
    combined_tasks = pd.DataFrame(combined_tasks_data, index=[name for name, _, _ in task_entries])
    
    # Create output directory
    output_root.mkdir(parents=True, exist_ok=False)
    
    # Write combined tasks
    write_tasks(combined_tasks, output_root)
    
    # Build episode indices for processing
    logging.info("Building episode-to-files indices...")
    success_ep_to_files = _build_episode_to_files_index(success_root)
    failed_ep_to_files = _build_episode_to_files_index(failed_root)
    
    # Prepare data writing
    contains_images = any(ft.get("dtype") == "image" for ft in success_info["features"].values())
    current_chunk = 0
    current_file = 0
    frame_offset = 0
    new_ep_counter = 0
    
    all_episode_records = []
    
    # Process each successful task
    for success_task_name, success_row in success_tasks.iterrows():
        success_task_idx = int(success_row["task_index"])
        
        if success_task_idx not in task_mapping:
            logging.warning(f"Skipping task '{success_task_name}' - no failed counterpart")
            continue
        
        failed_task_idx = task_mapping[success_task_idx]
        failed_task_name = f"Failed attempt to {success_task_name}"
        
        # Get episode counts
        success_count = success_counts.get(success_task_name, 0)
        
        logging.info(f"\nProcessing task '{success_task_name}':")
        logging.info(f"  Success episodes: {success_count}")
        
        # Get episode indices for both
        success_ep_indices = _get_episodes_for_task(success_episodes, success_task_name)
        failed_ep_indices = _get_episodes_for_task(failed_episodes, failed_task_name)
        
        if len(failed_ep_indices) < success_count:
            logging.warning(f"  Not enough failed episodes ({len(failed_ep_indices)}), using all available")
            failed_ep_indices = failed_ep_indices[:len(failed_ep_indices)]
        else:
            # Take exactly the same number of failed episodes
            failed_ep_indices = failed_ep_indices[:success_count]
        
        logging.info(f"  Taking {len(failed_ep_indices)} failed episodes")
        
        # Process successful episodes
        for src_ep_idx in success_ep_indices:
            file_paths = success_ep_to_files.get(src_ep_idx, [])
            df = _load_episode_data_from_files(file_paths, src_ep_idx)
            
            if len(df) == 0:
                logging.warning(f"  Success episode {src_ep_idx} has no data, skipping")
                continue
            
            # Get episode metadata
            ep_meta = success_episodes[success_episodes["episode_index"] == src_ep_idx].iloc[0]
            
            # Reindex
            length = len(df)
            df["episode_index"] = new_ep_counter
            df["index"] = np.arange(frame_offset, frame_offset + length)
            
            # Remap task_index to new combined index
            new_task_idx = task_name_to_new_index[success_task_name]
            df["task_index"] = new_task_idx
            
            # Write data
            dst_path = output_root / DEFAULT_DATA_PATH.format(
                chunk_index=current_chunk, file_index=current_file
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if contains_images:
                to_parquet_with_hf_images(df, dst_path)
            else:
                df.to_parquet(dst_path)
            
            # Create episode record
            ep_record = {
                "episode_index": new_ep_counter,
                "tasks": [success_task_name],
                "length": length,
                "data/chunk_index": current_chunk,
                "data/file_index": current_file,
                "dataset_from_index": frame_offset,
                "dataset_to_index": frame_offset + length,
            }
            
            # Copy video metadata if present
            for key in [k for k in ep_meta.index if k.startswith("videos/")]:
                ep_record[key] = ep_meta[key]
            
            all_episode_records.append(ep_record)
            
            # Advance counters
            frame_offset += length
            current_chunk, current_file = update_chunk_file_indices(
                current_chunk, current_file, success_info["chunks_size"]
            )
            new_ep_counter += 1
        
        # Process failed episodes
        for src_ep_idx in failed_ep_indices:
            file_paths = failed_ep_to_files.get(src_ep_idx, [])
            df = _load_episode_data_from_files(file_paths, src_ep_idx)
            
            if len(df) == 0:
                logging.warning(f"  Failed episode {src_ep_idx} has no data, skipping")
                continue
            
            # Get episode metadata
            ep_meta = failed_episodes[failed_episodes["episode_index"] == src_ep_idx].iloc[0]
            
            # Reindex
            length = len(df)
            df["episode_index"] = new_ep_counter
            df["index"] = np.arange(frame_offset, frame_offset + length)
            
            # Remap task_index to new combined index
            new_task_idx = task_name_to_new_index[failed_task_name]
            df["task_index"] = new_task_idx
            
            # Write data
            dst_path = output_root / DEFAULT_DATA_PATH.format(
                chunk_index=current_chunk, file_index=current_file
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if contains_images:
                to_parquet_with_hf_images(df, dst_path)
            else:
                df.to_parquet(dst_path)
            
            # Create episode record
            ep_record = {
                "episode_index": new_ep_counter,
                "tasks": [failed_task_name],
                "length": length,
                "data/chunk_index": current_chunk,
                "data/file_index": current_file,
                "dataset_from_index": frame_offset,
                "dataset_to_index": frame_offset + length,
            }
            
            # Copy video metadata if present
            for key in [k for k in ep_meta.index if k.startswith("videos/")]:
                ep_record[key] = ep_meta[key]
            
            all_episode_records.append(ep_record)
            
            # Advance counters
            frame_offset += length
            current_chunk, current_file = update_chunk_file_indices(
                current_chunk, current_file, success_info["chunks_size"]
            )
            new_ep_counter += 1
    
    # Write episodes metadata
    logging.info("\nWriting episodes metadata...")
    episodes_ds = Dataset.from_pandas(pd.DataFrame(all_episode_records))
    write_episodes(episodes_ds, output_root)
    
    # Copy and merge videos if present
    video_keys = [k for k, ft in success_info["features"].items() if ft["dtype"] == "video"]
    if len(video_keys) > 0:
        logging.info("Copying video files...")
        _copy_videos_for_episodes(
            success_root, failed_root, output_root,
            success_episodes, failed_episodes,
            all_episode_records, video_keys
        )
    
    # Aggregate stats
    logging.info("Aggregating statistics...")
    success_stats = load_stats(success_root)
    failed_stats = load_stats(failed_root)
    if success_stats and failed_stats:
        combined_stats = aggregate_stats([success_stats, failed_stats])
        write_stats(combined_stats, output_root)
    
    # Write final info.json
    final_info = {
        **success_info,
        "total_episodes": new_ep_counter,
        "total_frames": frame_offset,
        "total_tasks": len(combined_tasks),
        "splits": {"train": f"0:{new_ep_counter}"},
    }
    write_info(final_info, output_root)
    
    logging.info(f"\nMixed dataset created successfully at {output_root}")
    logging.info(f"Total episodes: {new_ep_counter}")
    logging.info(f"Total frames: {frame_offset}")
    logging.info(f"Total tasks: {len(combined_tasks)}")


def _copy_videos_for_episodes(
    success_root: Path,
    failed_root: Path,
    output_root: Path,
    success_episodes: pd.DataFrame,
    failed_episodes: pd.DataFrame,
    new_episode_records: List[dict],
    video_keys: List[str],
) -> None:
    """Copy video files referenced by episodes to output."""
    import shutil
    
    # This is a simplified version - in production you'd want to consolidate videos
    # For now, just copy the referenced video files maintaining their structure
    
    for record in new_episode_records:
        for key in video_keys:
            chunk_key = f"videos/{key}/chunk_index"
            file_key = f"videos/{key}/file_index"
            
            if chunk_key in record and file_key in record:
                src_chunk = int(record[chunk_key])
                src_file = int(record[file_key])
                
                # Determine source root based on episode tasks
                task_name = record["tasks"][0]
                if task_name.startswith("Failed attempt to"):
                    src_root = failed_root
                else:
                    src_root = success_root
                
                src_path = src_root / DEFAULT_VIDEO_PATH.format(
                    video_key=key, chunk_index=src_chunk, file_index=src_file
                )
                dst_path = output_root / DEFAULT_VIDEO_PATH.format(
                    video_key=key, chunk_index=src_chunk, file_index=src_file
                )
                
                if src_path.exists() and not dst_path.exists():
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(
        description="Create a mixed LibERO 10 dataset with equal successful and failed episodes per task."
    )
    parser.add_argument(
        "--success_root",
        type=str,
        default="/home/josh/phddev/lerobot-upstream/outputs/libero_10",
        help="Path to successful libero_10 dataset root.",
    )
    parser.add_argument(
        "--failed_root",
        type=str,
        default="/home/josh/phddev/lerobot-upstream/outputs/failed_libero10_50eps",
        help="Path to failed libero_10 dataset root.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the mixed dataset will be created.",
    )
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    success_root = Path(args.success_root).resolve()
    failed_root = Path(args.failed_root).resolve()
    output_root = Path(args.output_path).resolve()
    
    if not (success_root / INFO_PATH).exists():
        raise FileNotFoundError(f"Success dataset not found: {success_root}")
    if not (failed_root / INFO_PATH).exists():
        raise FileNotFoundError(f"Failed dataset not found: {failed_root}")
    if output_root.exists():
        raise FileExistsError(f"Output path already exists: {output_root}")
    
    create_mixed_dataset(success_root, failed_root, output_root)


if __name__ == "__main__":
    main()

