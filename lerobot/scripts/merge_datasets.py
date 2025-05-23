#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

# Helper functions
def load_jsonl(file_path: Path):
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def save_jsonl(data: list, file_path: Path):
    """Saves a list of dictionaries to a JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_json(file_path: Path):
    """Loads a JSON file into a dictionary."""
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_json(data: dict, file_path: Path):
    """Saves a dictionary to a JSON file with indentation."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def merge_single_source_into_target(source_path, target_path, target_info, target_episodes, target_tasks_list, target_episodes_stats):
    """Merge a single source dataset into target dataset"""
    
    # Create target directories if they don't exist
    meta_target_path = target_path / "meta"
    data_target_path = target_path / "data" / "chunk-000"
    videos_target_path_base = target_path / "videos" / "chunk-000"
    
    meta_target_path.mkdir(parents=True, exist_ok=True)
    data_target_path.mkdir(parents=True, exist_ok=True)
    # Video subdirs will be created as needed

    # 1. Load metadata from source
    source_info = load_json(source_path / "meta" / "info.json")
    source_episodes = load_jsonl(source_path / "meta" / "episodes.jsonl")
    source_tasks_list = load_jsonl(source_path / "meta" / "tasks.jsonl") 
    source_episodes_stats = load_jsonl(source_path / "meta" / "episodes_stats.jsonl")

    if not source_info or not source_episodes:
        print(f"Error: Source dataset metadata (info.json or episodes.jsonl) not found or incomplete in {source_path / 'meta'}")
        return target_info, target_episodes, target_tasks_list, target_episodes_stats
    
    if source_info.get("total_chunks", 1) > 1:
        print("Warning: Source dataset appears to have multiple chunks. This script is designed for single-chunk datasets. Merging might be incomplete or incorrect.")
    
    # 2. Initialize target metadata if it's None (first source)
    if target_info is None:
        print(f"Initializing new target dataset metadata based on first source.")
        target_info = {
            "codebase_version": source_info.get("codebase_version", "v2.1"),
            "robot_type": source_info.get("robot_type", "unknown"),
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "total_videos": 0,
            "total_chunks": 1, 
            "chunks_size": source_info.get("chunks_size", 1000), 
            "fps": source_info.get("fps", 30), 
            "splits": {"train": ""}, 
            "data_path": source_info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"),
            "video_path": source_info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"),
            "features": source_info.get("features", {}) 
        }
        target_episodes = []
        target_tasks_list = []
        target_episodes_stats = []

    # 3. Determine offsets
    episode_index_offset = target_info["total_episodes"]
    frame_index_offset = target_info["total_frames"] 

    # 4. Merge tasks and create task mapping
    merged_tasks_map = {t['task_index']: t['task'] for t in target_tasks_list}
    merged_descriptions_to_idx_map = {t['task']: t['task_index'] for t in target_tasks_list}

    source_old_task_idx_to_new_task_idx_map = {}
    
    current_max_merged_task_idx = -1
    if merged_tasks_map:
        current_max_merged_task_idx = max(merged_tasks_map.keys())
    
    next_available_task_idx = current_max_merged_task_idx + 1

    for src_task_obj in source_tasks_list:
        old_src_idx = src_task_obj['task_index']
        src_desc = src_task_obj['task']
        
        if src_desc in merged_descriptions_to_idx_map:
            source_old_task_idx_to_new_task_idx_map[old_src_idx] = merged_descriptions_to_idx_map[src_desc]
        else:
            merged_tasks_map[next_available_task_idx] = src_desc
            merged_descriptions_to_idx_map[src_desc] = next_available_task_idx
            source_old_task_idx_to_new_task_idx_map[old_src_idx] = next_available_task_idx
            next_available_task_idx += 1
    
    final_merged_tasks_for_jsonl = [{"task_index": idx, "task": desc} for idx, desc in merged_tasks_map.items()]
    final_merged_tasks_for_jsonl.sort(key=lambda x: x['task_index'])

    # 5. Process source episodes.jsonl
    updated_source_episodes = []
    for episode in source_episodes:
        new_episode = episode.copy()
        new_episode["episode_index"] += episode_index_offset
        updated_source_episodes.append(new_episode)

    # 6. Process source episodes_stats.jsonl
    updated_source_episodes_stats = []
    for stats_item in source_episodes_stats:
        new_stats = stats_item.copy()
        new_stats["episode_index"] += episode_index_offset

        if "stats" in new_stats:
            if "index" in new_stats["stats"]: 
                idx_stats = new_stats["stats"]["index"]
                for key in ["min", "max", "mean"]: 
                    if key in idx_stats and isinstance(idx_stats[key], list) and idx_stats[key]:
                        idx_stats[key][0] += frame_index_offset
            
            if "task_index" in new_stats["stats"]:
                task_idx_stats = new_stats["stats"]["task_index"]
                for key in ["min", "max"]: 
                     if key in task_idx_stats and isinstance(task_idx_stats[key], list) and task_idx_stats[key]:
                        old_idx = task_idx_stats[key][0]
                        task_idx_stats[key][0] = source_old_task_idx_to_new_task_idx_map.get(old_idx, old_idx) 
                if "mean" in task_idx_stats and isinstance(task_idx_stats["mean"], list) and task_idx_stats["mean"]:
                    old_idx_float = task_idx_stats["mean"][0]
                    old_idx_int = int(old_idx_float)
                    if old_idx_float == old_idx_int: 
                         task_idx_stats["mean"][0] = float(source_old_task_idx_to_new_task_idx_map.get(old_idx_int, old_idx_int))

        updated_source_episodes_stats.append(new_stats)

    # 7. Copy and rename data files (.parquet)
    source_data_files_path = source_path / "data" / "chunk-000"
    if source_data_files_path.is_dir():
        num_parquet_copied = 0
        print(f"Attempting to copy {source_info['total_episodes']} parquet episode files from {source_data_files_path}...")
        for i in range(source_info["total_episodes"]):
            old_episode_num_str = f"{i:06d}"
            new_episode_num_str = f"{i + episode_index_offset:06d}"
            src_file = source_data_files_path / f"episode_{old_episode_num_str}.parquet"
            tgt_file = data_target_path / f"episode_{new_episode_num_str}.parquet"
            if src_file.exists():
                shutil.copy2(src_file, tgt_file)
                num_parquet_copied +=1
            else:
                print(f"Warning: Source data file {src_file} not found.")
        print(f"Finished copying parquet files. {num_parquet_copied} files copied to {data_target_path}.")
    
    # 8. Copy and rename video files (.mp4)
    video_feature_keys = []
    if "features" in source_info:
        for f_key, f_props in source_info["features"].items():
            if f_props.get("dtype") == "video" and f_key.startswith("observation.images."):
                video_feature_keys.append(f_key) 
    
    if not video_feature_keys: 
        video_feature_keys = ["observation.images.head", "observation.images.wrist"]
        print("Warning: Video keys not found in source_info.features, using default head/wrist.")

    for video_feature_key in video_feature_keys:
        source_video_dir = source_path / "videos" / "chunk-000" / video_feature_key
        target_video_dir = videos_target_path_base / video_feature_key
        target_video_dir.mkdir(parents=True, exist_ok=True)

        if source_video_dir.is_dir():
            num_video_files_copied_for_stream = 0
            print(f"Attempting to copy {source_info['total_episodes']} '{video_feature_key}' video files from {source_video_dir}...")
            for i in range(source_info["total_episodes"]):
                old_episode_num_str = f"{i:06d}"
                new_episode_num_str = f"{i + episode_index_offset:06d}"
                src_file = source_video_dir / f"episode_{old_episode_num_str}.mp4"
                tgt_file = target_video_dir / f"episode_{new_episode_num_str}.mp4"
                if src_file.exists():
                    shutil.copy2(src_file, tgt_file)
                    num_video_files_copied_for_stream += 1
                else:
                    print(f"Warning: Source video file {src_file} not found for key {video_feature_key}.")
            print(f"Finished copying '{video_feature_key}' video files. {num_video_files_copied_for_stream} files copied to {target_video_dir}.")
        else:
            print(f"Warning: Source video directory {source_video_dir} not found for key {video_feature_key}.")

    # 9. Update target info.json
    target_info["total_episodes"] += source_info["total_episodes"]
    target_info["total_frames"] += source_info["total_frames"]
    target_info["total_tasks"] = len(final_merged_tasks_for_jsonl)
    
    num_video_streams = len(video_feature_keys) if video_feature_keys else 2 
    target_info["total_videos"] = target_info.get("total_videos",0) + \
                                 source_info.get("total_videos", num_video_streams * source_info["total_episodes"])

    if target_info["total_episodes"] > 0:
        target_info["splits"]["train"] = f"0:{target_info['total_episodes']}" 
    else:
        target_info["splits"]["train"] = ""

    # Return updated target metadata
    return target_info, target_episodes + updated_source_episodes, final_merged_tasks_for_jsonl, target_episodes_stats + updated_source_episodes_stats

def main():
    parser = argparse.ArgumentParser(
        description="Merge two LeRobot datasets (source1 and source2 into target). Assumes single-chunk datasets."
    )
    parser.add_argument("source1_dir", type=str, help="Path to the first source dataset directory.")
    parser.add_argument("source2_dir", type=str, help="Path to the second source dataset directory.")
    parser.add_argument("target_dir", type=str, help="Path to the target dataset directory. This dataset will be created.")
    args = parser.parse_args()

    source1_path = Path(args.source1_dir).resolve()
    source2_path = Path(args.source2_dir).resolve()
    target_path = Path(args.target_dir).resolve()

    if not source1_path.is_dir():
        print(f"Error: Source1 directory {source1_path} not found.")
        return
    if not source2_path.is_dir():
        print(f"Error: Source2 directory {source2_path} not found.")
        return

    # Initialize empty target metadata
    target_info = None
    target_episodes = []
    target_tasks_list = []
    target_episodes_stats = []

    # Merge source1 into target
    print(f"Merging dataset from {source1_path} into target...")
    target_info, target_episodes, target_tasks_list, target_episodes_stats = merge_single_source_into_target(
        source1_path, target_path, target_info, target_episodes, target_tasks_list, target_episodes_stats
    )

    # Merge source2 into target
    print(f"Merging dataset from {source2_path} into target...")
    target_info, target_episodes, target_tasks_list, target_episodes_stats = merge_single_source_into_target(
        source2_path, target_path, target_info, target_episodes, target_tasks_list, target_episodes_stats
    )

    # Save final merged metadata to target
    meta_target_path = target_path / "meta"
    save_jsonl(target_episodes, meta_target_path / "episodes.jsonl")
    save_jsonl(target_episodes_stats, meta_target_path / "episodes_stats.jsonl")
    save_jsonl(target_tasks_list, meta_target_path / "tasks.jsonl")
    save_json(target_info, meta_target_path / "info.json")

    print(f"Successfully merged datasets from {source1_path} and {source2_path} into {target_path}")

if __name__ == "__main__":
    main() 