#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd # For easier Parquet manipulation

# Helper functions (adapted from your merge script)
def load_jsonl(file_path: Path):
    data = []
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def save_jsonl(data: list, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_json(file_path: Path):
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_json(data: dict, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def update_parquet_file_for_filtering(file_path: Path, new_episode_index: int, global_frame_offset: int):
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        num_frames_in_episode = len(df)

        if 'episode_index' in df.columns:
            df['episode_index'] = new_episode_index
        else:
            print(f"Warning: 'episode_index' column not found in {file_path}. Creating it.")
            df['episode_index'] = new_episode_index
        
        if 'index' in df.columns:
            df['index'] = range(global_frame_offset, global_frame_offset + num_frames_in_episode)
        else:
            print(f"Warning: 'index' column not found in {file_path}. Creating it.")
            df['index'] = range(global_frame_offset, global_frame_offset + num_frames_in_episode)

        schema = table.schema
        updated_table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        pq.write_table(updated_table, file_path)
    except Exception as e:
        print(f"Error updating Parquet file {file_path}: {e}")

def filter_dataset(dataset_dir: Path, max_len: int):
    print(f"Processing dataset at: {dataset_dir}")
    print(f"Filtering episodes longer than: {max_len} frames.")
    
    meta_dir = dataset_dir / "meta"
    data_dir_base = dataset_dir / "data"
    videos_dir_base = dataset_dir / "videos"
    embeddings_dir_base = dataset_dir / "embeddings"

    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"

    info = load_json(info_path)
    original_episodes_list = load_jsonl(episodes_path)
    original_episodes_stats_list = load_jsonl(episodes_stats_path)

    if not info or not original_episodes_list:
        print("Error: info.json or episodes.jsonl not found or empty. Aborting.")
        return

    CHUNK_ID_STR = "chunk-000"
    data_chunk_dir = data_dir_base / CHUNK_ID_STR
    videos_chunk_dir = videos_dir_base / CHUNK_ID_STR
    embeddings_chunk_dir = embeddings_dir_base / CHUNK_ID_STR

    kept_episodes_info = [] 
    for ep_data in original_episodes_list:
        if ep_data["length"] <= max_len:
            kept_episodes_info.append({
                "original_index": ep_data["episode_index"],
                "length": ep_data["length"]
            })
    
    if not kept_episodes_info:
        print("Warning: No episodes met the criteria. The dataset will become empty.")
    else:
        print(f"Found {len(kept_episodes_info)} episodes to keep (out of {len(original_episodes_list)}).")

    kept_episodes_info.sort(key=lambda x: x["original_index"])
    old_to_new_idx_map = {ep_info["original_index"]: new_idx for new_idx, ep_info in enumerate(kept_episodes_info)}
    new_to_old_data_map = {new_idx: ep_info for new_idx, ep_info in enumerate(kept_episodes_info)}

    new_episodes_list = []
    for original_ep_data in original_episodes_list:
        original_idx = original_ep_data["episode_index"]
        if original_idx in old_to_new_idx_map:
            new_idx = old_to_new_idx_map[original_idx]
            updated_ep_data = original_ep_data.copy()
            updated_ep_data["episode_index"] = new_idx
            new_episodes_list.append(updated_ep_data)
    new_episodes_list.sort(key=lambda x: x["episode_index"]) 
    save_jsonl(new_episodes_list, episodes_path)
    print(f"Updated {episodes_path}")

    new_episodes_stats_list = []
    current_global_frame_offset_for_stats = 0 # Use a distinct counter for stats processing
    original_stats_dict = {s["episode_index"]: s for s in original_episodes_stats_list}

    for new_idx in range(len(kept_episodes_info)):
        old_idx_data = new_to_old_data_map[new_idx]
        original_idx = old_idx_data["original_index"]
        episode_length = old_idx_data["length"]

        if original_idx in original_stats_dict:
            stat_item_copy = original_stats_dict[original_idx].copy()
            # Ensure deep copy of the 'stats' dictionary to avoid modifying original_stats_dict
            stat_item_copy["stats"] = json.loads(json.dumps(original_stats_dict[original_idx]["stats"]))
            
            stat_item_copy["episode_index"] = new_idx
            
            if "stats" in stat_item_copy:
                stats_inner_dict = stat_item_copy["stats"]
                if "episode_index" in stats_inner_dict:
                    # ***** CORRECTED SECTION *****
                    stats_inner_dict["episode_index"]["min"] = [new_idx]
                    stats_inner_dict["episode_index"]["max"] = [new_idx]
                    stats_inner_dict["episode_index"]["mean"] = [float(new_idx)]
                    stats_inner_dict["episode_index"]["std"] = [0.0]
                    # 'count' is assumed to be correctly a list from original, e.g., [episode_length]
                    # If not, it should be: stats_inner_dict["episode_index"]["count"] = [episode_length]
                    if "count" not in stats_inner_dict["episode_index"] or not isinstance(stats_inner_dict["episode_index"]["count"], list):
                        print(f"Warning: Fixing count for episode_index {new_idx} in stats.")
                        stats_inner_dict["episode_index"]["count"] = [episode_length]

                if "index" in stats_inner_dict: # Global frame index
                    stats_inner_dict["index"]["min"] = [current_global_frame_offset_for_stats]
                    stats_inner_dict["index"]["max"] = [current_global_frame_offset_for_stats + episode_length - 1]
                    stats_inner_dict["index"]["mean"] = [current_global_frame_offset_for_stats + (episode_length - 1) / 2.0]
                    # Std of global indices for an episode is same as std of its frame_indices
                    if "frame_index" in stats_inner_dict and "std" in stats_inner_dict["frame_index"] and \
                       isinstance(stats_inner_dict["frame_index"]["std"], list):
                        stats_inner_dict["index"]["std"] = stats_inner_dict["frame_index"]["std"]
                    else: # Fallback if frame_index.std is missing or not a list
                        # This calculation for std of a sequence 0..N-1 might not be exactly what LeRobot expects
                        # but it's a reasonable placeholder. Or copy from a known good example.
                        # For simplicity, if original frame_index.std is bad, we make it [0.0] or similar
                        print(f"Warning: 'frame_index.std' for original episode {original_idx} is not a list or missing. Setting 'index.std' to a default.")
                        from statistics import pstdev
                        if episode_length > 1:
                             stats_inner_dict["index"]["std"] = [pstdev(range(episode_length))]
                        else:
                             stats_inner_dict["index"]["std"] = [0.0]


            new_episodes_stats_list.append(stat_item_copy)
            current_global_frame_offset_for_stats += episode_length
        else:
            print(f"Warning: Stats for original episode_index {original_idx} not found in episodes_stats.jsonl.")
    
    new_episodes_stats_list.sort(key=lambda x: x["episode_index"])
    save_jsonl(new_episodes_stats_list, episodes_stats_path)
    print(f"Updated {episodes_stats_path}")

    temp_suffix = "_temp_filter"
    temp_data_chunk_dir = data_dir_base / f"{CHUNK_ID_STR}{temp_suffix}"
    if temp_data_chunk_dir.exists(): shutil.rmtree(temp_data_chunk_dir)
    temp_data_chunk_dir.mkdir(parents=True, exist_ok=True)

    print("Processing Parquet files...")
    current_global_frame_offset_for_parquet = 0
    for new_idx in range(len(kept_episodes_info)):
        old_idx_data = new_to_old_data_map[new_idx]
        original_idx = old_idx_data["original_index"]
        episode_length = old_idx_data["length"]
        original_parquet_filename = Path(info["data_path"]).name.format(episode_chunk=0, episode_index=original_idx)
        original_parquet_path = data_chunk_dir / original_parquet_filename
        new_parquet_filename = Path(info["data_path"]).name.format(episode_chunk=0, episode_index=new_idx)
        temp_new_parquet_path = temp_data_chunk_dir / new_parquet_filename
        if original_parquet_path.exists():
            shutil.copy2(original_parquet_path, temp_new_parquet_path)
            update_parquet_file_for_filtering(temp_new_parquet_path, new_idx, current_global_frame_offset_for_parquet)
        else:
            print(f"Warning: Parquet file {original_parquet_path} not found for original episode {original_idx}.")
        current_global_frame_offset_for_parquet += episode_length
    
    if data_chunk_dir.exists(): shutil.rmtree(data_chunk_dir)
    os.rename(temp_data_chunk_dir, data_chunk_dir)
    print(f"Parquet files processed and moved to {data_chunk_dir}")

    video_keys = [k for k, v_info in info.get("features", {}).items() if v_info.get("dtype") == "video"]
    if not video_keys and videos_chunk_dir.exists():
        video_keys = [d.name for d in videos_chunk_dir.iterdir() if d.is_dir() and not d.name.endswith(temp_suffix)]
        if video_keys: print(f"Inferred video keys from directory structure: {video_keys}")

    print("Processing video files...")
    if videos_chunk_dir.exists() and video_keys:
        for video_key in video_keys:
            original_video_key_dir = videos_chunk_dir / video_key
            temp_video_key_dir = videos_chunk_dir / f"{video_key}{temp_suffix}"
            if temp_video_key_dir.exists(): shutil.rmtree(temp_video_key_dir)
            temp_video_key_dir.mkdir(parents=True, exist_ok=True)
            if original_video_key_dir.exists():
                for new_idx in range(len(kept_episodes_info)):
                    old_idx_data = new_to_old_data_map[new_idx]
                    original_idx = old_idx_data["original_index"]
                    video_filename_template = Path(info["video_path"]).name
                    original_video_filename = video_filename_template.format(episode_chunk=0, video_key=video_key, episode_index=original_idx)
                    original_video_path = original_video_key_dir / original_video_filename
                    new_video_filename = video_filename_template.format(episode_chunk=0, video_key=video_key, episode_index=new_idx)
                    temp_new_video_path = temp_video_key_dir / new_video_filename
                    if original_video_path.exists():
                        shutil.copy2(original_video_path, temp_new_video_path)
                shutil.rmtree(original_video_key_dir)
                os.rename(temp_video_key_dir, original_video_key_dir)
                print(f"Video files for key '{video_key}' processed and moved to {original_video_key_dir}")
    else:
        print("No video files or keys to process, or videos directory/chunk does not exist.")

    print("Processing embedding files...")
    if embeddings_chunk_dir.exists() and video_keys:
        for video_key in video_keys:
            original_embedding_key_dir = embeddings_chunk_dir / video_key
            temp_embedding_key_dir = embeddings_chunk_dir / f"{video_key}{temp_suffix}"
            if temp_embedding_key_dir.exists(): shutil.rmtree(temp_embedding_key_dir)
            temp_embedding_key_dir.mkdir(parents=True, exist_ok=True)
            if original_embedding_key_dir.exists():
                for new_idx in range(len(kept_episodes_info)):
                    old_idx_data = new_to_old_data_map[new_idx]
                    original_idx = old_idx_data["original_index"]
                    original_ep_emb_dirname = f"episode_{original_idx:06d}"
                    original_ep_emb_path = original_embedding_key_dir / original_ep_emb_dirname
                    new_ep_emb_dirname = f"episode_{new_idx:06d}"
                    temp_new_ep_emb_path = temp_embedding_key_dir / new_ep_emb_dirname
                    if original_ep_emb_path.exists() and original_ep_emb_path.is_dir():
                        shutil.copytree(original_ep_emb_path, temp_new_ep_emb_path)
                shutil.rmtree(original_embedding_key_dir)
                os.rename(temp_embedding_key_dir, original_embedding_key_dir)
                print(f"Embedding files for key '{video_key}' processed and moved to {original_embedding_key_dir}")
    elif embeddings_dir_base.exists():
         print("Embeddings directory exists but no video keys found/inferred to process associated embeddings.")
    else:
        print("No embedding files/directory to process.")
        
    new_total_episodes = len(kept_episodes_info)
    new_total_frames = sum(ep["length"] for ep in kept_episodes_info)
    num_video_streams = len(video_keys) if video_keys else 0
    new_total_videos = new_total_episodes * num_video_streams
    info["total_episodes"] = new_total_episodes
    info["total_frames"] = new_total_frames
    info["total_videos"] = new_total_videos
    if new_total_episodes > 0:
        info["splits"]["train"] = f"0:{new_total_episodes}"
    else:
        info["splits"]["train"] = "0:0"
    save_json(info, info_path)
    print(f"Updated {info_path}")
    print("Dataset filtering complete.")

def main():
    parser = argparse.ArgumentParser(description="Filter LeRobot dataset episodes by maximum length.")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to the LeRobot dataset directory.")
    parser.add_argument("--max_len", type=int, required=True, help="Maximum length of episodes to keep (inclusive).")
    args = parser.parse_args()
    if not args.dataset_path.is_dir():
        print(f"Error: Dataset directory {args.dataset_path} not found.")
        return
    if args.max_len < 0:
        print(f"Error: max_len must be non-negative.")
        return
    print("\nWARNING: This script will modify the dataset IN-PLACE.")
    print("It is STRONGLY recommended to BACKUP your dataset before proceeding.")
    proceed = input("Do you want to continue? (yes/no): ").lower()
    if proceed != 'yes':
        print("Operation cancelled by user.")
        return
    filter_dataset(args.dataset_path.resolve(), args.max_len)

if __name__ == "__main__":
    main()