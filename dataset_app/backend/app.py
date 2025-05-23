import argparse
import csv
import json
from io import StringIO
from pathlib import Path
import logging
import threading
from flask import Flask, jsonify, send_from_directory, abort, send_file, request

import numpy as np
import pandas as pd
from flask_cors import CORS

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import lerobot # To get the root path for templates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store lerobot_root_path
LEROBOT_ROOT_PATH = None

# Global variable to track ongoing merge operations
merge_operations = {}

# Helper functions from merge_datasets.py (copied to avoid import issues)
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
    import shutil
    
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

def merge_datasets_wrapper(source1_name, source2_name, target_name, operation_id):
    """
    Wrapper function that merges two datasets using the existing merge logic.
    This function is designed to be run in a separate thread.
    """
    global LEROBOT_ROOT_PATH, merge_operations
    
    try:
        # Update operation status
        merge_operations[operation_id]["status"] = "running"
        merge_operations[operation_id]["message"] = "Starting merge operation..."
        
        source1_path = Path(LEROBOT_ROOT_PATH) / source1_name
        source2_path = Path(LEROBOT_ROOT_PATH) / source2_name  
        target_path = Path(LEROBOT_ROOT_PATH) / target_name
        
        # Validate source paths exist
        if not source1_path.is_dir():
            raise ValueError(f"Source dataset '{source1_name}' not found")
        if not source2_path.is_dir():
            raise ValueError(f"Source dataset '{source2_name}' not found")
        if target_path.exists():
            raise ValueError(f"Target dataset '{target_name}' already exists")
        
        merge_operations[operation_id]["message"] = "Validating datasets..."
        
        # Initialize empty target metadata
        target_info = None
        target_episodes = []
        target_tasks_list = []
        target_episodes_stats = []
        
        # Merge source1 into target
        merge_operations[operation_id]["message"] = f"Merging dataset '{source1_name}'..."
        target_info, target_episodes, target_tasks_list, target_episodes_stats = merge_single_source_into_target(
            source1_path, target_path, target_info, target_episodes, target_tasks_list, target_episodes_stats
        )
        
        # Merge source2 into target  
        merge_operations[operation_id]["message"] = f"Merging dataset '{source2_name}'..."
        target_info, target_episodes, target_tasks_list, target_episodes_stats = merge_single_source_into_target(
            source2_path, target_path, target_info, target_episodes, target_tasks_list, target_episodes_stats
        )
        
        # Save final merged metadata
        merge_operations[operation_id]["message"] = "Saving merged metadata..."
        meta_target_path = target_path / "meta"
        save_jsonl(target_episodes, meta_target_path / "episodes.jsonl")
        save_jsonl(target_episodes_stats, meta_target_path / "episodes_stats.jsonl") 
        save_jsonl(target_tasks_list, meta_target_path / "tasks.jsonl")
        save_json(target_info, meta_target_path / "info.json")
        
        # Mark as completed
        merge_operations[operation_id]["status"] = "completed"
        merge_operations[operation_id]["message"] = f"Successfully merged datasets into '{target_name}'"
        
    except Exception as e:
        logger.error(f"Error during merge operation {operation_id}: {e}")
        merge_operations[operation_id]["status"] = "error"
        merge_operations[operation_id]["message"] = str(e)

# # Helper function to get dataset info without loading the full dataset object yet
# # This is a simplified version and might need to be more robust
def get_basic_dataset_info(dataset_path: Path):
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
            return {
                "repo_id": dataset_path.name,
                "total_episodes": info.get("total_episodes", 0),
                "total_frames": info.get("total_frames", 0),
                "fps": info.get("fps", 30),
                "codebase_version": info.get("codebase_version", "N/A")
            }
    return None

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes

    @app.route('/api/datasets', methods=['GET'])
    def list_datasets():
        global LEROBOT_ROOT_PATH
        if not LEROBOT_ROOT_PATH:
            logger.error("LEROBOT_ROOT_PATH is not set.")
            return jsonify({"error": "Dataset root path not configured"}), 500

        root_path = Path(LEROBOT_ROOT_PATH)
        datasets_info = []
        if root_path.exists() and root_path.is_dir():
            for dataset_dir in root_path.iterdir():
                if dataset_dir.is_dir():
                    # Check for meta/info.json to identify a lerobot dataset
                    info_file = dataset_dir / "meta" / "info.json"
                    if info_file.exists():
                        try:
                            basic_info = get_basic_dataset_info(dataset_dir)
                            if basic_info:
                                datasets_info.append(basic_info)
                        except Exception as e:
                            logger.error(f"Error reading dataset info for {dataset_dir.name}: {e}")
        else:
            logger.warning(f"LEROBOT_ROOT_PATH {LEROBOT_ROOT_PATH} does not exist or is not a directory.")
        return jsonify(datasets_info)

    @app.route('/api/datasets/<path:repo_id>/metadata', methods=['GET'])
    def get_dataset_metadata(repo_id: str):
        global LEROBOT_ROOT_PATH
        if not LEROBOT_ROOT_PATH:
            return jsonify({"error": "Dataset root path not configured"}), 500

        dataset_path = Path(LEROBOT_ROOT_PATH) / repo_id
        info_path = dataset_path / "meta" / "info.json"
        episodes_path = dataset_path / "meta" / "episodes.jsonl"

        if not dataset_path.is_dir() or not info_path.exists():
            return jsonify({"error": "Dataset not found"}), 404

        try:
            with open(info_path, 'r') as f:
                metadata = json.load(f)

            episodes_list = []
            if episodes_path.exists():
                 with open(episodes_path, 'r') as f:
                    for line in f:
                        episodes_list.append(json.loads(line))
            
            metadata["episodes_list"] = episodes_list
            # Add more metadata if needed, e.g., from tasks.jsonl
            return jsonify(metadata)
        except Exception as e:
            logger.error(f"Error loading metadata for {repo_id}: {e}")
            return jsonify({"error": f"Failed to load metadata for {repo_id}"}), 500


    def get_episode_data_from_dataset(dataset: LeRobotDataset, episode_index: int):
        columns = []
        selected_columns = [col for col, ft in dataset.features.items() if ft["dtype"] in ["float32", "int32"]]
        
        ignored_columns = []
        if "timestamp" in selected_columns:
            selected_columns.remove("timestamp")

        for column_name in list(selected_columns): # Iterate over a copy
            shape = dataset.features[column_name]["shape"]
            shape_dim = len(shape)
            if shape_dim > 1:
                selected_columns.remove(column_name)
                ignored_columns.append(column_name)

        header = ["timestamp"]
        for column_name in selected_columns:
            dim_state = dataset.meta.shapes[column_name][0]
            if "names" in dataset.features[column_name] and dataset.features[column_name]["names"]:
                column_names_for_header = dataset.features[column_name]["names"]
                while not isinstance(column_names_for_header, list):
                    column_names_for_header = list(column_names_for_header.values())[0]
            else:
                column_names_for_header = [f"{column_name}_{i}" for i in range(dim_state)]
            columns.append({"key": column_name, "value": column_names_for_header})
            header += column_names_for_header
        
        final_selected_columns_for_hf = ["timestamp"] + selected_columns

        from_idx = dataset.episode_data_index["from"][episode_index]
        to_idx = dataset.episode_data_index["to"][episode_index]
        
        # Ensure data is fetched as pandas DataFrame
        data_pd = (
            dataset.hf_dataset.select(range(from_idx, to_idx))
            .select_columns(final_selected_columns_for_hf)
            .with_format("pandas") # Ensure pandas format
        )
        
        # Convert to dict of lists, then back to DataFrame if necessary, or process directly
        # This step is to handle potential nested structures if not already flat
        data_dict = {}
        for col in final_selected_columns_for_hf:
            if col == "timestamp":
                 data_dict[col] = data_pd[col].tolist() # Direct conversion for simple columns
            else:
                # For columns that might be lists of arrays/tensors or similar
                # We need to flatten them into a list of lists for hstack
                if isinstance(data_pd[col].iloc[0], np.ndarray):
                    data_dict[col] = [arr.tolist() for arr in data_pd[col]]
                elif isinstance(data_pd[col].iloc[0], list):
                     data_dict[col] = data_pd[col].tolist()
                else: # Fallback for other types, assuming they can be directly used or need specific handling
                    data_dict[col] = data_pd[col].tolist()


        # Prepare rows for CSV ensuring all data is correctly formatted
        # Timestamp is the first column
        rows_for_csv = [data_dict["timestamp"]] 
        for col in selected_columns: # Exclude timestamp as it's already added
            # data_dict[col] is a list of lists (each inner list is a multi-dim value for one timestep)
            # We need to transpose it so that each list corresponds to one dimension over time
            transposed_col_data = list(map(list, zip(*data_dict[col])))
            rows_for_csv.extend(transposed_col_data)

        # Transpose back to get rows as list of [ts, val_dim1, val_dim2, ...]
        final_rows = list(map(list, zip(*rows_for_csv)))


        csv_buffer = StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(header)
        csv_writer.writerows(final_rows)
        csv_string = csv_buffer.getvalue()

        return csv_string, columns, ignored_columns

    @app.route('/api/datasets/<path:repo_id>/episodes/<int:episode_id>/data', methods=['GET'])
    def get_episode_data_route(repo_id: str, episode_id: int):
        global LEROBOT_ROOT_PATH
        if not LEROBOT_ROOT_PATH:
            return jsonify({"error": "Dataset root path not configured"}), 500

        dataset_root = Path(LEROBOT_ROOT_PATH) / repo_id
        if not dataset_root.is_dir():
            return jsonify({"error": "Dataset not found"}), 404

        try:
            # It's important that LeRobotDataset can find its files relative to its root
            # The `root` argument should be the specific path to the dataset directory.
            dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
        except Exception as e:
            logger.error(f"Failed to load LeRobotDataset {repo_id} from {dataset_root}: {e}")
            return jsonify({"error": f"Failed to load dataset {repo_id}. Check server logs."}), 500


        if episode_id < 0 or episode_id >= dataset.num_episodes:
            return jsonify({"error": "Episode not found"}), 404

        try:
            csv_str, columns_info, ignored_cols = get_episode_data_from_dataset(dataset, episode_id)
            
            videos_info = []
            if dataset.meta.video_keys:
                for key in dataset.meta.video_keys:
                    # Construct part of the path that is stable (chunk and video key)
                    # The full path will be /api/datasets/<repo_id>/videos/chunk-xxx/video_key_folder/episode_xxxxxx.mp4
                    video_chunk = dataset.meta.get_episode_chunk(episode_id)
                    video_key_folder_name = key # e.g. "observation.images.head"
                    
                    # The URL path should be relative to the /api/datasets/<repo_id>/videos/ prefix
                    # Example: videos/chunk-000/observation.images.head/episode_000000.mp4
                    # The filename itself: episode_000000.mp4
                    # The video key becomes part of the path
                    video_filename = f"episode_{episode_id:06d}.mp4"
                    
                    # Path for serving via Flask static route
                    # The part after /videos/ is what send_from_directory will get
                    api_video_url_path = f"chunk-{video_chunk:03d}/{video_key_folder_name}/{video_filename}"

                    videos_info.append({
                        "url": f"/api/datasets/{repo_id}/videos/{api_video_url_path}",
                        "filename": key, # This is the video stream name (e.g. observation.images.head)
                    })
            
            # Language instruction
            lang_instruction = None
            try:
                if dataset.meta.episodes and episode_id in dataset.meta.episodes:
                    tasks = dataset.meta.episodes[episode_id].get("tasks")
                    if tasks: # Assuming tasks is a list of strings
                        lang_instruction = tasks[0] if isinstance(tasks, list) and tasks else tasks

            except Exception as e:
                logger.warning(f"Could not retrieve language instruction for {repo_id} ep {episode_id}: {e}")


            return jsonify({
                "episode_id": episode_id,
                "csv_data": csv_str,
                "columns_info": columns_info,
                "ignored_columns": ignored_cols,
                "videos_info": videos_info,
                "language_instruction": lang_instruction,
                "fps": dataset.fps,
                "total_episodes_in_dataset": dataset.num_episodes,
                "all_episode_indices": list(range(dataset.num_episodes)) # For easy navigation
            })
        except Exception as e:
            logger.error(f"Error processing episode data for {repo_id} ep {episode_id}: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": "Failed to process episode data"}), 500

    @app.route('/api/datasets/<path:repo_id>/videos/<path:video_path_suffix>')
    def serve_video(repo_id: str, video_path_suffix: str):
        global LEROBOT_ROOT_PATH
        if not LEROBOT_ROOT_PATH:
            logger.error("LEROBOT_ROOT_PATH is not set, cannot serve video.")
            abort(500)

        # video_path_suffix is something like: chunk-000/observation.images.head/episode_000000.mp4
        # The actual video file is located at:
        # LEROBOT_ROOT_PATH / repo_id / videos / chunk-000 / observation.images.head / episode_000000.mp4
        
        # Construct the full path to the /videos/ directory for this dataset
        video_base_dir = Path(LEROBOT_ROOT_PATH) / repo_id / "videos"
        
        # The file to send is video_path_suffix relative to video_base_dir
        full_video_path = video_base_dir / video_path_suffix
        
        logger.info(f"Attempting to serve video: {full_video_path}")

        if not full_video_path.is_file():
            logger.error(f"Video file not found: {full_video_path}")
            abort(404)
        
        try:
            # send_from_directory expects the directory and then the filename separately
            # So, we need to provide video_base_dir and video_path_suffix
            return send_from_directory(video_base_dir, video_path_suffix, as_attachment=False)
        except FileNotFoundError:
            logger.error(f"send_from_directory could not find the file using base {video_base_dir} and suffix {video_path_suffix}.")
            abort(404)
        except Exception as e:
            logger.error(f"Error sending video file {full_video_path}: {e}")
            abort(500)

    @app.route('/api/datasets/merge', methods=['POST'])
    def merge_datasets():
        global LEROBOT_ROOT_PATH, merge_operations
        
        if not LEROBOT_ROOT_PATH:
            return jsonify({"error": "Dataset root path not configured"}), 500
            
        try:
            data = request.get_json()
            source1_name = data.get('source1')
            source2_name = data.get('source2') 
            target_name = data.get('target_name')
            
            if not all([source1_name, source2_name, target_name]):
                return jsonify({"error": "Missing required parameters: source1, source2, target_name"}), 400
                
            # Validate target name (basic validation)
            if not target_name.replace('_', '').replace('-', '').isalnum():
                return jsonify({"error": "Target name should only contain letters, numbers, hyphens, and underscores"}), 400
                
            # Generate operation ID
            import uuid
            operation_id = str(uuid.uuid4())
            
            # Initialize operation tracking
            merge_operations[operation_id] = {
                "status": "started",
                "message": "Initializing merge operation...",
                "source1": source1_name,
                "source2": source2_name, 
                "target_name": target_name
            }
            
            # Start merge operation in separate thread
            thread = threading.Thread(
                target=merge_datasets_wrapper,
                args=(source1_name, source2_name, target_name, operation_id)
            )
            thread.daemon = True
            thread.start()
            
            return jsonify({
                "operation_id": operation_id,
                "message": "Merge operation started"
            }), 202
            
        except Exception as e:
            logger.error(f"Error starting merge operation: {e}")
            return jsonify({"error": f"Failed to start merge operation: {str(e)}"}), 500

    @app.route('/api/datasets/merge/<operation_id>/status', methods=['GET'])
    def get_merge_status(operation_id):
        """Get the status of a merge operation"""
        global merge_operations
        
        if operation_id not in merge_operations:
            return jsonify({"error": "Operation not found"}), 404
            
        operation = merge_operations[operation_id]
        return jsonify({
            "operation_id": operation_id,
            "status": operation["status"],
            "message": operation["message"],
            "source1": operation["source1"],
            "source2": operation["source2"],
            "target_name": operation["target_name"]
        })

    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lerobot Dataset Visualizer Backend")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the backend server")
    parser.add_argument("--port", type=int, default=5001, help="Port for the backend server") # Changed default port
    parser.add_argument("--lerobot_root_path", type=str, required=True, help="Absolute path to the root of lerobot datasets (e.g., /path/to/lerobot/outputs)")
    args = parser.parse_args()

    LEROBOT_ROOT_PATH = args.lerobot_root_path
    logger.info(f"Lerobot dataset root path set to: {LEROBOT_ROOT_PATH}")

    # Verify LEROBOT_ROOT_PATH
    if not Path(LEROBOT_ROOT_PATH).is_dir():
        logger.error(f"Provided lerobot_root_path '{LEROBOT_ROOT_PATH}' is not a valid directory or does not exist.")
        exit(1)


    app = create_app()
    app.run(host=args.host, port=args.port, debug=True) 