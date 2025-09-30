#!/usr/bin/env python3
# /home/josh/phddev/lerobot-upstream/src/lerobot/scripts/update_mixed_instructions.py
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import json
import pandas as pd
from pathlib import Path


def update_mixed_dataset_instructions(dataset_path: str, success_task: str, fail_task: str):
    """
    Updates a LeRobot dataset to have different language instructions based on success flag.
    
    Args:
        dataset_path: Path to the dataset directory
        success_task: Language instruction for successful episodes
        fail_task: Language instruction for failed episodes
    """
    dataset_path = Path(dataset_path)
    meta_path = dataset_path / "meta"
    data_path = dataset_path / "data"
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    if not meta_path.exists():
        raise ValueError(f"Meta directory does not exist: {meta_path}")
    
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")
    
    # Step 1: Update tasks.jsonl with both tasks
    print("Updating tasks.jsonl...")
    tasks = [
        {"task_index": 0, "task": success_task},
        {"task_index": 1, "task": fail_task}
    ]
    
    with open(meta_path / "tasks.jsonl", "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    
    # Step 2: Read episodes.jsonl to get success flags
    print("Reading episodes.jsonl...")
    episodes = []
    episode_success_map = {}
    
    episodes_file = meta_path / "episodes.jsonl"
    if not episodes_file.exists():
        raise ValueError(f"episodes.jsonl not found: {episodes_file}")
    
    with open(episodes_file, "r") as f:
        for line in f:
            if line.strip():
                episode = json.loads(line.strip())
                episodes.append(episode)
                if "success" not in episode:
                    raise ValueError(f"Episode {episode['episode_index']} missing 'success' field")
                episode_success_map[episode["episode_index"]] = episode["success"]
    
    # Step 3: Update each parquet file with correct task_index
    print("Updating parquet files...")
    chunk_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("chunk-")]
    
    if not chunk_dirs:
        raise ValueError(f"No chunk directories found in: {data_path}")
    
    updated_count = 0
    for chunk_dir in chunk_dirs:
        parquet_files = list(chunk_dir.glob("episode_*.parquet"))
        
        for parquet_file in parquet_files:
            # Extract episode index from filename
            episode_index = int(parquet_file.stem.split("_")[-1])
            
            if episode_index not in episode_success_map:
                print(f"Warning: Episode {episode_index} not found in episodes.jsonl, skipping...")
                continue
            
            # Read the parquet file
            df = pd.read_parquet(parquet_file)
            
            # Update task_index based on success flag
            is_success = episode_success_map[episode_index]
            new_task_index = 0 if is_success else 1
            
            df["task_index"] = new_task_index
            
            # Save back to parquet
            df.to_parquet(parquet_file, index=False)
            
            updated_count += 1
            if updated_count % 10 == 0:
                print(f"Updated {updated_count} episodes...")
    
    # Step 4: Update episodes.jsonl with correct tasks
    print("Updating episodes.jsonl...")
    updated_episodes = []
    
    for episode in episodes:
        episode_index = episode["episode_index"]
        is_success = episode["success"]
        
        # Update the tasks list based on success flag
        if is_success:
            episode["tasks"] = [success_task]
        else:
            episode["tasks"] = [fail_task]
            
        updated_episodes.append(episode)
    
    # Write updated episodes.jsonl
    with open(episodes_file, "w") as f:
        for episode in updated_episodes:
            f.write(json.dumps(episode) + "\n")
    
    print("Dataset update completed!")
    
    # Print summary
    success_count = sum(1 for ep in episodes if ep["success"])
    fail_count = len(episodes) - success_count
    print(f"Summary:")
    print(f"  Total episodes: {len(episodes)}")
    print(f"  Successful episodes: {success_count} -> '{success_task}'")
    print(f"  Failed episodes: {fail_count} -> '{fail_task}'")
    print(f"  Updated parquet files: {updated_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Update LeRobot dataset with different language instructions based on success flag"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset directory (e.g., 'outputs/bl_mixed_100')"
    )
    parser.add_argument(
        "--success-task",
        type=str,
        default="Grasp a lego block and put it in the red area.",
        help="Language instruction for successful episodes (default: 'Grasp a lego block and put it in the red area.')"
    )
    parser.add_argument(
        "--fail-task", 
        type=str,
        default="Fail to grasp a lego block and put it in the red area.",
        help="Language instruction for failed episodes (default: 'Fail to grasp a lego block and put it in the red area.')"
    )
    
    args = parser.parse_args()
    
    try:
        update_mixed_dataset_instructions(args.dataset_path, args.success_task, args.fail_task)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
