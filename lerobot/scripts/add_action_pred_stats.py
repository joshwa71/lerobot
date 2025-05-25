import json
import argparse
from pathlib import Path
import numpy as np
import shutil

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

def calculate_done_stats(num_frames: int) -> dict:
    """
    Calculates mean and std for a binary 'done' signal.
    Assumes 'done' is 0 for all frames except the last one, which is 1.
    """
    if num_frames <= 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    if num_frames == 1: # Only one frame, which is the last one
        return {"min": 1.0, "max": 1.0, "mean": 1.0, "std": 0.0}

    # Create an array representing the 'done' signal for the episode
    done_signal = np.zeros(num_frames, dtype=float)
    done_signal[-1] = 1.0

    mean_done = np.mean(done_signal)
    std_done = np.std(done_signal)

    return {"min": 0.0, "max": 1.0, "mean": mean_done, "std": std_done}


def add_action_pred_stats_to_dataset(dataset_path_str: str):
    """
    Reads episodes_stats.jsonl from a LeRobot dataset, calculates statistics
    for 'action_pred' (action + done flag), and writes a new
    episodes_stats.jsonl file with the added statistics.

    Args:
        dataset_path_str (str): Path to the root of the LeRobot dataset.
    """
    dataset_path = Path(dataset_path_str).resolve()
    stats_file_path = dataset_path / "meta" / "episodes_stats.jsonl"
    output_stats_file_path = dataset_path / "meta" / "episodes_stats.jsonl"

    if not stats_file_path.exists():
        print(f"Error: episodes_stats.jsonl not found at {stats_file_path}")
        return

    print(f"Reading stats from: {stats_file_path}")
    episodes_stats_list = load_jsonl(stats_file_path)
    updated_episodes_stats_list = []

    for episode_data in episodes_stats_list:
        if "stats" not in episode_data or "action" not in episode_data["stats"]:
            print(f"Warning: Episode {episode_data.get('episode_index', 'N/A')} is missing 'stats' or 'stats.action'. Skipping.")
            updated_episodes_stats_list.append(episode_data)
            continue

        action_stats = episode_data["stats"]["action"]
        num_frames = int(action_stats.get("count", [0])[0]) # count is a list with one element

        if num_frames == 0:
            print(f"Warning: Episode {episode_data.get('episode_index', 'N/A')} has 0 frames in action stats. Skipping 'action_pred' calculation for this episode.")
            # Add a placeholder or default if needed, or just skip
            updated_episodes_stats_list.append(episode_data)
            continue

        done_stats = calculate_done_stats(num_frames)

        # Ensure all action_stats components are lists or convert them
        # Also handle cases where a stat might be a single number instead of a list
        current_action_min = np.array(action_stats["min"]).flatten().tolist()
        current_action_max = np.array(action_stats["max"]).flatten().tolist()
        current_action_mean = np.array(action_stats["mean"]).flatten().tolist()
        current_action_std = np.array(action_stats["std"]).flatten().tolist()


        action_pred_stats = {
            "min": current_action_min + [done_stats["min"]],
            "max": current_action_max + [done_stats["max"]],
            "mean": current_action_mean + [done_stats["mean"]],
            "std": current_action_std + [done_stats["std"]],
            "count": action_stats["count"] # Count remains the same
        }

        # Create a new dictionary for the updated episode to avoid modifying the original
        updated_episode_data = episode_data.copy()
        updated_episode_data["stats"] = episode_data["stats"].copy() # Ensure 'stats' dict is also copied
        updated_episode_data["stats"]["action_pred"] = action_pred_stats
        updated_episodes_stats_list.append(updated_episode_data)

    print(f"Writing updated stats to: {output_stats_file_path}")
    save_jsonl(updated_episodes_stats_list, output_stats_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adds 'action_pred' statistics (action + done flag) to a LeRobot dataset's episodes_stats.jsonl file."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the root directory of the LeRobot dataset (e.g., /path/to/lerobot/outputs/mixed_50)."
    )
    args = parser.parse_args()

    add_action_pred_stats_to_dataset(args.dataset_path)