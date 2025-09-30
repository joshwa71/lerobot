#!/usr/bin/env python
# /home/josh/phddev/lerobot-upstream/src/lerobot/scripts/convert_libero_to_lerobot.py
"""
Script to convert LIBERO datasets from HDF5 format to LeRobot dataset format.

Usage:
    python convert_libero_to_lerobot.py \
        --input-path /path/to/libero_dataset.hdf5 \
        --output-path /path/to/output/dataset \
        --repo-id username/dataset_name
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import h5py
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
LIBERO_FPS = 20
LIBERO_ROBOT_TYPE = "panda"
CODEBASE_VERSION = "v2.1"

# Feature mapping from LIBERO to LeRobot
LIBERO_TO_LEROBOT_FEATURES = {
    # Camera observations
    "agentview_rgb": "observation.images.head",
    "eye_in_hand_rgb": "observation.images.wrist",
    
    # Robot state observations - these will be concatenated into observation.state
    "ee_pos": None,  # Will be part of observation.state
    "ee_ori": None,  # Will be part of observation.state
    "gripper_states": None,  # Will be part of observation.state
    "joint_states": None,  # Will be part of observation.state
    
    # Actions
    "actions": "action",
}


def extract_libero_metadata(hdf5_path: Path) -> Dict[str, Any]:
    """Extract metadata from LIBERO HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        # Get problem info
        problem_info = json.loads(f["data"].attrs["problem_info"])
        language_instruction = problem_info["language_instruction"].strip('"')
        
        # Get environment metadata
        env_meta = json.loads(f["data"].attrs["env_args"])
        
        # Count episodes
        demos = sorted(list(f["data"].keys()))
        num_episodes = len(demos)
        
        # Get data shapes from first episode
        first_demo = f[f"data/{demos[0]}"]
        
        metadata = {
            "task": language_instruction,
            "robot_type": LIBERO_ROBOT_TYPE,
            "fps": LIBERO_FPS,
            "num_episodes": num_episodes,
            "env_meta": env_meta,
            "demos": demos,
            "shapes": {
                "agentview_rgb": first_demo["obs/agentview_rgb"].shape[1:],  # Remove time dimension
                "eye_in_hand_rgb": first_demo["obs/eye_in_hand_rgb"].shape[1:],
                "ee_pos": first_demo["obs/ee_pos"].shape[1:],
                "ee_ori": first_demo["obs/ee_ori"].shape[1:],
                "gripper_states": first_demo["obs/gripper_states"].shape[1:],
                "joint_states": first_demo["obs/joint_states"].shape[1:],
                "actions": first_demo["actions"].shape[1:],
            }
        }
        
    return metadata


def define_lerobot_features(metadata: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Define LeRobot features based on LIBERO metadata."""
    features = {}
    
    # Define action feature (7D: 3D pos + 3D ori + 1D gripper)
    features["action"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    }
    
    # Define observation.state feature
    # Concatenate: ee_pos (3) + ee_ori (3) + gripper_states (2) + joint_states (7) = 15
    features["observation.state"] = {
        "dtype": "float32", 
        "shape": (15,),
        "names": [
            "ee_pos_x", "ee_pos_y", "ee_pos_z",
            "ee_ori_x", "ee_ori_y", "ee_ori_z",
            "gripper_pos", "gripper_force",
            "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        ]
    }
    
    features["observation.images.head"] = {
        "dtype": "video",
        "shape": (128, 128, 3),
        "names": ["height", "width", "channel"],
    }
    features["observation.images.wrist"] = {
        "dtype": "video",
        "shape": (128, 128, 3),
        "names": ["height", "width", "channel"],
    }
    
    return features


def load_libero_episode(hdf5_file: h5py.File, demo_name: str) -> Dict[str, np.ndarray]:
    """Load a single episode from LIBERO HDF5 file."""
    demo_data = hdf5_file[f"data/{demo_name}"]
    
    episode = {
        "actions": demo_data["actions"][:],
        "agentview_rgb": demo_data["obs/agentview_rgb"][:],
        "eye_in_hand_rgb": demo_data["obs/eye_in_hand_rgb"][:],
        "ee_pos": demo_data["obs/ee_pos"][:],
        "ee_ori": demo_data["obs/ee_ori"][:],
        "gripper_states": demo_data["obs/gripper_states"][:],
        "joint_states": demo_data["obs/joint_states"][:],
        "dones": demo_data["dones"][:],
        "rewards": demo_data["rewards"][:]
    }
    
    return episode


def convert_episode_to_lerobot_format(
    episode_data: Dict[str, np.ndarray],
    task: str,
    fps: float
) -> Dict[str, Any]:
    """Convert LIBERO episode data to LeRobot format."""
    num_frames = len(episode_data["actions"])
    
    frames = []
    for i in range(num_frames):
        # Create observation.state by concatenating robot state info
        obs_state = np.concatenate([
            episode_data["ee_pos"][i],
            episode_data["ee_ori"][i],
            episode_data["gripper_states"][i],
            episode_data["joint_states"][i]
        ]).astype(np.float32)
        
        frame = {
            "action": episode_data["actions"][i].astype(np.float32),
            "observation.state": obs_state,
            "observation.images.head": episode_data["agentview_rgb"][i],  # Keep as uint8 HWC for now
            "observation.images.wrist": episode_data["eye_in_hand_rgb"][i],  # Keep as uint8 HWC for now
        }
        
        frames.append(frame)
    
    return frames, task


def convert_libero_to_lerobot(
    input_path: Path,
    output_path: Path,
    repo_id: str,
    force: bool = False
) -> None:
    """Main conversion function."""
    logger.info(f"Starting conversion of {input_path} to LeRobot format")
    
    # Extract metadata
    logger.info("Extracting metadata from LIBERO dataset...")
    metadata = extract_libero_metadata(input_path)
    logger.info(f"Found {metadata['num_episodes']} episodes with task: {metadata['task']}")
    
    # Define features for LeRobot
    features = define_lerobot_features(metadata)
    
    # Create LeRobot dataset
    logger.info("Creating LeRobot dataset...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=metadata["fps"],
        features=features,
        robot_type=metadata["robot_type"],
        root=output_path,
        use_videos=True,
    )
    
    # Add the task
    dataset.meta.add_task(metadata["task"])
    
    # Process each episode
    logger.info("Converting episodes...")
    with h5py.File(input_path, "r") as hdf5_file:
        for demo_idx, demo_name in enumerate(tqdm(metadata["demos"], desc="Processing episodes")):
            # Load episode data
            episode_data = load_libero_episode(hdf5_file, demo_name)
            
            # Convert to LeRobot format
            frames, task = convert_episode_to_lerobot_format(
                episode_data,
                metadata["task"],
                metadata["fps"]
            )
            
            # Add frames to dataset
            for frame_idx, frame in enumerate(frames):
                timestamp = frame_idx / metadata["fps"]
                dataset.add_frame(frame, task=task, timestamp=timestamp)
            
            # Save episode
            dataset.save_episode()
    
    logger.info(f"Successfully converted {metadata['num_episodes']} episodes")
    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Total frames: {dataset.meta.total_frames}")
    
    # Print dataset info
    logger.info("\nDataset Info:")
    logger.info(f"  Repository ID: {repo_id}")
    logger.info(f"  Robot Type: {metadata['robot_type']}")
    logger.info(f"  FPS: {metadata['fps']}")
    logger.info(f"  Episodes: {dataset.meta.total_episodes}")
    logger.info(f"  Frames: {dataset.meta.total_frames}")
    logger.info(f"  Task: {metadata['task']}")
    

def main():
    parser = argparse.ArgumentParser(
        description="Convert LIBERO HDF5 datasets to LeRobot format"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to the LIBERO HDF5 dataset file"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output directory for the LeRobot dataset"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID for the dataset (e.g., 'username/dataset_name')"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite if output directory already exists"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    
    if not args.input_path.suffix == ".hdf5":
        raise ValueError(f"Input file must be an HDF5 file, got: {args.input_path}")
    
    # Check output path
    if args.output_path.exists() and not args.force:
        raise ValueError(
            f"Output path already exists: {args.output_path}. "
            "Use --force to overwrite."
        )
    
    # Run conversion
    convert_libero_to_lerobot(
        input_path=args.input_path,
        output_path=args.output_path,
        repo_id=args.repo_id,
        force=args.force
    )


if __name__ == "__main__":
    main()