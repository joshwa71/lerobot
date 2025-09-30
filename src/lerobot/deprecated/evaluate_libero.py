#!/usr/bin/env python3


# /home/josh/phddev/lerobot-upstream/src/lerobot/scripts/evaluate_libero.py
# Copyright 2025

import argparse
import time
from typing import Optional

import torch
import numpy as np
import json
import os
from pathlib import Path

from lerobot.envs.configs import EnvTransformConfig
from lerobot.envs.configs import LiberoRemoteEnv as LiberoRemoteEnvConfig
from lerobot.scripts.rl.gym_manipulator import make_robot_env
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def evaluate_libero_remote(
    model_path: str,
    host: str,
    port: int,
    n_episodes: int,
    device: str,
    fps: int,
    control_time_s: float,
    use_gripper: bool,
    task_instruction: Optional[str] = None,
    record: bool = False,
    repo_id: Optional[str] = None,
):
    # Build env config and environment
    wrapper = EnvTransformConfig(
        control_time_s=control_time_s,
        use_gripper=use_gripper,
        display_cameras=False,
    )
    env_cfg = LiberoRemoteEnvConfig(host=host, port=port, fps=fps, device=device, wrapper=wrapper)
    env = make_robot_env(env_cfg)

    # Load policy config from model directory
    policy_cfg = PreTrainedConfig.from_pretrained(model_path)
    policy_cfg.device = device
    policy_cfg.pretrained_path = model_path
    # Prefer dataset stats from training for correct normalization
    policy = None
    train_cfg_path = os.path.join(model_path, "train_config.json")
    if os.path.isfile(train_cfg_path):
        try:
            with open(train_cfg_path, "r") as f:
                train_cfg = json.load(f)
            ds_repo_id = train_cfg.get("dataset", {}).get("repo_id")
            if ds_repo_id:
                # Try to resolve a local outputs dataset root relative to the model path
                ds = None
                try:
                    model_dir = Path(model_path).resolve()
                    # Find the nearest ancestor that contains the repo_id path
                    candidates = [p for p in model_dir.parents if (p / ds_repo_id).exists()]
                    if candidates:
                        ds_root = str((candidates[0] / ds_repo_id).resolve())
                        ds = LeRobotDataset(repo_id=ds_repo_id, root=ds_root)
                    else:
                        # Fallback to default behavior (HF cache or absolute repo id)
                        ds = LeRobotDataset(repo_id=ds_repo_id)
                except Exception:
                    ds = LeRobotDataset(repo_id=ds_repo_id)
                policy = make_policy(cfg=policy_cfg, ds_meta=ds.meta)
        except Exception:
            policy = None
    if policy is None:
        # Fallback to env-derived features
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    # Optional: set language instruction if supported (e.g., SmolVLA)
    # Ensure a non-empty default to avoid KeyError in language tokenization
    if hasattr(policy, "set_task_instruction"):
        try:
            instruction_base = task_instruction if task_instruction else "Complete the task"
            policy.set_task_instruction(instruction_base)
        except Exception:
            pass
    else:
        instruction_base = task_instruction if task_instruction else "Complete the task"

    successes = 0
    rewards_per_episode = []
    dataset = None
    dataset_image_keys = None

    # Determine expected image keys from model config
    expected_image_keys = [
        k for k, v in policy_cfg.input_features.items() if k.startswith("observation.images.")
    ]

    def align_obs_image_keys(observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        present = sorted([k for k in observation if k.startswith("observation.images.")])
        if not expected_image_keys:
            return observation
        # Map by index if missing keys
        for i, ek in enumerate(sorted(expected_image_keys)):
            if ek not in observation and i < len(present):
                observation[ek] = observation[present[i]]
        return observation

    for ep in range(n_episodes):
        obs, _ = env.reset()
        policy.reset()
        episode_reward = 0.0
        start_t = time.perf_counter()

        # Lazily create dataset at first episode if recording is enabled
        if record and dataset is None:
            if not repo_id:
                raise ValueError("--repo_id must be provided when --record is set")
            root_abs = Path(repo_id).resolve()
            if root_abs.exists():
                # Resume recording on existing dataset
                ds = LeRobotDataset(repo_id=repo_id, root=str(root_abs))
                dataset = ds
                # Use existing camera keys from metadata
                dataset_image_keys = ds.meta.camera_keys
            else:
                # Infer features from current observation and action space
                obs_no_batch = {k: (v.squeeze(0) if hasattr(v, "dim") and v.dim() > 0 else v) for k, v in obs.items()}
                state_shape = tuple(obs_no_batch["observation.state"].shape)
                action_dim = int(env.action_space.shape[0])
                image_keys = [k for k in obs_no_batch if k.startswith("observation.images.")]
                dataset_image_keys = image_keys
                features = {
                    "observation.state": {"dtype": "float32", "shape": state_shape, "names": None},
                    "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
                }
                for k in image_keys:
                    shp = tuple(obs_no_batch[k].shape[-3:])
                    # Define feature shape as HWC per LeRobot convention
                    if len(shp) == 3 and shp[0] in (1, 3, 4):
                        img_shape_hwc = (shp[1], shp[2], shp[0])
                    else:
                        img_shape_hwc = shp
                    features[k] = {
                        "dtype": "video",
                        "shape": img_shape_hwc,
                        "names": ["height", "width", "channel"],
                    }

                dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=fps,
                    features=features,
                    root=str(root_abs),
                    robot_type="libero_remote",
                    use_videos=True,
                )

        while True:
            # Inject language instruction expected by VLA policies
            # Batch size is 1 after wrappers; SmolVLA expects a list[str]
            if "task" not in obs:
                obs["task"] = [instruction_base]
            # Align image keys to what the policy expects
            obs = align_obs_image_keys(obs)
            with torch.inference_mode():
                action = policy.select_action(obs)

            # Pass torch tensor directly; TorchActionWrapper handles detach/cpu/np conversion
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            # Record frame if requested
            if record and dataset is not None:
                frame = {}
                # Observation state
                if "observation.state" in obs:
                    frame["observation.state"] = obs["observation.state"].detach().cpu().squeeze(0).float()
                # Images present at dataset creation time
                for k in (dataset_image_keys or []):
                    if k in obs:
                        frame[k] = obs[k].detach().cpu().squeeze(0)
                # Action used
                a_used = action
                if hasattr(a_used, "detach"):
                    a_used = a_used.detach()
                if hasattr(a_used, "cpu"):
                    a_used = a_used.cpu()
                frame["action"] = a_used.squeeze(0) if getattr(a_used, "dim", lambda: 0)() == 2 else a_used
                # Task string for this episode (may be overwritten on failure before saving)
                frame["task"] = instruction_base
                dataset.add_frame(frame)
            if terminated or truncated:
                break

        rewards_per_episode.append(episode_reward)
        # Treat any positive reward as success (sparse-reward convention)
        ep_success = episode_reward > 0.5
        successes += int(ep_success)

        # Save episode with success/failure task labeling
        if record and dataset is not None:
            try:
                if not ep_success:
                    failed_task = f"Failed attempt to {instruction_base}"
                    if isinstance(dataset.episode_buffer, dict) and "task" in dataset.episode_buffer:
                        dataset.episode_buffer["task"] = [failed_task] * dataset.episode_buffer["size"]
                dataset.save_episode()
            except Exception:
                try:
                    dataset.clear_episode_buffer()
                except Exception:
                    pass

        # Maintain target fps pacing for stability during evaluation
        if fps:
            elapsed = time.perf_counter() - start_t
            target = 1.0 / fps
            if elapsed < target:
                time.sleep(target - elapsed)

    success_rate = successes / max(1, n_episodes)
    avg_reward = float(np.mean(rewards_per_episode)) if rewards_per_episode else 0.0
    print(f"Episodes: {n_episodes}")
    print(f"Success rate: {success_rate:.2%} ({successes}/{n_episodes})")
    print(f"Average reward: {avg_reward:.3f}")
    # Cleanly close the environment / socket on client side
    try:
        env.close()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate a LeRobot policy in LIBERO via remote client")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model directory")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Remote server host")
    parser.add_argument("--port", type=int, default=5555, help="Remote server port")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Device")
    parser.add_argument("--fps", type=int, default=20, help="Target evaluation FPS")
    parser.add_argument("--control_time_s", type=float, default=20.0, help="Time limit per episode (sec)")
    parser.add_argument("--use_gripper", action="store_true", help="Enable gripper action channel")
    parser.add_argument("--task", type=str, default=None, help="Optional language instruction for VLA policies")
    parser.add_argument("--record", action="store_true", help="Record episodes to a LeRobot dataset")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help=(
            "Local path to save the dataset when --record is set. Example: "
            "--repo_id outputs/eval_libero_10_task_0_smolvla"
        ),
    )
    args = parser.parse_args()

    evaluate_libero_remote(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        n_episodes=args.episodes,
        device=args.device,
        fps=args.fps,
        control_time_s=args.control_time_s,
        use_gripper=bool(args.use_gripper),
        task_instruction=args.task,
        record=bool(args.record),
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()


