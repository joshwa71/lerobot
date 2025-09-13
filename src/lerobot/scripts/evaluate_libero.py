#!/usr/bin/env python3

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
            instruction = task_instruction if task_instruction else "Complete the task"
            policy.set_task_instruction(instruction)
        except Exception:
            pass

    successes = 0
    rewards_per_episode = []

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

        while True:
            # Inject language instruction expected by VLA policies
            # Batch size is 1 after wrappers; SmolVLA expects a list[str]
            if "task" not in obs:
                instruction = task_instruction if task_instruction else "Complete the task"
                obs["task"] = [instruction]
            # Align image keys to what the policy expects
            obs = align_obs_image_keys(obs)
            with torch.inference_mode():
                action = policy.select_action(obs)

            # Pass torch tensor directly; TorchActionWrapper handles detach/cpu/np conversion
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            if terminated or truncated:
                break

        rewards_per_episode.append(episode_reward)
        # Treat any positive reward as success (sparse-reward convention)
        successes += int(episode_reward > 0.5)

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
    )


if __name__ == "__main__":
    main()


