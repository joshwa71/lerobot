#!/usr/bin/env python3

import argparse
import time
from typing import Optional

import json
import os
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import EnvTransformConfig
from lerobot.envs.configs import LiberoRemoteEnv as LiberoRemoteEnvConfig
from lerobot.policies.factory import make_policy
from lerobot.scripts.rl.gym_manipulator import make_robot_env


def collect_libero_outcomes(
    model_path: str,
    host: str,
    port: int,
    desired_episodes: int,
    device: str,
    fps: int,
    control_time_s: float,
    use_gripper: bool,
    collect: str,
    repo_id: str,
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

    # Load policy
    policy_cfg = PreTrainedConfig.from_pretrained(model_path)
    policy_cfg.device = device
    policy_cfg.pretrained_path = model_path

    policy = None
    train_cfg_path = os.path.join(model_path, "train_config.json")
    if os.path.isfile(train_cfg_path):
        try:
            with open(train_cfg_path, "r") as f:
                train_cfg = json.load(f)
            ds_repo_id = train_cfg.get("dataset", {}).get("repo_id")
            if ds_repo_id:
                try:
                    model_dir = Path(model_path).resolve()
                    candidates = [p for p in model_dir.parents if (p / ds_repo_id).exists()]
                    if candidates:
                        ds_root = str((candidates[0] / ds_repo_id).resolve())
                        ds = LeRobotDataset(repo_id=ds_repo_id, root=ds_root)
                    else:
                        ds = LeRobotDataset(repo_id=ds_repo_id)
                except Exception:
                    ds = LeRobotDataset(repo_id=ds_repo_id)
                policy = make_policy(cfg=policy_cfg, ds_meta=ds.meta)
        except Exception:
            policy = None
    if policy is None:
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    # Instruction handling
    instruction_base = task_instruction if task_instruction else "Complete the task"
    if hasattr(policy, "set_task_instruction"):
        try:
            policy.set_task_instruction(instruction_base)
        except Exception:
            pass

    # Expected image keys for alignment
    expected_image_keys = [
        k for k, v in policy_cfg.input_features.items() if k.startswith("observation.images.")
    ]

    def align_obs_image_keys(observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        present = sorted([k for k in observation if k.startswith("observation.images.")])
        if not expected_image_keys:
            return observation
        for i, ek in enumerate(sorted(expected_image_keys)):
            if ek not in observation and i < len(present):
                observation[ek] = observation[present[i]]
        return observation

    # Dataset init deferred until first reset
    dataset = None
    dataset_image_keys = None
    root_abs = Path(repo_id).resolve()

    collected = 0
    trials = 0
    desired_is_success = collect == "success"

    while collected < desired_episodes:
        obs, _ = env.reset()
        policy.reset()
        episode_reward = 0.0
        trials += 1

        # Lazily set up dataset (create or resume)
        if dataset is None:
            if root_abs.exists():
                ds = LeRobotDataset(repo_id=repo_id, root=str(root_abs))
                dataset = ds
                dataset_image_keys = ds.meta.camera_keys
            else:
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

        start_step_t = time.perf_counter()
        while True:
            if "task" not in obs:
                obs["task"] = [instruction_base]
            obs = align_obs_image_keys(obs)
            with torch.inference_mode():
                action = policy.select_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)

            # Record frame into buffer
            frame = {}
            if "observation.state" in obs:
                frame["observation.state"] = obs["observation.state"].detach().cpu().squeeze(0).float()
            for k in (dataset_image_keys or []):
                if k in obs:
                    frame[k] = obs[k].detach().cpu().squeeze(0)
            a_used = action
            if hasattr(a_used, "detach"):
                a_used = a_used.detach()
            if hasattr(a_used, "cpu"):
                a_used = a_used.cpu()
            frame["action"] = a_used.squeeze(0) if getattr(a_used, "dim", lambda: 0)() == 2 else a_used
            frame["task"] = instruction_base
            dataset.add_frame(frame)

            # Maintain step pacing best-effort
            if fps:
                dt = time.perf_counter() - start_step_t
                target = 1.0 / fps
                if dt < target:
                    time.sleep(target - dt)
                start_step_t = time.perf_counter()

            if terminated or truncated:
                break

        # Decide outcome
        ep_success = episode_reward > 0.5
        matches = (ep_success and desired_is_success) or ((not ep_success) and (not desired_is_success))

        # Keep or discard the buffered episode
        if matches:
            try:
                if not ep_success:
                    failed_task = f"Failed attempt to {instruction_base}"
                    if isinstance(dataset.episode_buffer, dict) and "task" in dataset.episode_buffer:
                        dataset.episode_buffer["task"] = [failed_task] * dataset.episode_buffer["size"]
                dataset.save_episode()
                collected += 1
            except Exception:
                try:
                    dataset.clear_episode_buffer()
                except Exception:
                    pass
        else:
            try:
                dataset.clear_episode_buffer()
            except Exception:
                pass

    print(
        f"Collected {collected} {collect} episode(s) over {trials} trial episode(s)."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect only successful or failed LIBERO episodes with a LeRobot policy, "
            "recording them as a LeRobot dataset."
        )
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model directory")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Remote server host")
    parser.add_argument("--port", type=int, default=5555, help="Remote server port")
    parser.add_argument("--episodes", type=int, default=5, help="Number of desired episodes to collect")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Device")
    parser.add_argument("--fps", type=int, default=20, help="Target FPS")
    parser.add_argument("--control_time_s", type=float, default=20.0, help="Time limit per episode (sec)")
    parser.add_argument("--use_gripper", action="store_true", help="Enable gripper action channel")
    parser.add_argument("--task", type=str, default=None, help="Optional language instruction for VLA policies")
    parser.add_argument(
        "--collect",
        type=str,
        required=True,
        choices=["success", "fail"],
        help="Which episode outcome to collect.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help=(
            "Local path where the dataset will be recorded. Example: "
            "--repo_id /home/josh/phddev/lerobot-upstream/outputs/libero_collect_smolvla"
        ),
    )
    args = parser.parse_args()

    collect_libero_outcomes(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        desired_episodes=args.episodes,
        device=args.device,
        fps=args.fps,
        control_time_s=args.control_time_s,
        use_gripper=bool(args.use_gripper),
        collect=args.collect,
        repo_id=args.repo_id,
        task_instruction=args.task,
    )


if __name__ == "__main__":
    main()


