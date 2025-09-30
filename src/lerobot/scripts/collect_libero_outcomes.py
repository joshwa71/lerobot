#!/usr/bin/env python

# Copyright 2025
#
# Standalone script to evaluate a pretrained policy on LIBERO suites and optionally
# record episodes into a LeRobotDataset, filtering by outcome (success/fail/all).
#
# This script reuses the existing LeRobot infrastructure (policy factory, processors,
# env factory, and dataset writer) and does not modify existing code.

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.datasets.lerobot_dataset import LeRobotDataset


OutcomeFilter = Literal["success", "fail", "all"]


def _normalize_suite_name(s: str) -> str:
    s = s.strip().lower().replace(" ", "_")
    # common aliases
    if s in {"libero10", "libero_10", "libero 10"}:
        return "libero_10"
    return s


def _prepare_output_dir(root: Path | None, suite: str, outcome: str) -> Path:
    if root is None:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path("outputs") / "datasets" / f"{suite}_{outcome}_{ts}"
    root = root.resolve()
    if root.exists():
        # Avoid clobbering any existing path (even empty). Use a unique sibling path.
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        root = root.parent / f"{root.name}_{ts}"
    return root


def _infer_features_from_env(vec_env) -> tuple[dict, int]:
    """
    Inspect one reset observation to infer image/state/action shapes and fps.

    Returns (features_dict, fps)
    """
    obs, _ = vec_env.reset()
    # Expect obs["pixels"] to be dict with keys like {"image", "image2"}
    pixels = obs.get("pixels")
    if isinstance(pixels, dict):
        cam_keys = sorted(list(pixels.keys()))
    else:
        # Fallback to single camera named "image"
        cam_keys = ["image"]
    # Choose the first env in the batch
    sample_key = cam_keys[0]
    img_h, img_w, img_c = np.array(pixels[sample_key]).shape[-3:]

    # State
    state = obs.get("agent_pos")
    if state is None:
        raise ValueError("Expected 'agent_pos' in observation for pixels_agent_pos")
    state_dim = int(np.array(state).shape[-1])

    # Action
    act_dim = int(np.array(vec_env.single_action_space.shape).prod())

    # FPS: LIBERO datasets typically use 10 FPS
    fps = 10

    features = {
        f"observation.images.{k}": {
            "dtype": "image",
            "shape": (int(img_h), int(img_w), int(img_c)),
            "names": ["height", "width", "channel"],
            "fps": float(fps),
        }
        for k in cam_keys
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": ["state"],
        "fps": float(fps),
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (act_dim,),
        "names": ["actions"],
        "fps": float(fps),
    }
    return features, fps


def _build_policy_and_processors(
    model_path: str,
    device_str: str,
    env_cfg: LiberoEnvConfig,
) -> tuple[PreTrainedPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    # Load config from pretrained directory
    policy_cfg = PreTrainedConfig.from_pretrained(model_path)
    policy_cfg.pretrained_path = model_path
    # Make policy using env features (as eval pipeline does)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    # Pre/post processors: load from pretrained path with device override
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": device_str}},
    )
    return policy, preprocessor, postprocessor


def _select_action(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    env,
    obs: dict[str, Any],
) -> np.ndarray:
    obs_proc = preprocess_observation(obs)
    obs_proc = add_envs_task(env, obs_proc)
    obs_proc = preprocessor(obs_proc)
    with torch.inference_mode():
        action = policy.select_action(obs_proc)
    action = postprocessor(action)
    action_np: np.ndarray = action.to("cpu").numpy()
    return action_np


def _should_save(outcome: OutcomeFilter, is_success: bool) -> bool:
    if outcome == "all":
        return True
    if outcome == "success":
        return bool(is_success)
    if outcome == "fail":
        return not bool(is_success)
    return False


def _prefix_failed_task(task: str) -> str:
    return f"Failed attempt to {task}" if not task.startswith("Failed attempt to ") else task


def collect_for_task(
    task_group: str,
    task_id: int,
    vec_env,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset: LeRobotDataset | None,
    required_eps: int,
    outcome: OutcomeFilter,
    display_progress: bool = True,
) -> int:
    """
    Run repeated episodes on a single (suite, task_id) vec env, saving episodes into `dataset`
    if provided, until `required_eps` matching `outcome` are saved. Returns number saved.
    """
    saved = 0
    num_envs = vec_env.num_envs
    # Discover max steps from underlying envs
    max_steps = vec_env.call("_max_episode_steps")[0]

    # Obtain task strings for each sub-env
    def get_task_strings() -> list[str]:
        if hasattr(vec_env, "envs") and len(vec_env.envs) > 0:
            ts = []
            for e in vec_env.envs:
                t = getattr(e, "task_description", None) or getattr(e, "task", "")
                if isinstance(t, (tuple, list)):
                    t = t[0] if len(t) else ""
                ts.append(str(t))
            return ts
        # Fallback: call via vector API
        return [str(task_group)] * num_envs

    while saved < required_eps:
        obs, _ = vec_env.reset()
        task_strings = get_task_strings()

        # Each env accumulates frames for its episode until done
        per_env_frames: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
        done_mask = np.zeros((num_envs,), dtype=bool)

        for step in range(int(max_steps)):
            # Record current observations before stepping; pair with action below
            # (obs["pixels"] may be dict or array)
            # Prepare actions
            act = _select_action(policy, preprocessor, postprocessor, vec_env, obs)
            # Align
            assert act.ndim == 2 and act.shape[0] == num_envs

            # Record per-env frame data (using obs BEFORE step)
            pixels = obs.get("pixels")
            if isinstance(pixels, dict):
                cam_order = sorted(list(pixels.keys()))
            else:
                cam_order = ["image"]

            for i in range(num_envs):
                if done_mask[i]:
                    continue
                frame: dict[str, Any] = {"task": task_strings[i]}
                # Images
                if isinstance(pixels, dict):
                    for ck in cam_order:
                        frame[f"observation.images.{ck}"] = np.array(pixels[ck][i])
                else:
                    frame["observation.images.image"] = np.array(pixels[i])
                # State
                frame["observation.state"] = np.array(obs["agent_pos"][i], dtype=np.float32)
                # Action
                frame["action"] = np.array(act[i], dtype=np.float32)
                per_env_frames[i].append(frame)

            # Step env
            obs, reward, terminated, truncated, info = vec_env.step(act)
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            step_done = terminated | truncated
            # Success flags from final_info if present; else False
            successes = [False] * num_envs
            if isinstance(info, dict) and "final_info" in info and info["final_info"] is not None:
                final_info_list = info["final_info"]
                for i in range(num_envs):
                    finfo = final_info_list[i]
                    if finfo is not None and isinstance(finfo, dict):
                        successes[i] = bool(finfo.get("is_success", False))

            # Force completion at last step to ensure consistent termination and saving
            if (step + 1) >= int(max_steps):
                for i in range(num_envs):
                    if not done_mask[i]:
                        step_done[i] = True

            # Handle completed episodes
            for i in range(num_envs):
                if done_mask[i] or not step_done[i]:
                    continue
                is_success = successes[i]
                if _should_save(outcome, is_success):
                    if dataset is not None:
                        # Build episode_buffer using dataset helper to satisfy validation
                        frames = per_env_frames[i]
                        ep_len = len(frames)
                        episode_index = dataset.meta.total_episodes

                        ep_buffer: dict[str, Any] = dataset.create_episode_buffer(
                            episode_index=episode_index
                        )
                        ep_buffer["size"] = ep_len
                        ep_buffer["task"] = [
                            _prefix_failed_task(frames[0]["task"]) if not is_success else frames[0]["task"]
                        ] * ep_len

                        # Fill default timeline features
                        fps = float(dataset.fps)
                        ep_buffer["frame_index"] = list(range(ep_len))
                        ep_buffer["timestamp"] = [t / fps for t in range(ep_len)]

                        # Save images to disk and fill other modalities
                        for t, fr in enumerate(frames):
                            for key, val in fr.items():
                                if key == "task":
                                    continue
                                if key not in dataset.features:
                                    continue
                                if dataset.features[key]["dtype"] in ["image", "video"]:
                                    img_path = dataset._get_image_file_path(
                                        episode_index=episode_index, image_key=key, frame_index=t
                                    )
                                    if t == 0:
                                        img_path.parent.mkdir(parents=True, exist_ok=True)
                                    dataset._save_image(val, img_path)
                                    ep_buffer[key].append(str(img_path))
                                else:
                                    ep_buffer[key].append(val)

                        # Commit episode to dataset
                        dataset.save_episode(ep_buffer)
                        saved += 1
                # Mark env as done and ready for next episode
                done_mask[i] = True

            # Early exit if we've reached target
            if saved >= required_eps:
                break

        if display_progress:
            logging.info(
                f"[{task_group}:{task_id}] saved {saved}/{required_eps} ({outcome})"
            )

    return saved


def main():
    init_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pretrained policy on a LIBERO suite and optionally record episodes "
            "into a LeRobotDataset filtered by outcome."
        )
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model dir")
    parser.add_argument("--suite", type=str, default="libero_10", help="LIBERO suite name, e.g. 'libero_10'")
    parser.add_argument(
        "--record",
        action="store_true",
        help="If set, record episodes into a LeRobotDataset at --output_path",
    )
    parser.add_argument(
        "--save_outcome",
        type=str,
        choices=["success", "fail", "all"],
        default="fail",
        help="Which outcomes to save",
    )
    parser.add_argument(
        "--eps",
        type=int,
        default=50,
        help="Number of episodes to accumulate per task (matching --save_outcome)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Directory for the LeRobotDataset root. If omitted, a path under outputs/datasets is generated."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Vectorized env batch size per task")
    parser.add_argument("--async_envs", action="store_true", help="Use AsyncVectorEnv")
    parser.add_argument("--device", type=str, default="cuda", help="Override device (cpu/cuda)")

    args = parser.parse_args()

    suite = _normalize_suite_name(args.suite)
    outcome: OutcomeFilter = args.save_outcome  # type: ignore

    # Device detection and AMP policy handled by processors
    device = get_safe_torch_device(args.device, log=True)

    # Build env config for LIBERO
    env_cfg = LiberoEnvConfig(task=suite)

    # Create vectorized envs across all tasks in the suite
    envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=args.async_envs)
    if suite not in envs:
        raise RuntimeError(f"Expected suite '{suite}' in created envs, got keys: {list(envs.keys())}")
    suite_envs = envs[suite]  # dict[int, vec_env]

    # Build policy and processors
    policy, preprocessor, postprocessor = _build_policy_and_processors(
        model_path=args.model_path, device_str=device.type, env_cfg=env_cfg
    )

    # Optionally create dataset
    dataset: LeRobotDataset | None = None
    if args.record:
        out_root = _prepare_output_dir(Path(args.output_path) if args.output_path else None, suite, outcome)

        # Infer features and fps from one vec env (any task id)
        any_vec_env = next(iter(suite_envs.values()))
        features, fps = _infer_features_from_env(any_vec_env)

        # repo_id is a label; root is the actual directory
        repo_id = f"{suite}_{outcome}"
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=out_root,
            use_videos=False,
            image_writer_processes=0,
            image_writer_threads=8,
        )

    # Iterate all tasks in the suite
    total_per_task = args.eps
    for task_id, vec_env in suite_envs.items():
        saved = collect_for_task(
            task_group=suite,
            task_id=task_id,
            vec_env=vec_env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            required_eps=total_per_task,
            outcome=outcome,
        )
        logging.info(f"Finished task {suite}:{task_id} | saved {saved}/{total_per_task} episodes")

    logging.info("All tasks completed.")


if __name__ == "__main__":
    main()


