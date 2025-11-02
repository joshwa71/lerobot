#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import json
import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.scripts.lerobot_train import _sanitize_wandb_dict, update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, init_logging

from lerobot.policies.modules.memory_lite import split_memory_params


@dataclass
class SequentialOnlineConfig(TrainPipelineConfig):
    """Sequential online training over a list of dataset task indices.

    This extends the standard TrainPipelineConfig with online-specific knobs.
    - Only memory value parameters are trained (backbone frozen).
    - After each task, evaluate cumulatively on all seen tasks so far.
    """

    # List of dataset task indices to adapt on sequentially (e.g., 0..9 for LIBERO-10)
    online_task_ids: list[int] = field(default_factory=lambda: list(range(10)))

    # Steps to run per task during online adaptation
    online_steps_per_task: int = 200

    # Optional dataset->env task id mapping as a JSON string.
    # If empty and env.task == "libero_10", a default mapping is used.
    # Example CLI: --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'
    ds_to_env_map_json: str | None = None

    # Save a checkpoint after each task
    save_after_each_task: bool = True

    # Rebuild optimizer each task (False keeps momentum/state across tasks)
    reinit_optimizer_each_task: bool = False

    # Learning rate for memory value parameters (pk_value_param). Overrides any preset.
    memory_value_lr: float = 1e-3


def _default_libero10_map() -> dict[int, int]:
    return {0: 4, 1: 6, 2: 9, 3: 2, 4: 7, 5: 0, 6: 8, 7: 1, 8: 3, 9: 5}


def _build_dataloader_for_task(
    dataset, task_index_to_name: dict[int, str], dataset_task_id: int, batch_size: int, num_workers: int, device_type: str, drop_n_last_frames: int = 0
):
    """Create a dataloader that only draws episodes for the specified dataset task id."""
    if dataset.meta.tasks is None:
        raise ValueError("Dataset metadata has no tasks table; cannot filter by task indices.")

    all_episode_tasks = dataset.meta.episodes["tasks"]
    allowed_task_name = task_index_to_name[dataset_task_id]
    episode_indices = [i for i, tlist in enumerate(all_episode_tasks) if allowed_task_name in tlist]

    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=episode_indices,
        drop_n_last_frames=drop_n_last_frames,
        shuffle=True,
    )

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,  # sampler handles shuffling
        sampler=sampler,
        pin_memory=device_type == "cuda",
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def _freeze_to_memory_values_only(policy: PreTrainedPolicy) -> int:
    """Freeze all parameters except memory value tables (pk_value_param). Returns number of trainable params."""
    trainable = 0
    for p in policy.parameters():
        p.requires_grad = bool(getattr(p, "pk_value_param", False))
        if p.requires_grad:
            trainable += p.numel()
    return trainable


def _collect_task_index_to_name(dataset) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for task_name, row in dataset.meta.tasks.iterrows():
        mapping[int(row["task_index"])] = task_name
    return mapping


def _subset_envs(envs_all: dict[str, dict[int, Any]], suite_name: str, env_task_ids: list[int]) -> dict[str, dict[int, Any]]:
    suite_envs = envs_all.get(suite_name, {})
    return {suite_name: {tid: suite_envs[tid] for tid in env_task_ids if tid in suite_envs}}


@parser.wrap()
def sequential_train(cfg: SequentialOnlineConfig, accelerator: Accelerator | None = None):
    cfg.validate()

    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)

    is_main = accelerator.is_main_process
    if is_main:
        logging.info(colored("Sequential online adaptation", "yellow", attrs=["bold"]))
        logging.info(cfg.to_dict())

    # WandB setup
    if cfg.wandb.enable and cfg.wandb.project and is_main:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset
    if is_main:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset = make_dataset(cfg)

    # Policy
    if is_main:
        logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)

    # Attach pre/post processors with dataset stats and device overrides
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # Freeze everything except memory values
    num_trainable = _freeze_to_memory_values_only(policy)
    num_total = sum(p.numel() for p in policy.parameters())
    if is_main:
        logging.info(f"Trainable params (memory values only) = {num_trainable} / {num_total}")

    # Build optimizer/scheduler once, optionally reinit per task
    # Make the scheduler horizon equal to total steps across all tasks if we don't reinit per task.
    total_steps = cfg.online_steps_per_task * len(cfg.online_task_ids)
    sched_steps = cfg.online_steps_per_task if cfg.reinit_optimizer_each_task else total_steps
    cfg.steps = max(1, sched_steps)

    if is_main:
        logging.info("Creating optimizer (memory values only) with custom LR and no scheduler")
    # Build optimizer that only updates params with requires_grad=True (memory values)
    def _build_memory_optimizer(model: PreTrainedPolicy, lr: float) -> Optimizer:
        params = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    optimizer = _build_memory_optimizer(policy, cfg.memory_value_lr)
    lr_scheduler = None

    # Prepare with accelerator
    policy, optimizer, lr_scheduler = accelerator.prepare(policy, optimizer, lr_scheduler)

    # Eval envs: pre-create all envs for suite, later subset based on seen tasks
    eval_envs_all = None
    if cfg.env is not None:
        if is_main:
            logging.info("Creating eval envs")
        eval_envs_all = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    # Dataset task index -> name
    task_index_to_name = _collect_task_index_to_name(dataset)

    # Map dataset task indices to env task ids
    ds_to_env: dict[int, int] = {}
    if cfg.ds_to_env_map_json:
        # First try strict JSON
        try:
            parsed = json.loads(cfg.ds_to_env_map_json)
            ds_to_env = {int(k): int(v) for k, v in parsed.items()}
        except Exception as e:
            # Fallbacks: python-literal dict or simple comma-separated pairs
            tmp = cfg.ds_to_env_map_json.strip()
            # If provided as plain pairs like "0:4,1:6,...", wrap in braces
            if not (tmp.startswith("{") and tmp.endswith("}")):
                tmp = "{" + tmp + "}"
            try:
                parsed_py = ast.literal_eval(tmp)
                if isinstance(parsed_py, dict):
                    ds_to_env = {int(k): int(v) for k, v in parsed_py.items()}
                else:
                    raise ValueError("Parsed mapping is not a dict")
            except Exception as e2:
                if is_main:
                    logging.error(f"Failed to parse ds_to_env_map_json: {e2}")
                ds_to_env = {}
    if not ds_to_env and cfg.env is not None and cfg.env.task and "libero_10" in str(cfg.env.task):
        ds_to_env = _default_libero10_map()
    if not ds_to_env and is_main:
        logging.warning("No dataset->env mapping provided; cumulative evaluation will use dataset task ids directly.")

    # Training/eval trackers
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    global_step = 0

    # Pre-create a dict that will accumulate successes per env task for CL metrics
    seen_env_task_ids: list[int] = []

    # Iterate sequentially over dataset tasks
    for idx, dataset_task_id in enumerate(cfg.online_task_ids):
        if is_main:
            logging.info(colored(f"=== Online task {idx+1}/{len(cfg.online_task_ids)} | dataset_task_id={dataset_task_id}", "cyan", attrs=["bold"]))

        # Build per-task dataloader filtered by dataset_task_id
        drop_n_last = getattr(cfg.policy, "drop_n_last_frames", 0)
        dataloader = _build_dataloader_for_task(
            dataset,
            task_index_to_name,
            dataset_task_id,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            device_type=device.type,
            drop_n_last_frames=drop_n_last,
        )
        if hasattr(accelerator, "prepare_data_loader"):
            dataloader = accelerator.prepare_data_loader(dataloader, device_placement=False)
        else:
            dataloader = accelerator.prepare(dataloader)
        dl_iter = cycle(dataloader)

        # Optionally rebuild optimizer/scheduler per task
        if cfg.reinit_optimizer_each_task:
            # Re-freeze to be safe in case something toggled
            _freeze_to_memory_values_only(policy)
            # Recreate optimizer with the same custom LR, no scheduler
            optimizer = _build_memory_optimizer(accelerator.unwrap_model(policy), cfg.memory_value_lr)
            lr_scheduler = None
            policy, optimizer, lr_scheduler = accelerator.prepare(policy, optimizer, lr_scheduler)

        # One-task training loop
        policy.train()
        # Effective batch info for tracker display
        effective_bs = cfg.batch_size * accelerator.num_processes
        train_tracker = MetricsTracker(
            effective_bs,
            dataset.num_frames,
            dataset.num_episodes,
            train_metrics,
            initial_step=global_step,
            accelerator=accelerator,
        )

        for _ in range(cfg.online_steps_per_task):
            batch = next(dl_iter)
            batch = preprocessor(batch)
            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                accelerator=accelerator,
                lr_scheduler=lr_scheduler,
            )

            global_step += 1
            train_tracker.step()

            is_log_step = cfg.log_freq > 0 and global_step % cfg.log_freq == 0 and is_main
            if is_log_step:
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(_sanitize_wandb_dict(output_dict))
                    wandb_logger.log_dict(wandb_log_dict, global_step)
                train_tracker.reset_averages()

        # Save checkpoint after finishing this task
        if cfg.save_checkpoint and cfg.save_after_each_task and is_main:
            step_id = get_step_identifier(global_step, cfg.steps)
            logging.info(f"Checkpoint policy after task {idx+1} | step {global_step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, global_step)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=global_step,
                cfg=cfg,
                policy=accelerator.unwrap_model(policy),
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        # Cumulative evaluation up to this task
        if eval_envs_all is not None:
            # Extend seen env task list using mapping; fallback to dataset_task_id if no mapping
            env_tid = ds_to_env.get(dataset_task_id, dataset_task_id)
            if env_tid not in seen_env_task_ids:
                seen_env_task_ids.append(env_tid)

            env_subset = _subset_envs(eval_envs_all, cfg.env.task, seen_env_task_ids)

            if is_main:
                logging.info(colored(f"Evaluate on env tasks: {seen_env_task_ids}", "green"))

            with torch.no_grad(), accelerator.autocast():
                eval_info = eval_policy_all(
                    envs=env_subset,
                    policy=accelerator.unwrap_model(policy),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=cfg.eval.n_episodes,
                    videos_dir=(cfg.output_dir / "eval" / f"after_task_{idx+1}"),
                    max_episodes_rendered=0,
                    start_seed=cfg.seed,
                    max_parallel_tasks=cfg.env.max_parallel_tasks,
                )

            if is_main:
                # Log concise CL metrics to wandb
                overall = eval_info.get("overall", {})
                if wandb_logger:
                    log_dict = {
                        "num_tasks_seen": len(seen_env_task_ids),
                        "avg_sum_reward_seen": float(overall.get("avg_sum_reward", float("nan"))),
                        "avg_max_reward_seen": float(overall.get("avg_max_reward", float("nan"))),
                        "avg_pc_success_seen": float(overall.get("pc_success", float("nan"))),
                    }
                    # Per-task success (if available)
                    per_tasks = eval_info.get(cfg.env.task, {}) if cfg.env else {}
                    for tid, tinfo in per_tasks.items():
                        if isinstance(tinfo, dict) and "pc_success" in tinfo:
                            log_dict[f"success/task_{tid}"] = float(tinfo["pc_success"]) if tinfo["pc_success"] is not None else float("nan")
                    wandb_logger.log_dict(log_dict, global_step, mode="eval")

    # Cleanup
    if eval_envs_all:
        close_envs(eval_envs_all)

    if is_main:
        logging.info("End of sequential online training")


def main():
    sequential_train()


if __name__ == "__main__":
    main()


