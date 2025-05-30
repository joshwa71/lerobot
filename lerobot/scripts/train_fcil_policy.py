#!/usr/bin/env python

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

import logging
import time
import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import List, Dict, Any, Tuple 
import functools # Import functools for partial
from tqdm import tqdm

import torch
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata 
from lerobot.common.datasets.fcil_policy_dataset import FCILPolicyDataset, fcil_policy_collate_fn 
from lerobot.common.datasets.utils import cycle 
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.fcil_policy.configuration_fcil_policy import FCILPolicyConfig
from lerobot.common.policies.fcil_policy.modeling_fcil_policy import FCILPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig, TRAIN_CONFIG_NAME

logger = logging.getLogger(__name__)

@dataclass
class FCILPolicyTrainConfig(TrainPipelineConfig):
    policy: FCILPolicyConfig = field(default_factory=FCILPolicyConfig)
    
    success_dataset_repo_id: str = "lerobot/success_50_placeholder" 
    mixed_dataset_repo_id: str = "lerobot/mixed_50_placeholder"   
    dataset_root: Path | None = None
    train_val_split_ratio: float = 0.9
    eval_freq: int = 10

    def __post_init__(self):
        super().__post_init__() 
        if not self.job_name:
            self.job_name = f"fcil_policy_{self.policy.model_dim}md_{self.policy.n_layers}l"

    def validate(self):
        super().validate() 
        if not self.success_dataset_repo_id or not self.mixed_dataset_repo_id:
            raise ValueError("`success_dataset_repo_id` and `mixed_dataset_repo_id` must be provided for FCIL training.")
        

def update_fcil_policy(
    train_metrics: MetricsTracker,
    policy: FCILPolicy,
    batch: Tuple[Dict[str, Any], torch.Tensor],
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train() 

    policy_input_dict, target_action_and_done = batch
    
    def move_to_device(item):
        if isinstance(item, torch.Tensor):
            return item.to(device, non_blocking=True)
        elif isinstance(item, dict):
            return {k: move_to_device(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [move_to_device(i) if i is not None else None for i in item]
        return item 

    policy_input_dict_device = move_to_device(policy_input_dict)
    target_action_and_done_device = target_action_and_done.to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, enabled=use_amp) if use_amp else nullcontext():
        loss, output_dict = policy.forward((policy_input_dict_device, target_action_and_done_device))
        
    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm, error_if_nonfinite=False)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if lr_scheduler is not None:
        lr_scheduler.step()

    train_metrics.loss = loss.item() 
    if output_dict: 
        for k, v in output_dict.items():
            if k == "loss": continue 
            if hasattr(train_metrics, k) and isinstance(getattr(train_metrics,k), AverageMeter):
                getattr(train_metrics,k).update(v) 
            elif isinstance(v, (int, float)): 
                 logger.debug(f"Policy output metric '{k}' not in MetricsTracker's AverageMeters. Raw value: {v}")

    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@torch.no_grad() # Evaluation should not compute gradients
def evaluate_policy(policy: FCILPolicy, val_loader: DataLoader, device: torch.device, use_amp: bool = False) -> dict:
    policy.eval()

    all_batch_losses = []
    all_unweighted_losses = []
    all_recovery_proportions = []
    
    for batch in val_loader:
        policy_input_dict, target_action_and_done = batch
        
        def move_to_device(item):
            if isinstance(item, torch.Tensor):
                return item.to(device, non_blocking=True)
            elif isinstance(item, dict):
                return {k: move_to_device(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [move_to_device(i) if i is not None else None for i in item]
            return item 

        policy_input_dict_device = move_to_device(policy_input_dict)
        target_action_and_done_device = target_action_and_done.to(device, non_blocking=True)
        
        with torch.autocast(device_type=device.type, enabled=use_amp) if use_amp else nullcontext():
            loss, output_dict = policy.forward((policy_input_dict_device, target_action_and_done_device))

        all_batch_losses.append(loss.item())
        all_unweighted_losses.append(output_dict.get("unweighted_mse_loss", 0.0))
        all_recovery_proportions.append(output_dict.get("recovery_proportion", 0.0))

    # Aggregate metrics over all validation batches
    eval_metrics = {
        "loss": sum(all_batch_losses) / len(all_batch_losses) if all_batch_losses else 0.0,
        "unweighted_mse_loss": sum(all_unweighted_losses) / len(all_unweighted_losses) if all_unweighted_losses else 0.0,
        "recovery_proportion": sum(all_recovery_proportions) / len(all_recovery_proportions) if all_recovery_proportions else 0.0
    }
    return eval_metrics


@parser.wrap(config_path=None) 
def train_fcil_policy(cfg: FCILPolicyTrainConfig):
    if not cfg.resume:
        cfg.validate() 
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        from termcolor import colored 
        wandb_logger = WandBLogger(cfg)
    else:
        from termcolor import colored 
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True # type: ignore
    torch.backends.cuda.matmul.allow_tf32 = True # type: ignore

    logger.info("Attempting to load dataset_stats for policy normalization and dimension inference.")
    dataset_stats = None
    mixed_ds_meta_for_stats = None 
    try:
        mixed_ds_meta_for_stats = LeRobotDatasetMetadata(
            cfg.mixed_dataset_repo_id, 
            root=Path(cfg.dataset_root) / cfg.mixed_dataset_repo_id if cfg.dataset_root else None,
            revision=cfg.dataset.revision, 
        )
        dataset_stats = mixed_ds_meta_for_stats.stats
        
        if cfg.policy.state_dim is None and dataset_stats and "observation.state" in dataset_stats:
            cfg.policy.state_dim = dataset_stats["observation.state"]["mean"].shape[0]
            logger.info(f"Inferred policy.state_dim: {cfg.policy.state_dim} from dataset stats.")
        if cfg.policy.action_dim is None and dataset_stats and "action" in dataset_stats:
            cfg.policy.action_dim = dataset_stats["action"]["mean"].shape[0]
            logger.info(f"Inferred policy.action_dim: {cfg.policy.action_dim} from dataset stats.")

    except Exception as e:
        logger.warning(f"Could not load metadata/stats from {cfg.mixed_dataset_repo_id} to infer dims: {e}. Will rely on config values or fail later.")
    
    if cfg.policy.state_dim is None or cfg.policy.action_dim is None:
        raise ValueError(
            "policy.state_dim and policy.action_dim must be specified in the YAML configuration "
            "or be inferable from dataset_stats of mixed_dataset_repo_id."
        )

    logger.info("Creating FCILPolicyDataset")
    full_dataset = FCILPolicyDataset(
        config=cfg.policy, 
        success_dataset_repo_id=cfg.success_dataset_repo_id,
        mixed_dataset_repo_id=cfg.mixed_dataset_repo_id,
        root=cfg.dataset_root,
        revision=cfg.dataset.revision, 
    )

    num_total_episodes = len(full_dataset)
    if num_total_episodes == 0:
        raise ValueError("TrajectoryDataset is empty. Check repo_ids and dataset contents.")
    logging.info(f"TrajectoryDataset created with {num_total_episodes} total episodes.")

    # Split dataset
    num_train_episodes = int(num_total_episodes * cfg.train_val_split_ratio)
    num_val_episodes = num_total_episodes - num_train_episodes

    generator = torch.Generator().manual_seed(cfg.seed) if cfg.seed is not None else None
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [num_train_episodes, num_val_episodes],
        generator=generator
    )

    # Use functools.partial to pass the config to the collate_fn
    collate_fn_with_config = functools.partial(fcil_policy_collate_fn, config=cfg.policy)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True,
        collate_fn=collate_fn_with_config # USE THE CUSTOM COLLATE FN via partial
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
        collate_fn=collate_fn_with_config
    )

    dl_iter = cycle(train_loader)

    logger.info("Creating FCILPolicy")
    policy = make_policy(cfg=cfg.policy, ds_meta=mixed_ds_meta_for_stats) 
    policy.to(device)

    logger.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy) 
    grad_scaler = GradScaler(enabled=cfg.policy.use_amp and device.type == 'cuda')

    step = 0
    if cfg.resume:
        if not cfg.checkpoint_path or not cfg.checkpoint_path.exists():
            potential_last_checkpoint = cfg.output_dir / "checkpoints" / "last"
            if potential_last_checkpoint.exists() and potential_last_checkpoint.is_symlink():
                 cfg.checkpoint_path = potential_last_checkpoint.resolve()
                 logger.info(f"Resuming from last checkpoint: {cfg.checkpoint_path}")
            else:
                raise FileNotFoundError(f"Resume specified but checkpoint_path '{cfg.checkpoint_path}' not found or not set.")
        
        policy_checkpoint_file = cfg.checkpoint_path / "pretrained_model" / "model.safetensors"
        if policy_checkpoint_file.exists():
            state_dict = torch.load(policy_checkpoint_file, map_location=device)
            policy.load_state_dict(state_dict)
            logger.info(f"Loaded policy weights from {policy_checkpoint_file}")
        else:
            logger.warning(f"Policy checkpoint file not found at {policy_checkpoint_file}, policy initialized with random weights.")
        
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
        logger.info(f"Resumed training from step {step}.")


    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logger.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"num_learnable_params: {format_big_number(num_learnable_params)}")

    train_metrics_def = {
        "loss": AverageMeter("loss", ":.4f"),       
        "mse_loss": AverageMeter("mse", ":.4f"),    
        "unweighted_mse_loss": AverageMeter("unw_mse", ":.4f"),
        "recovery_proportion": AverageMeter("rec_prop", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size,
        len(train_dataset), 
        len(train_dataset), 
        train_metrics_def,
        initial_step=step
    )

    (cfg.output_dir / TRAIN_CONFIG_NAME).parent.mkdir(parents=True, exist_ok=True)
    cfg_dict_to_save = cfg.to_dict()
    with open(cfg.output_dir / TRAIN_CONFIG_NAME, "w") as f:
        json.dump(cfg_dict_to_save, f, indent=4)

    logger.info("Start FCIL policy training")
    progress_bar = tqdm(
        range(step, cfg.steps),
        desc="Training",
        initial=step,
        total=cfg.steps,
        dynamic_ncols=True,
        position=0,
        leave=True,
    )
    for current_iter_step in progress_bar:
        iter_start_time = time.perf_counter()
        batch = next(dl_iter) 
        train_tracker.dataloading_s = time.perf_counter() - iter_start_time

        train_tracker, output_dict = update_fcil_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm, 
            grad_scaler,
            lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )
        train_tracker.step() 

        is_log_step = cfg.log_freq > 0 and train_tracker.steps % cfg.log_freq == 0
        is_saving_step = train_tracker.steps % cfg.save_freq == 0 or train_tracker.steps == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and train_tracker.steps % cfg.eval_freq == 0
        
        if is_log_step:
            progress_bar.set_postfix(loss=f"{train_tracker.loss.avg:.4f}", lr=f"{train_tracker.lr.avg:.1e}")
            tqdm.write(f"Step {train_tracker.steps}/{cfg.steps} - {str(train_tracker)}")
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    for k_metric, v_metric in output_dict.items():
                        if k_metric != "loss" and isinstance(v_metric, (int, float)): 
                             wandb_log_dict[f"train_batch/{k_metric}"] = v_metric
                wandb_logger.log_dict(wandb_log_dict, train_tracker.steps, mode="train")
            train_tracker.reset_averages()
        
        if is_eval_step:
            tqdm.write(f"Evaluating policy after step {train_tracker.steps}...")
            eval_output_dict = evaluate_policy(policy, val_loader, device, use_amp=cfg.policy.use_amp)
            tqdm.write(f"Step {train_tracker.steps}/{cfg.steps} - Eval Metrics: {eval_output_dict}")
            if wandb_logger:
                wandb_logger.log_dict(eval_output_dict, train_tracker.steps, mode="eval")
        
        if cfg.save_checkpoint and is_saving_step:
            tqdm.write(f"Checkpoint policy after step {train_tracker.steps}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, train_tracker.steps)
            save_checkpoint(checkpoint_dir, train_tracker.steps, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

    progress_bar.close()
    logger.info("End of FCIL policy training")

if __name__ == "__main__":
    init_logging()
    from lerobot.common.policies.factory import make_policy 
    train_fcil_policy()