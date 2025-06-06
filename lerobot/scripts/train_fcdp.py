#!/usr/bin/env python

# lerobot/lerobot/scripts/train_fcdp.py

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
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import List, Dict, Any, Tuple
import functools
from tqdm import tqdm
from termcolor import colored
import torch
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.fcdp.fcdp_dataset import FCDPDataset, fcdp_collate_fn
from lerobot.common.datasets.utils import cycle
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.fcdp.configuration_fcdp import FCDPConfig
from lerobot.common.policies.fcdp.modeling_fcdp import FCDPPolicy
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
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig, TRAIN_CONFIG_NAME

logger = logging.getLogger(__name__)


def move_to_device(item, device):
    """Recursively move tensors to device."""
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, dict):
        return {k: move_to_device(v, device) for k, v in item.items()}
    elif isinstance(item, list):
        return [move_to_device(i, device) for i in item]
    else:
        return item


@dataclass
class FCDPTrainConfig(TrainPipelineConfig):
    policy: FCDPConfig = field(default_factory=FCDPConfig)
    
    success_dataset_repo_id: str = "lerobot/success_100"
    mixed_dataset_repo_id: str | None = "lerobot/mixed_50"
    dataset_root: Path | None = None
    train_val_split_ratio: float = 0.9
    eval_freq: int = 10

    def __post_init__(self):
        super().__post_init__()
        if not self.job_name:
            self.job_name = f"fcdp_{self.policy.model_dim}md_{self.policy.n_layers}l_{self.policy.horizon}h"

    def validate(self):
        super().validate()
        if not self.success_dataset_repo_id:
            raise ValueError("`success_dataset_repo_id` must be provided for FCDP training.")


def update_fcdp_policy(
    train_metrics: MetricsTracker,
    policy: FCDPPolicy,
    batch: Tuple[Dict[str, Any], torch.Tensor, torch.Tensor],
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()

    # Move batch to device
    policy_input_dict, target_action_trajectory, target_padding_mask = batch
    
    policy_input_dict = move_to_device(policy_input_dict, device)
    target_action_trajectory = target_action_trajectory.to(device, non_blocking=True)
    target_padding_mask = target_padding_mask.to(device, non_blocking=True)
    
    batch_on_device = (policy_input_dict, target_action_trajectory, target_padding_mask)
    
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch_on_device)
    
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: FCDPTrainConfig):
    cfg.validate()
    init_logging()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    # Create FCDPDataset
    dataset = FCDPDataset(
        config=cfg.policy,
        success_dataset_repo_id=cfg.success_dataset_repo_id,
        mixed_dataset_repo_id=cfg.mixed_dataset_repo_id,
        root=cfg.dataset_root,
        image_transforms=None,
        revision=None,
    )
    
    # Create train/val split
    train_size = int(cfg.train_val_split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Val dataset size: {len(val_dataset)}")

    logging.info("Creating policy")
    # Use make_policy to properly set up input/output features
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.success_ds_meta)
    policy = policy.to(device)

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{len(dataset)=} samples")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Create custom collate function
    collate_fn = functools.partial(fcdp_collate_fn, config=cfg.policy)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=device.type != "cpu",
        drop_last=False,
        collate_fn=collate_fn,
    )
    
    dl_iter = cycle(train_dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, len(train_dataset), len(train_dataset) // cfg.batch_size, train_metrics, initial_step=step
    )

    # Training loop
    progress_bar = tqdm(range(step, cfg.steps), desc="Training", leave=True, position=0)
    
    for current_step in progress_bar:
        start_data_loading_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_data_loading_time

        train_tracker, output_dict = update_fcdp_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler,
            lr_scheduler,
            cfg.policy.use_amp,
        )

        train_tracker.step += 1
        
        # Log
        if train_tracker.step % cfg.log_freq == 0:
            logging.info(
                f"Step {train_tracker.step} | "
                f"Loss: {train_tracker.loss.val:.3f} ({train_tracker.loss.avg:.3f}) | "
                f"LR: {train_tracker.lr.val:.2e} | "
                f"Grad: {train_tracker.grad_norm.val:.3f}"
            )

            if wandb_logger:
                log_dict = {
                    "train/loss": train_tracker.loss.avg,
                    "train/grad_norm": train_tracker.grad_norm.avg,
                    "train/lr": train_tracker.lr.val,
                    "train/update_s": train_tracker.update_s.avg,
                    "train/dataloading_s": train_tracker.data_loading_s.avg,
                    "step": train_tracker.step,
                }
                wandb_logger.log(log_dict)

        # Evaluate
        if cfg.eval_freq > 0 and train_tracker.step % cfg.eval_freq == 0:
            logging.info("Running validation...")
            policy.eval()
            val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation", leave=False):
                    policy_input_dict, target_action_trajectory, target_padding_mask = batch
                    
                    policy_input_dict = move_to_device(policy_input_dict, device)
                    target_action_trajectory = target_action_trajectory.to(device, non_blocking=True)
                    target_padding_mask = target_padding_mask.to(device, non_blocking=True)
                    
                    batch_on_device = (policy_input_dict, target_action_trajectory, target_padding_mask)
                    
                    with torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
                        loss, _ = policy.forward(batch_on_device)
                    
                    val_loss += loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss / val_steps
            logging.info(f"Validation loss: {avg_val_loss:.3f}")
            
            if wandb_logger:
                wandb_logger.log({"val/loss": avg_val_loss, "step": train_tracker.step})
            
            policy.train()

        # Save checkpoint
        if cfg.save_checkpoint and train_tracker.step % cfg.checkpoint_freq == 0:
            save_checkpoint(
                cfg.checkpoint_path, get_step_checkpoint_dir(train_tracker.step), train_tracker, policy, optimizer, lr_scheduler
            )
            update_last_checkpoint(cfg.checkpoint_path, train_tracker.step)

    # Final checkpoint
    if cfg.save_checkpoint:
        save_checkpoint(
            cfg.checkpoint_path, get_step_checkpoint_dir(train_tracker.step), train_tracker, policy, optimizer, lr_scheduler
        )
        update_last_checkpoint(cfg.checkpoint_path, train_tracker.step)

    logging.info("Training completed!")


if __name__ == "__main__":
    train()