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
from typing import List # Python 3.8 compatibility

import torch
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split # Added random_split
from safetensors.torch import load_model # For loading pretrained models

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.trajectory_dataset import TrajectoryDataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.fcil_classifier.configuration_fcil_classifier import FCILClassifierConfig
from lerobot.common.policies.fcil_classifier.modeling_fcil_classifier import FCILClassifierPolicy
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
from lerobot.configs.default import DatasetConfig # For type hinting

@dataclass
class FCILClassifierTrainConfig(TrainPipelineConfig):
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(repo_id="fcil_classifier/dummy_dataset_placeholder"))
    policy: FCILClassifierConfig = field(default_factory=FCILClassifierConfig)
    
    success_dataset_repo_id: str = "lerobot/so100_peg_insertion_cloned_teleop_success"
    mixed_dataset_repo_id: str = "lerobot/so100_peg_insertion_cloned_teleop_mixed"
    mixed_ds_identifier: str = "mixed"
    dataset_root: Path | None = None
    
    train_val_split_ratio: float = 0.9 # Ratio of data for training, rest for validation
    eval_freq: int = 1000 # Evaluate every N training steps

    image_feature_keys: List[str] = field(default_factory=lambda: ["observation.images.head", "observation.images.wrist"])
    use_embeddings: bool = False
    embedding_dim: int | None = None

    def __post_init__(self):
        self.checkpoint_path = None

    def validate(self):
        if self.resume:
            config_path_cli = parser.parse_arg("config_path")
            if not config_path_cli:
                policy_path_cli = parser.get_path_arg("policy")
                if policy_path_cli:
                    policy_path_obj = Path(policy_path_cli)
                    if policy_path_obj.name == "pretrained_model":
                        potential_config_path = policy_path_obj.parent / TRAIN_CONFIG_NAME
                        if potential_config_path.exists():
                             config_path_cli = str(potential_config_path)
                if not config_path_cli:
                    raise ValueError(
                        f"A config_path or a policy.path pointing to a checkpoint is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                    )
            if not Path(config_path_cli).resolve().exists():
                raise NotADirectoryError(
                    f"config_path '{config_path_cli}' is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )
            self.policy.pretrained_path = Path(config_path_cli).parent / "pretrained_model"
            self.checkpoint_path = Path(config_path_cli).parent

        if not self.job_name:
            self.job_name = f"{self.policy.type}_so100"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            if any(self.output_dir.iterdir()):
                raise FileExistsError(
                    f"Output directory {self.output_dir} already exists and is not empty, and resume is {self.resume}. "
                    f"Please change your output directory or set resume=True."
                )
        elif not self.output_dir:
            import datetime as dt
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train_fcil_classifier") / train_dir

        if self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()
        elif not self.use_policy_training_preset and self.optimizer is None:
             raise ValueError("Optimizer must be set when policy training presets are not used.")

        if not self.policy.image_feature_keys and self.image_feature_keys:
            self.policy.image_feature_keys = self.image_feature_keys
        self.policy.use_embeddings = self.use_embeddings
        self.policy.embedding_dim = self.embedding_dim


def update_classifier_policy(
    train_metrics: MetricsTracker,
    policy: FCILClassifierPolicy,
    batch: dict,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train() # Ensure model is in training mode

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, enabled=use_amp) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)

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
            if k == "loss": 
                continue
            if hasattr(train_metrics, k) and isinstance(getattr(train_metrics,k), AverageMeter):
                getattr(train_metrics,k).update(v) 
            elif isinstance(v, (int, float)):
                 logging.debug(f"Metric '{k}' from policy output not in MetricsTracker's AverageMeters.")

    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

@torch.no_grad() # Evaluation should not compute gradients
def evaluate_policy(policy: FCILClassifierPolicy, val_loader: DataLoader, device: torch.device, use_amp: bool = False) -> dict:
    policy.eval() # Ensure model is in evaluation mode
    
    # Reset policy's internal metrics before evaluation epoch
    if hasattr(policy, 'reset') and callable(getattr(policy, 'reset')):
        policy.reset() 

    all_batch_losses = []
    all_batch_accuracies = []
    all_batch_precisions = []
    all_batch_recalls = []
    all_batch_f1_scores = []
    
    for batch in val_loader:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)
        
        with torch.autocast(device_type=device.type, enabled=use_amp) if use_amp else nullcontext():
            loss, output_dict = policy.forward(batch) # policy.forward now returns batch-wise metrics

        all_batch_losses.append(loss.item())
        all_batch_accuracies.append(output_dict.get("accuracy", 0.0))
        all_batch_precisions.append(output_dict.get("precision", 0.0))
        all_batch_recalls.append(output_dict.get("recall", 0.0))
        all_batch_f1_scores.append(output_dict.get("f1_score", 0.0))

    # Aggregate metrics over all validation batches
    eval_metrics = {
        "loss": sum(all_batch_losses) / len(all_batch_losses) if all_batch_losses else 0.0,
        "accuracy": sum(all_batch_accuracies) / len(all_batch_accuracies) if all_batch_accuracies else 0.0,
        "precision": sum(all_batch_precisions) / len(all_batch_precisions) if all_batch_precisions else 0.0,
        "recall": sum(all_batch_recalls) / len(all_batch_recalls) if all_batch_recalls else 0.0,
        "f1_score": sum(all_batch_f1_scores) / len(all_batch_f1_scores) if all_batch_f1_scores else 0.0,
    }
    return eval_metrics


@parser.wrap(config_path=None)
def train_fcil_classifier(cfg: FCILClassifierTrainConfig):
    if not cfg.resume:
        cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        from termcolor import colored
        wandb_logger = WandBLogger(cfg)
    else:
        from termcolor import colored # Ensure termcolor is imported for offline logging too
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating TrajectoryDataset for classifier training")
    try:
        logging.info(f"Loading success dataset metadata from {cfg.success_dataset_repo_id} at {cfg.dataset_root}")
        success_root = Path(cfg.dataset_root) / cfg.success_dataset_repo_id if cfg.dataset_root else None
        example_ds_meta_success = LeRobotDatasetMetadata(cfg.success_dataset_repo_id, root=success_root)

        if cfg.policy.state_dim is None:
            cfg.policy.state_dim = example_ds_meta_success.features["observation.state"]["shape"][0]
        if cfg.policy.action_dim is None:
            cfg.policy.action_dim = example_ds_meta_success.features["action"]["shape"][0]

        if not cfg.policy.image_feature_keys and not cfg.use_embeddings and example_ds_meta_success.camera_keys:
            cfg.policy.image_feature_keys = example_ds_meta_success.camera_keys
            logging.info(f"Inferred image_feature_keys for raw images from success dataset: {cfg.policy.image_feature_keys}")
        elif not cfg.policy.image_feature_keys and cfg.image_feature_keys:
             cfg.policy.image_feature_keys = cfg.image_feature_keys
        
        mixed_root = Path(cfg.dataset_root) / cfg.mixed_dataset_repo_id if cfg.dataset_root else None
        example_ds_meta_mixed = LeRobotDatasetMetadata(cfg.mixed_dataset_repo_id, root=mixed_root)
        if example_ds_meta_success.stats and example_ds_meta_mixed.stats:
            combined_stats = aggregate_stats([example_ds_meta_success.stats, example_ds_meta_mixed.stats])
        elif example_ds_meta_success.stats:
            combined_stats = example_ds_meta_success.stats
            logging.warning("Using stats only from success_dataset_repo_id as mixed_dataset_repo_id has no stats.")
        elif example_ds_meta_mixed.stats:
            combined_stats = example_ds_meta_mixed.stats
            logging.warning("Using stats only from mixed_dataset_repo_id as success_dataset_repo_id has no stats.")
        else:
            combined_stats = None
            logging.warning("No stats found in either dataset metadata. Normalization layers will use defaults (infinity).")

    except Exception as e:
        logging.error(f"Could not load metadata to infer state/action dims or stats: {e}")
        if cfg.policy.state_dim is None or cfg.policy.action_dim is None :
            logging.error("Please ensure `state_dim` and `action_dim` are set in the config, or provide valid dataset_repo_ids for metadata loading. Defaults to 6.")
            if cfg.policy.state_dim is None: cfg.policy.state_dim = 6
            if cfg.policy.action_dim is None: cfg.policy.action_dim = 6
        if not cfg.policy.image_feature_keys and cfg.image_feature_keys:
            cfg.policy.image_feature_keys = cfg.image_feature_keys
        elif not cfg.policy.image_feature_keys and not cfg.use_embeddings:
            logging.warning("No image_feature_keys for raw images specified in config and could not infer.")
        combined_stats = None

    download_media = not cfg.use_embeddings and bool(cfg.policy.image_feature_keys)
    logging.info(f"Using state_dim: {cfg.policy.state_dim}, action_dim: {cfg.policy.action_dim}, feature_keys: {cfg.policy.image_feature_keys}, use_embeddings: {cfg.use_embeddings}, embedding_dim: {cfg.embedding_dim}")

    full_trajectory_dataset = TrajectoryDataset(
        repo_ids=[cfg.success_dataset_repo_id, cfg.mixed_dataset_repo_id],
        max_seq_len=cfg.policy.max_seq_len,
        image_feature_keys=cfg.policy.image_feature_keys,
        root=cfg.dataset_root,
        mixed_ds_identifier=cfg.mixed_ds_identifier,
        download_videos=download_media,
        use_embeddings=cfg.use_embeddings,
        embedding_dim=cfg.embedding_dim
    )

    num_total_episodes = len(full_trajectory_dataset)
    if num_total_episodes == 0:
        raise ValueError("TrajectoryDataset is empty. Check repo_ids and dataset contents.")
    logging.info(f"TrajectoryDataset created with {num_total_episodes} total episodes.")

    # Split dataset
    num_train_episodes = int(num_total_episodes * cfg.train_val_split_ratio)
    num_val_episodes = num_total_episodes - num_train_episodes
    
    generator = torch.Generator().manual_seed(cfg.seed) if cfg.seed is not None else None
    train_trajectory_dataset, val_trajectory_dataset = random_split(
        full_trajectory_dataset, 
        [num_train_episodes, num_val_episodes],
        generator=generator
    )
    logging.info(f"Split dataset: {len(train_trajectory_dataset)} train episodes, {len(val_trajectory_dataset)} val episodes.")

    train_loader = DataLoader(
        train_trajectory_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True, # Important for consistent batch sizes
    )
    dl_iter = cycle(train_loader)

    val_loader = DataLoader(
        val_trajectory_dataset,
        batch_size=cfg.batch_size, # Can be different from train_loader batch_size
        shuffle=False, # No need to shuffle validation data
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False, # Evaluate on all validation samples
    )

    logging.info("Creating FCILClassifierPolicy")
    policy = FCILClassifierPolicy(cfg.policy, dataset_stats=combined_stats)
    policy.to(device)

    logging.info("Creating optimizer and scheduler")
    optimizer = cfg.optimizer.build(policy.get_optim_params())
    lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler else None
    grad_scaler = GradScaler(enabled=cfg.policy.use_amp and device.type == 'cuda')

    step = 0
    if cfg.resume:
        if not cfg.checkpoint_path or not cfg.checkpoint_path.exists():
            potential_last_checkpoint = cfg.output_dir / "checkpoints" / "last"
            if potential_last_checkpoint.exists() and potential_last_checkpoint.is_symlink():
                 cfg.checkpoint_path = potential_last_checkpoint.resolve()
                 logging.info(f"Resuming from last checkpoint: {cfg.checkpoint_path}")
            else:
                raise FileNotFoundError(f"Resume specified but checkpoint_path '{cfg.checkpoint_path}' not found or not set.")
        policy_checkpoint_file = cfg.checkpoint_path / "pretrained_model" / "model.safetensors"
        if policy_checkpoint_file.exists():
            load_model(policy, policy_checkpoint_file, strict=True)
            logging.info(f"Loaded policy weights from {policy_checkpoint_file}")
        else:
            logging.warning(f"Policy checkpoint file not found at {policy_checkpoint_file}, policy initialized with random weights.")
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    if cfg.resume:
        cfg.validate()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f"num_learnable_params: {format_big_number(num_learnable_params)}")

    train_metrics_def = {
        "loss": AverageMeter("loss", ":.4f"),
        "accuracy": AverageMeter("acc", ":.3f"),
        "precision": AverageMeter("prec", ":.3f"),
        "recall": AverageMeter("rec", ":.3f"),
        "f1_score": AverageMeter("f1", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size,
        len(train_trajectory_dataset), # num_frames_per_epoch_train
        len(train_trajectory_dataset), # num_episodes for train
        train_metrics_def,
        initial_step=step
    )

    (cfg.output_dir / TRAIN_CONFIG_NAME).parent.mkdir(parents=True, exist_ok=True)
    cfg_dict_to_save = cfg.to_dict()
    if "dataset" in cfg_dict_to_save and cfg_dict_to_save["dataset"]["repo_id"] == "fcil_classifier/dummy_dataset_placeholder":
        del cfg_dict_to_save["dataset"]
    with open(cfg.output_dir / TRAIN_CONFIG_NAME, "w") as f:
        json.dump(cfg_dict_to_save, f, indent=4)

    logging.info("Start classifier training")
    for current_iter_step in range(step, cfg.steps):
        if current_iter_step % (cfg.log_freq if cfg.log_freq > 0 else 100) == 0 :
            logging.info(f"Overall Progress: Step {current_iter_step}/{cfg.steps}")
            
        iter_start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - iter_start_time

        train_tracker, output_dict = update_classifier_policy(
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
            logging.info(f"Step {train_tracker.steps}/{cfg.steps} - {str(train_tracker)}")
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    for k, v in output_dict.items():
                        if k != "loss" and isinstance(v, (int, float)):
                             wandb_log_dict[f"train_batch/{k}"] = v
                wandb_logger.log_dict(wandb_log_dict, train_tracker.steps, mode="train") # Explicitly train mode
            train_tracker.reset_averages()

        if is_eval_step:
            logging.info(f"Evaluating policy after step {train_tracker.steps}...")
            # policy.reset() # Reset internal metrics of the policy if it accumulates them
            eval_output_dict = evaluate_policy(policy, val_loader, device, use_amp=cfg.policy.use_amp)
            logging.info(f"Step {train_tracker.steps}/{cfg.steps} - Eval Metrics: {eval_output_dict}")
            if wandb_logger:
                wandb_logger.log_dict(eval_output_dict, train_tracker.steps, mode="eval")


        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {train_tracker.steps}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, train_tracker.steps)
            save_checkpoint(checkpoint_dir, train_tracker.steps, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

    logging.info("End of classifier training")

if __name__ == "__main__":
    init_logging()
    train_fcil_classifier()