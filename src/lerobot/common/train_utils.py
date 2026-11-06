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
import os
import shutil
from pathlib import Path

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.optim import (
    load_optimizer_state,
    load_scheduler_state,
    save_optimizer_state,
    save_scheduler_state,
)
from lerobot.policies import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.random_utils import load_rng_state, save_rng_state


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def save_training_step(step: int, save_dir: Path) -> None:
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    """Atomically point `last` at checkpoint_dir (new link under a temp name, then rename over)."""
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    tmp_link = checkpoint_dir.parent / f".{LAST_CHECKPOINT_LINK}.tmp"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    tmp_link.symlink_to(checkpoint_dir.relative_to(checkpoint_dir.parent))
    os.replace(tmp_link, last_checkpoint_dir)
    _fsync_dir(checkpoint_dir.parent)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
) -> None:
    """Write the checkpoint atomically (CLAUDE.md 9.4): into a hidden temp directory beside
    `checkpoint_dir`, fsynced, then renamed onto it. A kill mid-save leaves only the temp dir, never
    a partial checkpoint under a real step name. See _write_checkpoint for the layout."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = checkpoint_dir.parent / f".{checkpoint_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    _write_checkpoint(tmp_dir, step, cfg, policy, optimizer, scheduler, preprocessor, postprocessor)
    for dirpath, _, filenames in os.walk(tmp_dir):
        for name in filenames:
            fd = os.open(os.path.join(dirpath, name), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    if checkpoint_dir.exists():  # re-saving a step (e.g. after resuming from an earlier one)
        shutil.rmtree(checkpoint_dir)
    os.replace(tmp_dir, checkpoint_dir)
    _fsync_dir(checkpoint_dir.parent)


def _write_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
) -> None:
    """This function creates the following directory structure:

    005000/  #  training step at checkpoint
    ├── pretrained_model/
    │   ├── config.json  # policy config
    │   ├── model.safetensors  # policy weights
    │   ├── train_config.json  # train config
    │   ├── processor.json  # processor config (if preprocessor provided)
    │   └── step_*.safetensors  # processor state files (if any)
    └── training_state/
        ├── optimizer_param_groups.json  #  optimizer param groups
        ├── optimizer_state.safetensors  # optimizer state
        ├── rng_state.safetensors  # rng states
        ├── scheduler_state.json  # scheduler state
        └── training_step.json  # training step

    Args:
        cfg (TrainPipelineConfig): The training config used for this run.
        step (int): The training step at that checkpoint.
        policy (PreTrainedPolicy): The policy to save.
        optimizer (Optimizer | None, optional): The optimizer to save the state from. Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler to save the state from. Defaults to None.
        preprocessor: The preprocessor/pipeline to save. Defaults to None.
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)
    cfg.save_pretrained(pretrained_dir)
    if cfg.peft is not None:
        # When using PEFT, policy.save_pretrained will only write the adapter weights + config, not the
        # policy config which we need for loading the model. In this case we'll write it ourselves.
        policy.config.save_pretrained(pretrained_dir)
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)
    save_training_state(checkpoint_dir, step, optimizer, scheduler)

    # Save accumulated memory usage (if any) next to model.safetensors
    try:
        from lerobot.policies.modules.memory_lite import MLPPlusMemory

        usage = {}
        # Traverse modules to find memory wrappers and export their per-slot counts
        for module_name, module in policy.named_modules():
            mem_wrapper = getattr(module, "mlp", None)
            if isinstance(mem_wrapper, MLPPlusMemory) and hasattr(mem_wrapper, "mem"):
                mem = mem_wrapper.mem
                counts = getattr(mem, "usage_counts", None)
                batch_counts = getattr(mem, "usage_batch_counts", None)
                # Skip modules with no usage recorded at all
                if counts is None and batch_counts is None:
                    continue
                size = getattr(mem, "size", None)
                # Fallback sizes if needed
                if size is None:
                    if counts is not None:
                        size = len(counts)
                    elif batch_counts is not None:
                        size = len(batch_counts)
                total_list = counts.tolist() if counts is not None else [0] * int(size)
                batch_list = batch_counts.tolist() if batch_counts is not None else [0] * int(size)
                # Only persist non-zero entries to keep file small
                slot_counts = {}
                for i in range(int(size)):
                    total = int(total_list[i])
                    batches = int(batch_list[i])
                    if total > 0 or batches > 0:
                        slot_counts[f"value_slot_{i}"] = {
                            "total_accesses": total,
                            "batch_accesses": batches,
                        }
                if slot_counts:
                    usage[module_name] = slot_counts

        if usage:
            write_json({"per_module": usage}, pretrained_dir / "memory_usage.json")
    except Exception:
        # Never fail checkpointing due to optional usage logging
        pass


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> None:
    """
    Saves the training step, optimizer state, scheduler state, and rng state.

    Args:
        save_dir (Path): The directory to save artifacts to.
        train_step (int): Current training step.
        optimizer (Optimizer | None, optional): The optimizer from which to save the state_dict.
            Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler from which to save the state_dict.
            Defaults to None.
    """
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir)
    save_rng_state(save_dir)
    if optimizer is not None:
        save_optimizer_state(optimizer, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)


def load_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None
) -> tuple[int, Optimizer, LRScheduler | None]:
    """
    Loads the training step, optimizer state, scheduler state, and rng state.
    This is used to resume a training run.

    Args:
        checkpoint_dir (Path): The checkpoint directory. Should contain a 'training_state' dir.
        optimizer (Optimizer): The optimizer to load the state_dict to.
        scheduler (LRScheduler | None): The scheduler to load the state_dict to (can be None).

    Raises:
        NotADirectoryError: If 'checkpoint_dir' doesn't contain a 'training_state' dir

    Returns:
        tuple[int, Optimizer, LRScheduler | None]: training step, optimizer and scheduler with their
            state_dict loaded.
    """
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)

    return step, optimizer, scheduler
