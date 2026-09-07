#!/usr/bin/env python3
"""Shared scaffolding for the EXTERNAL continual-learning baselines (E67): RETAIN and O-LoRA.

Design rule (Josh, 7 Sep 26): these baselines live in scripts/baselines/ and IMPORT our code
(dataset/policy factories, the sequential trainer's per-task dataloader and paired-noise loss
evaluator, lerobot-train's update step) but never modify it. Everything protocol-critical is
taken from the same helpers the paper rows used, so the frames per task, the preprocessing and
the loss are identical to the naive/specialist/memory rows.

What lives here:
  * preemption contract helpers (CLAUDE.md 9.4): SIGTERM flag, atomic directory replace with
    fsync, timed saves;
  * the setup mirror of lerobot_sequential_train.sequential_train (accelerator, dataset,
    policy, processors) so both trainers build the model the same way as every other row;
  * per-task dataloader (imported verbatim) and the batch preparation used in the trainer loop.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator

from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_sequential_train import (  # our helpers, imported, unmodified
    _build_dataloader_for_task,
    _collect_task_index_to_name,
    _eval_loss_on_seen_tasks,
)
from lerobot.utils.utils import cycle

__all__ = [
    "Preempt",
    "install_sigterm_handler",
    "build_accelerator",
    "build_dataset_policy_processors",
    "task_dataloader",
    "prep_batch",
    "fsync_tree",
    "atomic_replace_dir",
    "write_json_atomic",
    "read_json",
    "timed",
    "update_last_symlink",
    "eval_loss_matrix_row",
    "_build_dataloader_for_task",
    "_collect_task_index_to_name",
]


# ---------------------------------------------------------------------------------------------
# Preemption contract
# ---------------------------------------------------------------------------------------------
class Preempt:
    """SIGTERM flag. The handler only sets the flag; the training loop checkpoints and exits 0."""

    def __init__(self) -> None:
        self.flag = False
        self.at: float | None = None

    def __call__(self, signum, frame) -> None:  # noqa: ANN001
        self.flag = True
        self.at = time.monotonic()
        print(f"[preempt] signal {signum} received - will checkpoint at the next step boundary", flush=True)


def install_sigterm_handler() -> Preempt:
    p = Preempt()
    signal.signal(signal.SIGTERM, p)
    return p


def fsync_tree(path: Path) -> None:
    """fsync every file under `path`, then the directories (durability of names + contents)."""
    path = Path(path)
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            if fp.is_symlink():
                continue
            fd = os.open(fp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    for root, dirs, files in os.walk(path, topdown=False):
        fd = os.open(root, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    parent_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def atomic_replace_dir(tmp_dir: Path, final_dir: Path) -> None:
    """Move a fully-written `tmp_dir` onto `final_dir`.

    A SIGKILL between the two renames leaves `<final>.old` (complete, previous state) and no
    `final`; readers must therefore fall back to `<final>.old` (see `resolve_state_dir`).
    """
    tmp_dir, final_dir = Path(tmp_dir), Path(final_dir)
    fsync_tree(tmp_dir)
    old = final_dir.with_name(final_dir.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if final_dir.exists():
        os.rename(final_dir, old)
    os.rename(tmp_dir, final_dir)
    fd = os.open(final_dir.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    if old.exists():
        shutil.rmtree(old)


def resolve_state_dir(final_dir: Path) -> Path | None:
    """`final_dir` if present, else `<final>.old` (interrupted replace), else None."""
    final_dir = Path(final_dir)
    if final_dir.exists():
        return final_dir
    old = final_dir.with_name(final_dir.name + ".old")
    if old.exists():
        return old
    return None


def write_json_atomic(path: Path, obj: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


class timed:
    """`with timed() as t: ...; t.s` -> elapsed seconds (checkpoint saves are logged, CLAUDE.md 9.4.7)."""

    def __enter__(self):
        self.t0 = time.monotonic()
        self.s = 0.0
        return self

    def __exit__(self, *a):
        self.s = time.monotonic() - self.t0
        return False


def update_last_symlink(checkpoints_dir: Path, step_dir: Path) -> None:
    last = Path(checkpoints_dir) / "last"
    if last.is_symlink() or last.exists():
        last.unlink()
    os.symlink(Path(step_dir).name, last)


# ---------------------------------------------------------------------------------------------
# Setup mirror of the sequential trainer
# ---------------------------------------------------------------------------------------------
def build_accelerator(cfg) -> Accelerator:
    from accelerate.utils import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    return Accelerator(
        gradient_accumulation_steps=getattr(cfg, "gradient_accumulation_steps", 1),
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )


def build_dataset_policy_processors(cfg, device):
    """Dataset + policy + pre/post processors, built exactly as sequential_train() builds them
    (dataset stats -> normalizer, rename map, device processor)."""
    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    processor_kwargs: dict[str, Any] = {}
    postprocessor_kwargs: dict[str, Any] = {}
    if cfg.policy.pretrained_path is None:
        raise ValueError("baselines always start from a checkpoint (--policy.path)")
    processor_kwargs["dataset_stats"] = dataset.meta.stats
    processor_kwargs["preprocessor_overrides"] = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
        "rename_observations_processor": {"rename_map": cfg.rename_map},
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
    if hasattr(policy, "precompute_task_embeddings"):
        policy.precompute_task_embeddings(dataset.meta)
    task_index_to_name = _collect_task_index_to_name(dataset)
    return dataset, policy, preprocessor, postprocessor, task_index_to_name


def task_dataloader(dataset, task_index_to_name, task_id: int, cfg, device):
    """The sequential trainer's per-task loader (episode-aware sampler over that task's episodes)."""
    drop_n_last = getattr(cfg.policy, "drop_n_last_frames", 0)
    dl = _build_dataloader_for_task(
        dataset,
        task_index_to_name,
        int(task_id),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device_type=device.type,
        drop_n_last_frames=drop_n_last,
    )
    return dl, cycle(dl)


def prep_batch(batch, dataset, preprocessor):
    for cam_key in dataset.meta.camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
    return preprocessor(batch)


def eval_loss_matrix_row(policy, accelerator, dataset, task_index_to_name, task_ids, device, preprocessor,
                         n_batches: int = 16, batch_size: int = 32, num_workers: int = 4, seed: int = 0):
    """One row of the paired-noise loss matrix, the E39/E65 instrument (imported, unmodified)."""
    return _eval_loss_on_seen_tasks(
        policy, accelerator, dataset, task_index_to_name, [int(t) for t in task_ids],
        batch_size=batch_size, num_workers=num_workers, device=device, n_batches=n_batches,
        preprocessor=preprocessor, seed=seed,
    )


def log_gpu_mem(tag: str) -> None:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 2**30
        peak = torch.cuda.max_memory_allocated() / 2**30
        logging.info(f"[{tag}] cuda allocated {alloc:.1f} GiB, peak {peak:.1f} GiB")


# ---------------------------------------------------------------------------------------------
# One optimisation step, structured exactly like lerobot_train.update_policy (forward under
# accelerator.autocast, accelerator.backward, and on the sync step: clip -> optimizer.step ->
# zero_grad -> scheduler.step). `extra_loss_fn` adds a differentiable regulariser (O-LoRA).
# ---------------------------------------------------------------------------------------------
def train_step(policy, batch, optimizer, lr_scheduler, accelerator, grad_clip_norm: float,
               extra_loss_fn=None):
    policy.train()
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
    penalty = None
    total = loss
    if extra_loss_fn is not None:
        penalty = extra_loss_fn()
        total = loss + penalty
    accelerator.backward(total)
    grad_norm = None
    lr = optimizer.param_groups[0]["lr"]
    synced = bool(accelerator.sync_gradients)
    if synced:
        if grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"), error_if_nonfinite=False)
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
        upd = getattr(unwrapped, "update", None)
        if callable(upd):
            upd()
        grad_norm = float(grad_norm)
    return float(loss.detach()), (None if penalty is None else float(penalty.detach())), grad_norm, lr, synced, output_dict


class RunningMeans:
    def __init__(self):
        self.d: dict[str, list[float]] = {}

    def add(self, **kv):
        for k, v in kv.items():
            if v is None:
                continue
            self.d.setdefault(k, []).append(float(v))

    def mean(self, k: str) -> float | None:
        v = self.d.get(k)
        return (sum(v) / len(v)) if v else None

    def reset(self):
        self.d = {}
