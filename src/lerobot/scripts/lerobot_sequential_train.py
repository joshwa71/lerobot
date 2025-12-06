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
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
import math
import os
import time
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

    # When true, use TF-only slot masking (no IDF stats required).
    # If enabled, this overrides TF-IDF behavior.
    tf_only: bool = False

    # Optional dataset->env task id mapping as a JSON string.
    # If empty and env.task == "libero_10", a default mapping is used.
    # Example CLI: --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'
    ds_to_env_map_json: str | None = None

    # Save a checkpoint after each task
    save_after_each_task: bool = True

    # Rebuild optimizer each task (False keeps momentum/state across tasks)
    reinit_optimizer_each_task: bool = False

    # Trainable subsets and learning rates for memory components
    # Values
    train_memory_value: bool = True
    # Learning rate for memory value parameters (pk_value_param).
    memory_value_lr: float = 1e-3
    # End LR for LR schedule; if None, use static LR
    memory_value_lr_end: float | None = None
    # LR scheduler type for memory values: "linear" or "cosine"
    memory_value_scheduler_type: str = "linear"

    # Keys
    train_memory_keys: bool = False
    memory_keys_lr: float = 1e-3
    # End LR for LR schedule; if None, use static LR
    memory_keys_lr_end: float | None = None
    # LR scheduler type for memory keys: "linear" or "cosine"
    memory_keys_scheduler_type: str = "linear"

    # Query projection (the memory query MLP linear projection)
    train_query_proj: bool = False
    query_proj_lr: float = 1e-3
    # End LR for LR schedule; if None, use static LR
    query_proj_lr_end: float | None = None
    # LR scheduler type for query projection: "linear" or "cosine"
    query_proj_scheduler_type: str = "linear"

    # TF-IDF gating to sparsify memory value updates
    tfidf_enable: bool = True
    # Number of memory value slots per module allowed to receive gradients each step
    tfidf_top_t: int = 128
    # Optional path to pretraining memory usage stats JSON (memory_usage.json)
    idf_stats_path: str | None = None
    # When true, ignore pretraining IDF stats and build IDF online from slot usage
    # observed during sequential training (across all tasks seen so far).
    use_online_idf_stats: bool = False

    # Exponent applied to IDF scores. >1 increases exploration (penalizes frequent slots more),
    # <1 decreases exploration. Default 1.0 = standard IDF.
    idf_exponent: float = 1.0


def _default_libero10_map() -> dict[int, int]:
    return {0: 4, 1: 6, 2: 9, 3: 2, 4: 7, 5: 0, 6: 8, 7: 1, 8: 3, 9: 5}


def _append_eval_results_jsonl(
    output_dir: Path,
    step: int,
    eval_info: dict,
):
    """
    Append a JSONL line with per-task success percentages to {output_dir}/eval/results.jsonl.
    Format: {"step": step, "task_0": success%, "task_1": success%, ...}
    """
    import numpy as np

    per_task = eval_info.get("per_task", [])
    if not per_task:
        return

    record = {"step": step}
    for entry in per_task:
        task_id = entry.get("task_id")
        metrics = entry.get("metrics", {})
        successes = metrics.get("successes", [])
        if successes:
            pc_success = float(np.mean(successes) * 100)
        else:
            pc_success = float("nan")
        record[f"task_{task_id}"] = pc_success

    results_path = Path(output_dir) / "eval" / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _build_memory_scheduler(
    optimizer: Optimizer,
    cfg: SequentialOnlineConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Build an LR scheduler for memory param groups.

    Each param group can have its own schedule (start_lr -> end_lr over total_steps),
    and its own scheduler type ("linear" or "cosine").
    If end_lr is None for a group, that group uses static LR.
    Returns None if all param groups use static LR.

    The scheduler resets LR to start values when created, so call this at the start of each task.
    """
    group_schedules: list[tuple[float, float, str]] = []

    if cfg.train_memory_value:
        start = cfg.memory_value_lr
        end = cfg.memory_value_lr_end if cfg.memory_value_lr_end is not None else start
        sched_type = getattr(cfg, "memory_value_scheduler_type", "linear")
        group_schedules.append((start, end, sched_type))

    if cfg.train_memory_keys:
        start = cfg.memory_keys_lr
        end = cfg.memory_keys_lr_end if cfg.memory_keys_lr_end is not None else start
        sched_type = getattr(cfg, "memory_keys_scheduler_type", "linear")
        group_schedules.append((start, end, sched_type))

    if cfg.train_query_proj:
        start = cfg.query_proj_lr
        end = cfg.query_proj_lr_end if cfg.query_proj_lr_end is not None else start
        sched_type = getattr(cfg, "query_proj_scheduler_type", "linear")
        group_schedules.append((start, end, sched_type))

    if len(group_schedules) != len(optimizer.param_groups):
        return None

    all_static = all(abs(start - end) < 1e-12 for start, end, _ in group_schedules)
    if all_static:
        return None

    for i, (start_lr, _, _) in enumerate(group_schedules):
        optimizer.param_groups[i]["lr"] = start_lr
        optimizer.param_groups[i]["initial_lr"] = start_lr

    def make_lr_lambda(start_lr: float, end_lr: float, total: int, scheduler_type: str):
        scheduler_type = (scheduler_type or "linear").lower()

        def lr_lambda(step: int) -> float:
            if total <= 1 or abs(start_lr) < 1e-12:
                return 1.0
            progress = min(step / max(total - 1, 1), 1.0)
            if scheduler_type == "linear":
                return 1.0 - progress * (1.0 - end_lr / start_lr)
            if scheduler_type == "cosine":
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                ratio_end = end_lr / start_lr
                return ratio_end + (1.0 - ratio_end) * cosine
            raise ValueError(f"Unknown scheduler type: {scheduler_type}. Expected 'linear' or 'cosine'.")

        return lr_lambda

    lambdas = [
        make_lr_lambda(start, end, total_steps, sched_type)
        for (start, end, sched_type) in group_schedules
    ]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)


def _reset_scheduler_for_task(
    optimizer: Optimizer,
    cfg: SequentialOnlineConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Reset optimizer LRs to start values and create a fresh scheduler for a new task.
    """
    group_lrs: list[float] = []
    if cfg.train_memory_value:
        group_lrs.append(cfg.memory_value_lr)
    if cfg.train_memory_keys:
        group_lrs.append(cfg.memory_keys_lr)
    if cfg.train_query_proj:
        group_lrs.append(cfg.query_proj_lr)

    for i, lr in enumerate(group_lrs):
        if i < len(optimizer.param_groups):
            optimizer.param_groups[i]["lr"] = lr

    return _build_memory_scheduler(optimizer, cfg, total_steps)


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
        shuffle=False,
        sampler=sampler,
        pin_memory=device_type == "cuda",
        drop_last=False,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def _freeze_to_selected_memory_params(
    policy: PreTrainedPolicy,
    train_memory_value: bool,
    train_memory_keys: bool,
    train_query_proj: bool,
) -> int:
    """Freeze all parameters except selected memory-related parameters. Returns number of trainable params."""
    trainable = 0
    for p in policy.parameters():
        if getattr(p, "pk_value_param", False) and train_memory_value:
            p.requires_grad = True
        elif getattr(p, "pk_keys_param", False) and train_memory_keys:
            p.requires_grad = True
        elif getattr(p, "pk_query_proj_param", False) and train_query_proj:
            p.requires_grad = True
        else:
            p.requires_grad = False
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


def _iter_memory_modules(unwrapped_policy: PreTrainedPolicy):
    """
    Yield tuples of (layer_index, mem_module, values_param, json_key) for all attached memory layers
    across the action expert and the VLM text backbone.

    json_key strings are aligned with memory_usage.json conventions:
      - Expert: "model.vlm_with_expert.lm_expert.layers.{i}"
      - VLM:    "model.vlm_with_expert.vlm.model.text_model.layers.{i}"
    """
    mems = []
    try:
        model = unwrapped_policy.model  # VLAFlowMatching
        expert = model.vlm_with_expert.lm_expert
        for li, layer in enumerate(expert.layers):
            mlp = getattr(layer, "mlp", None)
            # Lazy import to avoid circulars; type check by attribute presence
            mem = getattr(getattr(mlp, "mem", None), "values", None)
            if mem is not None and hasattr(mlp, "mem"):
                mem_module = mlp.mem
                values_param = mlp.mem.values
                json_key = f"model.vlm_with_expert.lm_expert.layers.{li}"
                mems.append((li, mem_module, values_param, json_key))
    except Exception:
        pass

    # Include VLM backbone (text_model) memory layers if present
    try:
        model = unwrapped_policy.model  # VLAFlowMatching
        vlm_text_model = model.vlm_with_expert.get_vlm_model().text_model
        for li, layer in enumerate(vlm_text_model.layers):
            mlp = getattr(layer, "mlp", None)
            mem = getattr(getattr(mlp, "mem", None), "values", None)
            if mem is not None and hasattr(mlp, "mem"):
                mem_module = mlp.mem
                values_param = mlp.mem.values
                json_key = f"model.vlm_with_expert.vlm.model.text_model.layers.{li}"
                mems.append((li, mem_module, values_param, json_key))
    except Exception:
        pass
    return mems


def _enable_memory_batch_logging(unwrapped_policy: PreTrainedPolicy, enable: bool = True):
    """
    Ensure per-batch slot indices are recorded during training by toggling mem.log_usage.
    """
    for _, mem_module, _, _ in _iter_memory_modules(unwrapped_policy):
        try:
            mem_module.log_usage = bool(enable)
        except Exception:
            pass


# In-memory accumulators for per-task memory slot usage (sequential adaptation)
# module_key -> task_id -> Counter(slot_idx -> total_count)
_per_task_totals = defaultdict(lambda: defaultdict(Counter))
# module_key -> task_id -> Counter(slot_idx -> batch_count)
_per_task_batches = defaultdict(lambda: defaultdict(Counter))
 # module_key -> task_id -> Counter(slot_idx -> total_update_count)
_per_task_update_totals = defaultdict(lambda: defaultdict(Counter))
# module_key -> task_id -> Counter(slot_idx -> batch_update_count)
_per_task_update_batches = defaultdict(lambda: defaultdict(Counter))


def _accumulate_task_usage_for_batch(unwrapped_policy: PreTrainedPolicy, task_id: int):
    """
    Accumulate per-task slot usage for the current batch.
    Assumes the current dataloader is filtered to a single dataset task id.
    """
    try:
        for _, mem, _, json_key in _iter_memory_modules(unwrapped_policy):
            idx = getattr(mem, "last_indices", None)
            if idx is None:
                continue
            # idx: (B*T, heads, knn) effectively, since memory flattens time; here we only need slot counts
            idx_flat = idx.reshape(-1).to(torch.long)
            num_slots = int(getattr(mem, "size", 0))
            if num_slots <= 0:
                continue
            counts = torch.bincount(idx_flat, minlength=num_slots)
            used = counts > 0
            if used.any():
                slots = used.nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
                vals = counts[used].detach().cpu().tolist()
                Tctr = _per_task_totals[json_key][int(task_id)]
                Bctr = _per_task_batches[json_key][int(task_id)]
                for s, v in zip(slots, vals):
                    Tctr[int(s)] += int(v)
                    Bctr[int(s)] += 1
    except Exception:
        # Never fail training due to optional logging
        pass


def _accumulate_task_updates_for_batch(unwrapped_policy: PreTrainedPolicy, task_id: int):
    """
    Accumulate per-task slot updates for the current batch.

    A slot is considered updated for a batch if it was selected by the TF-IDF
    gating as eligible to receive gradient updates in that step.
    """
    try:
        for _, mem, _, json_key in _iter_memory_modules(unwrapped_policy):
            idx = getattr(mem, "last_update_indices", None)
            if idx is None:
                continue
            idx_flat = idx.view(-1).to(torch.long)
            if idx_flat.numel() == 0:
                continue
            num_slots = int(getattr(mem, "size", 0))
            if num_slots <= 0:
                continue
            counts = torch.bincount(idx_flat, minlength=num_slots)
            used = counts > 0
            if used.any():
                slots = used.nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
                vals = counts[used].detach().cpu().tolist()
                Tctr = _per_task_update_totals[json_key][int(task_id)]
                Bctr = _per_task_update_batches[json_key][int(task_id)]
                for s, v in zip(slots, vals):
                    s_int = int(s)
                    Tctr[s_int] += int(v)
                    Bctr[s_int] += 1
    except Exception:
        # Never fail training due to optional logging
        pass


def _flush_per_task_usage(out_dir: Path, task_id: int | None = None):
    """
    Write JSON files under <out_dir>/memory_by_task/ summarizing per-task slot usage for each memory module.
    """
    try:
        out_dir = Path(out_dir) / "memory_by_task"
        out_dir.mkdir(parents=True, exist_ok=True)
        if task_id is not None:
            task_ids = [int(task_id)]
        else:
            seen = set()
            for by_task in _per_task_totals.values():
                seen.update(by_task.keys())
            task_ids = sorted(int(t) for t in seen)

        for t in task_ids:
            payload = {"per_module": {}}
            for json_key, by_task in _per_task_totals.items():
                tctr = by_task.get(int(t), Counter())
                uctr = _per_task_update_totals.get(json_key, {}).get(int(t), Counter())
                if not tctr and not uctr:
                    continue
                bctr = _per_task_batches.get(json_key, {}).get(int(t), Counter())
                bubctr = _per_task_update_batches.get(json_key, {}).get(int(t), Counter())
                slots_dict = {}
                all_slots = set(tctr.keys()) | set(uctr.keys())
                for s in sorted(all_slots):
                    s_int = int(s)
                    slots_dict[f"value_slot_{s_int}"] = {
                        "total_accesses": int(tctr.get(s_int, 0)),
                        "batch_accesses": int(bctr.get(s_int, 0)),
                        "total_updates": int(uctr.get(s_int, 0)),
                        "batch_updates": int(bubctr.get(s_int, 0)),
                    }
                if slots_dict:
                    # nest by task for clarity (module -> task -> slots)
                    payload["per_module"][json_key] = {f"task_{int(t)}": slots_dict}
            if payload["per_module"]:
                with open(out_dir / f"memory_usage_task_{int(t)}.json", "w") as f:
                    json.dump(payload, f)
    except Exception:
        # Don't crash if logging fails
        pass


# Online IDF accumulators shared across tasks (per module)
_online_idf_df_by_module: dict[str, torch.Tensor] = {}
_online_idf_total_batches: dict[str, int] = {}


def _accumulate_online_idf_stats_batch(unwrapped_policy: PreTrainedPolicy):
    """
    Update per-module DF counts from the current batch.

    This only updates document-frequency statistics; IDF tensors used for TF-IDF
    masking are recomputed separately (e.g. at task boundaries).
    """
    for _, mem, _, json_key in _iter_memory_modules(unwrapped_policy):
        idx = getattr(mem, "last_indices", None)
        if idx is None:
            continue
        try:
            num_slots = mem.values.shape[0] if hasattr(mem, "values") else getattr(mem, "size", None)
        except Exception:
            num_slots = getattr(mem, "size", None)
        if num_slots is None:
            continue
        num_slots = int(num_slots)
        if json_key not in _online_idf_df_by_module:
            _online_idf_df_by_module[json_key] = torch.zeros(num_slots, dtype=torch.float32)
            _online_idf_total_batches[json_key] = 0

        df_vec = _online_idf_df_by_module[json_key]
        if df_vec.numel() != num_slots:
            df_vec = torch.zeros(num_slots, dtype=torch.float32)
            _online_idf_df_by_module[json_key] = df_vec

        idx_flat = idx.reshape(-1).to(torch.long).detach().cpu()
        if idx_flat.numel() == 0:
            continue
        counts = torch.bincount(idx_flat, minlength=num_slots)
        used = counts > 0
        if used.any():
            df_vec[used] += 1.0
        _online_idf_total_batches[json_key] = _online_idf_total_batches.get(json_key, 0) + 1


def _init_online_idf_stats(unwrapped_policy: PreTrainedPolicy, idf_by_module: dict[str, torch.Tensor]):
    """
    Initialize per-module DF and IDF vectors for online IDF computation.

    DF counts and total batches are kept across tasks; IDF tensors are stored in idf_by_module
    and updated in-place by _update_online_idf_stats.
    """
    for _, mem, _, json_key in _iter_memory_modules(unwrapped_policy):
        try:
            num_slots = mem.values.shape[0] if hasattr(mem, "values") else getattr(mem, "size", None)
        except Exception:
            num_slots = getattr(mem, "size", None)
        if num_slots is None:
            continue
        num_slots = int(num_slots)
        if json_key not in _online_idf_df_by_module:
            _online_idf_df_by_module[json_key] = torch.zeros(num_slots, dtype=torch.float32)
            _online_idf_total_batches[json_key] = 0
        # Start with uniform IDF = 1.0 until enough usage has been observed.
        if json_key not in idf_by_module or idf_by_module[json_key].numel() != num_slots:
            idf_by_module[json_key] = torch.ones(num_slots, dtype=torch.float32)


def _update_online_idf_stats(unwrapped_policy: PreTrainedPolicy, idf_by_module: dict[str, torch.Tensor], idf_exponent: float = 1.0):
    """
    Recompute IDF vectors for each memory module from accumulated DF stats.

    Uses per-batch slot presence (batch-level DF) accumulated across all calls to
    `_accumulate_online_idf_stats_batch` and recomputes IDF as:
        idf_i = log((B + 1) / (DF_i + 1)) ^ idf_exponent
    where B is the number of batches seen so far for the module.
    idf_exponent > 1 increases exploration by penalizing frequent slots more.
    """
    for _, mem, _, json_key in _iter_memory_modules(unwrapped_policy):
        try:
            num_slots = mem.values.shape[0] if hasattr(mem, "values") else getattr(mem, "size", None)
        except Exception:
            num_slots = getattr(mem, "size", None)
        if num_slots is None:
            continue
        num_slots = int(num_slots)
        df_vec = _online_idf_df_by_module.get(json_key)
        total_batches = _online_idf_total_batches.get(json_key, 0)
        if df_vec is None or df_vec.numel() != num_slots or total_batches <= 0:
            # No usage stats yet for this module; keep existing IDF if present,
            # otherwise fall back to uniform IDF.
            if json_key not in idf_by_module or idf_by_module[json_key].numel() != num_slots:
                idf_by_module[json_key] = torch.ones(num_slots, dtype=torch.float32)
            continue

        B = float(total_batches)
        # IDF = log((B + 1) / (DF + 1)) ^ idf_exponent
        idf = torch.log((torch.tensor(B + 1.0) / (df_vec + 1.0)))
        if idf_exponent != 1.0:
            idf = idf ** idf_exponent
        idf_by_module[json_key] = idf


def _load_idf_from_usage_json(stats_path: Path, unwrapped_policy: PreTrainedPolicy, idf_exponent: float = 1.0):
    """
    Build per-module IDF vectors from a memory_usage.json file produced during pretraining.
    Returns dict: json_key -> torch.FloatTensor[idf_per_slot] on CPU.
    If a module is missing in the JSON, it is omitted (callers should fallback to uniform IDF).
    idf_exponent > 1 increases exploration by penalizing frequent slots more.
    """
    idf_by_module: dict[str, torch.Tensor] = {}
    try:
        with open(stats_path, "r") as f:
            data = json.load(f)
        per_module = data.get("per_module", {})
    except Exception:
        return idf_by_module

    # Determine which modules we actually have, to avoid building unnecessary tensors
    present = {json_key: (mem.values.shape[0] if hasattr(mem, "values") else mem.size) for _, mem, _, json_key in _iter_memory_modules(unwrapped_policy)}

    for json_key, num_slots in present.items():
        module_dict = per_module.get(json_key)
        if not isinstance(module_dict, dict):
            continue
        # Build DF vector and infer |B| as max(batch_accesses)
        df = torch.zeros(num_slots, dtype=torch.float32)
        max_batches = 0.0
        for slot_idx in range(num_slots):
            slot_key = f"value_slot_{slot_idx}"
            slot_info = module_dict.get(slot_key)
            if isinstance(slot_info, dict):
                bacc = int(slot_info.get("batch_accesses", 0))
                df[slot_idx] = float(bacc)
                if bacc > max_batches:
                    max_batches = float(bacc)
        # Guard against degenerate |B|
        if max_batches <= 0:
            continue
        # IDF = log((|B| + 1)/(DF + 1)) ^ idf_exponent
        idf = torch.log((torch.tensor(max_batches + 1.0) / (df + 1.0)))
        if idf_exponent != 1.0:
            idf = idf ** idf_exponent
        idf_by_module[json_key] = idf
    return idf_by_module


def _validate_idf_stats(unwrapped_policy: PreTrainedPolicy, idf_by_module: dict[str, torch.Tensor]):
    """
    Validate that IDF stats exist and cover all present memory modules (expert + VLM) with correct sizes.
    Raises a ValueError if validation fails.
    """
    if idf_by_module is None or len(idf_by_module) == 0:
        raise ValueError("TF-IDF is enabled but no IDF statistics were loaded.")

    present = {}
    for _, mem, _, json_key in _iter_memory_modules(unwrapped_policy):
        num_slots = mem.values.shape[0] if hasattr(mem, "values") else getattr(mem, "size", None)
        if num_slots is None:
            raise ValueError(f"Cannot determine number of slots for memory module: {json_key}")
        present[json_key] = int(num_slots)

    missing = [k for k in present.keys() if k not in idf_by_module]
    if missing:
        raise ValueError(f"Missing IDF statistics for modules: {missing}")

    mismatched = [k for k, n in present.items() if idf_by_module[k].numel() != n]
    if mismatched:
        raise ValueError(
            f"IDF size mismatch for modules: {[(k, idf_by_module[k].numel(), present[k]) for k in mismatched]}"
        )


def _compute_tfidf_top_indices_for_batch(
    unwrapped_policy: PreTrainedPolicy,
    idf_by_module: dict[str, torch.Tensor] | None,
    top_t: int,
    tf_only: bool,
) -> dict[torch.nn.Parameter, torch.Tensor]:
    """
    For each memory module, compute TF (or TF-IDF) over slots accessed in the current batch and
    return a dict mapping parameters -> 1D LongTensor of allowed row indices (top-t per module).
    The mask is always defined over value-slot indices; when memory keys are trainable, their
    gradients are masked using the corresponding key rows implied by these selected slots.
    If tf_only is True, IDF is taken as 1 for all slots and no IDF stats are required.
    """
    allowed_by_param: dict[torch.nn.Parameter, torch.Tensor] = {}
    for _, mem_module, values_param, json_key in _iter_memory_modules(unwrapped_policy):
        # last_indices exists only when mem.log_usage == True
        if not hasattr(mem_module, "last_indices") or mem_module.last_indices is None:
            # Fallback: allow all accessed slots if we can reconstruct from usage_counts delta; otherwise skip
            continue
        try:
            idx = mem_module.last_indices  # (B, heads, knn) on device
            idx_flat = idx.reshape(-1).to(torch.long)
            num_slots = mem_module.size
            # c(i): per-batch counts (TF numerator)
            counts = torch.bincount(idx_flat, minlength=num_slots).to(torch.float32)
            total_count = counts.sum()
            if total_count <= 0:
                continue
            tf = counts / total_count
            if tf_only:
                idf = torch.ones(num_slots, dtype=torch.float32, device=tf.device)
            else:
                idf = (idf_by_module or {}).get(json_key)
                if idf is None:
                    raise RuntimeError(f"Missing IDF statistics for module: {json_key}")
                if idf.numel() != num_slots:
                    raise RuntimeError(f"IDF size mismatch for module {json_key}: got {idf.numel()}, expected {num_slots}")
                idf = idf.to(device=tf.device, dtype=torch.float32)
            tfidf = tf * idf
            # Consider only slots with c(i) > 0
            used_mask = counts > 0
            if used_mask.any():
                tfidf_used = tfidf[used_mask]
                used_indices = used_mask.nonzero(as_tuple=False).view(-1)
                k = int(min(top_t, tfidf_used.numel()))
                if k <= 0:
                    continue
                vals, top_pos = torch.topk(tfidf_used, k=k, largest=True, sorted=False)
                top_indices = used_indices[top_pos]
                try:
                    # Record which slots are eligible for updates this step
                    mem_module.last_update_indices = top_indices.detach().cpu()
                except Exception:
                    pass
                # Always mask values by the selected slot indices
                allowed_by_param[values_param] = top_indices.detach()

                # If keys are trainable, mask their gradients to rows corresponding to the
                # selected value slots. Each value slot index encodes a pair of sub-keys
                # (i1, i2) for each head: i1 = slot // n_keys, i2 = slot % n_keys.
                keys_param = getattr(mem_module, "keys", None)
                if keys_param is not None and keys_param.requires_grad:
                    try:
                        n_keys = int(mem_module.n_keys)
                        # idx: (B, heads, knn)
                        idx = mem_module.last_indices
                        selected_mask = torch.zeros(num_slots, dtype=torch.bool, device=idx.device)
                        selected_mask[top_indices.to(device=idx.device)] = True
                        # Mask over (B, heads, knn) positions whose slots are selected
                        selected_per_bhk = selected_mask[idx]
                        if selected_per_bhk.any():
                            B, H, K = idx.shape
                            key_rows_list: list[torch.Tensor] = []
                            for h in range(H):
                                mh = selected_per_bhk[:, h, :]
                                if not mh.any():
                                    continue
                                s_h = idx[:, h, :][mh]
                                if s_h.numel() == 0:
                                    continue
                                s_h = torch.unique(s_h)
                                i1 = torch.div(s_h, n_keys, rounding_mode="floor")
                                i2 = s_h % n_keys
                                base = h * 2 * n_keys
                                key1 = base + i1
                                key2 = base + n_keys + i2
                                key_rows_list.append(key1)
                                key_rows_list.append(key2)
                            if key_rows_list:
                                key_rows = torch.unique(torch.cat(key_rows_list))
                                allowed_by_param[keys_param] = key_rows.detach()
                    except Exception:
                        # If key masking fails for any reason, fall back to unmasked keys.
                        pass
        except Exception:
            # Be robust: skip module on any failure
            raise
    return allowed_by_param


def _apply_gradient_mask_to_memory_values(allowed_by_param: dict[torch.nn.Parameter, torch.Tensor]):
    """
    Zero out gradients for all rows not in the allowed index set for each masked parameter.
    Should be called after backward, before gradient clipping and optimizer.step().
    """
    for p, allowed_rows in allowed_by_param.items():
        if p.grad is None:
            continue
        try:
            # p.grad: (num_slots, v_dim)
            num_slots = p.shape[0]
            device = p.grad.device
            mask = torch.zeros(num_slots, dtype=torch.bool, device=device)
            mask[allowed_rows.to(device=device)] = True
            # Zero out all rows not allowed
            p.grad[~mask] = 0
        except Exception:
            # If anything goes wrong, don't crash training
            continue


def _update_policy_with_tfidf(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    idf_by_module: dict[str, torch.Tensor] | None,
    top_t: int,
    tf_only: bool,
    lr_scheduler=None,
    lock=None,
):
    """
    Variant of update_policy that masks gradients for memory value tables to only top-t TF-IDF slots.
    """
    use_cuda_events = torch.cuda.is_available()
    if use_cuda_events:
        ev0 = torch.cuda.Event(enable_timing=True)
        ev_fwd = torch.cuda.Event(enable_timing=True)
        ev_bwd = torch.cuda.Event(enable_timing=True)
        ev_mask = torch.cuda.Event(enable_timing=True)
        ev_apply = torch.cuda.Event(enable_timing=True)
        ev_clip = torch.cuda.Event(enable_timing=True)
        ev_opt = torch.cuda.Event(enable_timing=True)
        ev_sched = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev0.record()
    wall0 = time.perf_counter()

    policy.train()
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
    if use_cuda_events:
        ev_fwd.record()

    accelerator.backward(loss)
    if use_cuda_events:
        ev_bwd.record()

    # Compute and apply TF-IDF gradient masks before clipping and step
    mask_build_s = 0.0
    mask_apply_s = 0.0
    try:
        unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
        if (idf_by_module is not None or tf_only) and top_t > 0:
            t0 = time.perf_counter()
            allowed = _compute_tfidf_top_indices_for_batch(unwrapped, idf_by_module, top_t, tf_only=tf_only)
            mask_build_s = time.perf_counter() - t0
            if use_cuda_events:
                ev_mask.record()
            if allowed:
                t1 = time.perf_counter()
                _apply_gradient_mask_to_memory_values(allowed)
                mask_apply_s = time.perf_counter() - t1
                if use_cuda_events:
                    ev_apply.record()
    except Exception:
        # Be conservative: if masking fails, continue without masking
        pass

    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"), error_if_nonfinite=False)
    if use_cuda_events:
        ev_clip.record()

    from contextlib import nullcontext
    with (lock if lock is not None else nullcontext()):
        optimizer.step()
    optimizer.zero_grad()
    if use_cuda_events:
        ev_opt.record()

    if lr_scheduler is not None:
        lr_scheduler.step()
    if use_cuda_events:
        ev_sched.record()

    # No special update hook beyond policy.update
    unwrapped_for_update = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    if hasattr(unwrapped_for_update, "update") and callable(getattr(unwrapped_for_update, "update")):
        unwrapped_for_update.update()

    # Timing aggregation
    step_wall_s = time.perf_counter() - wall0
    if use_cuda_events:
        ev_end.record()
        torch.cuda.synchronize()
        fwd_s = ev0.elapsed_time(ev_fwd) / 1000.0
        bwd_s = ev_fwd.elapsed_time(ev_bwd) / 1000.0
        # mask_build_s and mask_apply_s measured by wall clock (may include host ops)
        clip_s = ev_bwd.elapsed_time(ev_clip) / 1000.0
        opt_s = ev_clip.elapsed_time(ev_opt) / 1000.0
        sched_s = ev_opt.elapsed_time(ev_sched) / 1000.0 if lr_scheduler is not None else 0.0
        update_s = ev0.elapsed_time(ev_end) / 1000.0
    else:
        # Fallback to wall time breakdown (coarser but informative)
        fwd_s = float("nan")
        bwd_s = float("nan")
        clip_s = float("nan")
        opt_s = float("nan")
        sched_s = 0.0 if lr_scheduler is not None else 0.0
        update_s = step_wall_s

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = update_s
    train_metrics.fwd_s = fwd_s
    train_metrics.bwd_s = bwd_s
    train_metrics.mask_s = mask_build_s
    train_metrics.apply_mask_s = mask_apply_s
    train_metrics.clip_s = clip_s
    train_metrics.opt_s = opt_s
    train_metrics.sched_s = sched_s
    train_metrics.step_wall_s = step_wall_s
    return train_metrics, output_dict


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

    # Freeze everything except selected memory components
    num_trainable = _freeze_to_selected_memory_params(
        policy,
        train_memory_value=cfg.train_memory_value,
        train_memory_keys=cfg.train_memory_keys,
        train_query_proj=cfg.train_query_proj,
    )
    num_total = sum(p.numel() for p in policy.parameters())
    if is_main:
        logging.info(f"Trainable params (memory values only) = {num_trainable} / {num_total}")

    # Enable per-batch memory usage logging to allow TF computation
    try:
        _enable_memory_batch_logging(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), enable=True)
    except Exception:
        pass

    # Load IDF stats for TF-IDF gating (optional). Skipped when tf_only is True.
    idf_by_module = None
    if cfg.tfidf_enable and not cfg.tf_only:
        if cfg.use_online_idf_stats:
            # Online mode: initialize DF/IDF structures; no pretraining stats are used.
            idf_by_module = {}
            try:
                _init_online_idf_stats(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), idf_by_module)
                if is_main:
                    logging.info("Using online IDF statistics accumulated during sequential training.")
            except Exception:
                pass
        else:
            # Offline mode: load pretraining IDF stats from memory_usage.json.
            candidate_paths: list[Path] = []
            if cfg.idf_stats_path:
                candidate_paths.append(Path(cfg.idf_stats_path))
            # Try deriving from pretrained_path
            try:
                if cfg.policy.pretrained_path:
                    pp = Path(cfg.policy.pretrained_path)
                    candidate_paths.append(pp / "memory_usage.json")
                    candidate_paths.append(pp / "pretrained_model" / "memory_usage.json")
            except Exception:
                pass
            chosen = None
            for pth in candidate_paths:
                if pth is not None and pth.exists():
                    chosen = pth
                    break
            if chosen is None:
                raise FileNotFoundError("TF-IDF is enabled but no memory_usage.json path was found.")
            try:
                idf_by_module = _load_idf_from_usage_json(
                    chosen, accelerator.unwrap_model(policy, keep_fp32_wrapper=True), idf_exponent=cfg.idf_exponent
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load IDF stats from {chosen}: {e}")
            # Validate full coverage and shapes
            _validate_idf_stats(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), idf_by_module)
            if is_main:
                logging.info(f"Loaded IDF stats from: {chosen}")

    # Build optimizer/scheduler once, optionally reinit per task
    # Make the scheduler horizon equal to total steps across all tasks if we don't reinit per task.
    total_steps = cfg.online_steps_per_task * len(cfg.online_task_ids)
    sched_steps = cfg.online_steps_per_task if cfg.reinit_optimizer_each_task else total_steps
    cfg.steps = max(1, sched_steps)

    if is_main:
        logging.info("Creating optimizer (selected memory params) with custom LRs")

    # Build optimizer with distinct param groups for values/keys/query_proj
    def _build_memory_optimizer(model: PreTrainedPolicy, cfg_local: SequentialOnlineConfig) -> Optimizer:
        vals: list[torch.nn.Parameter] = []
        keys: list[torch.nn.Parameter] = []
        qproj: list[torch.nn.Parameter] = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if getattr(p, "pk_value_param", False):
                vals.append(p)
            elif getattr(p, "pk_keys_param", False):
                keys.append(p)
            elif getattr(p, "pk_query_proj_param", False):
                qproj.append(p)
        param_groups: list[dict] = []
        if cfg_local.train_memory_value and len(vals) > 0:
            param_groups.append({"params": vals, "lr": cfg_local.memory_value_lr, "weight_decay": 0.0})
        if cfg_local.train_memory_keys and len(keys) > 0:
            param_groups.append({"params": keys, "lr": cfg_local.memory_keys_lr, "weight_decay": 0.0})
        if cfg_local.train_query_proj and len(qproj) > 0:
            param_groups.append({"params": qproj, "lr": cfg_local.query_proj_lr, "weight_decay": 0.0})
        if len(param_groups) == 0:
            raise ValueError(
                "No trainable parameters selected. Enable at least one of --train_memory_value, --train_memory_keys, or --train_query_proj."
            )
        return optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    optimizer = _build_memory_optimizer(policy, cfg)
    lr_scheduler = _build_memory_scheduler(optimizer, cfg, cfg.online_steps_per_task)
    if is_main:
        if lr_scheduler is not None:
            logging.info("Linear LR schedule enabled (per-task reset)")
        else:
            logging.info("Using static LR (no schedule)")

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
        "preproc_s": AverageMeter("pre_s", ":.3f"),
        "fwd_s": AverageMeter("fwd_s", ":.3f"),
        "bwd_s": AverageMeter("bwd_s", ":.3f"),
        "mask_s": AverageMeter("mask_s", ":.3f"),
        "apply_mask_s": AverageMeter("apmsk_s", ":.3f"),
        "clip_s": AverageMeter("clip_s", ":.3f"),
        "opt_s": AverageMeter("opt_s", ":.3f"),
        "sched_s": AverageMeter("schd_s", ":.3f"),
        "step_wall_s": AverageMeter("step_s", ":.3f"),
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

        # Optionally rebuild optimizer state per task; always reset scheduler for fresh LR decay
        if cfg.reinit_optimizer_each_task:
            # Re-freeze to be safe in case something toggled
            _freeze_to_selected_memory_params(
                policy,
                train_memory_value=cfg.train_memory_value,
                train_memory_keys=cfg.train_memory_keys,
                train_query_proj=cfg.train_query_proj,
            )
            # Reset optimizer state in-place to avoid large CUDA re-allocations/fragmentation.
            try:
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    for p in group.get("params", []):
                        if p is None:
                            continue
                        if p.grad is not None:
                            p.grad = None
                        state = optimizer.state.get(p, None)
                        if state:
                            # Delete any tensor state (e.g., exp_avg, exp_avg_sq). They will be recreated lazily.
                            for k in list(state.keys()):
                                v = state[k]
                                if isinstance(v, torch.Tensor):
                                    del state[k]
                            optimizer.state[p] = {}
            except Exception:
                pass
            # Encourage allocator to release freed blocks and reduce fragmentation.
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Reset scheduler per task so each task gets the full LR decay from start_lr to end_lr
        lr_scheduler = _reset_scheduler_for_task(optimizer, cfg, cfg.online_steps_per_task)

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
            # Dataloading timing (wall clock)
            t0 = time.perf_counter()
            batch = next(dl_iter)
            train_tracker.dataloading_s = time.perf_counter() - t0

            # Preprocessing timing (wall clock)
            t1 = time.perf_counter()
            batch = preprocessor(batch)
            train_tracker.preproc_s = time.perf_counter() - t1
            if cfg.tfidf_enable or cfg.tf_only:
                train_tracker, output_dict = _update_policy_with_tfidf(
                    train_metrics=train_tracker,
                    policy=policy,
                    batch=batch,
                    optimizer=optimizer,
                    grad_clip_norm=cfg.optimizer.grad_clip_norm,
                    accelerator=accelerator,
                    idf_by_module=idf_by_module,
                    top_t=cfg.tfidf_top_t,
                    tf_only=cfg.tf_only,
                    lr_scheduler=lr_scheduler,
                )
            else:
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

            # Accumulate per-task memory usage for this batch (main process only)
            if is_main:
                try:
                    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
                    _accumulate_task_usage_for_batch(
                        unwrapped,
                        task_id=dataset_task_id,
                    )
                    _accumulate_task_updates_for_batch(
                        unwrapped,
                        task_id=dataset_task_id,
                    )
                except Exception:
                    pass

            # Update online IDF statistics (all ranks) for future TF-IDF masking steps.
            if cfg.tfidf_enable and not cfg.tf_only and cfg.use_online_idf_stats:
                try:
                    _accumulate_online_idf_stats_batch(
                        accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
                    )
                except Exception:
                    pass

        # After finishing this task, update IDF vectors from accumulated DF stats
        if cfg.tfidf_enable and not cfg.tf_only and cfg.use_online_idf_stats:
            try:
                _update_online_idf_stats(
                    accelerator.unwrap_model(policy, keep_fp32_wrapper=True),
                    idf_by_module=idf_by_module if idf_by_module is not None else {},
                    idf_exponent=cfg.idf_exponent,
                )
            except Exception:
                pass

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

        # Flush per-task memory usage for this task and clear accumulators (main only)
        if is_main:
            try:
                _flush_per_task_usage(cfg.output_dir, task_id=dataset_task_id)
            except Exception:
                pass
            _per_task_totals.clear()
            _per_task_batches.clear()
            _per_task_update_totals.clear()
            _per_task_update_batches.clear()

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
                step_id = get_step_identifier(global_step, cfg.steps)
                videos_dir = (cfg.output_dir / "eval" / f"videos_step_{step_id}") if is_main else None
                max_episodes_rendered = 4 if is_main else 0
                eval_info = eval_policy_all(
                    envs=env_subset,
                    policy=accelerator.unwrap_model(policy),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=cfg.eval.n_episodes,
                    videos_dir=videos_dir,
                    max_episodes_rendered=max_episodes_rendered,
                    start_seed=cfg.seed,
                    max_parallel_tasks=cfg.env.max_parallel_tasks,
                )

            if is_main:
                overall = eval_info.get("overall", {}) or {}
                avg_sum = overall.get("avg_sum_reward", float("nan"))
                avg_max = overall.get("avg_max_reward", float("nan"))
                pc_succ = overall.get("pc_success", float("nan"))
                logging.info(
                    f"Eval overall | tasks_seen={len(seen_env_task_ids)} "
                    f"avg_sum_reward={avg_sum:.3f} avg_max_reward={avg_max:.3f} pc_success={pc_succ:.2f}"
                )
                if wandb_logger:
                    log_dict = {
                        "num_tasks_seen": len(seen_env_task_ids),
                        "avg_sum_reward_seen": float(avg_sum),
                        "avg_max_reward_seen": float(avg_max),
                        "avg_pc_success_seen": float(pc_succ),
                    }
                    per_group = eval_info.get("per_group", {}) or {}
                    env_task_key = str(cfg.env.task) if cfg.env and cfg.env.task is not None else None
                    if env_task_key and env_task_key in per_group:
                        ginfo = per_group[env_task_key]
                        if isinstance(ginfo, dict) and "pc_success" in ginfo:
                            log_dict[f"success/{env_task_key}_overall"] = (
                                float(ginfo["pc_success"]) if ginfo["pc_success"] is not None else float("nan")
                            )
                    wandb_logger.log_dict(log_dict, global_step, mode="eval")
                    vpaths = overall.get("video_paths") if isinstance(overall, dict) else None
                    if vpaths:
                        wandb_logger.log_video(vpaths[0], global_step, mode="eval")

                _append_eval_results_jsonl(cfg.output_dir, global_step, eval_info)

    # Cleanup
    if eval_envs_all:
        close_envs(eval_envs_all)

    if is_main:
        logging.info("End of sequential online training")


def main():
    sequential_train()


if __name__ == "__main__":
    main()


