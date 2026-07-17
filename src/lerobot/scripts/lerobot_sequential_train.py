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

import numpy as np
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
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.common.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.scripts.lerobot_train import _sanitize_wandb_dict, update_policy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import cycle, format_big_number, init_logging

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

    # ---- Loss-based eval (--eval.type=loss) ----
    # Number of batches per seen task to average when computing loss-based eval
    # after each sequential task. Uses --batch_size for the batch dimension.
    eval_loss_n_batches: int = 20

    # Episodes per env for the eval that runs after the FINAL task only. 0 (default)
    # = use eval.n_episodes for every eval (unchanged behavior). >0 = intermediate
    # evals keep eval.n_episodes (cheap trajectory tracking) while the final
    # cumulative eval uses this larger count — at 20 eps a cell is +/-10pp near
    # p=0.4; 50 eps brings the headline number to +/-7pp for ~2.5x the cost of one
    # eval round instead of every round.
    eval_final_episodes: int = 0

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
    # How to compute the TF term before applying IDF.
    # "raw": count every retrieved slot equally.
    # "weighted": accumulate retrieval weights per slot.
    tf_idf_weighting_method: str = "raw"
    # Optional path to pretraining memory usage stats JSON (memory_usage.json)
    idf_stats_path: str | None = None
    # When true, ignore pretraining IDF stats and build IDF online from slot usage
    # observed during sequential training (across all tasks seen so far).
    use_online_idf_stats: bool = False

    # Exponent applied to IDF scores. >1 increases exploration (penalizes frequent slots more),
    # <1 decreases exploration. Default 1.0 = standard IDF.
    idf_exponent: float = 1.0

    # Denominator applied to pretraining DF stats when seeding online IDF.
    # Only used when both --use_online_idf_stats=true and --idf_stats_path are set.
    # Divides pretrain DF counts and total_batches by this value before seeding,
    # controlling how quickly sequential training overrides the pretraining prior.
    # E.g. if pretraining ran 100K batches and each sequential task is 3K steps,
    # setting denom=33 makes the pretrain prior "worth" ~one sequential task.
    idf_stats_denom: float = 1.0

    # ---- Prior-usefulness write protection (opt-in; default OFF = legacy behavior) ----
    # When True, down-weight value-slot updates to slots that PRIOR sequential tasks relied on.
    # For each slot s, usefulness u(s) = max over prior tasks of that task's peak-normalized read
    # profile; the per-batch TF-IDF write score is multiplied by (1 - u(s)) ** protect_beta, so
    # slots important to earlier tasks are pushed out of the top-t update set. This is the
    # task-identity-aware, importance-weighted, graded analogue of IDF. Rides on the TF-IDF mask,
    # so it requires tfidf_enable=True. Default False keeps the legacy mask byte-for-byte, so
    # existing scripts reproduce exactly.
    protect_prior_slots: bool = False
    # Sharpness of the protection gate; larger => protect deeper into each prior task's read core.
    # Only consulted when protect_prior_slots=True; 0.0 disables the gate even if the flag is set.
    protect_beta: float = 4.0
    # Protection mechanism (research_log E41).
    # - "rank" (default, legacy): the (1 - u)^beta factor multiplies the TF-IDF *ranking score*,
    #   so protection acts only by pushing slots out of the top-t mask. Per slot per batch this is
    #   binary (in mask = full gradient, out = zero) and cannot attenuate high-TF survivors or the
    #   diffuse low-u tail — the measured carrier of the rollout-level bleed (E41).
    # - "grad_scale": ranking is pure TF-IDF (no protection discount); each surviving slot's
    #   parameter UPDATE is multiplied by (1 - u(s)) ** beta via a post-optimizer-step blend
    #   (theta <- theta_pre + scale * (theta_post - theta_pre)) — exact per-slot LR scaling.
    #   NB implemented as a post-step blend rather than gradient scaling because Adam's
    #   normalization is invariant to a time-constant per-row gradient scale (m-hat and
    #   sqrt(v-hat) scale together), which would make naive grad scaling a no-op.
    #   Continuous per-slot write attenuation, no hard block; beta=0 or an empty store reproduces
    #   the unprotected mask exactly. Trainable keys are still masked, never scaled.
    protect_mode: str = "rank"
    # Normalization of a task's read profile when folding it into u(s) at its task boundary.
    # - "peak" (default, legacy): counts / max(counts). Degenerate for the sharp read
    #   distributions we measure (max/p99 ~ 10-16x => u ~ 0.03 at the core-50 boundary, so the
    #   gate only ever bites the top ~1% of slots; E39).
    # - "corefrac": counts / count_at_core50_boundary, clipped to 1. The boundary slot of the
    #   task's core-50 set (smallest slot set carrying half its read mass) gets u = 1; below it u
    #   decays proportionally with read density. Density-proportional, so (1 - u)^beta tracks the
    #   mass-weighted damage integral the protection is meant to suppress.
    protect_u_norm: str = "peak"
    # Hard-veto threshold on u (E42): slots with u >= protect_hard_u have their TF-IDF score
    # zeroed before top-t selection, in BOTH protect modes. 0 disables (default, legacy).
    # A vetoed slot never enters the mask, so it never receives gradient and never builds
    # optimizer momentum — airtight where the grad_scale blend can only attenuate. E42 measured
    # that at corefrac normalization u >= ~0.9 marks the prior tasks' true read cores (a few
    # hundred mask slots per layer per writer), carrying ~19-34% of the victim-perceived bleed.
    protect_hard_u: float = 0.0
    # Path to a JSON of {module_json_key: [slot_index, ...]} used to SEED the prior-usefulness
    # store with u = 1.0 at the listed slots before any sequential task trains (E42 addendum:
    # "freeze the generalist slots" — e.g. the top-K A-phase read-mass slots per layer). Seeded
    # slots behave exactly like a maximally-useful prior task from step 0: in "rank" mode their
    # score gets (1-1)^beta = 0, and with protect_hard_u > 0 they are removed from top-t
    # candidacy entirely (never in mask => no gradient, no momentum => frozen for the whole
    # sequential run). Later tasks' own profiles max-fold into the store as usual, so seeds are
    # never diluted. Empty string (default) = legacy behavior, no seeding.
    protect_seed_path: str = ""

    # ---- Optional visualization logging (WandB) ----
    # When enabled, build an interactive Plotly HTML visualization of:
    # - global memory usage (from pretraining memory_usage.json, if available)
    # - per-task memory usage (from <output_dir>/memory_by_task/*.json generated during sequential training)
    # and upload it to WandB at the end of the sequential run.
    log_full_memory_usage_viz: bool = True
    # Optional override for the heatmap grid side length. If None, inferred as ceil(sqrt(max_slots)).
    full_memory_usage_viz_grid_side: int | None = None
    # How to include Plotly JS in the HTML. "cdn" keeps the file small.
    full_memory_usage_viz_include_plotlyjs: str = "cdn"

    def validate(self):
        super().validate()
        if self.tf_idf_weighting_method not in {"raw", "weighted"}:
            raise ValueError(
                "tf_idf_weighting_method must be one of {'raw', 'weighted'}, "
                f"got {self.tf_idf_weighting_method!r}"
            )
        if self.protect_beta < 0:
            raise ValueError(f"protect_beta must be >= 0, got {self.protect_beta}")
        if self.protect_mode not in {"rank", "grad_scale"}:
            raise ValueError(
                f"protect_mode must be one of {{'rank', 'grad_scale'}}, got {self.protect_mode!r}"
            )
        if self.protect_u_norm not in {"peak", "corefrac"}:
            raise ValueError(
                f"protect_u_norm must be one of {{'peak', 'corefrac'}}, got {self.protect_u_norm!r}"
            )
        if not (0.0 <= self.protect_hard_u <= 1.0):
            raise ValueError(f"protect_hard_u must be in [0, 1], got {self.protect_hard_u}")
        if self.protect_seed_path:
            if not self.protect_prior_slots:
                raise ValueError(
                    "protect_seed_path requires protect_prior_slots=True (the seed lives in the "
                    "prior-usefulness store, which is only consulted when protection is enabled)."
                )
            if not os.path.isfile(self.protect_seed_path):
                raise ValueError(f"protect_seed_path does not exist: {self.protect_seed_path}")


def _render_cumulative_eval_bar_chart(
    eval_history: list[dict],
) -> "matplotlib.figure.Figure":
    """
    Render a grouped bar chart of per-task success rates across eval loops.

    eval_history: list of dicts, one per eval loop, each with:
        {"trained_task_idx": int, "per_task": {task_id: success_pct, ...}}

    Returns a matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_groups = len(eval_history)
    all_task_ids = sorted({tid for entry in eval_history for tid in entry["per_task"]})
    n_tasks = len(all_task_ids)
    task_id_to_pos = {tid: i for i, tid in enumerate(all_task_ids)}

    bar_width = 0.8 / max(n_tasks, 1)
    cmap = plt.get_cmap("tab10")
    colors = {tid: cmap(i % 10) for i, tid in enumerate(all_task_ids)}

    fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.5), 5))

    for group_idx, entry in enumerate(eval_history):
        for tid, success_pct in entry["per_task"].items():
            pos = task_id_to_pos[tid]
            x = group_idx + (pos - (n_tasks - 1) / 2) * bar_width
            ax.bar(x, success_pct, width=bar_width * 0.9, color=colors[tid],
                   label=f"Task {tid}" if group_idx == 0 or tid not in {
                       t for e in eval_history[:group_idx] for t in e["per_task"]
                   } else "")

    # Build x-tick labels
    x_labels = [f"After Task {entry['trained_task_idx']}" for entry in eval_history]
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Success %")
    ax.set_ylim(0, 105)
    ax.set_title("Cumulative Eval: Per-Task Success After Each Training Stage")

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    if unique_labels:
        ax.legend(unique_handles, unique_labels, loc="upper left",
                  fontsize=8, ncol=max(1, len(unique_labels) // 5 + 1))

    fig.tight_layout()
    return fig


def _render_loss_eval_chart(loss_history: list[dict]) -> "matplotlib.figure.Figure":
    """Line chart of per-task eval MSE (lower=better) after each training stage.

    loss_history: list of dicts, one per eval loop, each with:
        {"trained_task_idx": int, "per_task": {task_id: mse, ...}}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_groups = len(loss_history)
    all_task_ids = sorted({tid for e in loss_history for tid in e["per_task"]})
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.5), 5))
    x = list(range(n_groups))
    for i, tid in enumerate(all_task_ids):
        ys = [e["per_task"].get(tid, float("nan")) for e in loss_history]
        ax.plot(x, ys, marker="o", color=cmap(i % 10), label=f"Task {tid}")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"After Task {e['trained_task_idx']}" for e in loss_history],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_ylabel("MSE loss (lower = better)")
    ax.set_title("Cumulative Loss Eval: Per-Task MSE After Each Training Stage")
    ax.legend(loc="upper left", fontsize=8, ncol=max(1, len(all_task_ids) // 5 + 1))
    fig.tight_layout()
    return fig


def _append_loss_results_jsonl(
    output_dir: Path,
    step: int,
    per_task_loss: dict[int, float],
    baseline: dict[int, float],
):
    """Append a JSONL line with per-task eval MSE (and forgetting vs the just-trained
    baseline) to {output_dir}/eval/loss_results.jsonl.

    Format: {"step": step, "task_0": mse, "forget_0": mse-baseline, ...}
    """
    record: dict[str, float | int] = {"step": int(step)}
    for tid, v in per_task_loss.items():
        record[f"task_{tid}"] = float(v)
        base = baseline.get(int(tid))
        if base is not None:
            record[f"forget_{tid}"] = float(v) - float(base)

    results_path = Path(output_dir) / "eval" / "loss_results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a") as f:
        f.write(json.dumps(record) + "\n")


@torch.no_grad()
def _eval_loss_on_seen_tasks(
    policy: PreTrainedPolicy,
    accelerator: Accelerator,
    dataset,
    task_index_to_name: dict[int, str],
    seen_task_ids: list[int],
    *,
    batch_size: int,
    num_workers: int,
    device,
    n_batches: int,
    preprocessor,
    seed: int = 0,
) -> dict[int, float]:
    """Loss-based eval: mean flow-matching MSE for each seen task under the *current* policy.

    Mirrors the training forward (same language task-embedding conditioning) but runs in
    eval mode with no gradient/optimizer/TF-IDF updates. The per-(task, batch) RNG is
    seeded deterministically so the sampled flow-matching noise/time is identical across
    eval rounds, making the across-round deltas (i.e. forgetting) low-variance.

    Returns {dataset_task_id: mean_mse}. Runs identically on every rank; the caller logs
    on the main process only.
    """
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    was_training = unwrapped.training
    unwrapped.eval()
    cam_keys = list(dataset.meta.camera_keys)
    results: dict[int, float] = {}
    try:
        for t in seen_task_ids:
            t = int(t)
            name = task_index_to_name.get(t, "")
            try:
                dl = _build_dataloader_for_task(
                    dataset,
                    task_index_to_name,
                    t,
                    batch_size=batch_size,
                    num_workers=min(num_workers, 4),
                    device_type=device.type,
                    drop_n_last_frames=0,
                )
            except Exception as e:
                logging.warning(f"[loss-eval] could not build dataloader for task {t}: {e}")
                continue

            torch.manual_seed(seed + 7919 * (t + 1))  # reproducible frame sampling across rounds
            losses: list[float] = []
            it = iter(dl)
            for j in range(n_batches):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                for ck in cam_keys:
                    if ck in batch and batch[ck].dtype == torch.uint8:
                        batch[ck] = batch[ck].to(dtype=torch.float32) / 255.0
                batch = preprocessor(batch)
                B = batch[next(iter(batch))].shape[0]
                task_emb = None
                if hasattr(unwrapped, "get_task_embeddings"):
                    try:
                        task_emb = unwrapped.get_task_embeddings([name] * B)
                        if task_emb is not None:
                            task_emb = task_emb.to(device=device)
                    except Exception:
                        task_emb = None
                torch.manual_seed(10_000 * (t + 1) + j)  # paired noise/time across rounds
                with accelerator.autocast():
                    _, out = unwrapped.forward(batch, task_emb=task_emb)
                mse = out.get("mse_loss", out.get("loss"))
                if mse is not None:
                    losses.append(float(mse))
            del it, dl
            if losses:
                results[t] = sum(losses) / len(losses)
    finally:
        if was_training:
            unwrapped.train()
    return results


def _default_libero10_map() -> dict[int, int]:
    return {0: 4, 1: 6, 2: 9, 3: 2, 4: 7, 5: 0, 6: 8, 7: 1, 8: 3, 9: 5}


def _eval_n_episodes_for_task(cfg, task_pos: int) -> int:
    """Episode count for the rollout eval after the task at position `task_pos`.

    Returns cfg.eval_final_episodes for the LAST task when that override is set
    (>0); otherwise cfg.eval.n_episodes. Lets a run track the trajectory cheaply
    (e.g. 20 eps) while de-noising the headline final number (e.g. 50 eps).
    """
    if getattr(cfg, "eval_final_episodes", 0) > 0 and task_pos == len(cfg.online_task_ids) - 1:
        return int(cfg.eval_final_episodes)
    return int(cfg.eval.n_episodes)


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

    if "tasks" in dataset.meta.episodes.column_names:
        all_episode_tasks = dataset.meta.episodes["tasks"]
        allowed_task_name = task_index_to_name[dataset_task_id]
        episode_indices = [i for i, tlist in enumerate(all_episode_tasks) if allowed_task_name in tlist]
    else:
        episode_task_ids = getattr(dataset, "_episode_task_ids_cache", None)
        if episode_task_ids is None:
            episode_task_ids = defaultdict(set)
            for ep_idx, task_idx in zip(dataset.hf_dataset["episode_index"], dataset.hf_dataset["task_index"], strict=True):
                episode_task_ids[int(ep_idx)].add(int(task_idx))
            dataset._episode_task_ids_cache = episode_task_ids
        episode_indices = [
            ep_idx for ep_idx, task_ids in episode_task_ids.items() if dataset_task_id in task_ids
        ]

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


def _get_value_params(mem_module) -> list:
    """
    Get the value parameter(s) from a memory module.

    For value_type="vector": returns [mem.values]
    For value_type="lora": returns [mem.slot_down, mem.slot_up] (+ [mem.slot_bias]
    when the affine lora_slot_bias variant is enabled — all per-slot in dim 0, so
    the TF-IDF row mask applies uniformly).
    """
    value_type = getattr(mem_module, "value_type", "vector")
    if value_type == "vector":
        return [mem_module.values] if hasattr(mem_module, "values") else []
    elif value_type == "lora":
        params = []
        if hasattr(mem_module, "slot_down"):
            params.append(mem_module.slot_down)
        if hasattr(mem_module, "slot_up"):
            params.append(mem_module.slot_up)
        if hasattr(mem_module, "slot_bias"):
            params.append(mem_module.slot_bias)
        return params
    return []


def _iter_memory_modules(unwrapped_policy: PreTrainedPolicy):
    """
    Yield tuples of (layer_index, mem_module, value_params, json_key) for all attached memory layers
    on any policy backbone (smolvla expert/VLM text, pi05 action expert, etc.).

    Discovery is policy-agnostic: walks named_modules() looking for MLPPlusMemory
    wrappers and uses the wrapper's module path (minus the trailing ".mlp") as the
    json_key. Layer index is parsed from the path's `layers.{i}` segment.

    value_params is a list of parameters:
      - For vector mode: [values]
      - For lora mode: [slot_down, slot_up]
    """
    from lerobot.policies.modules.memory_lite import MLPPlusMemory
    import re

    mems = []
    seen_keys: set[str] = set()
    for module_name, module in unwrapped_policy.named_modules():
        if not isinstance(module, MLPPlusMemory):
            continue
        mem_module = getattr(module, "mem", None)
        if mem_module is None:
            continue
        value_params = _get_value_params(mem_module)
        if not value_params:
            continue
        # `module_name` looks like "model...layers.{i}.mlp"; strip the trailing ".mlp"
        # to align with the json conventions used in memory_usage.json.
        json_key = module_name[:-4] if module_name.endswith(".mlp") else module_name
        if json_key in seen_keys:
            continue
        seen_keys.add(json_key)
        match = re.search(r"layers\.(\d+)", json_key)
        li = int(match.group(1)) if match else 0
        mems.append((li, mem_module, value_params, json_key))
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


# ---- Prior-usefulness write protection accumulators (per module) ----
# Raw read counts for the CURRENT task (reset at each task boundary) and the cumulative
# usefulness vector u(s) = max over prior tasks of each task's peak-normalized read profile.
_protect_cur_counts_by_module: dict[str, torch.Tensor] = {}
_protect_usefulness_by_module: dict[str, torch.Tensor] = {}


def _accumulate_protect_counts_batch(unwrapped_policy: PreTrainedPolicy):
    """Accumulate per-slot read counts for the current task (used to build prior usefulness).

    Mirrors `_accumulate_online_idf_stats_batch` but keeps raw counts (not binary DF) so the
    per-task read profile can be peak-normalized at the task boundary.
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
        cur = _protect_cur_counts_by_module.get(json_key)
        if cur is None or cur.numel() != num_slots:
            cur = torch.zeros(num_slots, dtype=torch.float32)
            _protect_cur_counts_by_module[json_key] = cur
        idx_flat = idx.reshape(-1).to(torch.long).detach().cpu()
        if idx_flat.numel() == 0:
            continue
        cur += torch.bincount(idx_flat, minlength=num_slots).to(torch.float32)


def _core50_boundary_count(counts: torch.Tensor) -> float:
    """Read count of the slot at the task's core-50 boundary.

    The core-50 set is the smallest set of slots (taken hottest-first) carrying >= 50% of the
    task's total read mass; the boundary count is the count of its coldest member. Used by the
    "corefrac" usefulness normalization: u = counts / boundary, clipped to 1.
    """
    total = float(counts.sum().item())
    if total <= 0:
        return 0.0
    sorted_counts, _ = torch.sort(counts, descending=True)
    cum = torch.cumsum(sorted_counts, dim=0)
    k = int(torch.searchsorted(cum, 0.5 * total).item())
    k = min(k, sorted_counts.numel() - 1)
    return float(sorted_counts[k].item())


def _finalize_protect_usefulness(unwrapped_policy: PreTrainedPolicy, u_norm: str = "peak"):
    """Fold the just-finished task's read profile into the cumulative usefulness store.

    u(s) <- max(u(s), normalize(read_count)); then reset the current-task counts.
    normalize is counts/max (u_norm="peak", legacy) or min(1, counts/core50-boundary-count)
    (u_norm="corefrac"; see SequentialOnlineConfig.protect_u_norm).
    Call once at each task boundary, AFTER the task's training loop, so that while task W trains
    the store reflects only tasks strictly before W (W never protects against itself).
    """
    for _, mem, _, json_key in _iter_memory_modules(unwrapped_policy):
        cur = _protect_cur_counts_by_module.get(json_key)
        if cur is None:
            continue
        mx = float(cur.max().item()) if cur.numel() else 0.0
        if mx > 0:
            if u_norm == "corefrac":
                ref = _core50_boundary_count(cur)
                rnorm = (cur / ref).clamp_(max=1.0) if ref > 0 else cur / mx
            else:
                rnorm = cur / mx
            u = _protect_usefulness_by_module.get(json_key)
            if u is None or u.numel() != cur.numel():
                u = torch.zeros_like(cur)
            _protect_usefulness_by_module[json_key] = torch.maximum(u, rnorm)
        # Reset current-task counts for the next task.
        _protect_cur_counts_by_module[json_key] = torch.zeros_like(cur)


def _seed_protect_usefulness(unwrapped_policy: PreTrainedPolicy, path: str) -> dict[str, int]:
    """Seed the prior-usefulness store with u = 1.0 at the slots listed in `path`.

    JSON format: {module_json_key: [slot_index, ...]} — same module keys as the
    memory_by_task usage dumps. Seeds are max-folded, so calling this before the task loop
    makes the listed slots look like a maximally-useful prior task from step 0; combined with
    protect_hard_u > 0 they are structurally vetoed from the top-t mask for the whole run
    (the generalist-slot freeze, E42 addendum). Returns {json_key: n_seeded} for logging.
    """
    with open(path) as f:
        seed = json.load(f)
    seeded: dict[str, int] = {}
    for _, mem, _, json_key in _iter_memory_modules(unwrapped_policy):
        if json_key not in seed:
            continue
        num_slots = int(mem.size)
        idx = torch.as_tensor(seed[json_key], dtype=torch.long)
        if idx.numel() == 0:
            continue
        if int(idx.min()) < 0 or int(idx.max()) >= num_slots:
            raise ValueError(
                f"protect seed for {json_key} has out-of-range slot indices "
                f"(min {int(idx.min())}, max {int(idx.max())}, table {num_slots})"
            )
        u = _protect_usefulness_by_module.get(json_key)
        if u is None or u.numel() != num_slots:
            u = torch.zeros(num_slots, dtype=torch.float32)
        u = u.clone()
        u[idx] = 1.0
        _protect_usefulness_by_module[json_key] = u
        seeded[json_key] = int(idx.numel())
    missing = [k for k in seed if k not in seeded]
    if missing:
        raise ValueError(f"protect seed contains module keys not found on the policy: {missing}")
    return seeded


def _seed_online_idf_from_pretrain(
    stats_path: Path,
    unwrapped_policy: PreTrainedPolicy,
    denom: float = 1.0,
) -> bool:
    """
    Seed the online IDF DF accumulators with pretraining batch_accesses stats,
    divided by ``denom`` to control relative weight vs. sequential training stats.

    Populates the module-level ``_online_idf_df_by_module`` and
    ``_online_idf_total_batches`` dicts so that ``_init_online_idf_stats``
    (which skips modules already present) preserves the seeded values, and a
    subsequent ``_update_online_idf_stats`` call computes a non-uniform initial IDF.

    Returns True if at least one module was seeded.
    """
    try:
        with open(stats_path, "r") as f:
            data = json.load(f)
        per_module = data.get("per_module", {})
    except Exception:
        return False

    present: dict[str, int] = {}
    for _, mem, _, json_key in _iter_memory_modules(unwrapped_policy):
        try:
            num_slots = mem.values.shape[0] if hasattr(mem, "values") else getattr(mem, "size", None)
        except Exception:
            num_slots = getattr(mem, "size", None)
        if num_slots is not None:
            present[json_key] = int(num_slots)

    denom = max(denom, 1e-12)
    seeded_any = False

    for json_key, num_slots in present.items():
        module_dict = per_module.get(json_key)
        if not isinstance(module_dict, dict):
            continue
        df = torch.zeros(num_slots, dtype=torch.float32)
        max_batches = 0.0
        for slot_idx in range(num_slots):
            slot_key = f"value_slot_{slot_idx}"
            slot_info = module_dict.get(slot_key)
            if isinstance(slot_info, dict):
                bacc = float(slot_info.get("batch_accesses", 0))
                df[slot_idx] = bacc
                if bacc > max_batches:
                    max_batches = bacc
        if max_batches <= 0:
            continue
        _online_idf_df_by_module[json_key] = df / denom
        _online_idf_total_batches[json_key] = max_batches / denom
        seeded_any = True

    return seeded_any


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
    override_indices: dict[str, torch.Tensor] | None = None,
    override_weights: dict[str, torch.Tensor] | None = None,
    weighting_method: str = "raw",
    protect_usefulness_by_module: dict[str, torch.Tensor] | None = None,
    protect_beta: float = 0.0,
    protect_mode: str = "rank",
    protect_scale_out: dict[torch.nn.Parameter, torch.Tensor] | None = None,
    protect_hard_u: float = 0.0,
) -> dict[torch.nn.Parameter, torch.Tensor]:
    """
    For each memory module, compute TF (or TF-IDF) over slots accessed in the current batch and
    return a dict mapping parameters -> 1D LongTensor of allowed row indices (top-t per module).
    The mask is always defined over value-slot indices; when memory keys are trainable, their
    gradients are masked using the corresponding key rows implied by these selected slots.
    If tf_only is True, IDF is taken as 1 for all slots and no IDF stats are required.

    Protection: with protect_mode="rank" (legacy) the (1-u)^beta factor discounts the ranking
    score before top-t. With protect_mode="grad_scale" the ranking stays pure TF-IDF and, when
    `protect_scale_out` is provided, it is filled with value_param -> per-slot update scale
    vector (1-u)^beta (full num_slots, float32, CPU); the caller applies it to the optimizer
    step via _snapshot_protected_rows/_blend_protected_rows. Keys are never scaled, only masked.
    protect_hard_u > 0 additionally zeroes the score of slots with u >= threshold in BOTH modes
    (never in mask => no gradient, no momentum; see SequentialOnlineConfig.protect_hard_u).
    """
    allowed_by_param: dict[torch.nn.Parameter, torch.Tensor] = {}
    for _, mem_module, value_params, json_key in _iter_memory_modules(unwrapped_policy):
        # Use override indices (accumulated across micro-batches) if provided,
        # otherwise fall back to the module's last_indices from the most recent forward.
        if override_indices is not None and json_key in override_indices:
            idx = override_indices[json_key]
        elif hasattr(mem_module, "last_indices") and mem_module.last_indices is not None:
            idx = mem_module.last_indices
        else:
            continue
        try:
            idx_flat = idx.reshape(-1).to(torch.long)
            num_slots = mem_module.size
            if weighting_method == "weighted":
                if override_weights is not None and json_key in override_weights:
                    slot_weights = override_weights[json_key]
                else:
                    slot_weights = getattr(mem_module, "last_weights", None)
                if slot_weights is None:
                    raise RuntimeError(
                        f"Missing retrieval weights for weighted TF-IDF in module: {json_key}"
                    )
                weights_flat = slot_weights.reshape(-1).to(device=idx_flat.device, dtype=torch.float32)
                if weights_flat.numel() != idx_flat.numel():
                    raise RuntimeError(
                        f"Weight/index size mismatch for module {json_key}: "
                        f"{weights_flat.numel()} vs {idx_flat.numel()}"
                    )
                # c(i): per-batch retrieval mass assigned to slot i
                counts = torch.bincount(idx_flat, weights=weights_flat, minlength=num_slots).to(
                    torch.float32
                )
            else:
                # c(i): per-batch raw access counts (TF numerator)
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
            # Prior-usefulness protection, "rank" mode: multiply by (1 - u)**beta so slots
            # earlier tasks relied on are pushed out of the top-t update set (graded,
            # task-identity-aware). In "grad_scale" mode the ranking is left untouched and the
            # attenuation is applied to the gradients instead (see protect_scale_out below).
            if protect_usefulness_by_module is not None and protect_beta > 0 and protect_mode == "rank":
                u = protect_usefulness_by_module.get(json_key)
                if u is not None and u.numel() == num_slots:
                    u = u.to(device=tf.device, dtype=torch.float32)
                    tfidf = tfidf * (1.0 - u).clamp_(min=0.0).pow(protect_beta)
            # Hard veto (both modes, E42): remove slots at/above the u threshold from candidacy
            # entirely (not just zero their score — with top_t >= candidate count a zero-score
            # slot would still be selected). Never in mask => no gradient, no momentum.
            if protect_hard_u > 0 and protect_usefulness_by_module is not None:
                u_hard = protect_usefulness_by_module.get(json_key)
                if u_hard is not None and u_hard.numel() == num_slots:
                    keep = u_hard.to(device=tf.device, dtype=torch.float32) < protect_hard_u
                    tfidf = tfidf * keep
                    counts = counts * keep
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
                # Mask all value params by the selected slot indices
                # For vector mode: [values], for lora mode: [slot_down, slot_up]
                # (+ [slot_bias] for affine slots — same dim-0 slot indexing)
                for vp in value_params:
                    allowed_by_param[vp] = top_indices.detach()

                # "grad_scale" protection: emit the per-slot update scale (1-u)^beta for the
                # value params. Applied around optimizer.step() via _snapshot_protected_rows /
                # _blend_protected_rows, only to rows inside the mask (rest are zero-grad).
                if (
                    protect_scale_out is not None
                    and protect_mode == "grad_scale"
                    and protect_usefulness_by_module is not None
                    and protect_beta > 0
                ):
                    u = protect_usefulness_by_module.get(json_key)
                    if u is not None and u.numel() == num_slots:
                        scale = (1.0 - u.to(torch.float32)).clamp(min=0.0).pow(protect_beta)
                        for vp in value_params:
                            protect_scale_out[vp] = scale

                # If keys are trainable, mask their gradients to rows corresponding to the
                # selected value slots. Each value slot index encodes a pair of sub-keys
                # (i1, i2) for each head: i1 = slot // n_keys, i2 = slot % n_keys.
                keys_param = getattr(mem_module, "keys", None)
                if keys_param is not None and keys_param.requires_grad:
                    try:
                        n_keys = int(mem_module.n_keys)
                        # Use the same idx that was used for value selection (may be override_indices)
                        # idx shape: (B, heads, knn)
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


def _snapshot_protected_rows(
    allowed_by_param: dict[torch.nn.Parameter, torch.Tensor],
    scale_by_param: dict[torch.nn.Parameter, torch.Tensor] | None,
) -> list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Snapshot mask rows carrying a protection scale < 1 before optimizer.step().

    Returns [(param, rows, pre_values, row_scales)] for _blend_protected_rows. Applying the
    scale to the post-step delta (rather than the gradient) makes the attenuation exact
    per-slot LR scaling under any optimizer — Adam's update is invariant to a time-constant
    gradient scale, so gradient scaling would silently do ~nothing.
    """
    snap: list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    if not scale_by_param:
        return snap
    for p, allowed_rows in allowed_by_param.items():
        scale = scale_by_param.get(p)
        if scale is None or scale.numel() != p.shape[0]:
            continue
        try:
            rows = allowed_rows.to(device=p.device)
            sc = scale.to(device=p.device, dtype=torch.float32)[rows]
            need = sc < 1.0
            if not bool(need.any()):
                continue
            r = rows[need]
            snap.append((p, r, p.data[r].clone(), sc[need]))
        except Exception:
            continue
    return snap


def _blend_protected_rows(
    snap: list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: Optimizer | None = None,
):
    """theta[r] <- theta_pre[r] + scale * (theta_post[r] - theta_pre[r]), after optimizer.step().

    When `optimizer` is given, the row's Adam first moment (exp_avg) is scaled by the same
    factor. Without this the blend LEAKS (E42, measured): it rescales only steps where the row
    is in the snapshot (= in the top-t mask), but Adam keeps applying the row's momentum tail
    (~1/(1-beta1) steps) after the row leaves the churning mask — those steps are unblended and
    carried ~90% of the movement in the softprotect run (u=1.0 slots that should have been
    frozen moved at ~0.86x the unprotected rate). Scaling exp_avg at blend time attenuates the
    tail at its source, restoring the intended per-slot-LR semantics. exp_avg_sq is left
    untouched (it only normalizes step magnitude; shrinking it would inflate later steps).
    """
    opt = getattr(optimizer, "optimizer", optimizer)  # unwrap AcceleratedOptimizer
    with torch.no_grad():
        for p, rows, pre, sc in snap:
            try:
                scv = sc.to(dtype=p.dtype).view(-1, *([1] * (p.dim() - 1)))
                p.data[rows] = pre + scv * (p.data[rows] - pre)
                if opt is not None:
                    state = opt.state.get(p)
                    if state is not None and "exp_avg" in state:
                        state["exp_avg"][rows] *= scv.to(dtype=state["exp_avg"].dtype)
            except Exception:
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
    task_emb: torch.Tensor | None = None,
    accum_indices_bufs: dict[str, list[torch.Tensor]] | None = None,
    accum_weights_bufs: dict[str, list[torch.Tensor]] | None = None,
    weighting_method: str = "raw",
    protect_usefulness_by_module: dict[str, torch.Tensor] | None = None,
    protect_beta: float = 0.0,
    protect_mode: str = "rank",
    protect_hard_u: float = 0.0,
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
        loss, output_dict = policy.forward(batch, task_emb=task_emb)
    if use_cuda_events:
        ev_fwd.record()

    accelerator.backward(loss)
    if use_cuda_events:
        ev_bwd.record()

    # Only apply TF-IDF masking, clip, step, and zero_grad on the sync step
    # (last micro-batch of gradient accumulation). On intermediate micro-batches
    # gradients just accumulate.
    mask_build_s = 0.0
    mask_apply_s = 0.0
    if accelerator.sync_gradients:
        # Before TF-IDF masking: build concatenated indices from accumulated buffers
        # + current last_indices, so TF-IDF sees the full logical batch.
        # We pass these as override_indices rather than modifying modules, to avoid
        # polluting last_indices (which the outer loop uses for per-micro-batch stats).
        override_indices = None
        override_weights = None
        if accum_indices_bufs is not None:
            unwrapped_flush = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
            override_indices = {}
            if weighting_method == "weighted":
                override_weights = {}
            for _, mem, _, json_key in _iter_memory_modules(unwrapped_flush):
                bufs = accum_indices_bufs.get(json_key, [])
                cur = getattr(mem, "last_indices", None)
                if bufs and cur is not None:
                    override_indices[json_key] = torch.cat(bufs + [cur], dim=0)
                elif cur is not None:
                    override_indices[json_key] = cur
                if weighting_method == "weighted":
                    weight_bufs = accum_weights_bufs.get(json_key, []) if accum_weights_bufs is not None else []
                    cur_w = getattr(mem, "last_weights", None)
                    if weight_bufs and cur_w is not None:
                        override_weights[json_key] = torch.cat(weight_bufs + [cur_w], dim=0)
                    elif cur_w is not None:
                        override_weights[json_key] = cur_w
            accum_indices_bufs.clear()
            if accum_weights_bufs is not None:
                accum_weights_bufs.clear()

        # Compute and apply TF-IDF gradient masks before clipping and step
        unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
        protect_snap: list = []
        if (idf_by_module is not None or tf_only) and top_t > 0:
            t0 = time.perf_counter()
            protect_scale: dict[torch.nn.Parameter, torch.Tensor] | None = (
                {} if protect_mode == "grad_scale" else None
            )
            allowed = _compute_tfidf_top_indices_for_batch(
                unwrapped, idf_by_module, top_t, tf_only=tf_only,
                override_indices=override_indices,
                override_weights=override_weights,
                weighting_method=weighting_method,
                protect_usefulness_by_module=protect_usefulness_by_module,
                protect_beta=protect_beta,
                protect_mode=protect_mode,
                protect_scale_out=protect_scale,
                protect_hard_u=protect_hard_u,
            )
            mask_build_s = time.perf_counter() - t0
            if use_cuda_events:
                ev_mask.record()
            if allowed:
                t1 = time.perf_counter()
                _apply_gradient_mask_to_memory_values(allowed)
                protect_snap = _snapshot_protected_rows(allowed, protect_scale)
                mask_apply_s = time.perf_counter() - t1
                if use_cuda_events:
                    ev_apply.record()

        if grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"), error_if_nonfinite=False)
        if use_cuda_events:
            ev_clip.record()

        from contextlib import nullcontext
        with (lock if lock is not None else nullcontext()):
            optimizer.step()
        if protect_snap:
            # "grad_scale" protection: rescale the applied update on protected rows
            # (exact per-slot LR scaling; see _snapshot_protected_rows) and scale their
            # momentum so the attenuation survives mask churn (see _blend_protected_rows).
            _blend_protected_rows(protect_snap, optimizer)
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
        if accelerator.sync_gradients:
            clip_s = ev_bwd.elapsed_time(ev_clip) / 1000.0
            opt_s = ev_clip.elapsed_time(ev_opt) / 1000.0
            sched_s = ev_opt.elapsed_time(ev_sched) / 1000.0 if lr_scheduler is not None else 0.0
        else:
            clip_s = 0.0
            opt_s = 0.0
            sched_s = 0.0
        update_s = ev0.elapsed_time(ev_end) / 1000.0
    else:
        fwd_s = 0.0
        bwd_s = 0.0
        clip_s = 0.0
        opt_s = 0.0
        sched_s = 0.0
        update_s = step_wall_s

    # Only update metrics on sync steps to avoid diluting averages with
    # intermediate micro-batch zeros.
    if accelerator.sync_gradients:
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
        accelerator = Accelerator(
            gradient_accumulation_steps=getattr(cfg, "gradient_accumulation_steps", 1),
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
        )

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
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

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
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
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

    # Precompute task embeddings for language-conditioned memory queries
    if hasattr(policy, "precompute_task_embeddings"):
        policy.precompute_task_embeddings(dataset.meta)

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
            # Online mode: initialize DF/IDF structures, optionally seeded from
            # pretraining stats so that task 1 starts with a non-uniform IDF.
            idf_by_module = {}
            unwrapped_for_idf = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)

            # Seed from pretraining stats if both flags are set
            seeded = False
            if cfg.idf_stats_path:
                seed_path = Path(cfg.idf_stats_path)
                if seed_path.exists():
                    seeded = _seed_online_idf_from_pretrain(
                        seed_path, unwrapped_for_idf, denom=cfg.idf_stats_denom,
                    )
                    if is_main and seeded:
                        logging.info(
                            f"Seeded online IDF from pretraining stats: {seed_path} "
                            f"(denom={cfg.idf_stats_denom})"
                        )

            try:
                # _init_online_idf_stats skips modules already in the global
                # accumulators (i.e. those just seeded) and fills in the rest.
                _init_online_idf_stats(unwrapped_for_idf, idf_by_module)
                # If seeded, recompute IDF from the seeded DF so task 1 starts
                # with a meaningful (non-uniform) IDF vector.
                if seeded:
                    _update_online_idf_stats(
                        unwrapped_for_idf, idf_by_module, idf_exponent=cfg.idf_exponent,
                    )
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
    env_preprocessor = None
    env_postprocessor = None
    if cfg.env is not None and cfg.eval.type == "env":
        if is_main:
            logging.info("Creating eval envs")
        eval_envs_all = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
        # Upstream eval_policy_all() now requires env-side pre/post-processors.
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env, policy_cfg=cfg.policy
        )

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
    # Accumulator for the cumulative eval bar chart (one entry per eval loop)
    eval_bar_history: list[dict] = []
    # Loss-based eval (--eval.type=loss): per-task MSE history + per-task baseline
    # (a task's loss right after it was trained, used to report forgetting).
    loss_eval_history: list[dict] = []
    loss_baseline: dict[int, float] = {}

    # Seed the prior-usefulness store (generalist-slot freeze) before any task trains.
    if cfg.protect_prior_slots and cfg.protect_seed_path:
        seeded = _seed_protect_usefulness(
            accelerator.unwrap_model(policy, keep_fp32_wrapper=True), cfg.protect_seed_path
        )
        if accelerator.is_main_process:
            logging.info(
                f"Seeded prior-usefulness store from {cfg.protect_seed_path}: "
                + ", ".join(f"{k.rsplit('.', 1)[-1] if k.count('.') else k}={v}" for k, v in seeded.items())
                + f" (hard_u={cfg.protect_hard_u}, beta={cfg.protect_beta}, mode={cfg.protect_mode})"
            )

    # Iterate sequentially over dataset tasks
    for idx, dataset_task_id in enumerate(cfg.online_task_ids):
        # `idx` is rebound later in this block (memory diagnostics); keep the task
        # position under a stable name for the end-of-task eval.
        task_pos = idx
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
        grad_accum_steps = getattr(cfg, "gradient_accumulation_steps", 1)
        # Per-process effective batch size; MetricsTracker.step() multiplies by num_processes
        effective_bs = cfg.batch_size * grad_accum_steps
        train_tracker = MetricsTracker(
            effective_bs,
            dataset.num_frames,
            dataset.num_episodes,
            train_metrics,
            initial_step=global_step,
            accelerator=accelerator,
        )

        # Buffer for accumulating memory indices across micro-batches.
        # Passed to _update_policy_with_tfidf so TF-IDF top_t sees the full logical batch.
        _seq_accum_indices: dict[str, list[torch.Tensor]] = defaultdict(list)
        _seq_accum_weights: dict[str, list[torch.Tensor]] = defaultdict(list)

        def _seq_snapshot_indices(unwrapped):
            """Snapshot last_indices/last_weights into accumulation buffers for TF-IDF."""
            for _, mem, _, json_key in _iter_memory_modules(unwrapped):
                idx = getattr(mem, "last_indices", None)
                if idx is not None:
                    _seq_accum_indices[json_key].append(idx.clone())
                if cfg.tf_idf_weighting_method == "weighted":
                    weights = getattr(mem, "last_weights", None)
                    if weights is not None:
                        _seq_accum_weights[json_key].append(weights.clone())

        for _ in range(cfg.online_steps_per_task):
            output_dict = None

            for _micro in range(grad_accum_steps):
                with accelerator.accumulate(policy):
                    t0 = time.perf_counter()
                    batch = next(dl_iter)
                    train_tracker.dataloading_s = time.perf_counter() - t0

                    t1 = time.perf_counter()
                    for cam_key in dataset.meta.camera_keys:
                        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
                    batch = preprocessor(batch)
                    train_tracker.preproc_s = time.perf_counter() - t1

                    # Compute task embeddings for language-conditioned memory queries
                    task_emb = None
                    unwrapped_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
                    if hasattr(unwrapped_policy, "get_task_embeddings") and task_index_to_name:
                        try:
                            task_name = task_index_to_name.get(dataset_task_id, "")
                            B = batch[list(batch.keys())[0]].shape[0] if isinstance(batch, dict) else 1
                            task_names = [task_name] * B
                            task_emb = unwrapped_policy.get_task_embeddings(task_names)
                            if task_emb is not None:
                                task_emb = task_emb.to(device=device)
                        except Exception:
                            task_emb = None

                    # Snapshot indices BEFORE update so the buffer includes this micro-batch
                    # when _update_policy_with_tfidf reads it on the sync step.
                    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)

                    if cfg.tfidf_enable or cfg.tf_only:
                        # Forward sets last_indices; snapshot it for TF-IDF accumulation.
                        # We do forward+backward inside the update call, then snapshot.
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
                            task_emb=task_emb,
                            accum_indices_bufs=_seq_accum_indices if grad_accum_steps > 1 else None,
                            accum_weights_bufs=_seq_accum_weights if grad_accum_steps > 1 else None,
                            weighting_method=cfg.tf_idf_weighting_method,
                            protect_usefulness_by_module=(
                                _protect_usefulness_by_module if cfg.protect_prior_slots else None
                            ),
                            protect_beta=cfg.protect_beta,
                            protect_mode=cfg.protect_mode,
                            protect_hard_u=cfg.protect_hard_u,
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
                            task_emb=task_emb,
                            task_ids=None,
                        )

                # After each micro-batch forward: snapshot indices for TF-IDF accumulation
                # and accumulate task usage stats (main only, for JSON logging).
                # On sync steps with TF-IDF: buffer was consumed+cleared inside
                # _update_policy_with_tfidf, don't re-add.
                # On sync steps without TF-IDF or on non-sync steps: snapshot normally.
                # Clear on sync when TF-IDF is off to prevent unbounded growth.
                if accelerator.sync_gradients:
                    if not (cfg.tfidf_enable or cfg.tf_only):
                        _seq_accum_indices.clear()
                        _seq_accum_weights.clear()
                else:
                    _seq_snapshot_indices(unwrapped)
                if is_main:
                    _accumulate_task_usage_for_batch(unwrapped, task_id=dataset_task_id)
                    # Only accumulate update indices on sync steps (when TF-IDF actually ran)
                    if accelerator.sync_gradients:
                        _accumulate_task_updates_for_batch(unwrapped, task_id=dataset_task_id)

                # Accumulate online IDF stats per micro-batch (all ranks)
                if cfg.tfidf_enable and not cfg.tf_only and cfg.use_online_idf_stats:
                    _accumulate_online_idf_stats_batch(unwrapped)

                # Accumulate prior-usefulness read counts per micro-batch (all ranks)
                if cfg.protect_prior_slots:
                    _accumulate_protect_counts_batch(unwrapped)

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

        # After finishing this task, fold its read profile into the prior-usefulness store
        # so that subsequent tasks protect the slots this task relied on.
        if cfg.protect_prior_slots:
            try:
                _finalize_protect_usefulness(
                    accelerator.unwrap_model(policy, keep_fp32_wrapper=True),
                    u_norm=cfg.protect_u_norm,
                )
                if is_main:
                    logging.info(
                        f"Updated prior-usefulness protection store after task {idx + 1} "
                        f"(protect_beta={cfg.protect_beta}, mode={cfg.protect_mode}, "
                        f"u_norm={cfg.protect_u_norm}, hard_u={cfg.protect_hard_u})."
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
        # Flush per-task memory usage for this task and clear accumulators (main only)
        if is_main:
            try:
                _flush_per_task_usage(cfg.output_dir, task_id=dataset_task_id)
            except Exception:
                pass
            if wandb_logger:
                mem_step_id = get_step_identifier(global_step, cfg.steps)
                wandb_logger.log_memory_stats(cfg.output_dir, mem_step_id)
            _per_task_totals.clear()
            _per_task_batches.clear()
            _per_task_update_totals.clear()
            _per_task_update_batches.clear()

        # Cumulative evaluation up to this task
        if cfg.eval.type == "loss":
            # Loss-based eval: recompute flow-matching MSE on every task seen so far,
            # using the current (just-updated) policy. No env required.
            seen_ids = [int(t) for t in cfg.online_task_ids[: idx + 1]]
            if is_main:
                logging.info(colored(f"Loss eval on seen tasks: {seen_ids}", "green"))
            per_task_loss = _eval_loss_on_seen_tasks(
                policy,
                accelerator,
                dataset,
                task_index_to_name,
                seen_ids,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                device=device,
                n_batches=cfg.eval_loss_n_batches,
                preprocessor=preprocessor,
                seed=cfg.seed,
            )
            # Baseline = a task's loss the first time we see it (right after it trained).
            cur_tid = int(dataset_task_id)
            if cur_tid in per_task_loss:
                loss_baseline.setdefault(cur_tid, per_task_loss[cur_tid])

            if is_main:
                cur = per_task_loss.get(cur_tid, float("nan"))
                avg_seen = float(np.mean(list(per_task_loss.values()))) if per_task_loss else float("nan")
                prior_forget = {
                    tid: per_task_loss[tid] - loss_baseline.get(tid, per_task_loss[tid])
                    for tid in seen_ids
                    if tid != cur_tid and tid in per_task_loss
                }
                avg_forget = float(np.mean(list(prior_forget.values()))) if prior_forget else 0.0
                logging.info(
                    f"Loss eval | tasks_seen={len(seen_ids)} current_mse={cur:.4f} "
                    f"avg_mse_seen={avg_seen:.4f} avg_forgetting_prior={avg_forget:+.4f}"
                )
                for tid in seen_ids:
                    if tid in per_task_loss:
                        d = per_task_loss[tid] - loss_baseline.get(tid, per_task_loss[tid])
                        logging.info(
                            f"   task {tid} '{task_index_to_name.get(tid, '')[:34]}': "
                            f"mse={per_task_loss[tid]:.4f}  Δvs_trained={d:+.4f}"
                        )
                if wandb_logger:
                    log_dict = {
                        "num_tasks_seen": len(seen_ids),
                        "avg_mse_loss_seen": avg_seen,
                        "mse_loss_current": cur,
                        "avg_forgetting_prior": avg_forget,
                    }
                    for tid, v in per_task_loss.items():
                        log_dict[f"loss/task_{tid}"] = float(v)
                    for tid, d in prior_forget.items():
                        log_dict[f"forgetting/task_{tid}"] = float(d)
                    wandb_logger.log_dict(log_dict, global_step, mode="eval")

                _append_loss_results_jsonl(cfg.output_dir, global_step, per_task_loss, loss_baseline)
                loss_eval_history.append({
                    "trained_task_idx": cur_tid,
                    "per_task": {int(k): float(v) for k, v in per_task_loss.items()},
                })
                if wandb_logger:
                    try:
                        import matplotlib.pyplot as plt

                        fig = _render_loss_eval_chart(loss_eval_history)
                        wandb = wandb_logger._wandb
                        wandb.log({"eval/cumulative_loss_chart": wandb.Image(fig)}, step=global_step)
                        plt.close(fig)
                    except Exception as e:
                        logging.warning(f"Failed to log loss eval chart: {e}")
        elif eval_envs_all is not None:
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
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=_eval_n_episodes_for_task(cfg, task_pos),
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

                # Accumulate and log the cumulative eval bar chart
                per_task_results = {}
                for entry in (eval_info.get("per_task") or []):
                    tid = entry.get("task_id")
                    successes = (entry.get("metrics") or {}).get("successes", [])
                    if tid is not None and successes:
                        per_task_results[tid] = float(np.mean(successes) * 100)
                if per_task_results:
                    eval_bar_history.append({
                        "trained_task_idx": dataset_task_id,
                        "per_task": per_task_results,
                    })
                    if wandb_logger:
                        try:
                            import matplotlib.pyplot as plt
                            fig = _render_cumulative_eval_bar_chart(eval_bar_history)
                            wandb = wandb_logger._wandb
                            wandb.log({"eval/cumulative_success_chart": wandb.Image(fig)}, step=global_step)
                            plt.close(fig)
                        except Exception as e:
                            logging.warning(f"Failed to log cumulative eval bar chart: {e}")

    # Cleanup
    if eval_envs_all:
        close_envs(eval_envs_all)

    if is_main:
        # --- Build + log full memory usage visualization (optional) ---
        if wandb_logger is not None and getattr(cfg, "log_full_memory_usage_viz", True):
            # Per-task JSONs generated by this sequential run
            task_json_dir = Path(cfg.output_dir) / "memory_by_task"

            # Global JSON (typically from the pretrained checkpoint)
            global_json = None
            try:
                if cfg.idf_stats_path:
                    p = Path(str(cfg.idf_stats_path))
                    if p.is_file():
                        global_json = p
            except Exception:
                global_json = None
            if global_json is None:
                try:
                    pp = getattr(cfg.policy, "pretrained_path", None)
                    if pp:
                        cand = Path(pp) / "memory_usage.json"
                        if cand.is_file():
                            global_json = cand
                except Exception:
                    global_json = None

            if not task_json_dir.is_dir():
                logging.warning(
                    f"Skipping memory usage visualization: per-task JSON directory not found at {task_json_dir}"
                )
            else:
                wandb = wandb_logger._wandb

                # ---- 1) Interactive Plotly HTML (best-effort) ----
                try:
                    from lerobot.utils.memory_usage_viz import write_full_memory_usage_html

                    viz_dir = Path(cfg.output_dir) / "visualizations"
                    html_path = viz_dir / "full_memory_usage_viz.html"
                    html_path = write_full_memory_usage_html(
                        output_path=html_path,
                        global_json=global_json,
                        task_json_dir=task_json_dir,
                        grid_side=getattr(cfg, "full_memory_usage_viz_grid_side", None),
                        include_plotlyjs=getattr(cfg, "full_memory_usage_viz_include_plotlyjs", "cdn"),
                    )
                    logging.info(f"Wrote memory usage HTML to {html_path}")

                    # Attach the file to the run for easy download (Files tab)
                    try:
                        wandb.save(str(html_path), base_path=str(cfg.output_dir))
                    except Exception as e:
                        logging.warning(f"wandb.save for memory viz HTML failed: {e}")

                    # Embed directly in the run (Media panel) when supported
                    try:
                        if hasattr(wandb, "Html"):
                            try:
                                wandb_html = wandb.Html(str(html_path))
                            except Exception:
                                wandb_html = wandb.Html(html_path.read_text())
                            wandb.log(
                                {"eval/full_memory_usage_viz": wandb_html},
                                step=global_step,
                            )
                            logging.info("Logged interactive memory viz HTML to wandb")
                        elif hasattr(wandb, "Plotly"):
                            from lerobot.utils.memory_usage_viz import build_full_memory_usage_figure

                            fig = build_full_memory_usage_figure(
                                global_json=global_json,
                                task_json_dir=task_json_dir,
                                grid_side=getattr(cfg, "full_memory_usage_viz_grid_side", None),
                            )
                            wandb.log(
                                {"eval/full_memory_usage_viz": wandb.Plotly(fig)},
                                step=global_step,
                            )
                            logging.info("Logged interactive memory viz Plotly to wandb")
                    except Exception as e:
                        logging.warning(f"Failed to embed interactive memory viz in wandb: {e}")
                except Exception as e:
                    logging.warning(f"Failed to build interactive Plotly memory visualization: {e}")

                # ---- 2) Matplotlib IoU heatmaps + scalar metrics (robust fallback) ----
                try:
                    from lerobot.utils.memory_usage_viz import build_iou_images_and_metrics

                    iou_images, iou_metrics = build_iou_images_and_metrics(
                        global_json=global_json,
                        task_json_dir=task_json_dir,
                    )

                    if iou_images:
                        img_log = {}
                        for key, fig in iou_images.items():
                            img_log[key] = wandb.Image(fig)
                            import matplotlib.pyplot as plt
                            plt.close(fig)
                        wandb.log(img_log, step=global_step)
                        logging.info(f"Logged {len(iou_images)} IoU heatmap image(s) to wandb")

                    if iou_metrics:
                        wandb.log(iou_metrics, step=global_step)
                        logging.info(
                            f"Logged {len(iou_metrics)} memory IoU metrics to wandb. "
                            f"Overall mean IoU: {iou_metrics.get('memory_iou/all_modules_mean', 'N/A')}"
                        )
                except Exception as e:
                    logging.warning(f"Failed to build/log IoU heatmap images and metrics: {e}")

        logging.info("End of sequential online training")


def main():
    register_third_party_plugins()
    sequential_train()


if __name__ == "__main__":
    main()
