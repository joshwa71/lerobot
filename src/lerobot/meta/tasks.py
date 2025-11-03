#!/usr/bin/env python

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class TaskSplit:
    train: list[int]
    eval: list[int]


def default_task_split(ds: LeRobotDataset, num_eval: int = 5, seed: int = 0) -> TaskSplit:
    rng = random.Random(seed)
    tasks = list(range(ds.meta.total_tasks))
    rng.shuffle(tasks)
    eval_tasks = tasks[:num_eval]
    train_tasks = tasks[num_eval:]
    return TaskSplit(train=train_tasks, eval=eval_tasks)


def get_episode_indices_for_task(ds: LeRobotDataset, task_index: int) -> list[int]:
    # Episodes store task names (strings). Map to indices via dataset metadata.
    ep_indices = []
    for ep_idx, task_names in enumerate(ds.meta.episodes["tasks"]):
        # task_names can be a scalar or list; normalize to list
        names = task_names if isinstance(task_names, list) else [task_names]
        for name in names:
            idx = ds.meta.get_task_index(name)
            if idx is not None and int(idx) == int(task_index):
                ep_indices.append(ep_idx)
                break
    return ep_indices


def build_task_dataloader(
    ds: LeRobotDataset,
    task_index: int,
    frames_per_task: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    # Efficiently choose up to frames_per_task indices across episodes of this task
    episode_indices_to_use = get_episode_indices_for_task(ds, task_index)
    ranges = []
    for ep_idx in episode_indices_to_use:
        start = ds.meta.episodes["dataset_from_index"][ep_idx]
        end = ds.meta.episodes["dataset_to_index"][ep_idx]
        if end > start:
            ranges.append((start, end))

    # Compute a simple uniform sampling across episode ranges without enumerating all indices
    total = sum((end - start) for start, end in ranges)
    target = min(frames_per_task, total)
    indices: list[int] = []
    if total == 0 or target == 0:
        indices = []
    else:
        # Allocate proportional budget per episode
        remaining = target
        per_range = []
        for start, end in ranges:
            length = end - start
            n = max(0, (length * target) // total)
            per_range.append(max(0, int(n)))
        # Fix rounding by distributing leftovers
        assigned = sum(per_range)
        i = 0
        while assigned < target and i < len(per_range):
            per_range[i] += 1
            assigned += 1
            i += 1
        # Sample evenly within each episode range
        for (start, end), count in zip(ranges, per_range, strict=False):
            if count <= 0:
                continue
            step = max(1, (end - start) // count)
            cur = start
            added = 0
            while added < count and cur < end:
                indices.append(cur)
                cur += step
                added += 1
    if shuffle:
        random.shuffle(indices)

    subset = torch.utils.data.Subset(ds, indices)
    return torch.utils.data.DataLoader(
        subset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4,
    )


def cycle(loader: torch.utils.data.DataLoader) -> Iterator:
    saw_any = False
    while True:
        for b in loader:
            saw_any = True
            yield b
        if not saw_any:
            raise RuntimeError("Empty DataLoader in cycle(): check task selection and frames_per_task.")

