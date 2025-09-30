#!/usr/bin/env python

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler


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
    # Select episode indices for this task
    episode_indices_to_use = get_episode_indices_for_task(ds, task_index)
    sampler = EpisodeAwareSampler(
        dataset_from_indices=ds.meta.episodes["dataset_from_index"],
        dataset_to_indices=ds.meta.episodes["dataset_to_index"],
        episode_indices_to_use=episode_indices_to_use,
        shuffle=shuffle,
    )

    # Limit number of frames from this task per outer step
    indices = list(sampler)
    if shuffle:
        random.shuffle(indices)
    indices = indices[: frames_per_task]

    subset = torch.utils.data.Subset(ds, indices)
    return torch.utils.data.DataLoader(
        subset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2,
    )


def cycle(loader: torch.utils.data.DataLoader) -> Iterator:
    while True:
        for b in loader:
            yield b


