#!/usr/bin/env python

# Copyright 2025
# Licensed under the Apache License, Version 2.0

import argparse
from collections import Counter
from pathlib import Path

from lerobot.datasets.utils import load_episodes, load_tasks


def main():
    parser = argparse.ArgumentParser(
        description="Print dataset stats: total episodes and episodes per task (id and name)."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to dataset root directory (containing meta/, data/, videos/).",
    )
    parser.add_argument(
        "--as_csv",
        action="store_true",
        help="Output as CSV: task_index,task_name,episode_count",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    if not (root / "meta").exists():
        raise FileNotFoundError(f"Not a dataset root (missing meta/): {root}")

    episodes = load_episodes(root)
    tasks_df = load_tasks(root)

    name_to_index = {name: int(row.task_index) for name, row in tasks_df.iterrows()}

    # Count episodes per task (episode counted once per task if task appears in episode's task list)
    counter: Counter[int] = Counter()
    for row in episodes:
        ep_tasks = list(row.get("tasks", []) or [])
        for name in set(ep_tasks):
            idx = name_to_index.get(name)
            if idx is not None:
                counter[idx] += 1

    total_episodes = len(episodes)

    if args.as_csv:
        print("task_index,task_name,episode_count")
        for idx, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            task_name = tasks_df.index[idx]
            print(f"{idx},\"{task_name}\",{count}")
        return

    print(f"Total episodes: {total_episodes}")
    print("Episodes per task (sorted by count):")
    for idx, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        task_name = tasks_df.index[idx]
        print(f"  [{idx}] {task_name}: {count}")


if __name__ == "__main__":
    main()

