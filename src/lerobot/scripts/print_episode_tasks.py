#!/usr/bin/env python

# Copyright 2025
# Licensed under the Apache License, Version 2.0

import argparse
from pathlib import Path

from lerobot.datasets.utils import load_episodes, load_tasks


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print episode -> task mapping for a LeRobot v3.0 dataset (local root)."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help=(
            "Path to dataset root directory (containing meta/, data/, videos/)."
        ),
    )
    parser.add_argument(
        "--as_csv",
        action="store_true",
        help="Output as CSV lines: episode_index,task_index,task_name",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    if not (root / "meta").exists():
        raise FileNotFoundError(f"Not a dataset root (missing meta/): {root}")

    episodes = load_episodes(root)
    tasks_df = load_tasks(root)

    name_to_index = {name: int(row.task_index) for name, row in tasks_df.iterrows()}

    if args.as_csv:
        print("episode_index,task_index,task_name")
        for row in episodes:
            ep_idx = int(row["episode_index"]) if "episode_index" in row else -1
            ep_tasks = list(row.get("tasks", []) or [])
            for name in ep_tasks:
                idx = name_to_index.get(name)
                if idx is None:
                    continue
                print(f"{ep_idx},{idx},\"{name}\"")
        return

    for row in episodes:
        ep_idx = int(row["episode_index"]) if "episode_index" in row else -1
        ep_tasks = list(row.get("tasks", []) or [])
        indices = [name_to_index.get(name) for name in ep_tasks if name in name_to_index]
        indices_str = ", ".join(str(i) for i in indices)
        names_str = ", ".join(ep_tasks)
        print(f"episode {ep_idx}: task_indices=[{indices_str}] tasks=[{names_str}]")


if __name__ == "__main__":
    main()


