#!/usr/bin/env python

# Copyright 2025

import argparse
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import (
    DEFAULT_VIDEO_PATH,
    INFO_PATH,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_tasks,
    write_episode_stats,
    write_info,
    write_jsonlines,
    write_stats,
)


def _get_video_keys_from_info(info: dict) -> list[str]:
    return [k for k, ft in info["features"].items() if ft.get("dtype") == "video"]


def _format_data_path(info: dict, episode_index: int) -> str:
    chunk = episode_index // info["chunks_size"]
    return info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)


def _format_video_path(info: dict, episode_index: int, video_key: str) -> str:
    chunk = episode_index // info["chunks_size"]
    video_path = info.get("video_path") or DEFAULT_VIDEO_PATH
    return video_path.format(episode_chunk=chunk, video_key=video_key, episode_index=episode_index)


def _update_table_columns(
    table: pa.Table,
    new_episode_index: int,
    index_offset: int,
) -> pa.Table:
    num_rows = table.num_rows

    new_episode_col = pa.array(np.full((num_rows,), new_episode_index, dtype=np.int64))
    new_index_col = pa.array(np.arange(index_offset, index_offset + num_rows, dtype=np.int64))

    col_names = set(table.schema.names)
    if "episode_index" in col_names:
        table = table.set_column(table.schema.get_field_index("episode_index"), "episode_index", new_episode_col)
    else:
        table = table.append_column("episode_index", new_episode_col)

    if "index" in col_names:
        table = table.set_column(table.schema.get_field_index("index"), "index", new_index_col)
    else:
        table = table.append_column("index", new_index_col)

    return table


def _write_meta(
    out_root: Path,
    new_info: dict,
    new_episodes: list[dict[str, Any]],
    episodes_stats_serialized: list[tuple[int, dict]],
    tasks_src_root: Path,
) -> None:
    write_info(new_info, out_root)

    write_jsonlines(new_episodes, out_root / "meta/episodes.jsonl")

    stats_list = []
    for ep_idx, stats in episodes_stats_serialized:
        write_episode_stats(ep_idx, stats, out_root)
        stats_list.append(stats)
    if stats_list:
        aggregated = aggregate_stats(stats_list)
        write_stats(aggregated, out_root)

    # copy tasks.jsonl as-is
    src_tasks = tasks_src_root / "meta/tasks.jsonl"
    if src_tasks.is_file():
        (out_root / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_tasks, out_root / "meta/tasks.jsonl")


def remove_episodes(dataset_root: Path, output_dataset: Path, episodes_to_remove: list[int]):
    dataset_root = dataset_root.resolve()
    output_dataset = output_dataset.resolve()

    if (output_dataset / INFO_PATH).exists():
        raise FileExistsError(f"Output dataset already exists and contains metadata: {output_dataset}")

    info = load_info(dataset_root)
    video_keys = _get_video_keys_from_info(info)

    tasks, _ = load_tasks(dataset_root)
    episodes = load_episodes(dataset_root)
    stats = load_episodes_stats(dataset_root)

    remove_set = set(int(e) for e in episodes_to_remove)
    all_indices = sorted(episodes.keys())
    for e in remove_set:
        if e not in episodes:
            raise KeyError(f"Episode {e} not found in dataset.")

    kept_ep_order = [episodes[k] for k in all_indices if k not in remove_set]

    chunks_size = info["chunks_size"]
    total_frames = 0
    new_episodes_records: list[dict[str, Any]] = []
    episodes_stats_serialized: list[tuple[int, dict]] = []

    (output_dataset / "meta").mkdir(parents=True, exist_ok=True)
    (output_dataset / "data").mkdir(parents=True, exist_ok=True)
    if video_keys:
        (output_dataset / "videos").mkdir(parents=True, exist_ok=True)

    new_ep_idx = 0
    running_index_offset = 0

    for ep in kept_ep_order:
        old_ep_idx = int(ep["episode_index"]) if isinstance(ep, dict) else int(ep.episode_index)
        ep_length = int(ep["length"]) if isinstance(ep, dict) else int(ep.length)
        ep_tasks = ep["tasks"] if isinstance(ep, dict) else ep.tasks

        old_parquet_rel = _format_data_path(info, old_ep_idx)
        old_parquet_path = dataset_root / old_parquet_rel
        if not old_parquet_path.is_file():
            raise FileNotFoundError(f"Missing parquet: {old_parquet_path}")

        table = pq.read_table(old_parquet_path)
        table = _update_table_columns(
            table,
            new_episode_index=new_ep_idx,
            index_offset=running_index_offset,
        )

        new_chunk = new_ep_idx // chunks_size
        new_parquet_rel = info["data_path"].format(episode_chunk=new_chunk, episode_index=new_ep_idx)
        new_parquet_path = output_dataset / new_parquet_rel
        new_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, new_parquet_path)

        for vkey in video_keys:
            old_video_rel = _format_video_path(info, old_ep_idx, vkey)
            old_video_path = dataset_root / old_video_rel
            if not old_video_path.is_file():
                raise FileNotFoundError(f"Missing video file: {old_video_path}")
            new_video_rel = info["video_path"].format(episode_chunk=new_chunk, video_key=vkey, episode_index=new_ep_idx)
            new_video_path = output_dataset / new_video_rel
            new_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_video_path, new_video_path)

        new_episodes_records.append(
            {
                "episode_index": new_ep_idx,
                "tasks": list(set(ep_tasks)),
                "length": ep_length,
            }
        )

        if old_ep_idx not in stats:
            raise KeyError(f"Missing episode stats for episode {old_ep_idx} in {dataset_root}")
        episodes_stats_serialized.append((new_ep_idx, stats[old_ep_idx]))

        total_frames += ep_length
        running_index_offset += ep_length
        new_ep_idx += 1

    total_episodes = new_ep_idx
    total_chunks = math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0

    # total_tasks reflects tasks.jsonl
    total_tasks = len(tasks)
    new_info = {
        "codebase_version": info["codebase_version"],
        "robot_type": info.get("robot_type"),
        "total_episodes": total_episodes,
        "total_frames": int(total_frames),
        "total_tasks": total_tasks,
        "total_videos": len(video_keys) * total_episodes,
        "total_chunks": total_chunks,
        "chunks_size": chunks_size,
        "fps": info["fps"],
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": info["data_path"],
        "video_path": info.get("video_path"),
        "features": info["features"],
    }

    _write_meta(output_dataset, new_info, new_episodes_records, episodes_stats_serialized, dataset_root)


def parse_args():
    parser = argparse.ArgumentParser(description="Remove episodes from a LeRobot dataset and reindex.")
    parser.add_argument("dataset_root", type=str, help="Path to dataset root (contains meta/, data/, videos/)")
    parser.add_argument("episodes", type=int, nargs="+", help="Episode indices to remove")
    parser.add_argument(
        "--output_dataset",
        type=str,
        default=None,
        help="Path to output dataset root (default: dataset_root + '_filtered')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dataset = Path(args.output_dataset) if args.output_dataset else dataset_root.parent / f"{dataset_root.name}_filtered"
    remove_episodes(dataset_root, output_dataset, args.episodes)


if __name__ == "__main__":
    main()


