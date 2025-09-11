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
    write_task,
)


def _get_video_keys_from_info(info: dict) -> list[str]:
    return [k for k, ft in info["features"].items() if ft.get("dtype") == "video"]


def _assert_compatible_infos(info1: dict, info2: dict):
    if info1["fps"] != info2["fps"]:
        raise ValueError(f"Input datasets have different fps: {info1['fps']} vs {info2['fps']}")
    if info1.get("robot_type") != info2.get("robot_type"):
        raise ValueError(
            f"Input datasets have different robot_type: {info1.get('robot_type')} vs {info2.get('robot_type')}"
        )
    if info1["features"] != info2["features"]:
        raise ValueError("Input datasets have different features and cannot be merged safely.")


def _assert_same_tasks(tasks1: dict[int, str], tasks2: dict[int, str]):
    if tasks1 != tasks2:
        raise ValueError("Input datasets do not have identical tasks. Use merge_datasets_diff_task.py instead.")


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
    tasks_idx_to_text: dict[int, str],
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

    for idx in sorted(tasks_idx_to_text.keys()):
        write_task(idx, tasks_idx_to_text[idx], out_root)


def merge_datasets(dataset1: Path, dataset2: Path, output_dataset: Path):
    dataset1 = dataset1.resolve()
    dataset2 = dataset2.resolve()
    output_dataset = output_dataset.resolve()

    if (output_dataset / INFO_PATH).exists():
        raise FileExistsError(f"Output dataset already exists and contains metadata: {output_dataset}")

    info1 = load_info(dataset1)
    info2 = load_info(dataset2)
    _assert_compatible_infos(info1, info2)

    video_keys = _get_video_keys_from_info(info1)

    tasks1, _ = load_tasks(dataset1)
    tasks2, _ = load_tasks(dataset2)
    _assert_same_tasks(tasks1, tasks2)

    episodes1 = load_episodes(dataset1)
    episodes2 = load_episodes(dataset2)
    stats1 = load_episodes_stats(dataset1)
    stats2 = load_episodes_stats(dataset2)

    ds1_ep_order = [episodes1[k] for k in sorted(episodes1.keys())]
    ds2_ep_order = [episodes2[k] for k in sorted(episodes2.keys())]

    chunks_size = info1["chunks_size"]
    total_frames = 0
    new_episodes_records: list[dict[str, Any]] = []
    episodes_stats_serialized: list[tuple[int, dict]] = []

    (output_dataset / "meta").mkdir(parents=True, exist_ok=True)
    (output_dataset / "data").mkdir(parents=True, exist_ok=True)
    if video_keys:
        (output_dataset / "videos").mkdir(parents=True, exist_ok=True)

    new_ep_idx = 0
    running_index_offset = 0

    def process_source(
        src_root: Path,
        src_info: dict,
        src_eps_in_order: list[dict[str, Any]],
        src_stats_map: dict[int, dict],
    ):
        nonlocal new_ep_idx, running_index_offset, total_frames, new_episodes_records, episodes_stats_serialized

        for ep in src_eps_in_order:
            old_ep_idx = int(ep["episode_index"]) if isinstance(ep, dict) else int(ep.episode_index)
            ep_length = int(ep["length"]) if isinstance(ep, dict) else int(ep.length)
            ep_tasks = ep["tasks"] if isinstance(ep, dict) else ep.tasks

            old_parquet_rel = _format_data_path(src_info, old_ep_idx)
            old_parquet_path = src_root / old_parquet_rel
            if not old_parquet_path.is_file():
                raise FileNotFoundError(f"Missing parquet: {old_parquet_path}")

            table = pq.read_table(old_parquet_path)
            table = _update_table_columns(
                table,
                new_episode_index=new_ep_idx,
                index_offset=running_index_offset,
            )

            new_chunk = new_ep_idx // chunks_size
            new_parquet_rel = info1["data_path"].format(episode_chunk=new_chunk, episode_index=new_ep_idx)
            new_parquet_path = output_dataset / new_parquet_rel
            new_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, new_parquet_path)

            for vkey in video_keys:
                old_video_rel = _format_video_path(src_info, old_ep_idx, vkey)
                old_video_path = src_root / old_video_rel
                if not old_video_path.is_file():
                    raise FileNotFoundError(f"Missing video file: {old_video_path}")
                new_video_rel = info1["video_path"].format(episode_chunk=new_chunk, video_key=vkey, episode_index=new_ep_idx)
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

            if old_ep_idx not in src_stats_map:
                raise KeyError(f"Missing episode stats for episode {old_ep_idx} in {src_root}")
            episodes_stats_serialized.append((new_ep_idx, src_stats_map[old_ep_idx]))

            total_frames += ep_length
            running_index_offset += ep_length
            new_ep_idx += 1

    process_source(dataset1, info1, ds1_ep_order, stats1)
    process_source(dataset2, info2, ds2_ep_order, stats2)

    total_episodes = new_ep_idx
    total_chunks = math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0
    new_info = {
        "codebase_version": info1["codebase_version"],
        "robot_type": info1.get("robot_type"),
        "total_episodes": total_episodes,
        "total_frames": int(total_frames),
        "total_tasks": len(tasks1),
        "total_videos": len(video_keys) * total_episodes,
        "total_chunks": total_chunks,
        "chunks_size": chunks_size,
        "fps": info1["fps"],
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": info1["data_path"],
        "video_path": info1.get("video_path"),
        "features": info1["features"],
    }

    _write_meta(output_dataset, new_info, new_episodes_records, episodes_stats_serialized, tasks1)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge two LeRobot datasets with identical tasks.")
    parser.add_argument("dataset1", type=str, help="Path to first dataset root (contains meta/, data/, videos/)")
    parser.add_argument("dataset2", type=str, help="Path to second dataset root (contains meta/, data/, videos/)")
    parser.add_argument("output_dataset", type=str, help="Path to output dataset root (will be created)")
    return parser.parse_args()


def main():
    args = parse_args()
    merge_datasets(Path(args.dataset1), Path(args.dataset2), Path(args.output_dataset))


if __name__ == "__main__":
    main()


