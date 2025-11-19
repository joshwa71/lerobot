#!/usr/bin/env python

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _find_total_json(run_dir: Path) -> Path | None:
    """
    Try to find a memory_usage.json alongside a run directory.
    Looks under <run_dir>/pretrained_model/ and <run_dir>/.
    """
    candidates = [
        run_dir / "pretrained_model" / "memory_usage.json",
        run_dir / "memory_usage.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_totals(total_json: Path, module_key: str) -> Dict[int, int]:
    with open(total_json, "r") as f:
        data = json.load(f)
    per_module = data.get("per_module", {})
    module_dict = per_module.get(module_key, {})
    slot_to_total: Dict[int, int] = {}
    for slot_name, stats in module_dict.items():
        if not isinstance(stats, dict):
            continue
        if not slot_name.startswith("value_slot_"):
            continue
        s = int(slot_name.split("_")[-1])
        slot_to_total[s] = int(stats.get("total_accesses", 0))
    return slot_to_total


def _load_per_task(run_dir: Path) -> Tuple[Dict[str, Dict[int, Dict[int, int]]], List[int]]:
    """
    Returns:
      per_module: {module_key: {task_id: {slot_idx: total_accesses}}}
      task_ids: sorted list of task ids found
    """
    per_module: Dict[str, Dict[int, Dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    task_ids: set[int] = set()
    src_dir = run_dir / "memory_by_task"
    if not src_dir.exists():
        return {}, []
    for fp in sorted(src_dir.glob("memory_usage_task_*.json")):
        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        per_mod = data.get("per_module", {})
        for module_key, by_task in per_mod.items():
            if not isinstance(by_task, dict):
                continue
            for task_key, slot_dict in by_task.items():
                if not task_key.startswith("task_"):
                    continue
                tid = int(task_key.split("_")[-1])
                task_ids.add(tid)
                dst = per_module[module_key].setdefault(tid, {})
                for slot_name, stats in slot_dict.items():
                    if not isinstance(stats, dict):
                        continue
                    if not slot_name.startswith("value_slot_"):
                        continue
                    s = int(slot_name.split("_")[-1])
                    dst[s] = int(stats.get("total_accesses", 0)) + int(dst.get(s, 0))
    return per_module, sorted(task_ids)


def _sanitize_module_name(module_key: str) -> str:
    return module_key.replace(".", "_").replace("/", "_")


def plot_overlay(
    module_key: str,
    totals: Dict[int, int],
    per_task: Dict[int, Dict[int, int]],
    task_ids: List[int],
    topk: int,
    out_path: Path,
    figsize: Tuple[int, int] = (14, 6),
    show: bool = False,
):
    # Determine ranking by totals; if totals are missing, compute from per_task sum
    if not totals:
        totals = {}
        for tid, ctr in per_task.items():
            for s, v in ctr.items():
                totals[s] = totals.get(s, 0) + int(v)
    if not totals:
        raise RuntimeError("No totals available to rank slots.")
    ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:topk]
    slots = [s for s, _ in ranked]
    totals_vec = np.array([totals.get(s, 0) for s in slots], dtype=np.float64)

    # Build per-task matrix (tasks x slots)
    task_ids_sorted = sorted(task_ids)
    M = np.zeros((len(task_ids_sorted), len(slots)), dtype=np.float64)
    for i, tid in enumerate(task_ids_sorted):
        ctr = per_task.get(tid, {})
        for j, s in enumerate(slots):
            M[i, j] = ctr.get(s, 0)

    # Plot
    plt.figure(figsize=figsize)
    x = np.arange(len(slots))
    # Total background (bars)
    plt.bar(x, totals_vec, color="#A0A0A0", alpha=0.5, label="Total accesses")

    # Stacked per-task overlay
    bottom = np.zeros_like(totals_vec)
    cmap = plt.get_cmap("tab20")
    for i, tid in enumerate(task_ids_sorted):
        color = cmap(i % 20)
        plt.bar(x, M[i], bottom=bottom, color=color, alpha=0.9, linewidth=0.0, label=f"task {tid}")
        bottom += M[i]

    # Outline total on top for clarity
    plt.plot(x, totals_vec, color="k", linewidth=1.5, label="Total (outline)")

    plt.title(f"Memory slot accesses for {module_key} (top-{len(slots)})")
    plt.xlabel("Slot (ranked)")
    plt.ylabel("Access count")
    # Avoid cluttered xticks
    plt.xticks([], [])
    # Build a compact legend
    max_legend = 12
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(labels) > max_legend:
        # keep total + outline + first (max_legend-2) tasks
        keep = [0] + list(range(1, max_legend - 1)) + [len(labels) - 1]
        handles = [handles[i] for i in keep]
        labels = [labels[i] for i in keep]
    plt.legend(handles, labels, ncol=2, fontsize=8)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize per-task memory slot accesses over total.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing memory_by_task/")
    parser.add_argument("--module", type=str, default=None, help="Module key, e.g., model.vlm_with_expert.lm_expert.layers.3")
    parser.add_argument("--topk", type=int, default=200, help="Top-K slots to visualize by total accesses")
    parser.add_argument("--total-json", type=str, default=None, help="Optional path to memory_usage.json for totals")
    parser.add_argument("--list-modules", action="store_true", help="List available modules and exit")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    parser.add_argument("--figsize", type=str, default="14,6", help="Figure size W,H")
    parser.add_argument("--output", type=str, default=None, help="Optional output image path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    per_module, task_ids = _load_per_task(run_dir)
    if args.list_modules:
        mods = sorted(per_module.keys())
        if not mods:
            print("No per-task memory files found under memory_by_task/")
        else:
            print("Available modules:")
            for m in mods:
                print(m)
        return

    if not per_module:
        raise RuntimeError("No per-task memory files found under memory_by_task/.")

    module_key = args.module or sorted(per_module.keys())[0]
    if module_key not in per_module:
        available = ", ".join(sorted(per_module.keys()))
        raise ValueError(f"Module '{module_key}' not found. Available: {available}")

    if args.total_json:
        total_json = Path(args.total_json)
        if not total_json.exists():
            raise FileNotFoundError(f"total-json not found: {total_json}")
    else:
        total_json = _find_total_json(run_dir)

    if total_json is not None:
        totals = _load_totals(total_json, module_key)
    else:
        totals = {}  # will be derived from per-task sums

    W, H = (float(x) for x in args.figsize.split(","))
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = run_dir / "viz"
        out_path = out_dir / f"{_sanitize_module_name(module_key)}_top{int(args.topk)}.png"

    plot_overlay(
        module_key=module_key,
        totals=totals,
        per_task=per_module[module_key],
        task_ids=task_ids,
        topk=int(args.topk),
        out_path=out_path,
        figsize=(W, H),
        show=bool(args.show),
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


