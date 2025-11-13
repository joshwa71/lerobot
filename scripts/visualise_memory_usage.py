import argparse
import json
import os
import re
from typing import Dict, Any, Tuple, List

import matplotlib.pyplot as plt


SLOT_KEY_PATTERN = re.compile(r"^value_slot_(\d+)$")


def load_memory_usage(json_path: str) -> Dict[str, Dict[int, Dict[str, int]]]:
    """
    Load memory usage from a JSON file and return:
      { module_name: { slot_index: { 'total_accesses': int, 'batch_accesses': int } } }
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    per_module = data.get("per_module", data)

    modules: Dict[str, Dict[int, Dict[str, int]]] = {}
    for module_name, module_stats in per_module.items():
        if not isinstance(module_stats, dict):
            continue
        slots: Dict[int, Dict[str, int]] = {}
        for key, slot_stats in module_stats.items():
            match = SLOT_KEY_PATTERN.match(str(key))
            if match is None:
                continue
            if not isinstance(slot_stats, dict):
                continue
            slot_idx = int(match.group(1))
            total = int(slot_stats.get("total_accesses", 0))
            batch = int(slot_stats.get("batch_accesses", 0))
            slots[slot_idx] = {"total_accesses": total, "batch_accesses": batch}
        if slots:
            modules[module_name] = slots

    if not modules:
        raise ValueError("No slot usage found. Expected keys like 'value_slot_<idx>' under 'per_module'.")

    return modules


def plot_per_module_usage(
    modules: Dict[str, Dict[int, Dict[str, int]]],
    metric: str = "total_accesses",
) -> None:
    """
    For each module, plot a bar chart of slot index vs. the given metric.
    """
    module_items: List[Tuple[str, Dict[int, Dict[str, int]]]] = list(modules.items())
    module_items.sort(key=lambda x: x[0])

    for module_name, slots in module_items:
        slot_indices = sorted(slots.keys())
        values = [int(slots[i].get(metric, 0)) for i in slot_indices]

        plt.figure(figsize=(12, 4))
        plt.bar(slot_indices, values, width=0.9)
        plt.title(f"{module_name} — {metric.replace('_', ' ')}")
        plt.xlabel("Slot index")
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise memory slot usage per layer/module from memory_usage.json"
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to memory_usage.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = load_memory_usage(args.json_path)
    plot_per_module_usage(modules, metric="total_accesses")
    plt.show()


if __name__ == "__main__":
    main()

