import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go


def calculate_gini(array: np.ndarray) -> float:
    """Calculate the Gini coefficient of a numpy array."""
    x = array.astype(float).flatten()
    if x.size == 0:
        return 0.0
    if np.amin(x) < 0:
        x -= np.amin(x)
    x += 1e-8
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))


def calculate_entropy(array: np.ndarray) -> float:
    """Calculate the normalized Shannon entropy of a numpy array."""
    x = array.astype(float).flatten()
    if x.size == 0:
        return 0.0
    total = np.sum(x)
    if total == 0:
        return 0.0
    probs = x / total
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(array.size) if array.size > 0 else 0.0
    if max_entropy == 0:
        return 0.0
    return float(entropy / max_entropy)


def extract_task_slots(module_dict: Dict) -> Tuple[str, Dict[int, int]]:
    """
    Extract task name and slot->count mapping from a module dict.

    Expected shapes:
    - {"value_slot_0": {...}, ...}
    - {"task_X": {"value_slot_0": {...}, ...}}
    """
    keys = list(module_dict.keys())
    if not keys:
        raise ValueError("Module dict is empty.")

    if any(k.startswith("value_slot_") for k in keys):
        task_name = "unknown_task"
        slot_dict = module_dict
    else:
        task_like = [k for k in keys if k.startswith("task_")]
        if len(task_like) != 1:
            raise ValueError(
                "Could not uniquely determine task key in module dict. "
                f"Keys: {keys}"
            )
        task_name = task_like[0]
        slot_dict = module_dict[task_name]

    slots: Dict[int, int] = {}
    for slot_key, usage in slot_dict.items():
        if not slot_key.startswith("value_slot_"):
            continue
        try:
            idx = int(slot_key.replace("value_slot_", ""))
        except ValueError:
            continue
        count = int(usage.get("total_accesses", 0))
        slots[idx] = count

    return task_name, slots


def load_per_task_modules(
    directory: Path,
) -> Tuple[List[str], Dict[str, List[np.ndarray]], Dict[str, int]]:
    """
    Load all JSONs in a directory and return, for each module, aligned per-task arrays.

    Returns:
        task_names: global ordered list of task names across all files
        module_arrays: mapping module_name -> list of arrays (one per task in task_names)
        module_n_slots: mapping module_name -> number of slots for that module
    """
    json_files = sorted(p for p in directory.glob("*.json") if p.is_file())
    if not json_files:
        raise ValueError(f"No .json files found in {directory}")

    modules_slots: Dict[str, Dict[str, Dict[int, int]]] = {}
    modules_max_slot: Dict[str, int] = {}
    all_task_names: set[str] = set()

    for path in json_files:
        with path.open("r") as f:
            data = json.load(f)

        if "per_module" not in data:
            raise ValueError(f"'per_module' missing in {path}")

        for module_name, module_dict in data["per_module"].items():
            task_name, slots = extract_task_slots(module_dict)
            if task_name == "unknown_task":
                task_name = path.stem

            all_task_names.add(task_name)

            if module_name not in modules_slots:
                modules_slots[module_name] = {}
                modules_max_slot[module_name] = -1

            modules_slots[module_name][task_name] = slots

            if slots:
                max_slot_here = max(slots.keys())
                if max_slot_here > modules_max_slot[module_name]:
                    modules_max_slot[module_name] = max_slot_here

    if not modules_slots:
        raise ValueError("No module slot data collected from any JSON file.")

    task_names = sorted(all_task_names)

    module_arrays: Dict[str, List[np.ndarray]] = {}
    module_n_slots: Dict[str, int] = {}

    for module_name, task_slot_map in modules_slots.items():
        max_slot = modules_max_slot.get(module_name, -1)
        if max_slot < 0:
            continue
        n_slots = max_slot + 1
        module_n_slots[module_name] = n_slots

        arrays: List[np.ndarray] = []
        for task in task_names:
            slots = task_slot_map.get(task, {})
            arr = np.zeros(n_slots, dtype=float)
            for idx, val in slots.items():
                if 0 <= idx < n_slots:
                    arr[idx] = val
            arrays.append(arr)

        module_arrays[module_name] = arrays

    if not module_arrays:
        raise ValueError("No usable module arrays could be constructed.")

    return task_names, module_arrays, module_n_slots


def load_global_modules(
    json_path: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Load global per-module slot usage from a single JSON file."""
    with json_path.open("r") as f:
        data = json.load(f)

    if "per_module" not in data:
        raise ValueError(f"'per_module' missing in {json_path}")

    global_arrays: Dict[str, np.ndarray] = {}
    global_n_slots: Dict[str, int] = {}

    for module_name, module_dict in data["per_module"].items():
        slots: Dict[int, int] = {}
        for slot_key, usage in module_dict.items():
            if not isinstance(usage, dict):
                continue
            if not slot_key.startswith("value_slot_"):
                continue
            try:
                idx = int(slot_key.replace("value_slot_", ""))
            except ValueError:
                continue
            count = int(usage.get("total_accesses", 0))
            slots[idx] = count

        if not slots:
            continue

        max_slot = max(slots.keys())
        n_slots = max_slot + 1
        arr = np.zeros(n_slots, dtype=float)
        for idx, val in slots.items():
            if 0 <= idx < n_slots:
                arr[idx] = val

        global_arrays[module_name] = arr
        global_n_slots[module_name] = n_slots

    if not global_arrays:
        raise ValueError(f"No usable global module arrays could be constructed from {json_path}")

    return global_arrays, global_n_slots


def determine_grid_side(global_n_slots: int, prefer_side: int | None) -> int:
    """Determine the side length of the 2D grid."""
    if prefer_side is not None:
        return int(prefer_side)
    side = int(np.ceil(np.sqrt(global_n_slots)))
    return side


def compute_stats(array: np.ndarray) -> Tuple[float, int, int, float, float, float]:
    """Return (total_accesses, n_slots, unique_slots, gini, entropy, sparsity_pct)."""
    if array.size == 0:
        return 0.0, 0, 0, 0.0, 0.0, 0.0
    total = float(np.sum(array))
    n_slots = int(array.size)
    unique_slots = int(np.sum(array > 0))
    gini = calculate_gini(array)
    ent = calculate_entropy(array)
    zeros = float(np.sum(array == 0))
    sparsity = (zeros / n_slots) * 100.0
    return total, n_slots, unique_slots, gini, ent, sparsity


def compute_iou_matrix(task_arrays: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise intersection-over-union between task slot-usage vectors.
    Intersection = sum(min(a, b)), union = sum(max(a, b)) over slots.
    """
    n = len(task_arrays)
    iou = np.zeros((n, n), dtype=float)
    for i in range(n):
        a = task_arrays[i].astype(float)
        for j in range(i, n):
            b = task_arrays[j].astype(float)
            inter = np.minimum(a, b).sum()
            union = np.maximum(a, b).sum()
            val = float(inter / union) if union > 0 else 0.0
            iou[i, j] = val
            iou[j, i] = val
    return iou


def create_full_figure(
    module_names: List[str],
    task_names: List[str],
    global_arrays: Dict[str, np.ndarray],
    global_n_slots: Dict[str, int],
    module_arrays: Dict[str, List[np.ndarray]],
    module_n_slots: Dict[str, int],
    side: int,
) -> go.Figure:
    """Create a Plotly figure with module dropdown, global + per-task heatmaps and IoU matrix."""
    n_tasks = len(task_names)
    grid_size = side * side

    has_any_global = bool(global_arrays)
    has_any_tasks = n_tasks > 0

    colorscale = [
        [0.0, "rgb(0,0,0)"],
        [1.0, "rgb(253,231,37)"],
    ]

    task_cols = 3
    task_rows = int(np.ceil(n_tasks / task_cols)) if n_tasks > 0 else 0

    heatmap_px = side * 2
    stat_height_px = 40
    gap_px = 20
    margin_top = 120
    margin_bottom = 50
    margin_left = 60
    margin_right = 120

    global_section_height = heatmap_px + stat_height_px + gap_px if has_any_global else 0
    iou_section_height = n_tasks * 24 + stat_height_px + gap_px if has_any_tasks else 0
    task_section_height = (
        task_rows * (heatmap_px + stat_height_px + gap_px) if has_any_tasks else 0
    )

    total_content_height = global_section_height + iou_section_height + task_section_height
    fig_height = total_content_height + margin_top + margin_bottom
    fig_width = task_cols * (heatmap_px + gap_px) + margin_left + margin_right

    fig = go.Figure()

    module_max_log: Dict[str, float] = {}
    module_traces: Dict[str, List[go.Heatmap]] = {}
    module_layouts: Dict[str, dict] = {}
    module_annotations: Dict[str, List[dict]] = {}

    for m_idx, module_name in enumerate(module_names):
        global_arr = global_arrays.get(module_name)
        task_arrs = module_arrays.get(module_name, [])

        all_arrs: List[np.ndarray] = []
        if global_arr is not None:
            all_arrs.append(global_arr)
        all_arrs.extend(task_arrs)
        padded_arrays: List[np.ndarray] = []
        transformed_arrays: List[np.ndarray] = []
        max_log_val = 0.0

        for arr in all_arrs:
            padded = np.zeros(grid_size, dtype=float)
            size = min(arr.size, grid_size)
            if size > 0:
                padded[:size] = arr[:size]
            padded_arrays.append(padded)
            transformed = np.log1p(padded)
            transformed_arrays.append(transformed)
            if transformed.max() > max_log_val:
                max_log_val = float(transformed.max())

        if max_log_val <= 0.0:
            max_log_val = 1.0
        module_max_log[module_name] = max_log_val

        traces: List[go.Heatmap] = []
        layout_axes: dict = {}
        annotations: List[dict] = []
        axis_counter = 1

        def axis_name(n: int) -> Tuple[str, str]:
            if n == 1:
                return "x", "y"
            return f"x{n}", f"y{n}"

        def axis_key(n: int) -> Tuple[str, str]:
            if n == 1:
                return "xaxis", "yaxis"
            return f"xaxis{n}", f"yaxis{n}"

        y_cursor = 1.0 - (margin_top / fig_height)

        if global_arr is not None:
            global_y_top = y_cursor
            global_y_bottom = y_cursor - (heatmap_px / fig_height)
            global_x_left = margin_left / fig_width
            global_x_right = global_x_left + (heatmap_px / fig_width)

            xax, yax = axis_name(axis_counter)
            xkey, ykey = axis_key(axis_counter)
            axis_counter += 1

            layout_axes[xkey] = dict(
                domain=[global_x_left, global_x_right],
                anchor=yax,
                showticklabels=False,
                showgrid=False,
            )
            layout_axes[ykey] = dict(
                domain=[global_y_bottom, global_y_top],
                anchor=xax,
                showticklabels=False,
                showgrid=False,
                scaleanchor=xax,
                scaleratio=1,
            )

            padded_global = padded_arrays[0]
            transformed_global = transformed_arrays[0].reshape(side, side)
            counts_global = padded_global.reshape(side, side)
            customdata_global = np.arange(grid_size, dtype=int).reshape(side, side)

            global_heatmap = go.Heatmap(
                z=transformed_global,
                x=np.arange(side),
                y=np.arange(side),
                coloraxis="coloraxis",
                visible=(m_idx == 0),
                customdata=customdata_global,
                text=counts_global,
                hovertemplate=(
                    "Global | Slot %{customdata}<br>"
                    "Accesses %{text:.0f}<br>"
                    "log1p %{z:.3f}<extra></extra>"
                ),
                xaxis=xax,
                yaxis=yax,
            )
            traces.append(global_heatmap)

            g_total, g_slots, g_unique, g_gini, g_ent, g_sparsity = compute_stats(global_arr)
            stat_y = global_y_bottom - (stat_height_px * 0.5 / fig_height)
            annotations.append(
                dict(
                    x=(global_x_left + global_x_right) / 2,
                    y=stat_y,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    align="center",
                    font=dict(size=12),
                    text=(
                        f"<b>Global</b> | Total Accesses {int(g_total):,} | Unique Slots {g_unique:,} / {g_slots:,}<br>"
                        f"Gini {g_gini:.3f} | Ent {g_ent:.3f} | Unused {g_sparsity:.1f}%"
                    ),
                )
            )

            y_cursor = global_y_bottom - (stat_height_px + gap_px) / fig_height

        if n_tasks > 0 and task_arrs:
            iou_matrix = compute_iou_matrix(task_arrs)
            iou_size_px = n_tasks * 24
            iou_y_top = y_cursor
            iou_y_bottom = y_cursor - (iou_size_px / fig_height)
            iou_x_left = margin_left / fig_width
            iou_x_right = iou_x_left + (iou_size_px / fig_width)

            xax_iou, yax_iou = axis_name(axis_counter)
            xkey_iou, ykey_iou = axis_key(axis_counter)
            axis_counter += 1

            layout_axes[xkey_iou] = dict(
                domain=[iou_x_left, iou_x_right],
                anchor=yax_iou,
                showticklabels=True,
                tickvals=list(range(n_tasks)),
                ticktext=[t.replace("task_", "").replace("memory_usage_", "") for t in task_names],
                tickfont=dict(size=8),
                showgrid=False,
            )
            layout_axes[ykey_iou] = dict(
                domain=[iou_y_bottom, iou_y_top],
                anchor=xax_iou,
                showticklabels=True,
                tickvals=list(range(n_tasks)),
                ticktext=[t.replace("task_", "").replace("memory_usage_", "") for t in task_names],
                tickfont=dict(size=8),
                showgrid=False,
                scaleanchor=xax_iou,
                scaleratio=1,
            )

            iou_heatmap = go.Heatmap(
                z=iou_matrix,
                x=list(range(n_tasks)),
                y=list(range(n_tasks)),
                coloraxis="coloraxis2",
                visible=(m_idx == 0),
                hovertemplate="Task %{y} vs %{x}<br>IoU=%{z:.3f}<extra></extra>",
                xaxis=xax_iou,
                yaxis=yax_iou,
            )
            traces.append(iou_heatmap)

            iou_stat_y = iou_y_bottom - (stat_height_px * 0.5 / fig_height)
            mean_iou = float(np.mean(iou_matrix[np.triu_indices(n_tasks, k=1)])) if n_tasks > 1 else 0.0
            annotations.append(
                dict(
                    x=(iou_x_left + iou_x_right) / 2,
                    y=iou_stat_y,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    align="center",
                    font=dict(size=12),
                    text=f"<b>Task-Pair IoU</b> | Mean IoU {mean_iou:.3f}",
                )
            )

            y_cursor = iou_y_bottom - (stat_height_px + gap_px) / fig_height

        for t_idx, arr in enumerate(task_arrs):
            row = t_idx // task_cols
            col = t_idx % task_cols

            cell_y_top = y_cursor - row * ((heatmap_px + stat_height_px + gap_px) / fig_height)
            cell_y_bottom = cell_y_top - (heatmap_px / fig_height)
            cell_x_left = (margin_left + col * (heatmap_px + gap_px)) / fig_width
            cell_x_right = cell_x_left + (heatmap_px / fig_width)

            xax_t, yax_t = axis_name(axis_counter)
            xkey_t, ykey_t = axis_key(axis_counter)
            axis_counter += 1

            layout_axes[xkey_t] = dict(
                domain=[cell_x_left, cell_x_right],
                anchor=yax_t,
                showticklabels=False,
                showgrid=False,
            )
            layout_axes[ykey_t] = dict(
                domain=[cell_y_bottom, cell_y_top],
                anchor=xax_t,
                showticklabels=False,
                showgrid=False,
                scaleanchor=xax_t,
                scaleratio=1,
            )

            task_index = t_idx + (1 if global_arr is not None else 0)
            padded_task = padded_arrays[task_index]
            transformed_task = transformed_arrays[task_index].reshape(side, side)
            counts_task = padded_task.reshape(side, side)
            customdata_task = np.arange(grid_size, dtype=int).reshape(side, side)

            task_heatmap = go.Heatmap(
                z=transformed_task,
                x=np.arange(side),
                y=np.arange(side),
                coloraxis="coloraxis",
                visible=(m_idx == 0),
                customdata=customdata_task,
                text=counts_task,
                hovertemplate=(
                    f"{task_names[t_idx]} | Slot " "%{customdata}<br>"
                    "Accesses %{text:.0f}<br>"
                    "log1p %{z:.3f}<extra></extra>"
                ),
                xaxis=xax_t,
                yaxis=yax_t,
            )
            traces.append(task_heatmap)

            t_total, t_slots, t_unique, t_gini, t_ent, t_sparsity = compute_stats(arr)
            task_stat_y = cell_y_bottom - (stat_height_px * 0.5 / fig_height)
            short_name = task_names[t_idx].replace("task_", "").replace("memory_usage_", "")
            annotations.append(
                dict(
                    x=(cell_x_left + cell_x_right) / 2,
                    y=task_stat_y,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    align="center",
                    font=dict(size=11),
                    text=(
                        f"<b>{short_name}</b> | Accesses {int(t_total):,} | Unique {t_unique:,}<br>"
                        f"Gini {t_gini:.3f} | Ent {t_ent:.3f} | Unused {t_sparsity:.1f}%"
                    ),
                )
            )

        module_traces[module_name] = traces
        module_layouts[module_name] = layout_axes
        module_annotations[module_name] = annotations

    for module_name in module_names:
        for trace in module_traces[module_name]:
            fig.add_trace(trace)

    total_traces = len(fig.data)
    trace_offset = 0
    trace_ranges: Dict[str, Tuple[int, int]] = {}
    for module_name in module_names:
        n_traces = len(module_traces[module_name])
        trace_ranges[module_name] = (trace_offset, trace_offset + n_traces)
        trace_offset += n_traces

    buttons = []
    for m_idx, module_name in enumerate(module_names):
        visible = [False] * total_traces
        start, end = trace_ranges[module_name]
        for i in range(start, end):
            visible[i] = True

        update_layout = {
            "title.text": module_name,
            "coloraxis.cmax": module_max_log[module_name],
            "annotations": module_annotations[module_name],
        }
        update_layout.update(module_layouts[module_name])

        buttons.append(
            dict(
                label=module_name,
                method="update",
                args=[{"visible": visible}, update_layout],
            )
        )

    first_module = module_names[0]
    base_layout = dict(
        title=dict(text=first_module),
        coloraxis=dict(
            colorscale=colorscale,
            cmin=0,
            cmax=module_max_log[first_module],
            colorbar=dict(title="log1p(accesses)", x=1.02, len=0.4, y=0.8),
        ),
        coloraxis2=dict(
            colorscale="Blues",
            cmin=0.0,
            cmax=1.0,
            colorbar=dict(title="IoU", x=1.02, len=0.3, y=0.35),
        ),
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.0,
                y=1.02,
                xanchor="left",
                yanchor="bottom",
                direction="down",
                showactive=True,
            )
        ],
        height=fig_height,
        width=fig_width,
        margin=dict(t=margin_top, l=margin_left, r=margin_right, b=margin_bottom),
        template="plotly_white",
        annotations=module_annotations[first_module],
    )
    base_layout.update(module_layouts[first_module])
    fig.update_layout(**base_layout)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualise global and per-task memory slot usage with a shared module "
            "dropdown. For each module, show a global heatmap, task-pair IoU matrix, "
            "and per-task heatmaps in a grid with statistics."
        ),
    )
    parser.add_argument(
        "--global_json",
        "--global-stats",
        "--global_stats",
        dest="global_json",
        type=str,
        default=None,
        help="Optional: path to the global memory_usage.json file.",
    )
    parser.add_argument(
        "--task_json_dir",
        "--task-stats",
        "--task_stats",
        dest="task_json_dir",
        type=str,
        default=None,
        help="Optional: path to a directory containing per-task memory_usage_*.json files.",
    )
    parser.add_argument(
        "--grid-side",
        type=int,
        default=None,
        help=(
            "Optional: side length of the memory grid (e.g. 512). If not set, "
            "inferred as ceil(sqrt(max slot index + 1)) across modules."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="full_memory_usage_viz.html",
        help="Output HTML file for the interactive visualisation.",
    )

    args = parser.parse_args()

    if args.global_json is None and args.task_json_dir is None:
        parser.error(
            "At least one of --global_json/--global-stats or "
            "--task_json_dir/--task-stats must be provided."
        )

    global_arrays: Dict[str, np.ndarray] = {}
    global_n_slots: Dict[str, int] = {}
    task_names: List[str] = []
    module_arrays: Dict[str, List[np.ndarray]] = {}
    module_n_slots: Dict[str, int] = {}

    if args.global_json is not None:
        global_path = Path(args.global_json)
        if not global_path.is_file():
            raise SystemExit(f"Global JSON file not found: {global_path}")
        global_arrays, global_n_slots = load_global_modules(global_path)

    if args.task_json_dir is not None:
        task_dir = Path(args.task_json_dir)
        if not task_dir.is_dir():
            raise SystemExit(f"Per-task JSON directory not found: {task_dir}")
        task_names, module_arrays, module_n_slots = load_per_task_modules(task_dir)

    if global_arrays and module_arrays:
        global_modules = set(global_arrays.keys())
        task_modules = set(module_arrays.keys())
        module_names = sorted(global_modules & task_modules)

        if not module_names:
            raise SystemExit(
                "No overlapping modules between global JSON and per-task JSONs. "
                "Ensure you used matching logs for both inputs."
            )

        max_slots = 0
        for name in module_names:
            max_here = max(global_n_slots.get(name, 0), module_n_slots.get(name, 0))
            if max_here > max_slots:
                max_slots = max_here
    elif global_arrays:
        module_names = sorted(global_arrays.keys())
        max_slots = max(global_n_slots.values())
    else:
        module_names = sorted(module_arrays.keys())
        max_slots = max(module_n_slots.values())

    if max_slots <= 0:
        raise SystemExit("Computed maximum number of slots is non-positive.")

    side = determine_grid_side(max_slots, args.grid_side)

    fig = create_full_figure(
        module_names=module_names,
        task_names=task_names,
        global_arrays=global_arrays,
        global_n_slots=global_n_slots,
        module_arrays=module_arrays,
        module_n_slots=module_n_slots,
        side=side,
    )

    print(f"Saving visualisation to {args.output} ...")
    fig.write_html(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
