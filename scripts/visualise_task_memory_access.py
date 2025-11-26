#!/usr/bin/env python3
"""
Visualize memory slot usage across tasks from sequential training.

Usage:
    python scripts/visualise_task_memory_access.py /path/to/memory_by_task [--module MODULE_NAME]

This creates an interactive HTML visualization where you can hover over tasks
to highlight their slot usage and see overlap with other tasks.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def load_task_slots(memory_dir: Path) -> dict[str, dict[int, dict[int, dict]]]:
    """
    Load memory usage data from all task JSON files.
    
    Returns:
        dict: module_name -> task_id -> slot_id -> {total_accesses, batch_accesses}
    """
    data_by_module: dict[str, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    
    task_files = sorted(memory_dir.glob("memory_usage_task_*.json"))
    if not task_files:
        raise FileNotFoundError(f"No memory_usage_task_*.json files found in {memory_dir}")
    
    for task_file in task_files:
        task_id = int(task_file.stem.split("_")[-1])
        
        with open(task_file) as f:
            data = json.load(f)
        
        for module_name, module_data in data.get("per_module", {}).items():
            for task_key, slot_dict in module_data.items():
                for slot_key, access_info in slot_dict.items():
                    slot_id = int(slot_key.replace("value_slot_", ""))
                    data_by_module[module_name][task_id][slot_id] = access_info
    
    return dict(data_by_module)


def compute_overlap_stats(data_by_module: dict) -> dict[str, dict]:
    """Compute overlap statistics between tasks for each module."""
    stats = {}
    
    for module_name, task_data in data_by_module.items():
        task_ids = sorted(task_data.keys())
        task_slots = {tid: set(task_data[tid].keys()) for tid in task_ids}
        
        module_stats = {
            "task_ids": task_ids,
            "slots_per_task": {tid: len(slots) for tid, slots in task_slots.items()},
            "pairwise_overlap": {},
            "union_size": len(set().union(*task_slots.values())) if task_slots else 0,
        }
        
        for i, t1 in enumerate(task_ids):
            for t2 in task_ids[i+1:]:
                overlap = len(task_slots[t1] & task_slots[t2])
                union = len(task_slots[t1] | task_slots[t2])
                module_stats["pairwise_overlap"][(t1, t2)] = {
                    "overlap": overlap,
                    "union": union,
                    "jaccard": overlap / union if union > 0 else 0,
                }
        
        stats[module_name] = module_stats
    
    return stats


def create_slot_usage_figure(
    data_by_module: dict,
    module_name: str,
    normalize: bool = False,
) -> go.Figure:
    """
    Create interactive bar chart showing slot usage per task.
    
    Each task gets a different color. Hovering highlights that task's bars.
    """
    task_data = data_by_module[module_name]
    task_ids = sorted(task_data.keys())
    
    all_slots = set()
    for tid in task_ids:
        all_slots.update(task_data[tid].keys())
    all_slots = sorted(all_slots)
    
    if not all_slots:
        fig = go.Figure()
        fig.add_annotation(text="No slot data available", x=0.5, y=0.5, showarrow=False)
        return fig
    
    slot_to_idx = {s: i for i, s in enumerate(all_slots)}
    n_slots = len(all_slots)
    
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    ]
    
    fig = go.Figure()
    
    for i, task_id in enumerate(task_ids):
        slots = task_data[task_id]
        
        x_positions = []
        y_values = []
        hover_texts = []
        
        for slot_id, access_info in slots.items():
            x_positions.append(slot_to_idx[slot_id])
            if normalize:
                y_val = access_info["batch_accesses"]
            else:
                y_val = access_info["total_accesses"]
            y_values.append(y_val)
            hover_texts.append(
                f"Task {task_id}<br>"
                f"Slot: {slot_id}<br>"
                f"Total accesses: {access_info['total_accesses']:,}<br>"
                f"Batch accesses: {access_info['batch_accesses']:,}"
            )
        
        if not x_positions:
            continue
        
        sorted_indices = np.argsort(x_positions)
        x_positions = [x_positions[j] for j in sorted_indices]
        y_values = [y_values[j] for j in sorted_indices]
        hover_texts = [hover_texts[j] for j in sorted_indices]
        
        fig.add_trace(go.Bar(
            x=x_positions,
            y=y_values,
            name=f"Task {task_id}",
            marker_color=colors[i % len(colors)],
            opacity=0.7,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts,
        ))
    
    fig.update_layout(
        title=dict(
            text=f"Memory Slot Usage by Task<br><sub>{module_name}</sub>",
            x=0.5,
        ),
        xaxis_title="Slot Index (sorted by slot ID)",
        yaxis_title="Total Accesses" if not normalize else "Batch Accesses",
        barmode="overlay",
        hovermode="closest",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
        template="plotly_white",
    )
    
    n_ticks = min(20, n_slots)
    tick_step = max(1, n_slots // n_ticks)
    tick_vals = list(range(0, n_slots, tick_step))
    tick_texts = [str(all_slots[i]) for i in tick_vals]
    
    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_texts,
        tickangle=45,
    )
    
    return fig


def create_overlap_heatmap(data_by_module: dict, module_name: str) -> go.Figure:
    """Create a heatmap showing pairwise task overlap (Jaccard similarity)."""
    task_data = data_by_module[module_name]
    task_ids = sorted(task_data.keys())
    n_tasks = len(task_ids)
    
    task_slots = {tid: set(task_data[tid].keys()) for tid in task_ids}
    
    overlap_matrix = np.zeros((n_tasks, n_tasks))
    annotations = []
    
    for i, t1 in enumerate(task_ids):
        for j, t2 in enumerate(task_ids):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                intersection = len(task_slots[t1] & task_slots[t2])
                union = len(task_slots[t1] | task_slots[t2])
                jaccard = intersection / union if union > 0 else 0
                overlap_matrix[i, j] = jaccard
            
            annotations.append(dict(
                x=j,
                y=i,
                text=f"{overlap_matrix[i, j]:.2f}",
                showarrow=False,
                font=dict(color="white" if overlap_matrix[i, j] > 0.5 else "black"),
            ))
    
    fig = go.Figure(data=go.Heatmap(
        z=overlap_matrix,
        x=[f"Task {t}" for t in task_ids],
        y=[f"Task {t}" for t in task_ids],
        colorscale="Viridis",
        zmin=0,
        zmax=1,
        hovertemplate="Task %{y} vs Task %{x}<br>Jaccard: %{z:.3f}<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Task Slot Overlap (Jaccard Similarity)<br><sub>{module_name}</sub>",
            x=0.5,
        ),
        annotations=annotations,
        template="plotly_white",
    )
    
    return fig


def create_slot_histogram(data_by_module: dict, module_name: str) -> go.Figure:
    """Create histogram showing how many tasks share each slot."""
    task_data = data_by_module[module_name]
    task_ids = sorted(task_data.keys())
    
    slot_task_count = defaultdict(int)
    for tid in task_ids:
        for slot_id in task_data[tid].keys():
            slot_task_count[slot_id] += 1
    
    count_distribution = defaultdict(int)
    for slot_id, count in slot_task_count.items():
        count_distribution[count] += 1
    
    x_vals = sorted(count_distribution.keys())
    y_vals = [count_distribution[x] for x in x_vals]
    
    fig = go.Figure(data=go.Bar(
        x=[f"{x} task{'s' if x > 1 else ''}" for x in x_vals],
        y=y_vals,
        marker_color="#1f77b4",
        hovertemplate="Shared by %{x}<br>%{y} slots<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Slot Sharing Distribution<br><sub>{module_name}</sub>",
            x=0.5,
        ),
        xaxis_title="Number of Tasks Sharing Slot",
        yaxis_title="Number of Slots",
        template="plotly_white",
    )
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize memory slot usage across tasks from sequential training."
    )
    parser.add_argument(
        "memory_dir",
        type=Path,
        help="Path to memory_by_task directory containing memory_usage_task_*.json files",
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        help="Specific module to visualize. If not specified, creates separate plots for each.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML file path. Defaults to memory_dir/visualization.html",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Use batch_accesses instead of total_accesses for y-axis",
    )
    args = parser.parse_args()
    
    if not args.memory_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.memory_dir}")
    
    print(f"Loading data from {args.memory_dir}...")
    data_by_module = load_task_slots(args.memory_dir)
    
    if not data_by_module:
        raise ValueError("No module data found in the JSON files")
    
    print(f"Found {len(data_by_module)} modules:")
    for module_name in data_by_module:
        task_ids = sorted(data_by_module[module_name].keys())
        print(f"  {module_name}: tasks {task_ids}")
    
    stats = compute_overlap_stats(data_by_module)
    print("\nOverlap Statistics:")
    for module_name, module_stats in stats.items():
        print(f"\n{module_name}:")
        print(f"  Total unique slots across all tasks: {module_stats['union_size']}")
        print(f"  Slots per task: {module_stats['slots_per_task']}")
        for (t1, t2), overlap_info in module_stats["pairwise_overlap"].items():
            print(f"  Task {t1} vs {t2}: {overlap_info['overlap']}/{overlap_info['union']} = {overlap_info['jaccard']:.1%} overlap")
    
    modules_to_plot = [args.module] if args.module else list(data_by_module.keys())
    
    n_modules = len(modules_to_plot)
    n_plots_per_module = 3
    
    fig = make_subplots(
        rows=n_modules * n_plots_per_module,
        cols=1,
        subplot_titles=[
            title
            for module in modules_to_plot
            for title in [
                f"Slot Usage: {module.split('.')[-1]}",
                f"Overlap Heatmap: {module.split('.')[-1]}",
                f"Sharing Distribution: {module.split('.')[-1]}",
            ]
        ],
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25] * n_modules,
    )
    
    for i, module_name in enumerate(modules_to_plot):
        slot_fig = create_slot_usage_figure(data_by_module, module_name, args.normalize)
        overlap_fig = create_overlap_heatmap(data_by_module, module_name)
        hist_fig = create_slot_histogram(data_by_module, module_name)
        
        row_base = i * n_plots_per_module
        
        for trace in slot_fig.data:
            fig.add_trace(trace, row=row_base + 1, col=1)
        
        for trace in overlap_fig.data:
            fig.add_trace(trace, row=row_base + 2, col=1)
        
        for trace in hist_fig.data:
            fig.add_trace(trace, row=row_base + 3, col=1)
    
    fig.update_layout(
        height=400 * n_modules * n_plots_per_module,
        title=dict(
            text="Memory Slot Usage Analysis",
            x=0.5,
            font=dict(size=20),
        ),
        showlegend=True,
        template="plotly_white",
        barmode="overlay",
    )
    
    output_path = args.output or (args.memory_dir / "visualization.html")
    fig.write_html(str(output_path))
    print(f"\nVisualization saved to: {output_path}")
    
    for module_name in modules_to_plot:
        # Create a unique filename using more of the path
        module_short = module_name.replace("model.vlm_with_expert.", "").replace(".", "_")
        module_output = args.memory_dir / f"slot_usage_{module_short}.html"
        slot_fig = create_slot_usage_figure(data_by_module, module_name, args.normalize)
        slot_fig.update_layout(height=600, width=1200)
        slot_fig.write_html(str(module_output))
        print(f"Slot usage plot saved to: {module_output}")


if __name__ == "__main__":
    main()

