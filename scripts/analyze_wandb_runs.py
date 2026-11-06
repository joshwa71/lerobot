#!/usr/bin/env python3
"""Ad-hoc analysis of the pi05 libero_goal sequential runs.

Compares:
  - pretrain (libero_minus_goal, knn=36)
  - sequential top_t=512  (Entry 18 baseline)
  - sequential top_t=1536 (new run)
"""
import sys
import numpy as np
sys.path.insert(0, "/home/josh/lerobot/scripts")
from parse_wandb import WandbRun

BASE = "/home/josh/lerobot/outputs/train"
RUNS = {
    "pretrain":   f"{BASE}/libero_minus_goal_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_30k/wandb",
    "seq_t512":   f"{BASE}/libero_goal_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_30k/wandb",
    "seq_t1536":  f"{BASE}/libero_goal_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_30k_top_t_1536/wandb",
}

def last_val(run, key):
    vals = run.get_metric(key)
    return vals[-1][1] if vals else None

def mean_val(run, key, lo=None, hi=None):
    vals = run.get_metric(key)
    if not vals:
        return None
    xs = [v for (s, v) in vals if (lo is None or s >= lo) and (hi is None or s <= hi) and v is not None]
    return float(np.mean(xs)) if xs else None

def at_steps(run, key, steps, tol=200):
    """Value of key closest to each step in `steps`."""
    vals = run.get_metric(key)
    out = []
    for target in steps:
        best = None; bestd = 1e18
        for (s, v) in vals:
            d = abs(s - target)
            if d < bestd:
                bestd = d; best = v
        out.append(best if bestd <= tol else None)
    return out

runs = {name: WandbRun.from_wandb_dir(path) for name, path in RUNS.items()}
for name, r in runs.items():
    steps = r.steps()
    print(f"{name:12s}: {r.run_id}  records={len(r.history)} steps={steps[0]}..{steps[-1]}")
print()

# ---- Pretrain summary ----
pre = runs["pretrain"]
print("="*70)
print("PRETRAIN (libero_minus_goal, knn=36)  final/mean metrics")
print("="*70)
for key in ["train/mse_loss", "train/loss", "train/gate_mean",
            "train/gate_mean_L8","train/gate_mean_L10","train/gate_mean_L12","train/gate_mean_L14",
            "train/mem_usage_effnum_mean", "train/mem_used_frac_mean",
            "train/mem_usage_top1_share_mean",
            "train/routing_intra_task_support_mean", "train/routing_intra_task_entropy_mean",
            "train/routing_task_entropy_mean", "train/contrastive_loss_mean"]:
    lv = mean_val(pre, key, lo=28000)  # mean over last ~2k steps
    print(f"  {key:42s}: {lv}")
print()

# ---- Sequential comparison at task boundaries ----
TASK_STEPS = list(range(3000, 30001, 3000))
ORDER = [8,9,3,6,2,5,7,1,4,0]  # env id per task slot

for metric in ["train/mse_loss", "train/gate_mean", "train/mem_usage_effnum_mean",
               "train/mem_used_frac_mean", "train/mem_usage_top1_share_mean",
               "memory_iou/all_modules_mean",
               "memory_iou/layers.8","memory_iou/layers.10","memory_iou/layers.12","memory_iou/layers.14",
               "train/routing_intra_task_support_mean",
               "train/routing_intra_task_entropy_mean",
               "train/routing_task_entropy_mean",
               "eval/avg_pc_success_seen"]:
    print(f"--- {metric} (at each 3k task boundary) ---")
    print("           " + "".join(f"{s//1000:>7d}k" for s in TASK_STEPS))
    for name in ["seq_t512", "seq_t1536"]:
        vals = at_steps(runs[name], metric, TASK_STEPS)
        cells = "".join((f"{v:8.4f}" if isinstance(v,(int,float)) else f"{'--':>8}") for v in vals)
        print(f"  {name:9s}{cells}")
    print()

# ---- Per-layer gate & effnum, final ----
print("="*70)
print("PER-LAYER final (last step) gate / effnum / used_frac / top1")
print("="*70)
for name in ["seq_t512", "seq_t1536"]:
    r = runs[name]
    print(f"\n[{name}]")
    for L in [8,10,12,14]:
        g = last_val(r, f"train/gate_mean_L{L}")
        e = last_val(r, f"train/mem_usage_effnum_L{L}")
        uf = last_val(r, f"train/mem_used_frac_L{L}")
        t1 = last_val(r, f"train/mem_usage_top1_share_L{L}")
        iou = last_val(r, f"memory_iou/layers.{L}")
        gs = f"{g:.4f}" if g is not None else "NA"
        es = f"{e:.1f}" if e is not None else "NA"
        ufs = f"{uf:.4f}" if uf is not None else "NA"
        t1s = f"{t1:.5f}" if t1 is not None else "NA"
        ious = f"{iou:.4f}" if iou is not None else "NA"
        print(f"  L{L:2d}: gate={gs}  effnum={es}  used_frac={ufs}  top1={t1s}  read_IoU={ious}")

# ---- Mean gate / IoU over whole sequential run ----
print()
print("="*70)
print("RUN-MEAN (all steps) gate_mean, IoU, effnum")
print("="*70)
for name in ["seq_t512", "seq_t1536"]:
    r = runs[name]
    print(f"  {name:10s}: gate={mean_val(r,'train/gate_mean'):.4f}  "
          f"IoU={mean_val(r,'memory_iou/all_modules_mean'):.4f}  "
          f"effnum={mean_val(r,'train/mem_usage_effnum_mean'):.1f}  "
          f"used_frac={mean_val(r,'train/mem_used_frac_mean'):.4f}  "
          f"mse={mean_val(r,'train/mse_loss'):.4f}")
