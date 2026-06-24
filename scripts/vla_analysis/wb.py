#!/usr/bin/env python3
import sys, os
sys.path.insert(0, "/home/josh/lerobot/scripts")
from parse_wandb import WandbRun

BASE = "/home/josh/lerobot/outputs/train"
RUNS = {
  "sep5_PRE_40k":  f"{BASE}/libero_90_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k",
  "sep5_PROBE_10k":f"{BASE}/libero_90_pi05_8_10_12_14_probe10k_standard_c0.05_sep5.0_rq512",
  "ctrl_PRE_40k":  f"{BASE}/libero_90_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_40k",
}

def nearest(run, key, step):
    pts = run.get_metric(key)
    if not pts: return None
    return min(pts, key=lambda p: abs(p[0]-step))

def last(run, key):
    pts = run.get_metric(key)
    return pts[-1] if pts else None

runs = {}
for name, d in RUNS.items():
    wb = os.path.join(d, "wandb")
    if not os.path.isdir(wb):
        print(f"[skip] {name}: no wandb dir"); continue
    try:
        runs[name] = WandbRun.from_wandb_dir(wb)
        print(f"[ok] {name}: {runs[name]}")
    except Exception as e:
        print(f"[err] {name}: {e}")
print()

# Held-in eval trajectory
print("="*70)
print("HELD-IN EVAL (eval/pc_success) trajectory")
print("="*70)
for name, r in runs.items():
    pts = r.get_metric("eval/pc_success")
    print(f"  {name}:", [(s, round(v,3) if isinstance(v,(int,float)) else v) for s,v in pts])
print()

# key metric table at matched step
def show(name, r, step):
    keys = [
      ("mse", "train/mse_loss"),
      ("gate_mean", "train/gate_mean"),
      ("gate_L14", "train/gate_mean_L14"),
      ("q_intra_mean", "train/query_intra_sim_mean"),
      ("q_intra_L14", "train/query_intra_sim_L14"),
      ("q_inter_mean", "train/query_inter_sim_mean"),
      ("rout_sim_mean", "train/routing_inter_task_similarity_mean"),
      ("rout_sim_L14", "train/routing_inter_task_similarity_L14"),
      ("rout_sep_mean", "train/routing_inter_task_separation_mean"),
      ("supp_mean", "train/routing_intra_task_support_mean"),
      ("supp_L14", "train/routing_intra_task_support_L14"),
      ("supp_L8", "train/routing_intra_task_support_L8"),
      ("effnum_mean", "train/mem_usage_effnum_mean"),
      ("effnum_L14", "train/mem_usage_effnum_L14"),
      ("used_frac", "train/mem_used_frac_mean"),
      ("top1_share", "train/mem_usage_top1_share_mean"),
      ("contrastive", "train/contrastive_loss_mean"),
    ]
    print(f"\n--- {name} @ step~{step} ---")
    for label, k in keys:
        v = nearest(r, k, step)
        if v is None: print(f"   {label:>14s}: --"); continue
        s, val = v
        vs = f"{val:.4f}" if isinstance(val,(int,float)) else str(val)
        print(f"   {label:>14s}: {vs}   (@{s})")

print("="*70); print("PRETRAIN METRICS")
if "sep5_PRE_40k" in runs:
    show("sep5_PRE_40k", runs["sep5_PRE_40k"], 20000)
    show("sep5_PRE_40k", runs["sep5_PRE_40k"], 40000)
if "sep5_PROBE_10k" in runs:
    show("sep5_PROBE_10k", runs["sep5_PROBE_10k"], 10000)
if "ctrl_PRE_40k" in runs:
    show("ctrl_PRE_40k", runs["ctrl_PRE_40k"], 20000)
    show("ctrl_PRE_40k", runs["ctrl_PRE_40k"], 40000)
