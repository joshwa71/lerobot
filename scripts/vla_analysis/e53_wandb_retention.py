#!/usr/bin/env python3
"""E53 retention + wandb scalars for corefrac / spread-A (+ on-disk comparators).
Retention: per-env init (20ep) / peak / final (50ep) from eval/results.jsonl, means,
give-back. Wandb: per-task block-min & block-end train/mse_loss, grad-norm max,
LR peak (schedule verification), final logged memory_iou/all_modules_mean (for the
slots-pipeline validation). Output: outputs/analysis/e53/wandb_retention.json + stdout."""
import json, os, sys
sys.path.insert(0, "/home/josh/lerobot/scripts")
from parse_wandb import WandbRun

BASE = "/home/josh/lerobot/outputs/train"
RUNS = {
    "corefrac": f"{BASE}/libero_10_seq5_jw_layermax_compact_e9to12_v13to16_beta4corefrac_topt3072_lr2x_steps5k",
    "spreadA":  f"{BASE}/libero_10_seq5_jw_layermax_A_e2468_v10121416_beta4_topt3072_lr2x_steps5k",
    "foldin":   f"{BASE}/libero_10_seq5_jw_layermax_compact_e9to12_v13to16_beta4_topt3072_lr2x_steps5k",
    "plain":    f"{BASE}/libero_10_seq5_jw_layermax_compact_e9to12_v13to16_beta4_topt1536_steps5k",
    "comp":     f"{BASE}/libero_10_seq5_jw_arm1p_vlmknn16_beta4_topt3072_lr2x_steps5k",
}
ENV = {0: 4, 1: 6, 2: 9, 3: 2, 4: 7}
STEPS = [5000 * (k + 1) for k in range(5)]
LBL = {4: "two mugs", 6: "mug+pud", 9: "mug+micro", 2: "stove+moka", 7: "soup+cheese"}
OUT = "/home/josh/lerobot/outputs/analysis/e53/wandb_retention.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

out = {}
for name, rd in RUNS.items():
    row = {}
    # --- retention from eval/results.jsonl ---
    rp = os.path.join(rd, "eval", "results.jsonl")
    if os.path.exists(rp):
        M = {}
        for line in open(rp):
            d = json.loads(line)
            M[d["step"]] = d
        ret = {}
        for ordk, t in enumerate(range(5)):
            env = ENV[t]
            key = f"task_{env}"
            traj = [M[st][key] for st in STEPS if st in M and key in M[st] and st >= STEPS[ordk]]
            if traj:
                ret[f"t{t}/e{env}"] = {"init": traj[0], "peak": max(traj), "final": traj[-1], "traj": traj}
        inits = [v["init"] for v in ret.values()]
        finals = [v["final"] for v in ret.values()]
        row["retention"] = ret
        row["init_mean"] = round(sum(inits) / len(inits), 1)
        row["final_mean"] = round(sum(finals) / len(finals), 1)
        row["give_back"] = round(row["final_mean"] - row["init_mean"], 1)
    # --- wandb scalars ---
    wb = os.path.join(rd, "wandb")
    if os.path.isdir(wb):
        try:
            r = WandbRun.from_wandb_dir(wb)
            mse = r.get_metric("train/mse_loss") or []
            gn = r.get_metric("train/grad_norm") or []
            lr = r.get_metric("train/lr") or []
            iou = r.get_metric("memory_iou/all_modules_mean") or []
            blocks = {}
            for t in range(5):
                lo, hi = t * 5000, (t + 1) * 5000
                pts = [v for s, v in mse if lo < s <= hi]
                if pts:
                    blocks[f"t{t}"] = {"min": round(min(pts), 4), "end": round(pts[-1], 4)}
            row["block_mse"] = blocks
            row["block_min_mean"] = round(sum(b["min"] for b in blocks.values()) / len(blocks), 4) if blocks else None
            row["grad_norm_max"] = round(max(v for _, v in gn), 4) if gn else None
            row["lr_peak"] = max(v for _, v in lr) if lr else None
            row["memory_iou_final"] = round(iou[-1][1], 4) if iou else None
        except Exception as e:
            row["wandb_error"] = str(e)
    out[name] = row

for name, row in out.items():
    print(f"\n=== {name}")
    if "retention" in row:
        for k, v in row["retention"].items():
            print(f"  {k:8s} init {v['init']:5.1f} peak {v['peak']:5.1f} final {v['final']:5.1f}  traj {['%.0f' % x for x in v['traj']]}")
        print(f"  init_mean {row['init_mean']}  final_mean {row['final_mean']}  give_back {row['give_back']}")
    if "block_mse" in row:
        print(f"  block-min/end: " + "  ".join(f"{k}:{v['min']}/{v['end']}" for k, v in row["block_mse"].items()))
        print(f"  block_min_mean {row['block_min_mean']}  grad_norm_max {row['grad_norm_max']}  lr_peak {row['lr_peak']}  logged_iou_final {row['memory_iou_final']}")

json.dump(out, open(OUT, "w"), indent=1)
print("\nwrote", OUT)
