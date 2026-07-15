#!/usr/bin/env python3
import os, sys
sys.path.insert(0,"/home/josh/lerobot/scripts")
from parse_wandb import WandbRun
BASE="/home/josh/lerobot/outputs/train"
ENV={0:4,1:6,2:9,3:2,4:7}
LBL={4:"2mugs",6:"mug+pud",9:"mug+micro",2:"stove+moka",7:"soup+cheese"}
SEQ={
 "stageB":  (f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps5k_tasks5",5000),
 "affine":  (f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_affine_nogate_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps5k_tasks5",5000),
 "lr2x":    (f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_lr2x_steps5k_tasks5",5000),
 "steps7k": (f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps7k_tasks5",7000),
}
print("="*100)
print("SEQUENTIAL runs: per-block train/mse_loss (min, end) + gate + grad-norm max")
print("="*100)
for name,(d,bs) in SEQ.items():
    try: r=WandbRun.from_wandb_dir(os.path.join(d,"wandb"))
    except Exception as e: print(f"[err] {name}: {e}"); continue
    pts=r.get_metric("train/mse_loss")
    gn=r.get_metric("train/grad_norm")
    row=[]
    for k in range(5):
        lo,hi=bs*k+1,bs*(k+1)
        block=[v for s,v in pts if lo<=s<=hi]
        if not block: row.append(("--","--")); continue
        row.append((min(block),block[-1]))
    mins=" ".join(f"{ENV[k]}:{row[k][0]:.4f}" if row[k][0]!="--" else f"{ENV[k]}:--" for k in range(5))
    mean=sum(r0[0] for r0 in row if r0[0]!="--")/5
    print(f"\n{name:8s} block-min MSE  {mins}   MEAN={mean:.4f}")
    ends=" ".join(f"{ENV[k]}:{row[k][1]:.4f}" if row[k][1]!="--" else f"{ENV[k]}:--" for k in range(5))
    print(f"{'':8s} block-end MSE  {ends}")
    if gn: print(f"{'':8s} grad_norm max={max(v for _,v in gn):.4f} last={gn[-1][1]:.4f}")
    for gk in ["train/gate_mean","train/gate_mean_L14","train/gate_mean_L8"]:
        g=r.get_metric(gk)
        if g: print(f"{'':8s} {gk}: first={g[0][1]:.3f} last={g[-1][1]:.3f}")
    lr=r.get_metric("train/memory_value_lr") or r.get_metric("train/lr")
    if lr: print(f"{'':8s} value_lr: max={max(v for _,v in lr):.2e} min={min(v for _,v in lr):.2e}")

print()
print("="*100)
print("A-PHASES: libero_90 values-only 10k")
print("="*100)
APH={
 "A(stageB)": f"{BASE}/libero_90_pi05_8_10_12_14_frozenroute_rwarmupB_values10k_c0.05_sep5.0_noloc_rq512",
 "A(affine)": f"{BASE}/libero_90_pi05_8_10_12_14_frozenroute_affine_nogate_values10k_c0.05_sep5.0_noloc_rq512",
}
for name,d in APH.items():
    try: r=WandbRun.from_wandb_dir(os.path.join(d,"wandb"))
    except Exception as e: print(f"[err] {name}: {e}"); continue
    pts=r.get_metric("train/mse_loss")
    tail=[v for s,v in pts if s>=9000]
    ev=r.get_metric("eval/pc_success")
    print(f"\n{name}: mse@200={pts[0][1]:.4f}  mse@5k={min(v for s,v in pts if 4800<=s<=5200):.4f}  mse tail(9-10k) min={min(tail):.4f} last={pts[-1][1]:.4f}")
    print(f"   held-in eval: {[(s,round(v,1)) for s,v in ev] if ev else 'NONE LOGGED'}")
    for gk in ["train/gate_mean","train/gate_mean_L14"]:
        g=r.get_metric(gk)
        if g: print(f"   {gk}: first={g[0][1]:.3f} last={g[-1][1]:.3f}")
