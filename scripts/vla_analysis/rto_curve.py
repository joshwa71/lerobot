#!/usr/bin/env python3
"""Rank-2 RTO<->retention calibration curve (research_log Entry 32 appendix, test 1).

Builds per-task points from the 7 rank-2 libero_10 sequential runs:
  x = RTO (read-through-overwrite: frac of the task's read WEIGHT on slots updated by
      LATER tasks; per-layer + 4-layer mean)
  y = retention (final/init rollout success; also drop = final-init), plus basin-depth
      proxies (init, block-min train MSE) and L14 capacity proxies.

Writes outputs/analysis/rank2_rto_retention.json. The eventual [2,2,4,4] (rank-4)
sequential adds 10 points via the same extraction: ON the curve -> per-unit-overlap
destructiveness is rank-invariant; ABOVE -> specialization-drift/concentration; BELOW ->
basin-deepening dominates (the C pattern). Task-matched comparison (same env across
runs) is the low-noise read.

Usage: python rto_curve.py [--json-only]
"""
import json, os, sys, gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/josh/lerobot/scripts")
from slots import load_task, effnum, core_frac, overwrite_frac, LAYERS, ENV, ORDER  # noqa: E402
from parse_wandb import WandbRun  # noqa: E402

BASE = "/home/josh/lerobot/outputs/train"
P = "libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive"
# name -> (dir, steps_per_task, eval_eps)
RUNS = {
 "control":   (f"{BASE}/{P}_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_40k_top_t_1536", 3000, 50),
 "sep5":      (f"{BASE}/{P}_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536",            3000, 50),
 "protectB4": (f"{BASE}/{P}_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536_protect_beta4",        3000, 20),
 "C_steps5k": (f"{BASE}/{P}_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536_protect_beta4_steps5k", 5000, 20),
 "B_lr2x":    (f"{BASE}/{P}_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536_protect_beta4_lr2x",    3000, 20),
 "D_lr2x5k":  (f"{BASE}/{P}_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536_protect_beta4_lr2x_steps5k", 5000, 20),
 "A_beta8":   (f"{BASE}/{P}_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536_protect_beta8",        3000, 20),
}
LBL = {4:"two mugs",6:"mug+pud",9:"mug+micro",2:"stove+moka",7:"soup+cheese",0:"soup+sauce",
       8:"both mokas",1:"cheese+butter",3:"bowl+drawer",5:"book"}
ORD = {t:i for i,t in enumerate(ORDER)}
OUT = "/home/josh/lerobot/outputs/analysis/rank2_rto_retention.json"


def extract_run(name, run_dir, steps, eps):
    """Return the 10 per-task points for one run."""
    # --- slot side: RTO per layer + L14 capacity
    mbt = os.path.join(run_dir, "memory_by_task")
    data = {t: load_task(os.path.join(mbt, f"memory_usage_task_{t}.json")) for t in ORDER}
    rto, cap = {}, {}
    for t in ORDER:
        later = [u for u in ORDER if ORD[u] > ORD[t]]
        by_layer = {}
        for L in LAYERS:
            rids, racc, _ = data[t][L]
            upd = (np.array(sorted(set().union(*[set(data[u][L][2].tolist()) for u in later])), dtype=np.int64)
                   if later else np.array([], dtype=np.int64))
            by_layer[L] = overwrite_frac(rids, racc, upd)
        rto[t] = by_layer
        rids, racc, _ = data[t][14]
        cap[t] = (effnum(racc), core_frac(racc))
    del data; gc.collect()

    # --- eval side: init/peak/final per task (per-run step size)
    M = {}
    for line in open(os.path.join(run_dir, "eval/results.jsonl")):
        d = json.loads(line); M[int(d["step"])] = d
    STEPS = [steps * (k + 1) for k in range(10)]

    # --- wandb side: block-min MSE (basin-depth proxy)
    bm = {}
    try:
        r = WandbRun.from_wandb_dir(os.path.join(run_dir, "wandb"))
        pts = [(s, v) for s, v in r.get_metric("train/mse_loss") if isinstance(v, (int, float))]
        for k in range(10):
            vals = [v for s, v in pts if k * steps < s <= (k + 1) * steps]
            bm[k] = min(vals) if vals else None
    except Exception:
        bm = {k: None for k in range(10)}

    points = []
    for k, t in enumerate(ORDER):
        env = ENV[t]; st0 = STEPS[k]
        traj = [M[st][f"task_{env}"] for st in STEPS if st in M and f"task_{env}" in M[st] and st >= st0]
        if not traj:
            continue
        init, peak, final = traj[0], max(traj), traj[-1]
        m4 = float(np.mean(list(rto[t].values())))
        points.append(dict(
            run=name, rank_config="[2,2,2,2]", steps_per_task=steps, eval_eps=eps,
            task_index=t, order=k, env=env, label=LBL[env],
            rto_mean=round(m4, 4), rto_by_layer={str(L): round(v, 4) for L, v in rto[t].items()},
            init=init, peak=peak, final=final,
            ret_frac=round(final / init, 4) if init > 0 else None,
            drop=round(final - init, 2),
            blockmin_mse=round(bm[k], 4) if bm.get(k) is not None else None,
            L14_effnum=round(cap[t][0]), L14_core50=cap[t][1],
        ))
    return points


def main():
    allpts = []
    for name, (d, steps, eps) in RUNS.items():
        allpts += extract_run(name, d, steps, eps)
        print(f"[done] {name}", file=sys.stderr)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    payload = dict(
        meta=dict(
            created="2026-07-02",
            definition=("RTO = frac of task's read WEIGHT on slots updated by LATER tasks "
                        "(from memory_by_task JSONs); retention from eval/results.jsonl; "
                        "blockmin_mse = min train/mse_loss inside the task's own block. "
                        "Rank-4 test: extract the [2,2,4,4] sequential with the same code and "
                        "compare against these points (on/above/below the curve; task-matched)."),
            runs={n: dict(dir=d, steps_per_task=s, eval_eps=e) for n, (d, s, e) in RUNS.items()},
        ),
        points=allpts,
    )
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {OUT}  ({len(allpts)} task-points)", file=sys.stderr)

    if "--json-only" in sys.argv:
        return
    # ---- curve summary
    pts = [p for p in allpts if p["ret_frac"] is not None]
    print("\nRANK-2 CALIBRATION CURVE — RTO bins -> retention")
    print(f"{'RTO bin':>10} {'n':>3} {'mean ret%':>10} {'mean drop':>10} {'collapses(fin<=5)':>18}")
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        hi = lo + 0.2
        b = [p for p in pts if lo <= p["rto_mean"] < hi]
        if not b: continue
        mr = 100 * np.mean([p["ret_frac"] for p in b]); md = np.mean([p["drop"] for p in b])
        nc = sum(1 for p in b if p["final"] <= 5)
        print(f"{f'{lo:.0%}-{hi:.0%}':>10} {len(b):>3} {mr:>9.0f}% {md:>+10.1f} {nc:>18}")
    x = np.array([p["rto_mean"] for p in pts]); y = np.array([p["drop"] for p in pts])
    pear = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    spear = float(np.corrcoef(rx, ry)[0, 1])
    print(f"\ncorr(RTO, drop): pearson={pear:.3f}  spearman={spear:.3f}   (n={len(pts)})")
    print("\nTASK-MATCHED VIEW — env: (rto%, init->final) per run")
    for t in ORDER:
        env = ENV[t]
        row = [p for p in pts if p["env"] == env]
        row.sort(key=lambda p: p["rto_mean"])
        cells = "  ".join(f"{p['run'][:7]}({p['rto_mean']:.0%},{p['init']:.0f}->{p['final']:.0f})" for p in row)
        print(f"  e{env} {LBL[env]:<13} {cells}")


if __name__ == "__main__":
    main()
