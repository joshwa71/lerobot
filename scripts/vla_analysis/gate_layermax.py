#!/usr/bin/env python3
"""E50 automated warm-up gate for the layer-max chain.

Reads a held-out routing audit's memory_by_task JSONs and applies the certified-quality
bands per tower (calibrated on the E46-49 audit history at the n256 bank):
  EXPERT: family IoU <= 0.20 on >= 3 of 4 layers  AND  mean core50 >= 1200
          AND every layer's mean core50 >= 700
  VLM:    family IoU <= 0.25 on EVERY layer  AND  per-layer mean effnum >= 500
          AND min per-task effnum >= 150 (the ~2-draw palette-collapse tripwire)
famIoU = mean weighted-read IoU over the basket-family pairs (libero_10 task_index
4/5/7); core50 = slots carrying half a task's read mass; effnum = exp(entropy of the
read-mass distribution) - matching vlm_audit_analysis.py's definition so the historical
bands transfer.

Usage: gate_layermax.py <audit_run_name> <expert_layers_csv> <vlm_layers_csv>
Exit 0 = PASS (both towers), 1 = FAIL, 2 = data missing/unreadable.
"""
import json
import os
import sys

import numpy as np

BASE = "/home/josh/lerobot/outputs/train"
EXP_PREFIX = "model.paligemma_with_expert.gemma_expert.model.layers."
VLM_PREFIX = "model.paligemma_with_expert.paligemma.model.language_model.layers."
TASKS = list(range(10))
FAMILY = [(4, 5), (4, 7), (5, 7)]


def profiles(run_dir, t, mkey):
    d = json.load(open(os.path.join(run_dir, "memory_by_task", f"memory_usage_task_{t}.json")))
    node = d["per_module"][mkey]
    slots = node[next(iter(node))]
    idx, cnt = [], []
    for sk, st in slots.items():
        if st["total_accesses"]:
            idx.append(int(sk.rsplit("_", 1)[1]))
            cnt.append(st["total_accesses"])
    return np.array(idx, dtype=np.int64), np.array(cnt, dtype=np.float64)


def layer_stats(run_dir, mkey, nslots):
    raw = {t: profiles(run_dir, t, mkey) for t in TASKS}
    for idx, _ in raw.values():
        if len(idx):
            nslots = max(nslots, int(idx.max()) + 1)
    dense = {}
    for t in TASKS:
        idx, cnt = raw[t]
        a = np.zeros(nslots)
        if len(idx):
            a[idx] = cnt
        s = a.sum()
        dense[t] = a / s if s > 0 else a
    fam = []
    for i, j in FAMILY:
        mn = np.minimum(dense[i], dense[j]).sum()
        mx = np.maximum(dense[i], dense[j]).sum()
        fam.append(mn / max(mx, 1e-12))
    c50s, effs = [], []
    for t in TASKS:
        p = dense[t][dense[t] > 0]
        if not len(p):
            c50s.append(0)
            effs.append(0.0)
            continue
        order = np.sort(dense[t])[::-1]
        c50s.append(int(np.searchsorted(np.cumsum(order), 0.5)) + 1)
        effs.append(float(np.exp(-(p * np.log(p)).sum())))
    return float(np.mean(fam)), float(np.mean(c50s)), float(np.mean(effs)), float(np.min(effs))


def main():
    audit_run, exp_csv, vlm_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    run_dir = os.path.join(BASE, audit_run)
    n_json = len([f for f in os.listdir(os.path.join(run_dir, "memory_by_task"))
                  if f.endswith(".json")]) if os.path.isdir(os.path.join(run_dir, "memory_by_task")) else 0
    if n_json < 10:
        print(f"GATE: DATA MISSING ({n_json}/10 task JSONs in {run_dir})")
        sys.exit(2)
    # bank size from any JSON's config is not stored; infer from max slot index later is
    # unnecessary - the dense arrays only need an upper bound. Use 65536 (n256) which all
    # current arms share; a larger bank only wastes memory, never changes the stats.
    nslots = 65536

    exp_layers = [int(x) for x in exp_csv.split(",") if x.strip()]
    vlm_layers = [int(x) for x in vlm_csv.split(",") if x.strip()]

    exp_rows, vlm_rows = [], []
    for L in exp_layers:
        exp_rows.append((L, *layer_stats(run_dir, EXP_PREFIX + str(L), nslots)))
    for L in vlm_layers:
        vlm_rows.append((L, *layer_stats(run_dir, VLM_PREFIX + str(L), nslots)))

    print(f"{'tower':6s} {'L':>3s} {'famIoU':>7s} {'core50':>7s} {'effnum':>8s} {'min-eff':>8s}")
    for L, f, c, e, me in exp_rows:
        print(f"EXPERT {L:>3d} {f:7.3f} {c:7.0f} {e:8.0f} {me:8.0f}")
    for L, f, c, e, me in vlm_rows:
        print(f"VLM    {L:>3d} {f:7.3f} {c:7.0f} {e:8.0f} {me:8.0f}")

    exp_fam_ok = sum(1 for _, f, _, _, _ in exp_rows if f <= 0.20) >= min(3, len(exp_rows))
    exp_c50_mean = float(np.mean([c for _, _, c, _, _ in exp_rows]))
    exp_c50_layers_ok = all(c >= 700 for _, _, c, _, _ in exp_rows)
    exp_ok = exp_fam_ok and exp_c50_mean >= 1200 and exp_c50_layers_ok

    vlm_fam_ok = all(f <= 0.25 for _, f, _, _, _ in vlm_rows)
    vlm_eff_ok = all(e >= 500 for _, _, _, e, _ in vlm_rows)
    vlm_collapse_ok = all(me >= 150 for _, _, _, _, me in vlm_rows)
    vlm_ok = vlm_fam_ok and vlm_eff_ok and vlm_collapse_ok

    print(f"EXPERT gate: fam>=3/4<=0.20:{exp_fam_ok} c50-mean {exp_c50_mean:.0f}>=1200:"
          f"{exp_c50_mean >= 1200} c50-all>=700:{exp_c50_layers_ok} -> {'PASS' if exp_ok else 'FAIL'}")
    print(f"VLM    gate: fam-all<=0.25:{vlm_fam_ok} eff-all>=500:{vlm_eff_ok} "
          f"collapse-min>=150:{vlm_collapse_ok} -> {'PASS' if vlm_ok else 'FAIL'}")
    if exp_ok and vlm_ok:
        print("GATE: PASS")
        sys.exit(0)
    print("GATE: FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
