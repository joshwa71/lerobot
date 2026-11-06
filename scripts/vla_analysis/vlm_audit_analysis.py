#!/usr/bin/env python3
"""E44 VLM-memory routing audit analysis (the sweep gate).

Reads a held-out audit run's memory_by_task JSONs for the VLM modules
(...language_model.layers.{L}) and prints, per layer: per-task core50 / effnum, pairwise
weighted read IoU (family = libero_10 basket tasks 4/5/7, background = the rest), plus the
E21 collapse tripwire (per-task effnum — a router that routes on the constant instruction
alone collapses toward one ~64-slot mixture; state-conditional routing spans far more).

Gates (re-anchored for the 65,536-slot bank = 0.44x the expert table):
  PASS: famIoU <= ~0.25 AND per-task core50 >= ~650 AND per-task effnum >= ~500
  COLLAPSE: effnum <= ~150 (per-task-bias signature; kill the arm regardless of IoU)

Usage: python vlm_audit_analysis.py <audit_run_name> [layers=15,16]
"""
import json
import os
import sys

import numpy as np

BASE = "/home/josh/lerobot/outputs/train"
PREFIXES = {
    "vlm": "model.paligemma_with_expert.paligemma.model.language_model.layers.",
    "expert": "model.paligemma_with_expert.gemma_expert.model.layers.",
}
PREFIX = PREFIXES["vlm"]
FAMILY = [(4, 5), (4, 7), (5, 7)]


def profile(run_dir, t, L, n_slots):
    d = json.load(open(os.path.join(run_dir, "memory_by_task", f"memory_usage_task_{t}.json")))
    node = d["per_module"][PREFIX + str(L)]
    slots = node[next(iter(node))]
    a = np.zeros(n_slots)
    for sk, st in slots.items():
        v = st.get("total_accesses", 0)
        if v:
            a[int(sk.rsplit("_", 1)[1])] = v
    return a


def main():
    global PREFIX
    run = sys.argv[1]
    layers = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "15,16").split(",")]
    n_slots = int(sys.argv[3]) if len(sys.argv) > 3 else 256 * 256
    tower = sys.argv[4] if len(sys.argv) > 4 else "vlm"
    PREFIX = PREFIXES[tower]
    rd = os.path.join(BASE, run)
    tasks = sorted(
        int(f.rsplit("_", 1)[1][:-5])
        for f in os.listdir(os.path.join(rd, "memory_by_task"))
        if f.startswith("memory_usage_task_")
    )
    out = {}
    for L in layers:
        w = {}
        for t in tasks:
            a = profile(rd, t, L, n_slots)
            w[t] = a / max(a.sum(), 1e-12)
        print(f"\n==== {run} L{L} (bank {n_slots})")
        print(f"{'task':>5} {'core50':>7} {'effnum':>8}")
        for t in tasks:
            order = np.sort(w[t])[::-1]
            cum = np.cumsum(order)
            c50 = int(np.searchsorted(cum, 0.5)) + 1
            eff = 1.0 / (w[t] ** 2).sum()
            out[f"L{L}_t{t}"] = {"core50": c50, "effnum": float(eff)}
            flag = "  <<< COLLAPSE" if eff <= 150 else ("  (low)" if eff < 500 or c50 < 650 else "")
            print(f"{t:>5} {c50:>7} {eff:>8.0f}{flag}")
        ious = {}
        for i, a in enumerate(tasks):
            for b in tasks[i + 1:]:
                mn = np.minimum(w[a], w[b]).sum()
                mx = np.maximum(w[a], w[b]).sum()
                ious[(a, b)] = mn / max(mx, 1e-12)
        fam = [ious[p] for p in FAMILY if p in ious]
        bg = [v for k, v in ious.items() if k not in FAMILY]
        out[f"L{L}_famIoU"] = float(np.mean(fam)) if fam else None
        out[f"L{L}_bgIoU"] = float(np.mean(bg)) if bg else None
        fam_s = "/".join(f"{ious[p]:.3f}" for p in FAMILY if p in ious)
        print(f"famIoU mean={np.mean(fam):.3f} ({fam_s})  bgIoU={np.mean(bg):.3f}  "
              f"offdiag max={max(ious.values()):.3f}")
    dst = os.path.join(rd, f"{tower}_audit_summary.json" if tower != "vlm" else "vlm_audit_summary.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\nsaved {dst}")


if __name__ == "__main__":
    main()
