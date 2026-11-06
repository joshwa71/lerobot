#!/usr/bin/env python3
"""Routing-audit analysis — REAL-WORLD duplicate of vlm_audit_analysis.py (E44 gate input).

Reads a held-out audit run's memory_by_task JSONs for one tower's modules and writes
<audit_run>/{expert,vlm}_audit_summary.json with, per layer: per-task core50 / effnum, famIoU
(mean weighted read IoU over the FAMILY pairs) and bgIoU (all other pairs). Identical output
schema to the LIBERO script, so the chain gate and any downstream tooling read it unchanged.

Deltas vs the LIBERO script: the family is NOT the hardcoded libero_10 basket (4,5),(4,7),(5,7)
but AUDIT_FAMILY env ("1-3,0-4" in SEQ task ids; empty = no family -> famIoU None, bgIoU over
every pair); the base dir is AUDIT_BASE env; every print is None-safe; the task list is read
from the files present (any task count).

Usage: AUDIT_FAMILY=1-3 python vlm_audit_analysis_rw.py <audit_run> <layers csv> <n_slots> <vlm|expert>
"""
import json
import os
import sys

import numpy as np

BASE = os.environ.get("AUDIT_BASE", "/home/josh/lerobot/outputs/train")
PREFIXES = {
    "vlm": "model.paligemma_with_expert.paligemma.model.language_model.layers.",
    "expert": "model.paligemma_with_expert.gemma_expert.model.layers.",
}


def parse_family(s):
    fam = []
    for tok in (s or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, b = (int(x) for x in tok.split("-"))
        fam.append((min(a, b), max(a, b)))
    return fam


FAMILY = parse_family(os.environ.get("AUDIT_FAMILY", ""))


def profile(run_dir, t, L, n_slots, prefix):
    d = json.load(open(os.path.join(run_dir, "memory_by_task", f"memory_usage_task_{t}.json")))
    node = d["per_module"][prefix + str(L)]
    slots = node[next(iter(node))]
    a = np.zeros(n_slots)
    for sk, st in slots.items():
        v = st.get("total_accesses", 0)
        if v:
            a[int(sk.rsplit("_", 1)[1])] = v
    return a


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def main():
    run = sys.argv[1]
    layers = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "15,16").split(",")]
    n_slots = int(sys.argv[3]) if len(sys.argv) > 3 else 256 * 256
    tower = sys.argv[4] if len(sys.argv) > 4 else "vlm"
    prefix = PREFIXES[tower]
    rd = os.path.join(BASE, run)
    tasks = sorted(
        int(f.rsplit("_", 1)[1][:-5])
        for f in os.listdir(os.path.join(rd, "memory_by_task"))
        if f.startswith("memory_usage_task_")
    )
    print(f"[audit-rw] run={run} tower={tower} layers={layers} bank={n_slots} tasks={tasks} family={FAMILY}")
    out = {}
    for L in layers:
        w = {}
        for t in tasks:
            a = profile(rd, t, L, n_slots, prefix)
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
        out[f"L{L}_pairIoU"] = {f"{a}-{b}": float(v) for (a, b), v in ious.items()}
        fam_s = "/".join(f"{ious[p]:.3f}" for p in FAMILY if p in ious)
        worst = max(ious.items(), key=lambda kv: kv[1]) if ious else None
        print(f"famIoU mean={fmt(out[f'L{L}_famIoU'])} ({fam_s})  bgIoU={fmt(out[f'L{L}_bgIoU'])}  "
              f"offdiag max={fmt(worst[1] if worst else None)}{' at ' + str(worst[0]) if worst else ''}")
    dst = os.path.join(rd, f"{tower}_audit_summary.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\nsaved {dst}")


if __name__ == "__main__":
    main()
