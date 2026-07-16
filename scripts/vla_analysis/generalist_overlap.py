#!/usr/bin/env python3
"""E42 addendum: generalist-slot overlap analysis (the "freeze the A-phase core" decision).

Aggregates the A-phase (libero-90) per-slot read mass per layer from the audit sweep
(audit_libero90_usage_rwarmupB_A), builds top-{50,20,10,5}%-MASS generalist sets, then for
each sequential task (read profiles are arm-invariant under the frozen router; taken from the
softprotect run's JSONs) reports:
  - read overlap:   fraction of the task's read mass on the generalist set
  - write overlap:  fraction of the task's update-EVENT mass (JSON total_updates) on the set
                    (per-task realized-write deltas are gone with the deleted intermediates;
                    update events are the available proxy)
Interpretation: HIGH write overlap => freezing the set starves the writers (E19 regime);
LOW => a freeze is cheap (and its fit effect small either way) — the residual case for it is
the structural-anchor/off-trail story, not fit.

Saves outputs/analysis/e42/generalist_overlap.json and prints the table.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/josh/lerobot/scripts/vla_analysis")
from slots import PREFIX

BASE = "/home/josh/lerobot/outputs/train"
AUDIT = f"{BASE}/audit_libero90_usage_rwarmupB_A"
SEQ = f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_softprotect_cf_beta4_lr2x_steps5k_tasks5"
ENV = {0: 4, 1: 6, 2: 9, 3: 2, 4: 7}
N = 147456
LAYERS = [8, 10, 12, 14]
MASS_THRESHOLDS = [0.5, 0.2, 0.1, 0.05]


def profile(run_dir, t, L, field):
    d = json.load(open(os.path.join(run_dir, "memory_by_task", f"memory_usage_task_{t}.json")))
    node = d["per_module"][PREFIX + str(L)]
    slots = node[next(iter(node))]
    a = np.zeros(N)
    for sk, st in slots.items():
        v = st.get(field, 0)
        if v:
            a[int(sk.rsplit("_", 1)[1])] = v
    return a


out = {}
for L in LAYERS:
    agg = np.zeros(N)
    for t in range(90):
        agg += profile(AUDIT, t, L, "total_accesses")
    order = np.argsort(agg)[::-1]
    cum = np.cumsum(agg[order]) / max(agg.sum(), 1e-12)
    sets = {}
    for thr in MASS_THRESHOLDS:
        k = int(np.searchsorted(cum, thr)) + 1
        mask = np.zeros(N, dtype=bool)
        mask[order[:k]] = True
        sets[thr] = (mask, k)
    print(f"\n==== L{L}: A-phase aggregate — effnum {1.0 / ((agg / agg.sum()) ** 2).sum():.0f}, "
          + ", ".join(f"top-{int(100 * t)}%mass = {sets[t][1]} slots ({100 * sets[t][1] / N:.1f}% of table)"
                      for t in MASS_THRESHOLDS))
    out[f"L{L}_set_sizes"] = {str(t): int(sets[t][1]) for t in MASS_THRESHOLDS}
    hdr = f"{'task':>8} " + " ".join(f"read@{int(100 * t)}% wr@{int(100 * t)}%" for t in MASS_THRESHOLDS)
    print(hdr)
    for t in range(5):
        r = profile(SEQ, t, L, "total_accesses")
        w = profile(SEQ, t, L, "total_updates")
        r = r / max(r.sum(), 1e-12)
        w = w / max(w.sum(), 1e-12)
        cells = []
        rec = {}
        for thr in MASS_THRESHOLDS:
            m = sets[thr][0]
            ro, wo = float(r[m].sum()), float(w[m].sum())
            rec[str(thr)] = {"read": ro, "write": wo}
            cells.append(f"{100 * ro:7.1f}% {100 * wo:6.1f}%")
        print(f"t{t}(e{ENV[t]})  " + " ".join(cells))
        out[f"L{L}_t{t}"] = rec

os.makedirs("/home/josh/lerobot/outputs/analysis/e42", exist_ok=True)
json.dump(out, open("/home/josh/lerobot/outputs/analysis/e42/generalist_overlap.json", "w"), indent=1)
print("\nsaved outputs/analysis/e42/generalist_overlap.json")
