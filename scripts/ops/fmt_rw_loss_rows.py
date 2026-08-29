#!/usr/bin/env python3
"""Format the RW sequential's in-run loss rows: one line per boundary, drift vs just-trained.
Reads eval/loss_results.jsonl on stdin. SEEN=n skips the first n rows (for the streaming monitor).
baseline_t = task_t - forget_t (the value when task t had just trained)."""
import json, os, sys

seen = int(os.environ.get("SEEN", "0"))
rows = [json.loads(l) for l in sys.stdin if l.strip().startswith("{")]
for i, r in enumerate(rows):
    if i < seen:
        continue
    parts = []
    for t in range(10):
        k = f"task_{t}"
        if k not in r:
            continue
        v = r[k]
        f = r.get(f"forget_{t}", 0.0)
        base = v - f
        parts.append(f"t{t} {v:.5f} jt" if abs(f) < 1e-12 else f"t{t} {v:.5f} {100*f/base:+5.1f}%")
    print(f"[seq row {i+1} @ step {r['step']}] " + "   ".join(parts))
