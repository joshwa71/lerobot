#!/usr/bin/env python3
"""E53 checkpoint group diff (rebuild of the lost E39 scratchpad ckpt_diff.py).
For each run, for each consecutive per-task checkpoint pair, stream every tensor and
report per GROUP: n tensors changed, max |delta|. Groups: slot values, memory router
(keys/query_proj/film), memory other (gate/value_proj/swilu), backbone LM, backbone
expert, vision, other. Proves the values-only claim (only slot_* moves) per block.
Output: outputs/analysis/e53/ckpt_diff_e53.json (+ stdout table)."""
import json, os
import torch
from safetensors import safe_open

BASE = "/home/josh/lerobot/outputs/train"
RUNS = {
    "corefrac": f"{BASE}/libero_10_seq5_jw_layermax_compact_e9to12_v13to16_beta4corefrac_topt3072_lr2x_steps5k",
    "spreadA":  f"{BASE}/libero_10_seq5_jw_layermax_A_e2468_v10121416_beta4_topt3072_lr2x_steps5k",
}
CKPTS = ["005000", "010000", "015000", "020000", "025000"]
OUT = "/home/josh/lerobot/outputs/analysis/e53/ckpt_diff_e53.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def group_of(k):
    if ".mlp.mem.slot_" in k:
        return "slot_values"
    if ".mlp.mem." in k:
        if any(s in k for s in ("keys", "query_proj", "film", "query")):
            return "mem_router"
        return "mem_other"  # gate / value_proj / swilu etc.
    if "vision_tower" in k:
        return "vision"
    if "gemma_expert" in k:
        return "backbone_expert"
    if "paligemma" in k:
        return "backbone_lm"
    return "other"


results = {}
for name, rd in RUNS.items():
    results[name] = {}
    for a, b in zip(CKPTS[:-1], CKPTS[1:]):
        pa = os.path.join(rd, "checkpoints", a, "pretrained_model", "model.safetensors")
        pb = os.path.join(rd, "checkpoints", b, "pretrained_model", "model.safetensors")
        agg = {}
        with safe_open(pa, framework="pt") as fa, safe_open(pb, framework="pt") as fb:
            keys = set(fa.keys())
            assert keys == set(fb.keys()), f"key mismatch {a}->{b}"
            for k in keys:
                ta = fa.get_tensor(k)
                tb = fb.get_tensor(k)
                d = (tb.float() - ta.float()).abs().max().item()
                g = agg.setdefault(group_of(k), {"n_total": 0, "n_changed": 0, "max_abs": 0.0})
                g["n_total"] += 1
                if d > 0:
                    g["n_changed"] += 1
                    g["max_abs"] = max(g["max_abs"], d)
                del ta, tb
        results[name][f"{a}->{b}"] = agg
        row = "  ".join(f"{g}:{v['n_changed']}/{v['n_total']}(max{v['max_abs']:.3g})" for g, v in sorted(agg.items()))
        print(f"[{name}] {a}->{b}: {row}", flush=True)

json.dump(results, open(OUT, "w"), indent=1)
print("wrote", OUT)
