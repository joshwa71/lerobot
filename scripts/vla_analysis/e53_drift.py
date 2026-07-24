#!/usr/bin/env python3
"""E53 delivered-damage: read-mass-weighted relative value drift on victim slot bands,
for the corefrac and spread-A runs (E52 comparators: outputs/analysis/e52/core_drift.json).
Bands per victim task T per module: core50 (top slots to 50% read mass), shoulder
(50->90% band), full (entire read footprint) — full == the task-perceived field change
(the old scratchpad field_change.py read). Baseline = T's own-block checkpoint; drift
walked over each later per-task checkpoint.
Output: outputs/analysis/e53/core_drift_e53.json"""
import json, os
import torch
from safetensors import safe_open

BASE = "/home/josh/lerobot/outputs/train"
RUNS = {
    "corefrac": f"{BASE}/libero_10_seq5_jw_layermax_compact_e9to12_v13to16_beta4corefrac_topt3072_lr2x_steps5k",
    "spreadA":  f"{BASE}/libero_10_seq5_jw_layermax_A_e2468_v10121416_beta4_topt3072_lr2x_steps5k",
}
CKPT = {0: "005000", 1: "010000", 2: "015000", 3: "020000", 4: "025000"}
VICTIMS = [(0, "e4"), (1, "e6"), (2, "e9"), (3, "e2")]
OUT = "/home/josh/lerobot/outputs/analysis/e53/core_drift_e53.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def short_mod(m):
    l = m.rsplit(".", 1)[-1]
    return ("E" if "gemma_expert" in m else "V") + l


def load_bands(run_dir, t):
    """per module: {core: [(sid,acc)...], shoulder: [...], full: [...]} by read mass"""
    p = os.path.join(run_dir, "memory_by_task", f"memory_usage_task_{t}.json")
    d = json.load(open(p))["per_module"]
    bands = {}
    for mod, tasks in d.items():
        st = tasks.get(f"task_{t}")
        if st is None:
            continue
        items = []
        for slot, s in st.items():
            a = s.get("total_accesses", 0)
            if a:
                items.append((int(slot.rsplit("_", 1)[-1]), a))
        items.sort(key=lambda kv: -kv[1])
        tot = sum(a for _, a in items)
        acc, core, shoulder = 0, [], []
        for sid, a in items:
            if acc < 0.5 * tot:
                core.append((sid, a))
            elif acc < 0.9 * tot:
                shoulder.append((sid, a))
            acc += a
        bands[mod] = {"core": core, "shoulder": shoulder, "full": items}
    return bands


def slot_tensors(run_dir, ck, mod):
    p = os.path.join(run_dir, "checkpoints", ck, "pretrained_model", "model.safetensors")
    out = {}
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if ".mlp.mem.slot_" in k and k.startswith(mod + "."):
                out[k] = f.get_tensor(k)
    return out


def per_slot_mat(tensors):
    mats, n = [], None
    for k in sorted(tensors):
        t = tensors[k].float()
        t = t.reshape(t.shape[0], -1)
        n = t.shape[0] if n is None else n
        assert t.shape[0] == n, (k, t.shape)
        mats.append(t)
    return torch.cat(mats, dim=1)


results = {}
for name, rd in RUNS.items():
    results[name] = {}
    mods = sorted(load_bands(rd, 0).keys(), key=lambda m: ("gemma_expert" not in m, int(m.rsplit(".", 1)[-1])))
    for t, env in VICTIMS:
        bands = load_bands(rd, t)
        base_ck = CKPT[t]
        later = [CKPT[k] for k in range(t + 1, 5)]
        results[name][env] = {}
        for mod in mods:
            b = bands[mod]
            row = {bn_: {} for bn_ in ("core", "shoulder", "full") if b[bn_]}
            # load each checkpoint's full slot matrix ONCE; bands are index slices
            base_all = per_slot_mat(slot_tensors(rd, base_ck, mod))
            sel = {}
            for band_name in row:
                items = b[band_name]
                ids = torch.tensor([sid for sid, _ in items])
                w = torch.tensor([float(a) for _, a in items])
                base = base_all[ids]
                sel[band_name] = (ids, w / w.sum(), base, base.norm(dim=1).clamp(min=1e-8))
            del base_all
            for ck in later:
                cur_all = per_slot_mat(slot_tensors(rd, ck, mod))
                for band_name, (ids, w, base, bn) in sel.items():
                    rel = (cur_all[ids] - base).norm(dim=1) / bn
                    row[band_name][ck] = round(float((w * rel).sum()), 4)
                del cur_all
            del sel
            results[name][env][short_mod(mod)] = row
        core_final = {m: (v.get("core", {}).get(later[-1]) if later else None)
                      for m, v in results[name][env].items()} if later else {}
        print(f"[{name}] {env}: core-final " + "  ".join(
            f"{m}:{(v*100 if v is not None else 0):.0f}%" for m, v in core_final.items()), flush=True)

json.dump(results, open(OUT, "w"), indent=1)
print("wrote", OUT)
