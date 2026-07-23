#!/usr/bin/env python3
"""E52 delivered-damage: read-mass-weighted relative value drift on victim cores.
For victim task T with core-50 C_m per module m (from T's own block JSON, read mass w):
  drift_k(m) = sum_{s in C_m} w_s * ||V_k(s)-V_own(s)|| / ||V_own(s)|| / sum w_s
V(s) = concat of all slot tensors' row s for that module. Baseline = T's own-block
checkpoint; k walks the later per-task checkpoints. Runs: foldin + plain (comp
intermediates deleted; its recorded numbers are the comparator)."""
import json, os
import torch
from safetensors import safe_open
from collections import defaultdict

BASE = "/home/josh/lerobot/outputs/train"
RUNS = {
    "foldin": f"{BASE}/libero_10_seq5_jw_layermax_compact_e9to12_v13to16_beta4_topt3072_lr2x_steps5k",
    "plain":  f"{BASE}/libero_10_seq5_jw_layermax_compact_e9to12_v13to16_beta4_topt1536_steps5k",
}
CKPT = {0: "005000", 1: "010000", 2: "015000", 3: "020000", 4: "025000"}
VICTIMS = [(0, "e4"), (1, "e6"), (2, "e9"), (3, "e2")]
OUT = "/home/josh/lerobot/outputs/analysis/e52/core_drift.json"


def short_mod(m):
    l = m.rsplit(".", 1)[-1]
    return ("E" if "gemma_expert" in m else "V") + l


def load_cores(run_dir, t, topfrac=0.5):
    p = os.path.join(run_dir, "memory_by_task", f"memory_usage_task_{t}.json")
    d = json.load(open(p))["per_module"]
    cores = {}
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
        acc, core = 0, []
        for sid, a in items:
            core.append((sid, a)); acc += a
            if acc >= topfrac * tot:
                break
        cores[mod] = core
    return cores


def slot_tensors(run_dir, ck, mod):
    """returns {tensor_name: tensor} for this module's slot tensors"""
    p = os.path.join(run_dir, "checkpoints", ck, "pretrained_model", "model.safetensors")
    out = {}
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if ".mlp.mem.slot_" in k and k.startswith(mod + "."):
                out[k] = f.get_tensor(k)
    return out


def per_slot_mat(tensors):
    """concat all slot tensors row-wise -> (n_slots, D) float32"""
    mats = []
    n = None
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
    # module list from t0 json
    mods = sorted(load_cores(rd, 0).keys(), key=lambda m: ("gemma_expert" not in m, int(m.rsplit(".", 1)[-1])))
    for t, env in VICTIMS:
        cores = load_cores(rd, t)
        base_ck = CKPT[t]
        later = [CKPT[k] for k in range(t + 1, 5)]
        results[name][env] = {}
        for mod in mods:
            core = cores[mod]
            ids = torch.tensor([sid for sid, _ in core])
            w = torch.tensor([float(a) for _, a in core])
            w = w / w.sum()
            base = per_slot_mat(slot_tensors(rd, base_ck, mod))[ids]
            bn = base.norm(dim=1).clamp(min=1e-8)
            row = {}
            for ck in later:
                cur = per_slot_mat(slot_tensors(rd, ck, mod))[ids]
                rel = (cur - base).norm(dim=1) / bn
                row[ck] = round(float((w * rel).sum()), 4)
                del cur
            results[name][env][short_mod(mod)] = row
            del base
        print(f"[{name}] {env}: " + "  ".join(
            f"{m}:{list(v.values())[-1]*100:.0f}%" for m, v in results[name][env].items()), flush=True)

json.dump(results, open(OUT, "w"), indent=1)
print("wrote", OUT)
