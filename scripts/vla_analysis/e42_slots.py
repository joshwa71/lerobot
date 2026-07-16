#!/usr/bin/env python3
"""E42 slot autopsy over the 3 landed arms (+ stageB/lr2x anchors), L14 and L8.

Per arm/block: mask-updated count (JSON), realized-written count (ckpt-diff |d|>eps),
displacement stats on realized writes, update-event concentration (ev/slot p50/p90),
self-coverage (task read mass on its own realized writes), and the full bleed matrix
(field change each block causes as perceived by every PRIOR task's read distribution).
For softp: e9-block per-slot |d| split by the corefrac-u the run actually used
(max over t0,t1 profiles), contrasted with lr2x's same split (blend verification).
"""
import os, sys, json, gc
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, "/home/josh/lerobot/scripts/vla_analysis")
from slots import PREFIX

BASE = "/home/josh/lerobot/outputs/train"
A_B = f"{BASE}/libero_90_pi05_8_10_12_14_frozenroute_rwarmupB_values10k_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model"
A_AFF = f"{BASE}/libero_90_pi05_8_10_12_14_frozenroute_affine_nogate_values10k_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model"
S5 = ["005000", "010000", "015000", "020000", "025000"]
# NB stageB's per-task checkpoints were deleted on this box (final only) -> excluded;
# affine (bias arm, 1x LR) is the matched-LR displacement anchor for bs64 instead.
RUNS = {
 "affine": (f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_affine_nogate_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps5k_tasks5", S5),
 "lr2x":   (f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_lr2x_steps5k_tasks5", S5),
 "bs64":   (f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_bs64accum2_steps5k_tasks5", S5),
 "softp":  (f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_softprotect_cf_beta4_lr2x_steps5k_tasks5", S5),
 "top3k":  (f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_3072_protect_beta4_steps5k_tasks5", S5),
 "steps7k":(f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps7k_tasks5", ["007000","014000","021000","028000","035000"]),
}
ENV = {0: 4, 1: 6, 2: 9, 3: 2, 4: 7}
N = 147456
EPS = 1e-6


def key(L, t):
    return f"model.paligemma_with_expert.gemma_expert.model.layers.{L}.mlp.mem.{t}"


def load_slot(path, L):
    with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
        d = f.get_tensor(key(L, "slot_down")).float().reshape(N, -1)
        u = f.get_tensor(key(L, "slot_up")).float().reshape(N, -1)
    return torch.cat([d, u], dim=1)


def task_json(run_dir, t):
    return json.load(open(os.path.join(run_dir, "memory_by_task", f"memory_usage_task_{t}.json")))


def profiles(run_dir, t, L):
    d = task_json(run_dir, t)
    node = d["per_module"][PREFIX + str(L)]
    slots = node[next(iter(node))]
    acc = np.zeros(N); upd = np.zeros(N)
    for sk, st in slots.items():
        i = int(sk.rsplit("_", 1)[1])
        if st["total_accesses"]:
            acc[i] = st["total_accesses"]
        if st["total_updates"]:
            upd[i] = st["total_updates"]
    del d; gc.collect()
    return acc, upd


def corefrac_u(counts):
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts)
    s = np.sort(counts)[::-1]
    cum = np.cumsum(s)
    k = min(int(np.searchsorted(cum, 0.5 * total)), len(s) - 1)
    ref = s[k]
    return np.clip(counts / ref, 0, 1) if ref > 0 else counts / counts.max()


out = {}
for name, (rd, steps) in RUNS.items():
    for L in [14, 8]:
        print("=" * 110)
        print(f"RUN {name}  L{L}")
        acc = {}; updj = {}
        for t in range(5):
            acc[t], updj[t] = profiles(rd, t, L)
        w = {t: torch.tensor(acc[t] / max(acc[t].sum(), 1e-9), dtype=torch.float32) for t in range(5)}
        a_ckpt = A_AFF if name == "affine" else A_B
        paths = [a_ckpt] + [os.path.join(rd, "checkpoints", s, "pretrained_model") for s in steps]
        prev = load_slot(paths[0], L)
        for b in range(5):
            cur = load_slot(paths[b + 1], L)
            delta = (cur - prev).norm(dim=1)
            base_norm = prev.norm(dim=1)
            real = (delta > EPS).nonzero().flatten()
            du = delta[real]
            evs = updj[b][updj[b] > 0]
            mask_n = int((updj[b] > 0).sum())
            cov = float((w[b][real]).sum())  # self-coverage: read mass on realized writes
            line = (f"  t{b}(e{ENV[b]}): maskupd={mask_n:6d} realized={len(real):6d} "
                    f"|d| p50={du.median():.3f} mean={du.mean():.3f} p90={du.kthvalue(max(1,int(0.9*len(real)))).values:.3f} "
                    f"ev/slot p50={np.median(evs):.0f} p90={np.percentile(evs,90):.0f} selfcov={100*cov:.1f}%")
            print(line)
            out[f"{name}_L{L}_t{b}"] = {"maskupd": mask_n, "realized": int(len(real)),
                                        "d_p50": float(du.median()), "d_mean": float(du.mean()),
                                        "ev_p50": float(np.median(evs)), "ev_p90": float(np.percentile(evs, 90)),
                                        "selfcov": cov}
            # bleed: field change perceived by every PRIOR task
            for p in range(b):
                num = float((w[p] * delta).sum()); den = float((w[p] * base_norm).sum())
                order = torch.argsort(w[p], descending=True)
                cw = torch.cumsum(w[p][order], 0)
                core = order[: int(torch.searchsorted(cw, 0.5)) + 1]
                numc = float((w[p][core] * delta[core]).sum()); denc = float((w[p][core] * base_norm[core]).sum())
                print(f"      bleed onto t{p}(e{ENV[p]}): full={100*num/den:.2f}%  core50={100*numc/denc:.2f}%")
                out[f"{name}_L{L}_t{b}_bleed_t{p}"] = {"full": num / den, "core50": numc / denc}
            # softp/lr2x: e9-block displacement by u-decile (blend verification)
            if b == 2 and name in ("softp", "lr2x"):
                ucf = np.maximum(corefrac_u(acc[0]), corefrac_u(acc[1]))
                ut = torch.tensor(ucf, dtype=torch.float32)
                print(f"      e9-block |d| by u(prior-usefulness) bin over MASK-updated slots:")
                uids = torch.tensor(np.nonzero(updj[2])[0])
                for lo, hi in [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 1.01)]:
                    sel = uids[(ut[uids] >= lo) & (ut[uids] < hi)]
                    if len(sel) == 0:
                        continue
                    dd = delta[sel]
                    print(f"        u in [{lo},{hi}): n={len(sel):5d}  |d| p50={dd.median():.4f} mean={dd.mean():.4f}  "
                          f"expected_scale={(1-min(hi,1.0))**4:.4f}..{(1-lo)**4:.4f}")
            prev = cur
        del prev, cur
        gc.collect()

json.dump(out, open("/home/josh/lerobot/outputs/analysis/e42/slots_summary.json", "w"), indent=1)
print("saved outputs/analysis/e42/slots_summary.json")
