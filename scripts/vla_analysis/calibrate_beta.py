#!/usr/bin/env python3
"""Offline (quantile, beta) calibration for grad_scale protection (E41).

Replays the sequential exposure structure from the lr2x run's block JSONs + checkpoint
displacements, under the corefrac u-norm, and prices each beta:

  For each writer block W with prior tasks P(W):
    u(s)      = max over p in P(W) of corefrac-normalized read counts of p   (the real store)
    scale(s)  = (1 - u(s))^beta                                              (per-slot LR mult)
    bleed(p)  = sum_s w_p(s) * |delta_W(s)|            (damage integrand, per victim p)
    kept(p)   = sum_s w_p(s) * |delta_W(s)| * scale(s)
    cost(W)   = 1 - sum_s m_W(s) * |delta_W(s)| * scale(s) / sum_s m_W(s) * |delta_W(s)|
               (writer's own read-mass-weighted write retention loss)

delta_W from checkpoint diffs (L14 + L8), w/m from memory_by_task JSONs. Exposure topology is
arm-invariant (measured), so the lr2x run calibrates for the composed lr2x+protect run.
"""
import os, sys, json, gc
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, "/home/josh/lerobot/scripts/vla_analysis")
from slots import PREFIX

BASE = "/home/josh/lerobot/outputs/train"
RUN = f"{BASE}/libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_lr2x_steps5k_tasks5"
A_CKPT = f"{BASE}/libero_90_pi05_8_10_12_14_frozenroute_rwarmupB_values10k_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model"
STEPS = ["005000", "010000", "015000", "020000", "025000"]
ENV = {0: 4, 1: 6, 2: 9, 3: 2, 4: 7}
LAYERS = [14, 8]
NSLOTS = 147456
BETAS = [0.5, 1, 2, 4, 8, 16]


def key(L, t):
    return f"model.paligemma_with_expert.gemma_expert.model.layers.{L}.mlp.mem.{t}"


def load_slot(path, L):
    with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
        d = f.get_tensor(key(L, "slot_down")).float().reshape(NSLOTS, -1)
        u = f.get_tensor(key(L, "slot_up")).float().reshape(NSLOTS, -1)
    return torch.cat([d, u], dim=1)


def task_profile(t, L):
    d = json.load(open(os.path.join(RUN, "memory_by_task", f"memory_usage_task_{t}.json")))
    node = d["per_module"][PREFIX + str(L)]
    slots = node[next(iter(node))]
    counts = np.zeros(NSLOTS)
    for sk, st in slots.items():
        if st["total_accesses"]:
            counts[int(sk.rsplit("_", 1)[1])] = st["total_accesses"]
    del d
    gc.collect()
    return counts


def corefrac_u(counts):
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts)
    s = np.sort(counts)[::-1]
    cum = np.cumsum(s)
    k = min(int(np.searchsorted(cum, 0.5 * total)), len(s) - 1)
    ref = s[k]
    return np.clip(counts / ref, 0, 1) if ref > 0 else counts / counts.max()


print(f"{'L':>3} {'writer':>7} {'victim':>7} {'beta':>5} {'bleed kept %':>12} {'writer cost %':>13}")
summary = {}
for L in LAYERS:
    counts = {t: task_profile(t, L) for t in range(5)}
    w = {t: counts[t] / counts[t].sum() for t in range(5)}          # victim read-mass
    ucf = {t: corefrac_u(counts[t]) for t in range(5)}
    paths = [A_CKPT] + [os.path.join(RUN, "checkpoints", s, "pretrained_model") for s in STEPS]
    prev = load_slot(paths[0], L)
    for W in range(5):
        cur = load_slot(paths[W + 1], L)
        delta = (cur - prev).norm(dim=1).numpy()
        prev = cur
        if W == 0:
            continue
        u_store = np.max(np.stack([ucf[p] for p in range(W)]), axis=0)
        for beta in BETAS:
            scale = np.power(np.clip(1 - u_store, 0, 1), beta)
            # writer's own cost: its read-mass-weighted write retention
            own = w[W] * delta
            cost = 1 - (own * scale).sum() / max(own.sum(), 1e-12)
            key_s = (L, W, beta)
            summary[key_s] = {"cost": cost, "bleed_kept": {}}
            for p in range(W):
                bl = w[p] * delta
                kept = (bl * scale).sum() / max(bl.sum(), 1e-12)
                summary[key_s]["bleed_kept"][p] = kept
                print(f"{L:>3} t{W}(e{ENV[W]})  t{p}(e{ENV[p]})  {beta:>5} {100*kept:>11.1f}% {100*cost:>12.1f}%")
    del prev, cur
    gc.collect()

# aggregate view: the e9 block (W=2) is the measured bleed channel
print("\n==== AGGREGATE: e9's block (W=2), mean over L14+L8, victims e4+e6 ====")
print(f"{'beta':>5} {'bleed kept %':>13} {'e9 cost %':>10}")
for beta in BETAS:
    kepts, costs = [], []
    for L in LAYERS:
        s = summary[(L, 2, beta)]
        costs.append(s["cost"])
        kepts += list(s["bleed_kept"].values())
    print(f"{beta:>5} {100*np.mean(kepts):>12.1f}% {100*np.mean(costs):>9.1f}%")
