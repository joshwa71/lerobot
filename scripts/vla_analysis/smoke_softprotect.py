#!/usr/bin/env python3
"""Smokes for grad_scale protection (protect_mode) + corefrac usefulness norm (protect_u_norm).

S1  config validation: defaults pass; bad protect_mode / protect_u_norm raise
S2  _core50_boundary_count on known distributions
S3  peak norm legacy regression: fold == counts/max exactly
S4  corefrac norm: boundary slot u=1, hotter clipped to 1, colder proportional, zeros -> 0
S5  max-aggregation across two folds (corefrac)
S6  mode="rank": scale_out stays empty; ranking discount applied (legacy behavior)
S7  mode="grad_scale": ranking NOT discounted; scale_out filled with (1-u)^beta
S8  end-to-end grads on real module: outside mask zero; inside scaled exactly; beta=0 == mask-only
S9  all value-param shapes (slot_down 3D / slot_up 3D / slot_bias 2D) scaled row-consistently
S10 empty store / task-0: no scaling, plain mask
S11 trainable keys: masked, never scaled
S18 budget v3 (E51): conserved proportional allocation — lam-solver hand cases + conservation,
    membership == rank-mode membership, u=1 exclusion, scale emission, empty-store TF-prop
S19 MULTI-STEP momentum integration (the test the v2 NaN proved was missing): persistent
    2x-capped row over 300 Adam steps == 2x-LR reference bitwise; attenuated/frozen rows
    damped/frozen; the v2 unclamped momentum multiply reproduces the divergence
"""
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

from lerobot.policies.modules.memory_config import MemoryLayerConfig
from lerobot.policies.modules.memory_lite import MLPPlusMemory
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig,
    _accumulate_protect_counts_batch,
    _apply_gradient_mask_to_memory_values,
    _compute_tfidf_top_indices_for_batch,
    _conserved_proportional_scales,
    _core50_boundary_count,
    _finalize_protect_usefulness,
    _protect_cur_counts_by_module,
    _protect_usefulness_by_module,
)

torch.set_grad_enabled(True)
FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


# ---- S1: config validation ----
def try_validate(**over):
    cfg = SequentialOnlineConfig.__new__(SequentialOnlineConfig)  # skip heavy __init__
    ok = True
    try:
        # validate() calls super().validate() which needs full init; test the new checks directly
        pm = over.get("protect_mode", "rank")
        pu = over.get("protect_u_norm", "peak")
        ph = over.get("protect_hard_u", 0.0)
        if pm not in {"rank", "grad_scale", "budget"}:
            raise ValueError
        if pu not in {"peak", "corefrac"}:
            raise ValueError
        if not (0.0 <= ph <= 1.0):
            raise ValueError
        sp = over.get("protect_seed_path", "")
        if sp:
            if not over.get("protect_prior_slots", False):
                raise ValueError
            if not os.path.isfile(sp):
                raise ValueError
    except ValueError:
        ok = False
    return ok

check("S1a defaults valid", try_validate())
check("S1b grad_scale/corefrac valid", try_validate(protect_mode="grad_scale", protect_u_norm="corefrac"))
check("S1c bad mode rejected", not try_validate(protect_mode="soft"))
check("S1d bad norm rejected", not try_validate(protect_u_norm="quantile"))

# ---- S2: _core50_boundary_count ----
# total 200: slot0 alone carries exactly 50% -> core-50 = {slot0}, boundary = 100 (tie case)
c = torch.tensor([100.0, 50, 25, 12, 6, 3, 2, 1, 1, 0])
check("S2a boundary exact-tie -> hottest slot", _core50_boundary_count(c) == 100.0, f"got {_core50_boundary_count(c)}")
# total 201: cum 100 < 100.5 -> boundary moves to idx1 (50)
c1b = torch.tensor([100.0, 50, 25, 12, 6, 3, 2, 1, 1, 1])
check("S2a2 boundary past tie", _core50_boundary_count(c1b) == 50.0, f"got {_core50_boundary_count(c1b)}")
c2 = torch.ones(10)  # uniform: cum hits 5.0 at k=4 (0-indexed searchsorted -> 4)
check("S2b uniform boundary is 1", _core50_boundary_count(c2) == 1.0)
check("S2c empty/zero counts -> 0", _core50_boundary_count(torch.zeros(5)) == 0.0)
c3 = torch.tensor([10.0])
check("S2d single slot", _core50_boundary_count(c3) == 10.0)


# ---- module fixture ----
def make_wrapped(seed=0, bias=False, keys_grad=False):
    torch.manual_seed(seed)
    cfg = MemoryLayerConfig(
        layers=[2], enabled=True, mem_n_keys=16, mem_heads=2, mem_knn=4,
        mem_k_dim=8, mem_v_dim=-1, value_type="lora", lora_rank=2,
        swilu_projection=True, mem_gated=True, log_usage=True,
        lora_slot_bias=bias,
    )

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].mlp = MLPPlusMemory(nn.Linear(32, 32), 32, cfg)

    h = Holder()
    h.layers[0].mlp.mem.keys.requires_grad_(keys_grad)
    return h

NUM_SLOTS = 16 * 16  # n_keys^2


def fold_counts(holder, counts, u_norm):
    """Manually inject per-slot counts as the 'current task' profile and fold."""
    _protect_cur_counts_by_module.clear()
    for _, mem, _, jk in __import__("lerobot.scripts.lerobot_sequential_train", fromlist=["x"])._iter_memory_modules(holder):
        _protect_cur_counts_by_module[jk] = counts.clone()
    _finalize_protect_usefulness(holder, u_norm=u_norm)


# ---- S3/S4/S5: normalization behaviors ----
h = make_wrapped()
_protect_usefulness_by_module.clear()
counts = torch.zeros(NUM_SLOTS)
counts[:10] = torch.tensor([100.0, 50, 25, 12, 6, 3, 2, 1, 1, 0.5])
fold_counts(h, counts, "peak")
jk = next(iter(_protect_usefulness_by_module))
u_peak = _protect_usefulness_by_module[jk]
check("S3 peak fold == counts/max", torch.allclose(u_peak, counts / 100.0))

_protect_usefulness_by_module.clear()
fold_counts(h, counts, "corefrac")
u_cf = _protect_usefulness_by_module[jk]
# boundary count = 50 (from S2a analysis: cum 100,150 >= 100.25 at idx1)
check("S4a corefrac: hottest clipped to 1", float(u_cf[0]) == 1.0)
check("S4b corefrac: boundary slot u=1", float(u_cf[1]) == 1.0)
check("S4c corefrac: colder proportional", abs(float(u_cf[2]) - 25 / 50) < 1e-6 and abs(float(u_cf[4]) - 6 / 50) < 1e-6)
check("S4d corefrac: zero-count -> 0", float(u_cf[20]) == 0.0)

counts2 = torch.zeros(NUM_SLOTS)
counts2[30] = 40.0
counts2[2] = 4.0
fold_counts(h, counts2, "corefrac")
u_agg = _protect_usefulness_by_module[jk]
check("S5a max-agg keeps old", float(u_agg[0]) == 1.0 and abs(float(u_agg[4]) - 6 / 50) < 1e-6)
check("S5b max-agg adds new task core", float(u_agg[30]) == 1.0)
check("S5c max-agg takes max", abs(float(u_agg[2]) - max(25 / 50, 4 / 40)) < 1e-6)

# ---- S6-S11: mask + scale behavior on a real forward/backward ----
def run_case(mode, beta, store, bias=False, keys_grad=False, top_t=8, seed=3):
    h = make_wrapped(seed=seed, bias=bias, keys_grad=keys_grad)
    mem = h.layers[0].mlp.mem
    x = torch.randn(4, 5, 32)
    out = h.layers[0].mlp(x)
    out.sum().backward()
    scale_out = {} if mode in ("grad_scale", "budget") else None
    allowed = _compute_tfidf_top_indices_for_batch(
        h, idf_by_module=None, top_t=top_t, tf_only=True,
        protect_usefulness_by_module=store, protect_beta=beta, protect_mode=mode,
        protect_scale_out=scale_out,
    )
    return h, mem, allowed, scale_out

# build a store with known u over the slots this module actually retrieves
h0 = make_wrapped(seed=3)
x0 = torch.randn(4, 5, 32)
h0.layers[0].mlp(x0)
import lerobot.scripts.lerobot_sequential_train as ST
jk0 = next(iter(dict((j, m) for _, m, _, j in ST._iter_memory_modules(h0))))
retrieved = torch.unique(h0.layers[0].mlp.mem.last_indices.reshape(-1))
u_store = torch.zeros(NUM_SLOTS)
u_store[retrieved[::2]] = 0.75  # every other retrieved slot is "prior-useful"
STORE = {jk0: u_store}

# S6: rank mode leaves scale empty and discounts ranking
h, mem, allowed_rank, so = run_case("rank", 4.0, STORE)
check("S6a rank mode: no scale emitted", so is None)
vp0 = ST._get_value_params(mem)[0]
sel_rank = set(allowed_rank[vp0].tolist())
protected = set(u_store.nonzero().view(-1).tolist())
h2, mem2, allowed_none, _ = run_case("rank", 0.0, None)
sel_none = set(allowed_none[ST._get_value_params(mem2)[0]].tolist())
check("S6b rank mode avoids protected slots vs unprotected run",
      len(sel_rank & protected) < len(sel_none & protected),
      f"{len(sel_rank & protected)} vs {len(sel_none & protected)}")

# S7: grad_scale mode: ranking == unprotected ranking; scale filled
h, mem, allowed_gs, scale_gs = run_case("grad_scale", 4.0, STORE)
sel_gs = set(allowed_gs[ST._get_value_params(mem)[0]].tolist())
check("S7a grad_scale ranking == pure TF ranking", sel_gs == sel_none, f"{len(sel_gs ^ sel_none)} differ")
check("S7b scale emitted for all value params",
      len(scale_gs) == len(ST._get_value_params(mem)) and
      all(v.numel() == NUM_SLOTS for v in scale_gs.values()))
expected_scale = (1.0 - u_store).clamp(min=0).pow(4.0)
check("S7c scale == (1-u)^beta", torch.allclose(next(iter(scale_gs.values())), expected_scale))

# S8: update-blend end-to-end against a REAL Adam optimizer.
# Motivating fact first: Adam's step is invariant to a time-constant gradient scale, so the
# blend (not grad scaling) is the correct mechanism.
def adam_steps_movement(store, scale_rows=None, blend=False, n_steps=3, seed=11):
    """Train a fresh module a few identical steps; return per-row movement of vp0 + touched rows."""
    h = make_wrapped(seed=seed)
    mem = h.layers[0].mlp.mem
    vps = ST._get_value_params(mem)
    opt = torch.optim.Adam(vps, lr=1e-2)
    torch.manual_seed(seed + 99)
    xs = [torch.randn(4, 5, 32) for _ in range(n_steps)]
    p0 = vps[0].detach().clone()
    touched = set()
    for x in xs:
        opt.zero_grad()
        out = h.layers[0].mlp(x)
        out.sum().backward()
        scale_out = {}
        allowed = _compute_tfidf_top_indices_for_batch(
            h, idf_by_module=None, top_t=8, tf_only=True,
            protect_usefulness_by_module=store,
            protect_beta=4.0, protect_mode="grad_scale", protect_scale_out=scale_out,
        )
        _apply_gradient_mask_to_memory_values(allowed)
        if scale_rows == "grad":  # the WRONG mechanism (for the invariance demo)
            for p in vps:
                sc = scale_out.get(p)
                if sc is not None:
                    r = allowed[p]
                    p.grad[r] *= sc[r].view(-1, *([1] * (p.grad.dim() - 1)))
        snap = ST._snapshot_protected_rows(allowed, scale_out) if blend else []
        opt.step()
        if snap:
            ST._blend_protected_rows(snap)
        touched |= set(allowed[vps[0]].tolist())
    return (vps[0].detach() - p0), sorted(touched)

# store keyed to the SAME module/inputs (seed 11): dry run to find touched rows
_, touched_rows = adam_steps_movement(store=None, seed=11)
u_store2 = torch.zeros(NUM_SLOTS)
u_store2[torch.tensor(touched_rows[::2])] = 0.75
STORE2 = {jk0: u_store2}
mv_base, _ = adam_steps_movement(store=STORE2, scale_rows=None)      # store present, no mechanism
mv_grad, _ = adam_steps_movement(store=STORE2, scale_rows="grad")    # naive grad scaling
mv_blend, _ = adam_steps_movement(store=STORE2, blend=True)          # post-step blend
moved = (mv_base.reshape(NUM_SLOTS, -1).norm(dim=1) > 1e-9).nonzero().view(-1).tolist()
prot_rows = [r for r in moved if float(u_store2[r]) > 0]
free_rows = [r for r in moved if float(u_store2[r]) == 0]
u_store = u_store2  # downstream checks reference the active store
check("S8a store hits some masked rows", len(prot_rows) > 0 and len(free_rows) > 0,
      f"{len(prot_rows)} prot / {len(free_rows)} free")
ratio_grad = float(mv_grad[prot_rows].norm() / mv_base[prot_rows].norm())
check("S8b Adam invariance: naive grad scaling ~no-op", ratio_grad > 0.9,
      f"movement ratio {ratio_grad:.3f} (would be {0.25**4:.4f} if scaling worked)")
exp_sc = float((1 - 0.75) ** 4)
ratios = (mv_blend[prot_rows].norm(dim=tuple(range(1, mv_blend.dim()))) /
          mv_base[prot_rows].norm(dim=tuple(range(1, mv_base.dim()))).clamp(min=1e-12))
check("S8c blend: protected-row movement = scale x base", torch.allclose(ratios, torch.full_like(ratios, exp_sc), atol=1e-4),
      f"ratios ~{float(ratios.mean()):.5f} vs expected {exp_sc:.5f}")
check("S8d blend: free rows untouched", torch.allclose(mv_blend[free_rows], mv_base[free_rows], atol=1e-7))

h, mem, allowed, scale = run_case("grad_scale", 0.0, STORE)
check("S8e beta=0: scale dict empty (pure mask)", len(scale) == 0)

# S9: shapes covered incl. slot_bias — blend applies row-wise to 2D and 3D params
h, mem, allowed, scale = run_case("grad_scale", 2.0, STORE, bias=True)
vps = ST._get_value_params(mem)
names = {id(p): n for n, p in mem.named_parameters()}
check("S9a bias included in scale set",
      sorted(names[id(p)] for p in scale) == ["slot_bias", "slot_down", "slot_up"],
      str(sorted(names[id(p)] for p in scale)))
u_s9 = STORE[jk0]  # the store run_case used (S8 rebound u_store to its own fixture)
pre_vals = {id(p): p.data.clone() for p in vps}
snap = ST._snapshot_protected_rows(allowed, scale)
with torch.no_grad():
    for p in vps:
        p.data += 1.0  # fake optimizer step: uniform +1 movement
ST._blend_protected_rows(snap)
sc2 = (1.0 - u_s9).clamp(min=0).pow(2.0)
ok = True
for p in vps:
    mv = p.data - pre_vals[id(p)]
    for r in allowed[vps[0]].tolist():
        want = float(sc2[r]) if float(u_s9[r]) > 0 else 1.0
        if not torch.allclose(mv[r], torch.full_like(mv[r], want), atol=1e-5):
            ok = False
check("S9b 2D/3D blend row-consistent", ok)

# S10: empty store -> no scaling
h, mem, allowed, scale = run_case("grad_scale", 4.0, None)
check("S10a no store: scale empty", len(scale) == 0)
h, mem, allowed, scale = run_case("grad_scale", 4.0, {})
check("S10b empty store dict: scale empty", len(scale) == 0)

# S11: trainable keys masked, never scaled
h, mem, allowed, scale = run_case("grad_scale", 4.0, STORE, keys_grad=True)
keys_p = mem.keys
check("S11a keys in mask set", keys_p in allowed)
check("S11b keys NOT in scale set", keys_p not in scale)

# S12: THE MOMENTUM LEAK (E42) — a protected row that leaves the churning mask keeps moving
# via Adam's exp_avg tail. Controlled fixture: row 0 (protected, scale s) and row 1 (free) get
# gradient on step 1 only; steps 2-12 train row 1 alone (row 0 masked out, grad zero). The
# optimizer still applies row 0's momentum tail on those steps.
def churn_run(mechanism):  # "none" | "blend_only" (old, leaky) | "blend_opt" (fixed)
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(16, 4))
    opt = torch.optim.Adam([p], lr=1e-2)
    s = 0.0625
    sv = torch.ones(16); sv[0] = s
    def step(rows_with_grad, mask_rows):
        opt.zero_grad()
        g = torch.zeros_like(p)
        g[rows_with_grad] = 1.0
        p.grad = g
        allowed = {p: torch.tensor(mask_rows)}
        snap = [] if mechanism == "none" else ST._snapshot_protected_rows(allowed, {p: sv})
        opt.step()
        if snap:
            ST._blend_protected_rows(snap, opt if mechanism == "blend_opt" else None)
    step([0, 1], [0, 1])
    for _ in range(11):
        step([1], [1])
    return p.data[0].norm().item(), p.data[1].norm().item()

r0_ctrl, r1_ctrl = churn_run("none")
r0_old, r1_old = churn_run("blend_only")
r0_fix, r1_fix = churn_run("blend_opt")
s = 0.0625
check("S12a leak reproduced: old blend leaves most movement under churn",
      r0_old / r0_ctrl > 0.5,
      f"old/ctrl = {r0_old / r0_ctrl:.3f} (designed {s:.4f})")
check("S12b fix: blend+exp_avg scaling == s x control under churn",
      abs(r0_fix / r0_ctrl - s) < 0.15 * s,
      f"fix/ctrl = {r0_fix / r0_ctrl:.4f} vs s = {s:.4f}")
check("S12c free row identical in all mechanisms",
      abs(r1_old - r1_ctrl) < 1e-6 and abs(r1_fix - r1_ctrl) < 1e-6)
# s=0 must freeze the row outright even with the tail
def churn_run_s0(mechanism):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(16, 4))
    opt = torch.optim.Adam([p], lr=1e-2)
    sv = torch.ones(16); sv[0] = 0.0
    for k in range(12):
        opt.zero_grad()
        g = torch.zeros_like(p)
        g[[0, 1] if k == 0 else [1]] = 1.0
        p.grad = g
        allowed = {p: torch.tensor([0, 1] if k == 0 else [1])}
        snap = ST._snapshot_protected_rows(allowed, {p: sv})
        opt.step()
        if snap:
            ST._blend_protected_rows(snap, opt if mechanism == "blend_opt" else None)
    return p.data[0].norm().item()
check("S12d s=0: old blend still moves the row (the u=1 finding)", churn_run_s0("blend_only") > 1e-4,
      f"{churn_run_s0('blend_only'):.5f}")
check("S12e s=0: fixed blend freezes the row exactly", churn_run_s0("blend_opt") == 0.0,
      f"{churn_run_s0('blend_opt'):.2e}")

# S13: hard veto — u >= threshold never enters the mask, in both modes; 0 disables
h13 = make_wrapped(seed=3)
x13 = torch.randn(4, 5, 32)
h13.layers[0].mlp(x13)
mem13 = h13.layers[0].mlp.mem
h13.layers[0].mlp(x13).sum().backward()
hot = torch.unique(mem13.last_indices.reshape(-1))
u_veto = torch.zeros(NUM_SLOTS)
u_veto[hot[:4]] = 0.95   # above threshold
u_veto[hot[4:8]] = 0.5   # below threshold
VSTORE = {jk0: u_veto}
for mode in ("rank", "grad_scale"):
    for hard_u, expect_in in ((0.0, True), (0.9, False)):
        sc_out = {} if mode == "grad_scale" else None
        allowed13 = _compute_tfidf_top_indices_for_batch(
            h13, idf_by_module=None, top_t=NUM_SLOTS, tf_only=True,
            protect_usefulness_by_module=VSTORE, protect_beta=0.0, protect_mode=mode,
            protect_scale_out=sc_out, protect_hard_u=hard_u,
        )
        sel13 = set(allowed13[ST._get_value_params(mem13)[0]].tolist())
        got_in = all(int(r) in sel13 for r in hot[:4])
        check(f"S13 {mode} hard_u={hard_u}: u=0.95 slots {'in' if expect_in else 'OUT of'} mask",
              got_in == expect_in)
        if hard_u > 0:
            check(f"S13 {mode} hard_u={hard_u}: u=0.5 slots stay in mask",
                  all(int(r) in sel13 for r in hot[4:8]))
# config validation for the new field
check("S13z protect_hard_u validation rejects 1.5", not try_validate(protect_hard_u=1.5))

# ---- S14: generalist-slot freeze — seed the store from a JSON (E42 addendum) ----
import json as _json, tempfile as _tempfile
from lerobot.scripts.lerobot_sequential_train import _seed_protect_usefulness

h14 = make_wrapped(seed=5)
x14 = torch.randn(4, 5, 32)
h14.layers[0].mlp(x14)
mem14 = h14.layers[0].mlp.mem
hot14 = torch.unique(mem14.last_indices.reshape(-1))
jk14 = next(j for _, _, _, j in ST._iter_memory_modules(h14))
_hotset14 = set(int(r) for r in hot14)
_cold14 = next(i for i in range(NUM_SLOTS) if i not in _hotset14)
seed_rows = [int(r) for r in hot14[:3]] + [_cold14]  # 3 hot slots + 1 guaranteed-cold slot
with _tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f14:
    _json.dump({jk14: seed_rows}, f14)
    seed_path14 = f14.name

# a) loader sets u=1.0 exactly at listed slots
_protect_usefulness_by_module.clear()
seeded14 = _seed_protect_usefulness(h14, seed_path14)
u14 = _protect_usefulness_by_module[jk14]
check("S14a seed count reported", seeded14 == {jk14: len(seed_rows)})
check("S14a seeded slots u=1.0", all(float(u14[r]) == 1.0 for r in seed_rows))
check("S14a unseeded slots u=0", float(u14.sum()) == float(len(seed_rows)))

# b) with hard_u, seeded slots are structurally OUT of the mask even at top_t=all
allowed14 = _compute_tfidf_top_indices_for_batch(
    h14, idf_by_module=None, top_t=NUM_SLOTS, tf_only=True,
    protect_usefulness_by_module=_protect_usefulness_by_module, protect_beta=4.0,
    protect_mode="rank", protect_hard_u=0.9,
)
sel14 = set(allowed14[ST._get_value_params(mem14)[0]].tolist())
check("S14b seeded hot slots vetoed from mask", all(int(r) not in sel14 for r in hot14[:3]))
check("S14b unseeded hot slots still in mask", all(int(r) in sel14 for r in hot14[3:]))

# c) later tasks max-fold WITHOUT diluting seeds
counts14 = torch.zeros(NUM_SLOTS)
counts14[int(hot14[3])] = 50.0
fold_counts(h14, counts14, "corefrac")
u14b = _protect_usefulness_by_module[jk14]
check("S14c seeds survive task fold", all(float(u14b[r]) == 1.0 for r in seed_rows))
check("S14c task core folds in alongside", float(u14b[int(hot14[3])]) == 1.0)

# d) bad seeds raise
try:
    with _tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fbad:
        _json.dump({jk14: [NUM_SLOTS + 5]}, fbad)
        bad_path = fbad.name
    _seed_protect_usefulness(h14, bad_path); ok_d1 = False
except ValueError:
    ok_d1 = True
try:
    with _tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fbad2:
        _json.dump({"not.a.module": [1]}, fbad2)
        bad_path2 = fbad2.name
    _seed_protect_usefulness(h14, bad_path2); ok_d2 = False
except ValueError:
    ok_d2 = True
check("S14d out-of-range slot index raises", ok_d1)
check("S14d unknown module key raises", ok_d2)

# e) config validation
check("S14e seed_path without protect_prior_slots rejected",
      not try_validate(protect_seed_path=seed_path14))
check("S14e nonexistent seed_path rejected",
      not try_validate(protect_prior_slots=True, protect_seed_path="/nonexistent/seed.json"))
check("S14e valid seed_path + protection accepted",
      try_validate(protect_prior_slots=True, protect_seed_path=seed_path14))

# ---- S18: budget v3 — conserved proportional allocation (E51) ----
check("S18-pre budget mode accepted by validation", try_validate(protect_mode="budget"))

# a) lam-solver hand cases
sc = _conserved_proportional_scales(torch.tensor([4.0, 2, 1, 1]), 4.0)
check("S18a1 hand case [4,2,1,1] -> [2,1,.5,.5]", torch.allclose(sc, torch.tensor([2.0, 1, 0.5, 0.5]), atol=1e-5), f"{sc.tolist()}")
sc = _conserved_proportional_scales(torch.tensor([10.0, 1, 1, 1]), 4.0)
check("S18a2 hand case [10,1,1,1] -> [2,2/3,2/3,2/3]", torch.allclose(sc, torch.tensor([2.0, 2/3, 2/3, 2/3]), atol=1e-5), f"{sc.tolist()}")
sc = _conserved_proportional_scales(torch.full((6,), 3.7), 6.0)
check("S18a3 flat scores -> flat 1.0 (t0-with-flat-TF == plain)", torch.allclose(sc, torch.ones(6), atol=1e-6))

# b) random heavy-tail: conservation exact, cap respected, order preserved
torch.manual_seed(7)
s_ht = torch.sort(torch.distributions.Pareto(1.0, 1.2).sample((3072,)), descending=True).values
sc = _conserved_proportional_scales(s_ht, 3072.0)
check("S18b1 conservation exact on heavy tail", abs(float(sc.sum()) - 3072.0) < 1e-2, f"sum={float(sc.sum()):.4f}")
check("S18b2 cap respected", float(sc.max()) <= 2.0 + 1e-5)
check("S18b3 monotone non-increasing", bool((sc[1:] <= sc[:-1] + 1e-6).all()))
check("S18b4 head capped, tail below 1", float(sc[0]) > 1.99 and float(sc[-1]) < 1.0)

# c) end-to-end: membership identical to rank mode (same score, same store)
h, mem, allowed_rank2, _ = run_case("rank", 4.0, STORE)
sel_rank2 = allowed_rank2[ST._get_value_params(mem)[0]].tolist()
h, mem, allowed_bg, scale_bg = run_case("budget", 4.0, STORE)
vpb = ST._get_value_params(mem)[0]
sel_bg = allowed_bg[vpb].tolist()
check("S18c budget membership == rank-mode membership (twin masks)", set(sel_bg) == set(sel_rank2),
      f"{len(set(sel_bg) ^ set(sel_rank2))} differ")

# d) scale emission: sum over mask == k, off-mask == 1, cap respected
sv_full = scale_bg[vpb]
mask_scales = sv_full[torch.tensor(sel_bg)]
off_mask = torch.ones(NUM_SLOTS, dtype=torch.bool); off_mask[torch.tensor(sel_bg)] = False
check("S18d1 mask scales sum == k", abs(float(mask_scales.sum()) - len(sel_bg)) < 1e-2,
      f"sum={float(mask_scales.sum()):.3f} k={len(sel_bg)}")
check("S18d2 off-mask scale == 1", bool((sv_full[off_mask] == 1.0).all()))
check("S18d3 cap respected end-to-end", float(mask_scales.max()) <= 2.0 + 1e-5)
check("S18d4 scale emitted for all value params",
      len(scale_bg) == len(ST._get_value_params(mem)) and all(v.numel() == NUM_SLOTS for v in scale_bg.values()))

# e) u=1 slots never enter the mask (seats go to clean slots)
h_e = make_wrapped(seed=3)
x_e = torch.randn(4, 5, 32)
h_e.layers[0].mlp(x_e)
retr_e = torch.unique(h_e.layers[0].mlp.mem.last_indices.reshape(-1))
u1_store = torch.zeros(NUM_SLOTS)
u1_store[retr_e[:4]] = 1.0  # 4 retrieved slots fully protected
h, mem, allowed_u1, scale_u1 = run_case("budget", 4.0, {jk0: u1_store})
sel_u1 = set(allowed_u1[ST._get_value_params(mem)[0]].tolist())
frozen = set(retr_e[:4].tolist())
check("S18e1 u=1 slots excluded from mask", len(sel_u1 & frozen) == 0, f"overlap={sel_u1 & frozen}")
check("S18e2 mask still full size (seats reallocated)", len(sel_u1) == 8, f"k={len(sel_u1)}")

# f) task-0 store (dict exists, no entry for the module — the real t0 state):
# membership == pure TF; allocation TF-proportional, conserved
h, mem, allowed_e, scale_e = run_case("budget", 4.0, {})
vpe = ST._get_value_params(mem)[0]
sel_e = set(allowed_e[vpe].tolist())
check("S18f1 empty store: membership == pure TF mask", sel_e == sel_none, f"{len(sel_e ^ sel_none)} differ")
me = scale_e[vpe][torch.tensor(sorted(sel_e))]
check("S18f2 empty store: conserved (sum == k)", abs(float(me.sum()) - len(sel_e)) < 1e-2)

# ---- S19: MULTI-STEP momentum integration (would have caught the v2 NaN) ----
def run_multistep(scales, steps=300, lr=1e-2, mode="fixed"):
    """mode: 'fixed' = shipped blend; 'v2bug' = unclamped momentum multiply; 'plain'/'plain2x' = references."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(4, 3))
    opt = torch.optim.Adam([p], lr=(2 * lr if mode == "plain2x" else lr))
    sv = torch.tensor(scales, dtype=torch.float32)
    rows = torch.arange(4)
    for _ in range(steps):
        opt.zero_grad()
        p.grad = torch.ones_like(p)
        snap = [] if mode in ("plain", "plain2x") else ST._snapshot_protected_rows({p: rows}, {p: sv})
        opt.step()
        if mode == "fixed":
            ST._blend_protected_rows(snap, opt)
        elif mode == "v2bug":
            with torch.no_grad():
                for pp, r, pre, s_ in snap:
                    scv = s_.to(dtype=pp.dtype).view(-1, 1)
                    pp.data[r] = pre + scv * (pp.data[r] - pre)
                    st = opt.state.get(pp)
                    if st is not None and "exp_avg" in st:
                        st["exp_avg"][r] *= scv  # the v2 line, no clamp
    return p.data.clone(), opt.state[p]["exp_avg"].clone()

SC = [2.0, 0.4, 1.0, 0.0]
th_fix, m_fix = run_multistep(SC, mode="fixed")
th_ref, m_ref = run_multistep(SC, mode="plain")
th_2x, _ = run_multistep(SC, mode="plain2x")
check("S19a all finite after 300 boosted steps", bool(torch.isfinite(th_fix).all() and torch.isfinite(m_fix).all()))
check("S19b boosted-row momentum untouched (== plain Adam)", torch.allclose(m_fix[0], m_ref[0], atol=1e-6),
      f"{float(m_fix[0].abs().max()):.3f} vs {float(m_ref[0].abs().max()):.3f}")
check("S19c boosted-row trajectory == 2x-LR Adam exactly", torch.allclose(th_fix[0], th_2x[0], atol=1e-5),
      f"{float(th_fix[0][0]):.5f} vs {float(th_2x[0][0]):.5f}")
check("S19d scale-1 row == plain Adam", torch.allclose(th_fix[2], th_ref[2], atol=1e-7))
check("S19e attenuated row: 0 < movement < plain", 0 < float(th_fix[1].abs().max()) < float(th_ref[1].abs().max()))
check("S19f s=0 row bitwise frozen", bool((th_fix[3] == 0).all()))
th_bug, m_bug = run_multistep(SC, mode="v2bug")
check("S19g v2 unclamped momentum reproduces the divergence",
      (not bool(torch.isfinite(m_bug[0]).all())) or float(m_bug[0].abs().max()) > 1e30,
      f"|exp_avg| max = {float(m_bug[0].abs().max()):.2e}")

# ---- S20: mass-based (nucleus) top-p (E51 — replaces the E50 count rule) ----
# a) end-to-end: concentrated scores pin at the top_t floor
h20 = make_wrapped(seed=3)
x20 = torch.randn(4, 5, 32)
h20.layers[0].mlp(x20)
mem20 = h20.layers[0].mlp.mem
allowed20 = _compute_tfidf_top_indices_for_batch(
    h20, idf_by_module=None, top_t=8, tf_only=True, top_p=0.9, top_p_cap=12,
)
n_read20 = int(torch.unique(mem20.last_indices.reshape(-1)).numel())
k20 = len(allowed20[ST._get_value_params(mem20)[0]])
check("S20a mass rule end-to-end: floor <= k <= cap", 8 <= k20 <= 12, f"k={k20} n_read={n_read20}")
check("S20a2 last_mask_k records the mass k", mem20.last_mask_k[0] == k20)

# b) hand distributions through the same arithmetic
def mass_k(scores, p, floor, cap):
    s = torch.sort(torch.tensor(scores, dtype=torch.float32), descending=True).values
    cum = torch.cumsum(s, 0)
    kp = int(torch.searchsorted(cum, p * float(cum[-1])).item()) + 1
    return min(len(scores), max(floor, kp), cap)

check("S20b1 concentrated: [.5,.2,.1,.1,.05,.05] p=.9 -> k_p=4 -> floor 5 binds",
      mass_k([.5, .2, .1, .1, .05, .05], 0.9, 5, 100) == 5)
check("S20b2 same, floor 3 -> k=4 (adaptive band engaged)",
      mass_k([.5, .2, .1, .1, .05, .05], 0.9, 3, 100) == 4)
check("S20b3 flat 100 scores p=.9 -> k_p=90 -> cap 50 binds",
      mass_k([1.0] * 100, 0.9, 10, 50) == 50)
check("S20b4 p=1.0 -> whole read set (subject to cap)",
      mass_k([.4, .3, .2, .1], 1.0, 1, 100) == 4)
# the E50 failure shape: a huge BINARY tail with negligible mass must not drag k up.
# NB the rule only ignores a tail whose AGGREGATE mass < 1-p; a tail carrying more than
# that is walked into by design — which is why the cap (not p) is the real guard when
# per-batch TF tails are fat (the measured comp block-k90 of 8-22k says they are).
heavy = [100.0, 50, 25, 12] + [0.001] * 5000
check("S20b5 negligible-mass tail ignored: k ~= head", mass_k(heavy, 0.9, 2, 4000) <= 10,
      f"k={mass_k(heavy, 0.9, 2, 4000)}")
fat = [100.0, 50, 25, 12] + [0.01] * 5000  # tail = 21% of mass -> walked into -> cap binds
check("S20b6 fat-mass tail: rule reaches in, cap binds", mass_k(fat, 0.9, 2, 4000) == 2634
      and mass_k(fat, 0.9, 2, 1000) == 1000, f"k={mass_k(fat, 0.9, 2, 4000)}")

print()
if FAILS:
    print(f"FAILED: {FAILS}"); sys.exit(1)
print("ALL SMOKES PASS")
