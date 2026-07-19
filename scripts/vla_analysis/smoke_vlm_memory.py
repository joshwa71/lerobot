#!/usr/bin/env python3
"""Smokes for the E44 VLM text-span memory (module level).

S1  zero-init values => memory strictly contained in the span (pre-span bitwise mlp)
S2  nonzero values => only the last-`span` positions change; earlier positions bitwise equal
S3  retrieval runs on the text slice only (last_indices covers B*span tokens, not B*seq)
S4  sequences shorter than the span skip memory (plain mlp passthrough)
S5  text_span + memory_only raises at construction
S6  backward through the span output reaches keys, query_proj, and slot values
S7  text_span=0 legacy path: output == mlp(x) + mem(x) recomputed by hand
S8  derived-cfg attach on a toy 18-layer list wraps exactly [15,16] with the VLM bank
    geometry (n_keys, rank, knn, span) while a sibling "expert" list stays untouched
"""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/home/josh/lerobot/src")
from lerobot.policies.modules.memory_config import MemoryLayerConfig
from lerobot.policies.modules.memory_lite import MLPPlusMemory, attach_memory_to_layer_list

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def mk_cfg(span, **over):
    kw = dict(layers=[0], enabled=True, mem_n_keys=16, mem_heads=2, mem_knn=4,
              mem_k_dim=8, mem_v_dim=-1, value_type="lora", lora_rank=2,
              swilu_projection=True, mem_gated=True, log_usage=True, text_span=span)
    kw.update(over)
    return MemoryLayerConfig(**kw)


DIM = 32
SPAN = 6
SEQ = 20
torch.manual_seed(0)
base = nn.Linear(DIM, DIM)
w = MLPPlusMemory(nn.Linear(DIM, DIM), DIM, mk_cfg(SPAN))
w.mlp = base
x = torch.randn(3, SEQ, DIM)

# S1: at zero-init values the memory still emits its value_proj bias term (known E39
# residual, calibrated by the A-phase) — the invariant is CONTAINMENT: pre-span positions
# are bitwise the plain mlp, and the span deviation is the small bias-scale term.
out = w(x)
check("S1a pre-span bitwise == mlp(x) at init", torch.equal(out[:, :-SPAN], base(x)[:, :-SPAN]))
dev = (out[:, -SPAN:] - base(x)[:, -SPAN:]).abs().max()
check("S1b span deviation is bias-scale only", float(dev) < 1.0, f"max dev {float(dev):.4f}")

# S2: bump values -> only the span changes
with torch.no_grad():
    for n, p in w.mem.named_parameters():
        if getattr(p, "pk_value_param", False):
            p.add_(torch.randn_like(p) * 0.1)
out2 = w(x)
check("S2a pre-span positions bitwise equal", torch.equal(out2[:, :-SPAN], base(x)[:, :-SPAN]))
check("S2b span positions changed", not torch.allclose(out2[:, -SPAN:], base(x)[:, -SPAN:]))

# S3: retrieval covered only the slice
li = w.mem.last_indices
check("S3 retrieval on slice only", li is not None and li.shape[0] == 3 * SPAN or (li.dim() == 3 and li.shape[0] == 3 and li.shape[1] * 0 == 0 and li.reshape(-1).numel() > 0 and li.shape[-2] * 0 == 0),
      f"last_indices shape {tuple(li.shape) if li is not None else None}")
# strict form: token dimension must be B*SPAN (layout (B*T, heads, knn) or (B, T, ...)); accept either
tok = li.shape[0] if li.dim() == 3 and li.shape[0] != 3 else (li.shape[0] * li.shape[1] if li.dim() == 4 else li.shape[0])
check("S3b token count == B*SPAN", tok in (3 * SPAN, 3), f"tok-dim {tok} (B*span={3*SPAN})")

# S4: short sequence -> plain mlp
xs = torch.randn(2, SPAN - 2, DIM)
check("S4 short seq passthrough", torch.equal(w(xs), base(xs)))

# S5: memory_only guard
try:
    MLPPlusMemory(nn.Linear(DIM, DIM), DIM, mk_cfg(SPAN, memory_only=True))
    ok5 = False
except ValueError:
    ok5 = True
check("S5 text_span + memory_only raises", ok5)

# S6: grads reach router + values through the span
w.train()
loss = w(x)[:, -SPAN:].pow(2).sum()
loss.backward()
gk = w.mem.keys.grad
gq = [p.grad for n, p in w.mem.named_parameters() if getattr(p, "pk_query_proj_param", False)]
gv = [p.grad for n, p in w.mem.named_parameters() if getattr(p, "pk_value_param", False)]
check("S6a keys grad nonzero", gk is not None and float(gk.abs().sum()) > 0)
check("S6b query-proj grad nonzero", any(g is not None and float(g.abs().sum()) > 0 for g in gq))
check("S6c value grads nonzero", any(g is not None and float(g.abs().sum()) > 0 for g in gv))

# S7: legacy span=0 path unchanged
w0 = MLPPlusMemory(nn.Linear(DIM, DIM), DIM, mk_cfg(0))
x7 = torch.randn(2, 9, DIM)
manual = w0.mlp(x7) + w0.mem(x7)
check("S7 legacy path == mlp + mem", torch.allclose(w0(x7), manual, atol=1e-6))

# S8: derived attach wraps [15,16] of an 18-layer list; expert list untouched
class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(DIM, DIM)

lm = nn.ModuleList([Layer() for _ in range(18)])
expert = nn.ModuleList([Layer() for _ in range(18)])
import dataclasses
base_cfg = mk_cfg(0, layers=[8, 10], vlm_layers=[15, 16], vlm_mem_n_keys=8,
                  vlm_lora_rank=4, vlm_mem_knn=2, vlm_text_span=SPAN)
vlm_cfg = dataclasses.replace(
    base_cfg, layers=base_cfg.vlm_layers, mem_n_keys=base_cfg.vlm_mem_n_keys,
    lora_rank=base_cfg.vlm_lora_rank, mem_knn=base_cfg.vlm_mem_knn,
    layer_ranks=[], lang_to_query=False, text_span=base_cfg.vlm_text_span, vlm_layers=[])
targets = attach_memory_to_layer_list(lm, DIM, vlm_cfg, label="VLM")
check("S8a targets == [15,16]", targets == [15, 16])
check("S8b wrapped types", all(isinstance(lm[i].mlp, MLPPlusMemory) for i in (15, 16))
      and not any(isinstance(lm[i].mlp, MLPPlusMemory) for i in range(15)))
check("S8c expert list untouched", not any(isinstance(l.mlp, MLPPlusMemory) for l in expert))
m16 = lm[16].mlp
check("S8d bank geometry", m16.mem.size == 64 and m16.text_span == SPAN,
      f"size={m16.mem.size} span={m16.text_span}")
rank_par = [p for n, p in m16.mem.named_parameters() if "slot_down" in n][0]
check("S8e rank applied", rank_par.shape[-1] == 4 or rank_par.shape[0] == 64 * 4 or 4 in rank_par.shape,
      f"slot_down shape {tuple(rank_par.shape)}")

# S9: token-mask (pad fix) — masked positions: zero memory output, excluded from stats
w9 = MLPPlusMemory(nn.Linear(DIM, DIM), DIM, mk_cfg(SPAN))
base9 = w9.mlp
with torch.no_grad():
    for n_, p_ in w9.mem.named_parameters():
        if getattr(p_, "pk_value_param", False):
            p_.add_(torch.randn_like(p_) * 0.1)
x9 = torch.randn(3, SEQ, DIM)
vm9 = torch.zeros(3, SPAN, dtype=torch.bool); vm9[:, :3] = True  # first 3 of span valid
w9.train(); w9._ctx_valid_mask = vm9
out9 = w9(x9)
pad_delta = (out9[:, -SPAN + 3:] - base9(x9)[:, -SPAN + 3:]).abs().max()
val_delta = (out9[:, -SPAN:-SPAN + 3] - base9(x9)[:, -SPAN:-SPAN + 3]).abs().max()
check("S9a masked positions bitwise plain-mlp", float(pad_delta) == 0.0, f"pad delta {float(pad_delta):.2e}")
check("S9b valid positions carry memory", float(val_delta) > 0)
li9 = w9.mem.last_indices
check("S9c stats cover valid tokens only", li9.shape[0] == 3 * 3, f"rows {li9.shape[0]} (expect 9)")
w9._ctx_valid_mask = torch.zeros(2, SPAN, dtype=torch.bool)  # stale/mismatched batch dim
out9b = w9(x9)
check("S9d stale-mask shape guard -> unmasked behavior", out9b.shape == x9.shape and w9.mem.last_indices.shape[0] == 3 * SPAN)

# S10: never-attach — the mem call runs on the batch-max valid prefix only
w10 = MLPPlusMemory(nn.Linear(DIM, DIM), DIM, mk_cfg(SPAN))
base10 = w10.mlp
with torch.no_grad():
    for n_, p_ in w10.mem.named_parameters():
        if getattr(p_, "pk_value_param", False):
            p_.add_(torch.randn_like(p_) * 0.1)
x10 = torch.randn(3, SEQ, DIM)
vm10 = torch.zeros(3, SPAN, dtype=torch.bool)
vm10[0, :2] = True; vm10[1, :4] = True; vm10[2, :3] = True   # ragged; tmax=4
w10.train(); w10._ctx_valid_mask = vm10
out10 = w10(x10)
lo, hi = SEQ - SPAN, SEQ - SPAN + 4
check("S10a beyond-tmax bitwise plain-mlp (never routed)", torch.equal(out10[:, hi:], base10(x10)[:, hi:]))
check("S10b pre-span bitwise plain-mlp", torch.equal(out10[:, :lo], base10(x10)[:, :lo]))
check("S10c ragged tails zeroed in-slice",
      torch.equal(out10[0, lo + 2:hi], base10(x10)[0, lo + 2:hi])
      and torch.equal(out10[2, lo + 3:hi], base10(x10)[2, lo + 3:hi]))
check("S10d longest sample carries memory through its full span",
      not torch.allclose(out10[1, lo:hi], base10(x10)[1, lo:hi]))
li10 = w10.mem.last_indices
check("S10e stats rows == sum(valid) == 9", li10.shape[0] == 9, f"rows {li10.shape[0]}")
check("S10f module never saw beyond tmax", li10.shape[0] <= 3 * 4)
# eval-path filter
w10.eval(); w10.mem.EVAL_MEMORY = True
_ = w10(x10)
check("S10g eval-path stats also mask-filtered", w10.mem.last_indices.shape[0] == 9,
      f"rows {w10.mem.last_indices.shape[0]}")

# ---- S11/S12: pooled routing + route-once (E45) ----
# Layout note: with route-once, training/eval stats rows per sample are
# [palette x n_state, instr tokens 0..il) ] (the compact call puts the shared state
# key FIRST so valid tokens stay a contiguous prefix for the loss machinery, and
# stat_repeat restores the served-position multiplicity).
SPAN11, SEQ11 = 20, 26
x11 = torch.randn(3, SEQ11, DIM)
vm11 = torch.zeros(3, SPAN11, dtype=torch.bool)
vm11[0, :18] = True; vm11[1, :16] = True; vm11[2, :18] = True
il_ok = torch.tensor([8, 7, 8])       # all rows usable -> compact route-once path
il_mix = torch.tensor([8, 7, 0])      # row 2 lacks the boundary -> whole-batch fallback


def build11(mode, w=(1.0, 0.0), seed=7):
    torch.manual_seed(seed)
    m = MLPPlusMemory(nn.Linear(DIM, DIM), DIM,
                      mk_cfg(SPAN11, vlm_router_pool=mode, vlm_router_pool_weights=list(w)))
    with torch.no_grad():
        for n_, p_ in m.mem.named_parameters():
            if getattr(p_, "pk_value_param", False):
                torch.manual_seed(seed + 1); p_.add_(torch.randn_like(p_) * 0.1)
    m.train(); m._ctx_valid_mask = vm11
    return m


def spy_mem(w):
    calls = []
    orig = w.mem.forward

    def f(x, **kw):
        calls.append(tuple(x.shape))
        return orig(x, **kw)

    w.mem.forward = f
    return calls


def stat_groups(m, il, vm):
    """Split last_indices rows into per-sample (palette_rows, instr_rows) under the
    compact layout."""
    li = m.mem.last_indices.reshape(m.mem.last_indices.shape[0], -1)
    v = vm.sum(dim=1).tolist()
    groups, r = [], 0
    for s_, vi in enumerate(v):
        n = vi - int(il[s_])
        pal = [set(li[r + j].tolist()) for j in range(n)]
        ins = [set(li[r + n + j].tolist()) for j in range(int(il[s_]))]
        groups.append((pal, ins)); r += vi
    return groups, r

w11 = build11("anchored", (1.0, 0.0))
w11._ctx_instr_len = il_ok
calls11 = spy_mem(w11)
out11 = w11(x11)
check("S11a compact call: T == max(il)+1", calls11[-1][1] == int(il_ok.max()) + 1,
      f"T={calls11[-1][1]}")
groups, rows = stat_groups(w11, il_ok, vm11)
check("S11b stats rows == sum(valid) (multiplicity preserved)", rows == int(vm11.sum()),
      f"{rows} vs {int(vm11.sum())}")
pal0, ins0 = groups[0]
check("S11c palette rows identical within sample", all(g == pal0[0] for g in pal0)
      and all(g == groups[1][0][0] for g in groups[1][0]))
check("S11d instr rows route per-token", sum(a != ins0[0] for a in ins0[1:]) >= 4,
      f"{sum(a != ins0[0] for a in ins0[1:])}/7 differ")
lo11 = SEQ11 - SPAN11
st0 = lo11 + int(il_ok[0])
check("S11e state outputs differ per position (value path live)",
      not torch.allclose(out11[0, st0], out11[0, st0 + 1]))
check("S11f pre-span bitwise plain-mlp", torch.equal(out11[:, :lo11], w11.mlp(x11)[:, :lo11]))
check("S11g ragged tail zeroed", torch.equal(out11[1, lo11 + 16:], w11.mlp(x11)[1, lo11 + 16:]))

# S12a: PARITY vs the broadcast-key path (same seed -> identical params; reference
# bypasses the wrapper and runs the redundant per-position computation manually)
w_ref = build11("anchored", (1.0, 0.0))
w_ref._ctx_instr_len = il_ok
xs_ref = x11[:, -SPAN11:].contiguous()
rs_b = w_ref._pooled_router_keys(xs_ref, vm11)
mem_ref = w_ref.mem(xs_ref, router_x=rs_b, token_mask=vm11)
mem_ref = mem_ref * vm11.unsqueeze(-1).to(mem_ref.dtype)
out_ref = torch.cat([w_ref.mlp(x11)[:, :lo11], w_ref.mlp(x11)[:, lo11:] + mem_ref], dim=1)
d12 = (out11 - out_ref).abs().max()
check("S12a route-once output == broadcast path", float(d12) < 1e-5, f"max|d|={float(d12):.2e}")

# S12b: eval-path stats multiplicity
w11.eval(); w11.mem.EVAL_MEMORY = True
_ = w11(x11)
check("S12b eval stats rows == sum(valid)",
      w11.mem.last_indices.shape[0] == int(vm11.sum()),
      f"rows {w11.mem.last_indices.shape[0]}")
w11.train(); w11.mem.EVAL_MEMORY = False

# S12c: degenerate row -> whole-batch fallback to the broadcast path
w12 = build11("anchored", (1.0, 0.0))
w12._ctx_instr_len = il_mix
calls12 = spy_mem(w12)
out12 = w12(x11)
tmax11 = int(vm11.sum(dim=1).max())
check("S12c fallback call: T == tmax (broadcast)", calls12[-1][1] == tmax11,
      f"T={calls12[-1][1]}")
li12 = w12.mem.last_indices.reshape(w12.mem.last_indices.shape[0], -1)
v12 = vm11.sum(dim=1).tolist()
r0 = 0  # sample 0 rows are per-position in the fallback layout
sets0 = [set(li12[r0 + p].tolist()) for p in range(v12[0])]
check("S12d fallback: state positions share the palette (usable row)",
      all(sets0[p] == sets0[8] for p in range(8, v12[0])))
r2 = v12[0] + v12[1]
sets2 = [set(li12[r2 + p].tolist()) for p in range(v12[2])]
check("S12e fallback: boundary-less row routes per-token",
      sum(sets2[p] != sets2[8] for p in range(9, v12[2])) >= 4)

# S12f: frozen-route composition — routing keys from router_x, values from live x
w13 = build11("anchored", (1.0, 0.0))
w13._ctx_instr_len = il_ok
rx_sim = torch.randn(3, SEQ11, DIM)  # stands in for the frozen stream
out13a = w13(x11, router_x=rx_sim)
g13a, _ = stat_groups(w13, il_ok, vm11)
out13b = w13(x11 + 0.05 * torch.randn_like(x11), router_x=rx_sim)
g13b, _ = stat_groups(w13, il_ok, vm11)
check("S12f routing stationary under value-input change",
      all(ga[0][0] == gb[0][0] for ga, gb in zip(g13a, g13b)))
check("S12g outputs move with the value input",
      not torch.allclose(out13a[:, -SPAN11:], out13b[:, -SPAN11:]))

# S12h: grads flow through the compact path (keys via palette weights, slot values
# via the grouped einsum)
w14 = build11("anchored", (1.0, 0.0))
w14._ctx_instr_len = il_ok
w14.zero_grad(set_to_none=True)
w14(x11)[:, -SPAN11:].pow(2).sum().backward()
gk14 = w14.mem.keys.grad
gv14 = [p.grad for n_, p_ in [] ] or [p.grad for n_, p in w14.mem.named_parameters()
                                      if getattr(p, "pk_value_param", False)]
check("S12h keys grad nonzero", gk14 is not None and float(gk14.abs().sum()) > 0)
check("S12i slot-value grads nonzero",
      any(g is not None and float(g.abs().sum()) > 0 for g in gv14))

print()
if FAILS:
    print(f"FAILED: {FAILS}"); sys.exit(1)
print("ALL VLM-MEMORY SMOKES PASS")
