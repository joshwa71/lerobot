#!/usr/bin/env python3
"""Smokes for value_input_noise_* (E57 off-trail lever: training-only noise on the x
consumed by the LoRA slot transforms; router/gate/swilu/MLP stay clean).

S1  no override / p=0 / sigma=0: forward BITWISE == baseline, train and eval modes
S2  flag ON, EVAL mode: BITWISE == baseline (training-only mechanism)
S3  flag ON, TRAIN mode: output differs; retrieved indices IDENTICAL (router untouched)
S4  per-layer threading via attach_memory_to_layer_list: sigma=0 layer bitwise clean,
    sigma>0 layer differs; length-mismatch raises
S5  grads flow to slot_down/slot_up through the noised input
S6  amp=[1,1] == amp unset; amp draw deterministic under seed
S7  apply_shared_palette (route-once) path: train differs on valid rows, padded rows
    stay zero, eval bitwise clean
S8  scale sanity: measured output perturbation grows with sigma
"""
import sys

import torch
import torch.nn as nn

from lerobot.policies.modules.memory_config import MemoryLayerConfig
from lerobot.policies.modules.memory_lite import (
    HashingMemoryLite,
    MLPPlusMemory,
    attach_memory_to_layer_list,
)

torch.set_grad_enabled(True)
FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def make_cfg(**over):
    cfg = MemoryLayerConfig(
        layers=[2], enabled=True, mem_n_keys=16, mem_heads=2, mem_knn=4,
        mem_k_dim=8, mem_v_dim=-1, value_type="lora", lora_rank=2,
        swilu_projection=True, mem_gated=True, log_usage=True,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def build(seed=0, sigma=None, **over):
    torch.manual_seed(seed)
    m = HashingMemoryLite(input_dim=32, output_dim=32, cfg=make_cfg(**over),
                          value_noise_sigma_override=sigma)
    # nonzero slot_up so the value path has signal (init is zero)
    with torch.no_grad():
        m.slot_up.normal_(0, 0.05)
    return m


x = torch.randn(4, 3, 32)

# ---- S1: off == baseline (train + eval) ----
base = build(seed=0)
off1 = build(seed=0)                                         # no override
off2 = build(seed=0, sigma=0.0, value_input_noise_p=0.5)     # sigma 0
off3 = build(seed=0, sigma=0.5, value_input_noise_p=0.0)     # p 0
for mode in ("train", "eval"):
    getattr(base, mode)()
    torch.manual_seed(11)
    y0 = base(x.clone())
    for tag, m in (("no-override", off1), ("sigma0", off2), ("p0", off3)):
        getattr(m, mode)()
        torch.manual_seed(11)
        y = m(x.clone())
        check(f"S1 {tag} bitwise ({mode})", torch.equal(y0, y))

# ---- S2: flag on, eval mode == baseline ----
on = build(seed=0, sigma=0.5, value_input_noise_p=0.5)
on.eval(); base.eval()
torch.manual_seed(11); y0 = base(x.clone())
torch.manual_seed(11); y1 = on(x.clone())
check("S2 eval-mode bitwise", torch.equal(y0, y1))

# ---- S3: flag on, train mode: output differs, indices identical ----
on.train(); base.train()
torch.manual_seed(11); y0 = base(x.clone()); i0 = base.last_indices.clone()
torch.manual_seed(11); y1 = on(x.clone()); i1 = on.last_indices.clone()
check("S3a train output differs", not torch.equal(y0, y1),
      f"max|dy|={(y0 - y1).abs().max():.4f}")
check("S3b retrieval indices identical", torch.equal(i0, i1))

# ---- S4: per-layer threading via attach ----
class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(32, 32)

layers = nn.ModuleList([Layer() for _ in range(4)])
cfg = make_cfg(layers=[1, 3], value_input_noise_p=0.5,
               value_input_noise_sigma=[0.0, 0.8])
torch.manual_seed(0)
attach_memory_to_layer_list(layers, dim=32, cfg=cfg, label="SMOKE")
for li, expect_noise in ((1, False), (3, True)):
    m = layers[li].mlp
    with torch.no_grad():
        m.mem.slot_up.normal_(0, 0.05)
    m.train()
    torch.manual_seed(7); ya = m(x.clone())
    torch.manual_seed(7); yb = m(x.clone())
    # noised layer consumes extra RNG -> same seed still gives same draw; instead
    # compare against the module with noise force-disabled
    p_saved = m.mem.value_noise_p
    m.mem.value_noise_p = 0.0
    torch.manual_seed(7); yc = m(x.clone())
    m.mem.value_noise_p = p_saved
    check(f"S4 L{li} determinism under seed", torch.equal(ya, yb))
    check(f"S4 L{li} {'differs from' if expect_noise else 'matches'} clean",
          (not torch.equal(ya, yc)) if expect_noise else torch.equal(ya, yc))
try:
    bad = make_cfg(layers=[1, 3], value_input_noise_sigma=[0.5])
    attach_memory_to_layer_list(nn.ModuleList([Layer() for _ in range(4)]), dim=32,
                                cfg=bad, label="SMOKE")
    check("S4 length mismatch raises", False)
except ValueError:
    check("S4 length mismatch raises", True)

# ---- S5: grads reach slot params through the noised input ----
on.train()
on.zero_grad(set_to_none=True)
torch.manual_seed(11)
out = on(x.clone().requires_grad_(False))
out.square().mean().backward()
gd = on.slot_down.grad
gu = on.slot_up.grad
check("S5 grads on slot_down", gd is not None and gd.abs().sum() > 0)
check("S5 grads on slot_up", gu is not None and gu.abs().sum() > 0)

# ---- S6: amp semantics ----
a1 = build(seed=0, sigma=0.5, value_input_noise_p=0.5, value_input_noise_amp=[1.0, 1.0])
a2 = build(seed=0, sigma=0.5, value_input_noise_p=0.5)
a1.train(); a2.train()
torch.manual_seed(13); y1 = a1(x.clone())
torch.manual_seed(13); y2 = a2(x.clone())
check("S6a amp [1,1] == unset", torch.equal(y1, y2))
a3 = build(seed=0, sigma=0.5, value_input_noise_p=0.5, value_input_noise_amp=[0.5, 1.5])
a3.train()
torch.manual_seed(13); y3a = a3(x.clone())
torch.manual_seed(13); y3b = a3(x.clone())
check("S6b amp draw deterministic under seed", torch.equal(y3a, y3b))

# ---- S7: palette path (route-once value application) ----
pal_on = build(seed=0, sigma=0.8, value_input_noise_p=0.5)
pal_off = build(seed=0)
B, N = 2, 5
x_pos = torch.randn(B, N, 32)
pos_mask = torch.ones(B, N, dtype=torch.bool)
pos_mask[:, -2:] = False  # padded tail
K = pal_on.heads * pal_on.knn
idx_row = torch.randint(0, pal_on.size, (B, K))
w_row = torch.softmax(torch.randn(B, K), dim=-1)
router_key = torch.randn(B, 32)
for mode, expect_diff in (("eval", False), ("train", True)):
    getattr(pal_on, mode)(); getattr(pal_off, mode)()
    torch.manual_seed(17)
    ya = pal_off.apply_shared_palette(x_pos.clone(), pos_mask, idx_row, w_row, router_key)
    torch.manual_seed(17)
    yb = pal_on.apply_shared_palette(x_pos.clone(), pos_mask, idx_row, w_row, router_key)
    check(f"S7 palette {mode} {'differs' if expect_diff else 'bitwise'}",
          (not torch.equal(ya, yb)) if expect_diff else torch.equal(ya, yb))
    check(f"S7 palette {mode} padded rows zero", bool((yb[:, -2:].abs().sum() == 0).item()))

# ---- S8: perturbation grows with sigma ----
mags = []
for sig in (0.1, 0.5, 1.5):
    m = build(seed=0, sigma=sig, value_input_noise_p=1.0)
    m.train()
    torch.manual_seed(19); yn = m(x.clone())
    m.value_noise_p = 0.0
    torch.manual_seed(19); yc = m(x.clone())
    mags.append(float((yn - yc).abs().mean()))
check("S8 perturbation monotone in sigma", mags[0] < mags[1] < mags[2],
      f"{[round(v, 5) for v in mags]}")

print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + str(FAILS)} "
      f"({len(FAILS)} fails)")
sys.exit(1 if FAILS else 0)
