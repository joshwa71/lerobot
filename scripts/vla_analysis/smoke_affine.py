#!/usr/bin/env python3
"""Smokes for lora_slot_bias (affine slots) + eval_final_episodes.

S1  flag OFF: state-dict keys unchanged, _slot_param_names unchanged
S2  flag ON @ init: forward BITWISE == flag-off (zero bias)
S3  bias moves output; grads land ONLY on retrieved slot rows
S4  legacy checkpoint load (missing slot_bias) -> strict=False, bias stays zero, forward == off
S5  trainer integration: _get_value_params includes slot_bias; TF-IDF row mask zeroes it
S6  optimizer tags (pk_value_param, fixed_lr)
S7  composes with mem_gated=False
S8  composes with lora_rank_override
S9  _eval_n_episodes_for_task selection logic
S10 offload branch parity with nonzero bias
"""
import sys
from types import SimpleNamespace

import torch

from lerobot.policies.modules.memory_config import MemoryLayerConfig
from lerobot.policies.modules.memory_lite import HashingMemoryLite

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
        swilu_projection=True, mem_gated=True, log_usage=False,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def build(seed=0, **over):
    torch.manual_seed(seed)
    m = HashingMemoryLite(input_dim=32, output_dim=32, cfg=make_cfg(**over))
    m.eval()
    return m


x = torch.randn(2, 3, 32)

# ---- S1: flag off unchanged ----
m_off = build(seed=0)
check("S1a no slot_bias key when off", "slot_bias" not in dict(m_off.named_parameters()))
check("S1b slot names off", m_off._slot_param_names() == ("slot_down", "slot_up"))

# ---- S2: flag on @ init == off ----
m_on = build(seed=0, lora_slot_bias=True)
shared_equal = all(
    torch.equal(dict(m_off.named_parameters())[k], v)
    for k, v in m_on.named_parameters() if k != "slot_bias"
)
check("S2a same-seed shared params identical", shared_equal)
check("S2b slot names on", m_on._slot_param_names() == ("slot_down", "slot_up", "slot_bias"))
check("S2c bias zero-init", float(m_on.slot_bias.abs().max()) == 0.0)
with torch.no_grad():
    y_off = m_off(x.clone())
    y_on = m_on(x.clone())
check("S2d forward bitwise equal @ zero bias", torch.equal(y_off, y_on),
      f"maxdiff={float((y_off - y_on).abs().max()):.2e}")

# ---- S3: bias moves output; grads only on retrieved rows ----
with torch.no_grad():
    m_on.slot_bias.normal_(0, 0.5)
    y_biased = m_on(x.clone())
check("S3a nonzero bias changes output", not torch.equal(y_off, y_biased),
      f"delta={float((y_off - y_biased).abs().max()):.2e}")
m_on.train()
out = m_on(x.clone())
out.sum().backward()
g = m_on.slot_bias.grad
retrieved = set()
m_on.log_usage = True
with torch.no_grad():
    m_on(x.clone())
retrieved = set(m_on.last_indices.reshape(-1).tolist())
rows_with_grad = set(torch.nonzero(g.abs().sum(dim=1) > 0).view(-1).tolist())
check("S3b bias grad exists", g is not None and float(g.abs().max()) > 0)
check("S3c grads only on retrieved rows", rows_with_grad.issubset(retrieved),
      f"{len(rows_with_grad)} grad rows / {len(retrieved)} retrieved")
m_on.log_usage = False
m_on.eval()
m_on.zero_grad()

# ---- S4: legacy checkpoint load ----
m_legacy = build(seed=0, lora_slot_bias=True)
res = m_legacy.load_state_dict(m_off.state_dict(), strict=False)
check("S4a missing == {slot_bias}", list(res.missing_keys) == ["slot_bias"] and not res.unexpected_keys,
      f"missing={res.missing_keys} unexpected={res.unexpected_keys}")
with torch.no_grad():
    y_leg = m_legacy(x.clone())
check("S4b legacy-load forward == off", torch.equal(y_leg, y_off))

# ---- S5: trainer integration ----
from lerobot.scripts.lerobot_sequential_train import (
    _apply_gradient_mask_to_memory_values,
    _eval_n_episodes_for_task,
    _get_value_params,
)

vps = _get_value_params(m_on)
names = {id(p): n for n, p in m_on.named_parameters()}
vp_names = [names[id(p)] for p in vps]
check("S5a _get_value_params includes bias", vp_names == ["slot_down", "slot_up", "slot_bias"], str(vp_names))
m_on.slot_bias.grad = torch.ones_like(m_on.slot_bias)
allowed = torch.tensor([1, 5, 9])
_apply_gradient_mask_to_memory_values({m_on.slot_bias: allowed})
g = m_on.slot_bias.grad
nz = torch.nonzero(g.abs().sum(dim=1) > 0).view(-1).tolist()
check("S5b TF-IDF row mask applies to bias", nz == [1, 5, 9], str(nz))

# ---- S6: tags ----
check("S6 pk_value_param + fixed_lr", getattr(m_on.slot_bias, "pk_value_param", False)
      and getattr(m_on.slot_bias, "fixed_lr", None) is not None)

# ---- S7: no-gate composes ----
m_ng = build(seed=1, lora_slot_bias=True, mem_gated=False)
check("S7a gating is None", m_ng.gating is None)
with torch.no_grad():
    y_ng = m_ng(x.clone())
check("S7b no-gate forward runs", y_ng.shape == (2, 3, 32) and torch.isfinite(y_ng).all())

# ---- S8: rank override composes ----
torch.manual_seed(2)
m_r4 = HashingMemoryLite(input_dim=32, output_dim=32, cfg=make_cfg(lora_slot_bias=True), lora_rank_override=4)
check("S8 rank-4 + bias shapes", m_r4.slot_up.shape[1] == 4 and m_r4.slot_bias.shape == (m_r4.size, m_r4.v_dim))

# ---- S9: eval_final_episodes selection ----
def cfgN(final):
    return SimpleNamespace(eval_final_episodes=final, online_task_ids=[0, 1, 2, 3, 4],
                           eval=SimpleNamespace(n_episodes=20))

check("S9a default off -> 20 everywhere",
      [_eval_n_episodes_for_task(cfgN(0), i) for i in range(5)] == [20] * 5)
check("S9b final=50 -> 20,20,20,20,50",
      [_eval_n_episodes_for_task(cfgN(50), i) for i in range(5)] == [20, 20, 20, 20, 50])

# ---- S10: offload parity with nonzero bias ----
m_a = build(seed=3, lora_slot_bias=True)
with torch.no_grad():
    m_a.slot_bias.normal_(0, 0.5)
m_b = build(seed=3, lora_slot_bias=True)
m_b.load_state_dict(m_a.state_dict())
m_b._slots_offloaded = True  # CPU test: gather path exercises the offload branch
with torch.no_grad():
    ya = m_a(x.clone()); yb = m_b(x.clone())
check("S10 offload branch parity", torch.allclose(ya, yb, atol=0, rtol=0),
      f"maxdiff={float((ya - yb).abs().max()):.2e}")

print()
if FAILS:
    print(f"FAILED: {FAILS}"); sys.exit(1)
print("ALL SMOKES PASS")
