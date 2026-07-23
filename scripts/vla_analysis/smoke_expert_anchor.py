#!/usr/bin/env python3
"""Smokes for the E52 expert-anchor pooled routing (module level).

S1a flag "" and flag "text"-with-no-anchor-set are BITWISE identical (same weights)
S1b flag "text" + anchor set + B=0 is bitwise identical (inert at zero weight)
S2  mix arithmetic == hand-computed B*nrm(W_a@a) + (1-B)*nrm(tok), rescaled to token RMS
S3  anchor at B=0.5 changes the retrieval indices (routing actually moves)
S4  per-row validity: invalid rows' outputs bitwise == the no-anchor run
S5  stale/mismatched anchor batch -> silent per-token fallback (+ warning flag), no crash
S6  backward reaches anchor_proj (grad nonzero via retrieval score weights)
S7  anchor_proj params carry pk_query_proj_param (router group: warm-up-trainable)
S8  overwrite semantics: second set_expert_anchor wins
S9  guards: bad mode raises; weight out of [0,1] raises; text_span>0 disables structurally
S10 with router_x given, the MIX applies to router_x (value path keeps live x)
"""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/home/josh/lerobot/src")
from lerobot.policies.modules.memory_config import MemoryLayerConfig
from lerobot.policies.modules.memory_lite import MLPPlusMemory

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


DIM, SRC, SEQ, B = 32, 48, 10, 3


def mk_cfg(**over):
    kw = dict(layers=[0], enabled=True, mem_n_keys=16, mem_heads=2, mem_knn=4,
              mem_k_dim=8, mem_v_dim=-1, value_type="lora", lora_rank=2,
              swilu_projection=True, mem_gated=True, log_usage=True, text_span=0,
              expert_anchor_src_dim=SRC)
    kw.update(over)
    return MemoryLayerConfig(**kw)


def mk_pair():
    """flag-off and flag-on wrappers sharing every common weight."""
    torch.manual_seed(0)
    w_off = MLPPlusMemory(nn.Linear(DIM, DIM), DIM, mk_cfg())
    torch.manual_seed(0)
    w_on = MLPPlusMemory(nn.Linear(DIM, DIM), DIM, mk_cfg(expert_anchor_pool="text"))
    sd = w_off.state_dict()
    missing, unexpected = w_on.load_state_dict(sd, strict=False)
    assert not unexpected and all("anchor_proj" in k for k in missing), (missing, unexpected)
    return w_off, w_on


torch.manual_seed(7)
x = torch.randn(B, SEQ, DIM)
anchor = torch.randn(B, SRC)
valid = torch.ones(B, dtype=torch.bool)

# S1a
w_off, w_on = mk_pair()
check("S1a flag-on w/o anchor == flag-off (bitwise)", torch.equal(w_off(x), w_on(x)))

# S1b
w_on.expert_anchor_w = 0.0
w_on.set_expert_anchor(anchor, valid)
check("S1b anchor set + B=0 bitwise inert", torch.equal(w_off(x), w_on(x)))
w_on.expert_anchor_w = 0.5

# S2 mix arithmetic
mixed = w_on._mix_expert_anchor(x)
bf = x.float()
tok_rms = bf.pow(2).mean(-1).sqrt()
mean_rms = tok_rms.mean()
ap = w_on.anchor_proj(anchor).float()
ap = ap / ap.pow(2).mean(-1, keepdim=True).sqrt()
tn = bf / tok_rms.unsqueeze(-1)
exp = ((0.5 * ap.unsqueeze(1) + 0.5 * tn) * mean_rms).to(x.dtype)
check("S2 mix == hand-computed", torch.allclose(mixed, exp, atol=1e-5),
      f"max dev {(mixed - exp).abs().max():.2e}")

# S3 routing moves
w_on.set_expert_anchor(None)
_ = w_on(x)
idx_plain = w_on.mem.last_indices.clone() if w_on.mem.last_indices is not None else None
w_on.set_expert_anchor(anchor, valid)
_ = w_on(x)
idx_anch = w_on.mem.last_indices.clone()
check("S3 retrieval indices change under the anchor",
      idx_plain is not None and not torch.equal(idx_plain, idx_anch))

# S4 per-row validity
v2 = torch.tensor([True, False, False])
w_on.set_expert_anchor(anchor, v2)
out_mixedrows = w_on(x)
w_on.set_expert_anchor(None)
out_plain = w_on(x)
check("S4a invalid rows bitwise == no-anchor", torch.equal(out_mixedrows[1:], out_plain[1:]))
check("S4b valid row differs", not torch.equal(out_mixedrows[0], out_plain[0]))

# S5 stale batch
w_on.set_expert_anchor(torch.randn(B + 2, SRC), torch.ones(B + 2, dtype=torch.bool))
try:
    out_stale = w_on(x)
    ok = torch.equal(out_stale, out_plain) and w_on._warned_anchor_stale
except Exception as e:  # noqa: BLE001
    ok = False
check("S5 stale anchor -> fallback + warning, no crash", ok)

# S6 grads reach anchor_proj
w_on.set_expert_anchor(anchor, valid)
w_on.zero_grad(set_to_none=True)
out = w_on(x)
out.sum().backward()
g = w_on.anchor_proj.weight.grad
check("S6 anchor_proj grad nonzero", g is not None and float(g.abs().max()) > 0)

# S7 tag
check("S7 pk_query_proj_param tag", getattr(w_on.anchor_proj.weight, "pk_query_proj_param", False))

# S8 overwrite
a2 = torch.randn(B, SRC)
w_on.set_expert_anchor(anchor, valid)
w_on.set_expert_anchor(a2, valid)
m2 = w_on._mix_expert_anchor(x)
ap2 = w_on.anchor_proj(a2).float()
ap2 = ap2 / ap2.pow(2).mean(-1, keepdim=True).sqrt()
exp2 = ((0.5 * ap2.unsqueeze(1) + 0.5 * tn) * mean_rms).to(x.dtype)
check("S8 second set_expert_anchor wins", torch.allclose(m2, exp2, atol=1e-5))

# S9 guards
try:
    MLPPlusMemory(nn.Linear(DIM, DIM), DIM, mk_cfg(expert_anchor_pool="bogus"))
    ok = False
except ValueError:
    ok = True
check("S9a bad mode raises", ok)
try:
    MLPPlusMemory(nn.Linear(DIM, DIM), DIM, mk_cfg(expert_anchor_pool="text", expert_anchor_weight=1.5))
    ok = False
except ValueError:
    ok = True
check("S9b weight out of range raises", ok)
w_vlm = MLPPlusMemory(nn.Linear(DIM, DIM), DIM, mk_cfg(expert_anchor_pool="text", text_span=4))
check("S9c text_span>0 disables structurally", w_vlm.expert_anchor == "" and w_vlm.anchor_proj is None)

# S10 mix applies to router_x
seen = {}
orig_mem = w_on.mem.forward


def spy(xx, **kw):
    seen["router_x"] = kw.get("router_x")
    seen["x"] = xx
    return orig_mem(xx, **kw)


w_on.mem.forward = spy
rx = torch.randn(B, SEQ, DIM)
w_on.set_expert_anchor(anchor, valid)
_ = w_on(x, router_x=rx)
exp_rx = w_on._mix_expert_anchor(rx)
check("S10a router_x is the anchored composite", torch.allclose(seen["router_x"], exp_rx, atol=1e-5))
check("S10b value path keeps live x", torch.equal(seen["x"], x))
w_on.mem.forward = orig_mem

print(f"\n{15 - len(FAILS)}/15 PASS" if not FAILS else f"\nFAILURES: {FAILS}")
sys.exit(1 if FAILS else 0)
