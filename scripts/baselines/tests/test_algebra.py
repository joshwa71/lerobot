#!/usr/bin/env python3
"""Local (GPU-free) tests of the baseline algebra: O-LoRA rank concatenation + orthogonality
penalty + freezing, and the RETAIN interpolation. Needs torch + peft only (no lerobot).
Run: python scripts/baselines/tests/test_algebra.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olora.olora_common import (OrthPenalty, adapter_key, adapter_tensors, algebra_check, check_key_format,  # noqa: E402
                                concat_adapters, l1_of_adapter_B, load_adapter_tensors, lora_layers,
                                set_trainable_adapter)
from retain.retain_merge import cpu_copy, merge_in_place  # noqa: E402

from peft import LoraConfig, get_peft_model  # noqa: E402


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 48)
        self.v_proj = nn.Linear(64, 48)
        self.state_proj = nn.Linear(8, 48)   # in_features < r*t -> excluded from the penalty
        self.out = nn.Linear(48, 5)

    def forward(self, x, s):
        return self.out(torch.tanh(self.q_proj(x) + self.v_proj(x) + self.state_proj(s)))


def test_olora():
    torch.manual_seed(0)
    r, n_tasks = 4, 3
    base = Tiny()
    cfg = LoraConfig(r=r, lora_alpha=1, target_modules=r"(q_proj|v_proj|state_proj)", lora_dropout=0.0, bias="none")
    m = get_peft_model(base, cfg, adapter_name="olora_t0")
    check_key_format(m, "olora_t0")
    names = ["olora_t0"]
    for k in range(1, n_tasks):
        m.add_adapter(f"olora_t{k}", LoraConfig(**{**cfg.to_dict()}))
        names.append(f"olora_t{k}")
    # give every adapter random, non-zero weights (B is zero-initialised)
    with torch.no_grad():
        for _, mod in lora_layers(m):
            for a in names:
                mod.lora_A[a].weight.normal_()
                mod.lora_B[a].weight.normal_()
    m.base_model.set_adapter(names)
    n_train = set_trainable_adapter(m, "olora_t2")
    per = sum(p.numel() for n, p in m.named_parameters() if "olora_t2" in n.split("."))
    assert n_train == per > 0, (n_train, per)
    assert all((("olora_t2" in n.split(".")) == p.requires_grad) for n, p in m.named_parameters())

    # concatenation: sum of adapters == one rank-12 adapter, padded to 16 -> zero rows/cols
    tensors, r_tot, scaling = concat_adapters(m, names, export_rank=r * 4)
    assert r_tot == r * n_tasks and abs(scaling - 0.25) < 1e-12, (r_tot, scaling)
    rel = algebra_check(m, names, tensors, n_modules=3)
    assert rel < 1e-5, rel
    for k, t in tensors.items():
        if k.endswith("lora_A.weight"):
            assert torch.all(t[r_tot:] == 0) and t.shape[0] == r * 4
        else:
            assert torch.all(t[:, r_tot:] == 0) and t.shape[1] == r * 4
    # end-to-end: module output with all adapters == module output with the concatenated one
    x = torch.randn(3, 64)
    _, mod = [(n, mm) for n, mm in lora_layers(m) if n.endswith("q_proj")][0]
    y_multi = mod(x)
    m.add_adapter("_chk", LoraConfig(**{**cfg.to_dict(), "r": r_tot, "lora_alpha": scaling * r_tot}))
    unp, _, _ = concat_adapters(m, names, None)
    load_adapter_tensors(m, "_chk", unp)
    m.base_model.set_adapter(["_chk"])
    y_one = mod(x)
    assert torch.allclose(y_multi, y_one, atol=1e-5, rtol=1e-5), (y_multi - y_one).abs().max()
    m.base_model.set_adapter(names)
    m.delete_adapter("_chk")
    set_trainable_adapter(m, "olora_t2")

    # penalty: matches a hand computation, excludes state_proj
    pen = OrthPenalty(m, "olora_t2", ["olora_t0", "olora_t1"], r"(^|\.)state_proj$")
    assert [e[0] for e in pen.excluded] == ["state_proj"], pen.excluded
    manual = 0.0
    for n, mod in lora_layers(m):
        if n.endswith("state_proj"):
            continue
        P = torch.cat([mod.lora_A["olora_t0"].weight, mod.lora_A["olora_t1"].weight], 0)
        manual += (P @ mod.lora_A["olora_t2"].weight.t()).abs().sum()
    assert torch.allclose(pen(), manual), (pen(), manual)
    # the penalty's gradient reaches only the current adapter's A
    pen().backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert "olora_t2" in n.split(".") and "lora_A" in n, n
    # driving the penalty to zero makes the rows orthogonal (sanity of sign/shape)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        (0.5 * pen()).backward()
        opt.step()
    assert pen().item() < 1e-2 * manual.item(), pen().item()
    # frozen adapters untouched
    assert l1_of_adapter_B(m, "olora_t0") > 0
    print("O-LoRA algebra: OK  (rank_concat", r_tot, "scaling", scaling, "algebra rel err", f"{rel:.1e})")


def test_retain():
    torch.manual_seed(1)
    m = Tiny().to(torch.bfloat16)
    prev = cpu_copy(m)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    ft = cpu_copy(m)
    st = merge_in_place(m, prev, 0.5)
    for n, p in m.named_parameters():
        ref = (0.5 * prev[n].float() + 0.5 * ft[n].float()).to(torch.bfloat16)
        assert torch.equal(p.data, ref), n
    assert abs(st["largest_tensor"]["ratio_fp32"] - 0.5) < 1e-6, st
    assert abs(st["ratio_merged_over_ft"] - 0.5) < 1e-6, st
    print("RETAIN merge: OK ", {k: round(v, 5) if isinstance(v, float) else v for k, v in st.items() if k != "largest_tensor"})


if __name__ == "__main__":
    test_olora()
    test_retain()
    print("ALL OK")
