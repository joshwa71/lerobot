#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import nn


@dataclass
class LoraAttachConfig:
    enable: bool = False
    r: int = 4
    alpha: float = 16.0
    dropout: float = 0.05
    # Regexes matched against module qualified names in the policy
    # Example defaults cover attention and MLP projections plus small policy heads
    target_modules_regex: list[str] = field(
        default_factory=lambda: [
            r"self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
            r"mlp\.(up_proj|down_proj|gate_proj)$",
            r"(?:^|\.)state_proj$",
            r"(?:^|\.)action_.*",
        ]
    )


class LoRALinear(nn.Module):
    """Lightweight LoRA adapter around a frozen Linear layer.

    Forward: y = base(x) + scale * (x @ A @ B)
    A: in_features x r; B: r x out_features; scale = alpha / r
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects a Linear base module")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)

        # Freeze base parameters
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA parameters are initialized to zeros (B) and small random (A)
        # Following common practice: A small init helps stability
        device = self.base.weight.device
        # Match base weight dtype to avoid upcasting large activations
        dtype = self.base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.r, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.r, self.out_features, device=device, dtype=dtype))

        # Kaiming uniform on A; keep B at zeros to start from identity behavior
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r == 0:
            return y
        x_d = self.dropout(x)
        x_proj = x_d.to(self.lora_A.dtype) @ self.lora_A
        delta = x_proj @ self.lora_B
        delta = delta.to(y.dtype)
        return y + self.scaling * delta

    # Expose common Linear attributes for compatibility with downstream code
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)


def _match_any(name: str, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if re.search(pat, name):
            return True
    return False


def _replace_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    setattr(parent, child_name, new_module)


def attach_lora(policy: nn.Module, cfg: LoraAttachConfig) -> nn.Module:
    """Attach LoRA adapters in-place to Linear modules matching regex patterns.

    Returns the same policy instance for chaining.
    """
    if not cfg.enable:
        return policy

    target_patterns = cfg.target_modules_regex or []

    # Walk named_modules to get qualified names and parent references
    # We reconstruct parent modules by splitting the qualified name
    replaced = 0
    for qual_name, module in list(policy.named_modules()):
        if isinstance(module, nn.Linear) and _match_any(qual_name, target_patterns):
            # Find parent
            if "." in qual_name:
                parent_name, child_name = qual_name.rsplit(".", 1)
                parent = dict(policy.named_modules())[parent_name]
            else:
                parent = policy
                child_name = qual_name

            lora_layer = LoRALinear(module, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)
            _replace_module(parent, child_name, lora_layer)
            replaced += 1

    # Freeze all parameters, then unfreeze LoRA-specific params only
    for p in policy.parameters():
        p.requires_grad = False
    for _, mod in policy.named_modules():
        if isinstance(mod, LoRALinear):
            if hasattr(mod, "lora_A"):
                mod.lora_A.requires_grad = True
            if hasattr(mod, "lora_B"):
                mod.lora_B.requires_grad = True

    if replaced == 0:
        # It is fine if no module matched, but warn via print to keep dependency-free
        print("[LoRA] No target modules matched. Check target_modules_regex patterns.")
    else:
        print(f"[LoRA] Attached LoRA to {replaced} Linear modules.")

    return policy


def iter_trainable_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in module.parameters() if p.requires_grad)


