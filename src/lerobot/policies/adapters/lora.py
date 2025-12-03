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
from typing import Iterable

import torch
from torch import nn

from lerobot.configs.lora import LoraAttachConfig


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

    # Handle parameter freezing based on train_lora_only flag
    if cfg.train_lora_only:
        # Freeze all base parameters, only train LoRA adapters
        for p in policy.parameters():
            p.requires_grad = False
        for _, mod in policy.named_modules():
            if isinstance(mod, LoRALinear):
                if hasattr(mod, "lora_A"):
                    mod.lora_A.requires_grad = True
                if hasattr(mod, "lora_B"):
                    mod.lora_B.requires_grad = True
    else:
        # Keep original trainability of base parameters, also enable LoRA parameters
        # LoRA parameters are already trainable by default from LoRALinear.__init__
        # but base parameters inside LoRALinear are frozen; we need to respect original
        # trainability which is handled by LoRALinear freezing only its own base
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
        mode_str = "train_lora_only" if cfg.train_lora_only else "train_lora_and_base"
        print(f"[LoRA] Attached LoRA to {replaced} Linear modules ({mode_str}).")

    return policy


def iter_trainable_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in module.parameters() if p.requires_grad)


