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

from dataclasses import dataclass, field


@dataclass
class LoraAttachConfig:
    enable: bool = False
    r: int = 4
    alpha: float = 16.0
    dropout: float = 0.05
    # When True, freeze all base model parameters and only train LoRA adapters.
    # When False, train both LoRA adapters and any originally trainable base parameters.
    train_lora_only: bool = True
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

