#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from lerobot.meta.configs import InnerOptConfig


@dataclass
class TaskResult:
    delta: dict[str, torch.Tensor]
    metrics: dict[str, float] | None = None


class MetaAlgorithm:
    def get_trainable(self, model: nn.Module) -> list[nn.Parameter]:
        return [p for p in model.parameters() if p.requires_grad]

    def restore_parameters(self, model: nn.Module, ref: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in ref:
                    p.copy_(ref[n])

    def snapshot(self, model: nn.Module) -> dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def inner_optimizer(self, params: Iterable[nn.Parameter], cfg: InnerOptConfig) -> torch.optim.Optimizer:
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


