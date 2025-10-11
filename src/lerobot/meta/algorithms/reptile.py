#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import time
import logging

from lerobot.meta.algorithms.base import MetaAlgorithm, TaskResult
from lerobot.meta.configs import ReptileConfig, InnerOptConfig


@dataclass
class Reptile(MetaAlgorithm):
    cfg: ReptileConfig

    def adapt(
        self,
        model,
        support_iter: Iterator[dict],
        steps: int,
        inner_cfg: InnerOptConfig,
        preprocessor,
    ) -> TaskResult:
        theta = self.snapshot(model)
        opt = self.inner_optimizer([p for p in model.parameters() if p.requires_grad], inner_cfg)
        # Ensure training mode for modules with training-time behavior
        was_training = model.training if hasattr(model, "training") else True
        if hasattr(model, "train"):
            model.train()
        loss_sum = 0.0
        for inner_idx in range(1, steps + 1):
            batch = next(support_iter)
            batch = preprocessor(batch)
            loss, _ = model.forward(batch)
            opt.zero_grad()
            loss.backward()
            # optional grad clipping
            if inner_cfg.grad_clip_norm and inner_cfg.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], inner_cfg.grad_clip_norm, error_if_nonfinite=False)
            opt.step()
            t5 = time.perf_counter()
            loss_sum += float(loss.item())
            # Best-effort flush for timing clarity
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        with torch.no_grad():
            delta = {}
            for n, p in model.named_parameters():
                if p.requires_grad and n in theta:
                    # Store deltas on CPU to lower GPU memory pressure
                    d = (p - theta[n]).detach().to("cpu")
                    delta[n] = d.clone()
        # restore initial params to keep model at meta-parameters before outer step aggregation
        self.restore_parameters(model, theta)
        # Restore original training mode
        if hasattr(model, "train") and not was_training:
            model.eval()
        avg_loss = loss_sum / max(1, steps)
        return TaskResult(delta=delta, metrics={"inner_avg_loss": avg_loss})

    def outer_step(self, model, task_results: list[TaskResult]) -> None:
        if not task_results:
            return
        # Average deltas and apply
        with torch.no_grad():
            # Prepare accumulator per param
            acc = {}
            for res in task_results:
                for n, d in res.delta.items():
                    if n not in acc:
                        acc[n] = d.to(acc.get(n, d).device).clone()
                    else:
                        acc[n].add_(d.to(acc[n].device))
            scale = self.cfg.meta_step_size / max(1, len(task_results))
            for n, p in model.named_parameters():
                if p.requires_grad and n in acc:
                    # Move delta to parameter device on the fly
                    p.add_(acc[n].to(p.device), alpha=scale)