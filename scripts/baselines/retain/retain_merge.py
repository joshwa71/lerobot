#!/usr/bin/env python3
"""RETAIN weight interpolation, kept lerobot-free so it can be unit-tested and used offline.

    theta~ = (1 - alpha) * theta_prev + alpha * theta_ft        (Yadav et al. 2026, Eq. 2 / Eq. 4)

`merge_in_place` mixes a live model with a CPU copy of its pre-task parameters (fp32 math, cast
back to the parameter dtype). `merge_safetensors` does the same between two saved
`model.safetensors` files (audit / re-merge at another alpha). Statistics returned by both include
an exactness witness: on the largest tensor the fine-tune moved, |merged - prev|_1 / |ft - prev|_1
computed in fp32 must equal alpha (tied/unused tensors such as lm_head never move and are skipped).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def cpu_copy(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()}


@torch.no_grad()
def merge_in_place(model, theta_prev: dict[str, torch.Tensor], alpha: float) -> dict:
    sum_ft_prev = 0.0
    sum_merged_prev = 0.0
    n_el = 0
    largest = None
    for n, p in model.named_parameters():
        if n not in theta_prev:
            raise KeyError(f"parameter {n} missing from theta_prev")
        prev = theta_prev[n].to(device=p.device, dtype=torch.float32)
        ft = p.data.float()
        merged32 = (1.0 - alpha) * prev + alpha * ft
        d_ft = (ft - prev).abs().sum().item()
        d_m = (merged32 - prev).abs().sum().item()
        sum_ft_prev += d_ft
        sum_merged_prev += d_m
        n_el += p.numel()
        if d_ft > 0 and (largest is None or p.numel() > largest[1]):
            largest = (n, p.numel(), d_m / d_ft)
        p.data.copy_(merged32.to(p.dtype))
        del prev, ft, merged32
    return {
        "mean_abs_ft_minus_prev": sum_ft_prev / max(n_el, 1),
        "mean_abs_merged_minus_prev": sum_merged_prev / max(n_el, 1),
        "ratio_merged_over_ft": (sum_merged_prev / sum_ft_prev) if sum_ft_prev > 0 else float("nan"),
        "largest_tensor": ({"name": largest[0], "numel": largest[1], "ratio_fp32": largest[2]} if largest
                           else {"name": None, "numel": 0, "ratio_fp32": float("nan")}),
        "n_params": n_el,
    }


@torch.no_grad()
def merge_safetensors(prev_dir: Path, ft_dir: Path, out_dir: Path, alpha: float) -> dict:
    """Interpolate two `pretrained_model` dirs into a third (config/processors copied from ft_dir)."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    prev_dir, ft_dir, out_dir = Path(prev_dir), Path(ft_dir), Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for f in ft_dir.iterdir():
        if f.name != "model.safetensors" and f.is_file():
            shutil.copy2(f, tmp / f.name)
    merged: dict[str, torch.Tensor] = {}
    stats = {"alpha": alpha, "n_params": 0, "sum_ft_prev": 0.0, "sum_merged_prev": 0.0, "largest": None}
    with safe_open(str(prev_dir / "model.safetensors"), framework="pt") as fp, \
         safe_open(str(ft_dir / "model.safetensors"), framework="pt") as ff:
        kp, kf = set(fp.keys()), set(ff.keys())
        if kp != kf:
            raise RuntimeError(f"key sets differ: only-prev={sorted(kp-kf)[:3]} only-ft={sorted(kf-kp)[:3]}")
        for k in sorted(kf):
            a, b = fp.get_tensor(k), ff.get_tensor(k)
            if a.shape != b.shape:
                raise RuntimeError(f"{k}: shape {a.shape} != {b.shape}")
            if not a.is_floating_point():
                if not torch.equal(a, b):
                    raise RuntimeError(f"{k}: non-float tensor differs")
                merged[k] = b
                continue
            a32, b32 = a.float(), b.float()
            m32 = (1.0 - alpha) * a32 + alpha * b32
            d_ft = (b32 - a32).abs().sum().item()
            d_m = (m32 - a32).abs().sum().item()
            stats["n_params"] += a.numel()
            stats["sum_ft_prev"] += d_ft
            stats["sum_merged_prev"] += d_m
            if d_ft > 0 and (stats["largest"] is None or a.numel() > stats["largest"][1]):
                stats["largest"] = (k, a.numel(), d_m / d_ft)
            merged[k] = m32.to(a.dtype)
    save_file(merged, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    stats["ratio_merged_over_ft"] = stats["sum_merged_prev"] / stats["sum_ft_prev"] if stats["sum_ft_prev"] > 0 else float("nan")
    with open(tmp / "retain_merge.json", "w") as f:
        json.dump({**stats, "prev": str(prev_dir), "ft": str(ft_dir)}, f, indent=2)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    a = ap.parse_args()
    print(json.dumps(merge_safetensors(a.prev, a.ft, a.out, a.alpha), indent=2))
