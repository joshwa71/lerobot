#!/usr/bin/env python3
"""O-LoRA helpers (Wang et al., Findings of EMNLP 2023, arXiv 2310.14152).

Method (paper Sec. 3.2 + official code, github.com/cmnfriend/O-LoRA):
  * one LoRA pair {A_t, B_t} per task t; all earlier pairs {A_i, B_i | i < t} are FROZEN but stay
    ACTIVE, so the forward pass is  W x + s * sum_i B_i A_i x   (s = alpha/r, identical for all i);
  * training objective  L_task + lambda_1 * sum_{i<t} L_orth(A_i, A_t)  with, in the official
    implementation,  L_orth = | A_i A_t^T |_1  (sum of absolute entries of the r x r Gram block;
    the paper's Eq. 8 writes the squared Frobenius norm — the code is what produced the numbers);
    lambda_1 = 0.5, lambda_2 (an L2 on the new pair) = 0 for the reported orders;
  * inference uses the sum of all adapters, i.e. no task identity; parameters grow by one
    adapter per task (the official code carries the earlier pairs as one frozen block of rank
    r*(t-1) plus the trainable rank-r pair — rank concatenation).

Checkpoint format. Because every adapter shares the scaling s, the sum of k rank-r adapters IS one
rank-(k*r) LoRA:  B_cat = [B_1 ... B_k],  A_cat = [A_1; ...; A_k].  At every task boundary we
export that single adapter, zero-padded to `export_rank` (= r * n_tasks) so all boundaries have
the same shapes, as an ordinary PEFT checkpoint. The existing instruments — the 4-seed rollout
campaign (`--policy.use_peft=true`) and the adapter-swap loss matrix (`mse_matrix_peft.py`) — then
run on it unchanged. Zero rows/columns contribute nothing, and the concatenation is exact up to
fp32 summation order (checked at every boundary, see `export_check`).

This module never modifies our code: it only enumerates peft's `LoraLayer` modules.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import torch
from peft import get_peft_model_state_dict
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file, save_file

PEFT_PREFIX = "base_model.model."


def lora_layers(peft_model) -> list[tuple[str, LoraLayer]]:
    """(name within peft_model, module) for every LoRA-wrapped layer, in module order."""
    return [(n, m) for n, m in peft_model.named_modules() if isinstance(m, LoraLayer)]


def adapter_key(module_name: str, which: str) -> str:
    return f"{module_name}.lora_{which}.weight"


def adapter_tensors(peft_model, adapter: str, to_cpu: bool = True) -> dict[str, torch.Tensor]:
    """{key: tensor} for one adapter, keys in peft's save format (adapter name stripped)."""
    out = {}
    for name, mod in lora_layers(peft_model):
        if adapter not in mod.lora_A:
            continue
        a = mod.lora_A[adapter].weight.detach()
        b = mod.lora_B[adapter].weight.detach()
        out[adapter_key(name, "A")] = a.cpu().clone() if to_cpu else a
        out[adapter_key(name, "B")] = b.cpu().clone() if to_cpu else b
    return out


def check_key_format(peft_model, adapter: str) -> None:
    """Our key derivation must coincide with peft's own state-dict keys for that adapter."""
    ours = set(adapter_tensors(peft_model, adapter, to_cpu=False).keys())
    theirs = set(get_peft_model_state_dict(peft_model, adapter_name=adapter).keys())
    if ours != theirs:
        missing = sorted(theirs - ours)[:5]
        extra = sorted(ours - theirs)[:5]
        raise RuntimeError(f"adapter key format mismatch: missing={missing} extra={extra}")


@torch.no_grad()
def load_adapter_tensors(peft_model, adapter: str, tensors: dict[str, torch.Tensor]) -> int:
    n = 0
    for name, mod in lora_layers(peft_model):
        if adapter not in mod.lora_A:
            continue
        for which, lin in (("A", mod.lora_A[adapter]), ("B", mod.lora_B[adapter])):
            t = tensors[adapter_key(name, which)]
            if tuple(t.shape) != tuple(lin.weight.shape):
                raise RuntimeError(f"{adapter_key(name, which)}: shape {tuple(t.shape)} != {tuple(lin.weight.shape)}")
            lin.weight.copy_(t.to(device=lin.weight.device, dtype=lin.weight.dtype))
            n += 1
    if n == 0:
        raise RuntimeError(f"no LoRA layers carry adapter {adapter}")
    return n


def set_trainable_adapter(peft_model, adapter: str) -> int:
    """Only `adapter`'s A/B are trainable; the base and every other adapter are frozen."""
    n = 0
    for name, p in peft_model.named_parameters():
        parts = name.split(".")
        is_lora = any(part.startswith("lora_") for part in parts)
        p.requires_grad = bool(is_lora and adapter in parts)
        if p.requires_grad:
            n += p.numel()
    return n


def l1_of_adapter_B(peft_model, adapter: str) -> float:
    return sum(m.lora_B[adapter].weight.detach().float().abs().sum().item()
               for _, m in lora_layers(peft_model) if adapter in m.lora_B)


class OrthPenalty:
    """lambda-free sum over modules of |A_prev @ A_cur^T|_1, A_prev = frozen rows of all earlier tasks.

    Modules matching `exclude_regex` (on the module name with the peft prefix stripped) get no
    penalty: rank r rows cannot be mutually orthogonal in fewer than r*t input dimensions
    (pi0.5's 32-input state/action projections), so the constraint is unsatisfiable there.
    """

    def __init__(self, peft_model, cur: str, prev: list[str], exclude_regex: str | None):
        self.terms: list[tuple[str, torch.Tensor, torch.nn.Parameter]] = []
        self.excluded: list[tuple[str, int]] = []
        self.unsatisfiable: list[tuple[str, int, int]] = []
        pat = re.compile(exclude_regex) if exclude_regex else None
        for name, mod in lora_layers(peft_model):
            short = name[len(PEFT_PREFIX):] if name.startswith(PEFT_PREFIX) else name
            if cur not in mod.lora_A:
                continue
            prev_here = [a for a in prev if a in mod.lora_A]
            if not prev_here:
                continue
            in_features = mod.lora_A[cur].weight.shape[1]
            if pat is not None and pat.search(short):
                self.excluded.append((short, int(in_features)))
                continue
            P = torch.cat([mod.lora_A[a].weight.detach() for a in prev_here], dim=0).float()  # (r*(t-1), in)
            if P.shape[0] + mod.lora_A[cur].weight.shape[0] > in_features:
                self.unsatisfiable.append((short, int(in_features), int(P.shape[0] + mod.lora_A[cur].weight.shape[0])))
            self.terms.append((short, P, mod.lora_A[cur].weight))

    def __call__(self) -> torch.Tensor:
        tot = None
        for _, P, A in self.terms:
            v = torch.mm(P, A.float().t()).abs().sum()
            tot = v if tot is None else tot + v
        if tot is None:
            return torch.zeros((), device=self.terms[0][2].device if self.terms else "cpu")
        return tot

    @torch.no_grad()
    def stats(self) -> dict:
        per = []
        for short, P, A in self.terms:
            G = torch.mm(P, A.float().t()).abs()
            per.append((short, float(G.max()), float(G.mean()), float(G.sum())))
        if not per:
            return {"n_modules": 0, "total_l1": 0.0, "max_abs_entry": 0.0}
        return {
            "n_modules": len(per),
            "total_l1": sum(p[3] for p in per),
            "max_abs_entry": max(p[1] for p in per),
            "mean_abs_entry": sum(p[2] for p in per) / len(per),
            "worst_modules": sorted(per, key=lambda p: -p[1])[:5],
            "excluded": self.excluded,
            "unsatisfiable": self.unsatisfiable,
        }


@torch.no_grad()
def concat_adapters(peft_model, adapters: list[str], export_rank: int | None) -> tuple[dict[str, torch.Tensor], int, float]:
    """Rank-concatenate `adapters` (in order) into one LoRA; zero-pad to `export_rank` rows/cols.
    Returns (tensors in peft save format, unpadded rank, common scaling)."""
    tensors: dict[str, torch.Tensor] = {}
    r_tot = None
    scaling = None
    for name, mod in lora_layers(peft_model):
        present = [a for a in adapters if a in mod.lora_A]
        if not present:
            continue
        s = {float(mod.scaling[a]) for a in present}
        if len(s) != 1:
            raise RuntimeError(f"{name}: adapters have different scalings {s}; concatenation is not exact")
        s = s.pop()
        scaling = s if scaling is None else scaling
        if abs(s - scaling) > 1e-12:
            raise RuntimeError("scaling differs across modules")
        A = torch.cat([mod.lora_A[a].weight.detach().float() for a in present], dim=0)   # (r_tot, in)
        B = torch.cat([mod.lora_B[a].weight.detach().float() for a in present], dim=1)   # (out, r_tot)
        r_here = A.shape[0]
        r_tot = r_here if r_tot is None else r_tot
        if r_here != r_tot:
            raise RuntimeError(f"{name}: rank {r_here} != {r_tot}")
        if export_rank is not None:
            if export_rank < r_tot:
                raise RuntimeError(f"export_rank {export_rank} < concatenated rank {r_tot}")
            A_p = torch.zeros(export_rank, A.shape[1], dtype=A.dtype)
            A_p[:r_tot] = A.cpu()
            B_p = torch.zeros(B.shape[0], export_rank, dtype=B.dtype)
            B_p[:, :r_tot] = B.cpu()
            A, B = A_p, B_p
        else:
            A, B = A.cpu(), B.cpu()
        tensors[adapter_key(name, "A")] = A.contiguous()
        tensors[adapter_key(name, "B")] = B.contiguous()
    if r_tot is None:
        raise RuntimeError("no LoRA layers found")
    return tensors, r_tot, scaling


@torch.no_grad()
def algebra_check(peft_model, adapters: list[str], tensors: dict[str, torch.Tensor], n_modules: int = 8) -> float:
    """sum_i B_i A_i x  vs  B_cat A_cat x  on random x, fp32, for a few modules. Returns max rel err."""
    worst = 0.0
    layers = [(n, m) for n, m in lora_layers(peft_model) if any(a in m.lora_A for a in adapters)]
    g = torch.Generator().manual_seed(0)
    for name, mod in layers[:: max(1, len(layers) // n_modules)][:n_modules]:
        in_f = mod.lora_A[adapters[0]].weight.shape[1]
        x = torch.randn(4, in_f, generator=g)
        dev = mod.lora_A[adapters[0]].weight.device
        x = x.to(dev)
        ref = None
        for a in adapters:
            if a not in mod.lora_A:
                continue
            y = mod.lora_B[a].weight.detach().float() @ (mod.lora_A[a].weight.detach().float() @ x.t())
            ref = y if ref is None else ref + y
        A = tensors[adapter_key(name, "A")].to(dev)
        B = tensors[adapter_key(name, "B")].to(dev)
        got = B @ (A @ x.t())
        rel = float((got - ref).abs().max() / (ref.abs().max() + 1e-12))
        worst = max(worst, rel)
    return worst


def write_export_checkpoint(pretrained_dir: Path, peft_model, adapters: list[str], export_rank: int,
                            base_model_path: str, train_cfg, preprocessor, postprocessor) -> dict:
    """Ordinary PEFT checkpoint dir (adapter_config.json, adapter_model.safetensors, policy
    config.json, processors, train_config.json) holding the rank-concatenated adapter."""
    pretrained_dir = Path(pretrained_dir)
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    tensors, r_tot, scaling = concat_adapters(peft_model, adapters, export_rank)
    rel = algebra_check(peft_model, adapters, tensors)
    peft_cfg = copy.deepcopy(peft_model.peft_config[adapters[0]])
    peft_cfg.r = int(export_rank)
    peft_cfg.lora_alpha = float(scaling * export_rank) if (scaling * export_rank) % 1 else int(scaling * export_rank)
    peft_cfg.inference_mode = True
    peft_cfg.base_model_name_or_path = str(base_model_path)
    peft_cfg.rank_pattern = {}
    peft_cfg.alpha_pattern = {}
    peft_cfg.save_pretrained(str(pretrained_dir))
    save_file(tensors, str(pretrained_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    # policy config (use_peft=True on it since the wrap), processors, train config — as save_checkpoint does
    peft_model.config.save_pretrained(pretrained_dir)
    if train_cfg is not None:
        train_cfg.save_pretrained(pretrained_dir)
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)
    n_params = sum(t.numel() for t in tensors.values())
    return {"rank_concat": r_tot, "export_rank": export_rank, "scaling": scaling,
            "lora_alpha_export": peft_cfg.lora_alpha, "n_export_params": n_params,
            "algebra_max_rel_err": rel, "n_tensors": len(tensors)}


def load_export_tensors(pretrained_dir: Path) -> dict[str, torch.Tensor]:
    return load_file(str(Path(pretrained_dir) / "adapter_model.safetensors"))
