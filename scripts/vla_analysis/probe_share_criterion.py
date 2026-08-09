#!/usr/bin/env python3
"""Share-criterion probe (E61 addendum 5, 9 Aug 26).

Question: can we PREDICT, before training, which adjacent memory-attach layers can
share one K/V table (E61 sharing) and which need their own? Hypothesis (Josh):
layers whose router inputs look SIMILAR -> share; layers whose inputs DIFFER ->
dedicate. Under frozen-route/prepass every router reads the memory-free stage-1
features, so the router-input similarity between two layers is measurable on the
stage-1 checkpoint with a forward-pass probe — no training.

VALIDATION REQUIREMENT (pre-registered): whatever metric we adopt must separate
  - expert pair (6,8):   SHAREABLE  (held/improved in E61)
  - expert pair (10,12): NOT        (e7 38->22 in E61)
and call the VLM pairs (7,9)/(11,13) shareable (they held).
Warning from the log: E61's own overlap stats (per-step mask overlap 2-3%,
block-scale site-bleed 17-43%) do NOT separate the good pair from the bad one —
so a metric passing this validation is a finding, not a formality.

Metrics per adjacent layer pair (i,j), per tower (VLM restricted to language-field
token slices — image/pad tokens are not what production routers key on):
  raw_cos      mean per-token cos(x_i[t], x_j[t])           (residual-stream similarity)
  cent_cos     same after subtracting each layer's batch-mean over tokens
               (removes the shared dominant direction that inflates raw cosine)
  rel_delta    mean ||x_j[t]-x_i[t]|| / ||x_i[t]||          (how much the stream moves)
  task_rsa     Pearson corr of the two layers' 10x10 task-centroid cosine matrices
               (upper triangle) — do the layers organize TASKS the same way?
Plus per-layer inter/intra task cosine (querystats convention) for context.

Invocation (SequentialOnlineConfig probe convention, cf. run_e59_querystats_subL7.sh):
  SC_EXPERT_LAYERS='[2,4,6,8,10,12,14,16]' SC_VLM_LAYERS='[3,5,7,9,11,13,15]' \
  SC_NB=8 OUT=.../share_criterion_stage1.json \
  python probe_share_criterion.py --policy.path=<stage1> --dataset.repo_id=libero_10 ...
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from accelerate import Accelerator
from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig,
    _build_dataloader_for_task,
    _collect_task_index_to_name,
)

# language field = last 200 prefix positions (tokenizer_max_length); instruction is
# its head. instr16 is guaranteed-real text; lang60 additionally covers the
# state-as-text region for typical prompts (may include a few pads on short
# prompts — reported separately, instr16 is the clean primary).
VLM_SLICES = {"instr16": (-200, -184), "lang60": (-200, -140)}


def _pair_stats(xi: torch.Tensor, xj: torch.Tensor) -> dict:
    """xi, xj: [N_tokens, D] float32 on GPU. Returns per-batch accumulable sums."""
    n = xi.shape[0]
    raw = torch.nn.functional.cosine_similarity(xi, xj, dim=-1)
    ci = xi - xi.mean(dim=0, keepdim=True)
    cj = xj - xj.mean(dim=0, keepdim=True)
    cent = torch.nn.functional.cosine_similarity(ci, cj, dim=-1)
    rel = (xj - xi).norm(dim=-1) / xi.norm(dim=-1).clamp_min(1e-6)
    return {
        "n": n,
        "raw_sum": raw.sum().item(),
        "cent_sum": cent.sum().item(),
        "rel_sum": rel.sum().item(),
    }


def _finalize_pairs(acc: dict) -> dict:
    out = {}
    for pair, a in acc.items():
        n = max(a["n"], 1)
        out[pair] = {
            "raw_cos": a["raw_sum"] / n,
            "cent_cos": a["cent_sum"] / n,
            "rel_delta": a["rel_sum"] / n,
            "n_tokens": a["n"],
        }
    return out


def _task_geometry(pooled: dict, labels: np.ndarray, layers: list) -> tuple[dict, dict]:
    """pooled: {L: [N, D] fp32 np}. Returns (per-layer inter/intra, centroid-cos matrices)."""
    tasks = sorted(set(labels.tolist()))
    per_layer, cmats = {}, {}
    for L in layers:
        X = pooled[L]
        Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-6, None)
        cents = np.stack([Xn[labels == t].mean(axis=0) for t in tasks])
        cn = cents / np.clip(np.linalg.norm(cents, axis=1, keepdims=True), 1e-6, None)
        cmat = cn @ cn.T
        iu = np.triu_indices(len(tasks), k=1)
        inter = float(cmat[iu].mean())
        intras = []
        for t in tasks:
            Z = Xn[labels == t]
            c = Z.mean(axis=0)
            c /= max(np.linalg.norm(c), 1e-6)
            intras.append(float((Z @ c).mean()))
        per_layer[L] = {"inter": inter, "intra": float(np.mean(intras))}
        cmats[L] = cmat[iu]
    return per_layer, cmats


def _rsa(cmats: dict, pairs: list) -> dict:
    out = {}
    for i, j in pairs:
        a, b = cmats[i], cmats[j]
        out[f"{i}-{j}"] = float(np.corrcoef(a, b)[0, 1])
    return out


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    cfg.validate()
    accelerator = Accelerator()
    device = accelerator.device

    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    processor_kwargs, postprocessor_kwargs = {}, {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if hasattr(policy, "precompute_task_embeddings"):
        policy.precompute_task_embeddings(dataset.meta)
    policy = accelerator.prepare(policy)
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    unwrapped.eval()
    task_index_to_name = _collect_task_index_to_name(dataset)

    exp_layers = json.loads(os.environ.get("SC_EXPERT_LAYERS", "[2,4,6,8,10,12,14,16]"))
    vlm_layers = json.loads(os.environ.get("SC_VLM_LAYERS", "[3,5,7,9,11,13,15]"))
    nb = int(os.environ.get("SC_NB", "8"))
    out_path = os.environ["OUT"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    exp_pairs = list(zip(exp_layers[:-1], exp_layers[1:]))
    vlm_pairs = list(zip(vlm_layers[:-1], vlm_layers[1:]))

    exp_mods = unwrapped.model.paligemma_with_expert.gemma_expert.model.layers
    vlm_mods = unwrapped.model.paligemma_with_expert.paligemma.model.language_model.layers

    cap_exp: dict[int, torch.Tensor] = {}
    cap_vlm: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(store, L):
        def hook(module, args, kwargs=None):
            x = args[0]
            if x.dim() == 3:
                store[L] = x.detach().float()
        return hook

    for L in exp_layers:
        hooks.append(exp_mods[L].mlp.register_forward_pre_hook(make_hook(cap_exp, L)))
    for L in vlm_layers:
        hooks.append(vlm_mods[L].mlp.register_forward_pre_hook(make_hook(cap_vlm, L)))

    # accumulators
    acc_exp = {f"{i}-{j}": {"n": 0, "raw_sum": 0.0, "cent_sum": 0.0, "rel_sum": 0.0} for i, j in exp_pairs}
    acc_vlm = {
        sl: {f"{i}-{j}": {"n": 0, "raw_sum": 0.0, "cent_sum": 0.0, "rel_sum": 0.0} for i, j in vlm_pairs}
        for sl in VLM_SLICES
    }
    pooled_exp = {L: [] for L in exp_layers}
    pooled_vlm = {sl: {L: [] for L in vlm_layers} for sl in VLM_SLICES}
    labels = []

    cam_keys = list(dataset.meta.camera_keys)
    tasks = sorted(task_index_to_name.keys())
    torch.manual_seed(1000)
    for t in tasks:
        name = task_index_to_name.get(t, "")
        try:
            dl = _build_dataloader_for_task(
                dataset, task_index_to_name, t,
                batch_size=cfg.batch_size, num_workers=cfg.num_workers, device_type=device.type,
            )
        except ValueError as e:
            print(f"[sc] task {t}: skipped ({e})", flush=True)
            continue
        it = iter(dl)
        n = 0
        for _ in range(nb):
            try:
                batch = next(it)
            except StopIteration:
                break
            for ck in cam_keys:
                if ck in batch and batch[ck].dtype == torch.uint8:
                    batch[ck] = batch[ck].to(torch.float32) / 255.0
            batch = preprocessor(batch)
            B = batch[next(iter(batch))].shape[0]
            te = unwrapped.get_task_embeddings([name] * B) if hasattr(unwrapped, "get_task_embeddings") else None
            if te is not None:
                te = te.to(device)
            cap_exp.clear()
            cap_vlm.clear()
            with torch.no_grad(), accelerator.autocast():
                unwrapped.forward(batch, task_emb=te)
            if len(cap_exp) < len(exp_layers) or len(cap_vlm) < len(vlm_layers):
                print(f"[sc] task {t}: incomplete capture, skipping batch", flush=True)
                continue

            # expert tower: all suffix tokens
            for i, j in exp_pairs:
                xi = cap_exp[i].reshape(-1, cap_exp[i].shape[-1])
                xj = cap_exp[j].reshape(-1, cap_exp[j].shape[-1])
                s = _pair_stats(xi, xj)
                a = acc_exp[f"{i}-{j}"]
                for k in ("raw_sum", "cent_sum", "rel_sum"):
                    a[k] += s[k]
                a["n"] += s["n"]
            for L in exp_layers:
                pooled_exp[L].append(cap_exp[L].mean(dim=1).cpu().numpy().astype(np.float32))

            # VLM tower: language-field slices only
            for sl, (a0, a1) in VLM_SLICES.items():
                for i, j in vlm_pairs:
                    xi = cap_vlm[i][:, a0:a1, :].reshape(-1, cap_vlm[i].shape[-1])
                    xj = cap_vlm[j][:, a0:a1, :].reshape(-1, cap_vlm[j].shape[-1])
                    s = _pair_stats(xi, xj)
                    a = acc_vlm[sl][f"{i}-{j}"]
                    for k in ("raw_sum", "cent_sum", "rel_sum"):
                        a[k] += s[k]
                    a["n"] += s["n"]
                for L in vlm_layers:
                    pooled_vlm[sl][L].append(
                        cap_vlm[L][:, a0:a1, :].mean(dim=1).cpu().numpy().astype(np.float32)
                    )
            labels.append(np.full(B, t, dtype=np.int32))
            n += 1
        print(f"[sc] task {t} ({name[:40]}): {n} batches", flush=True)

    for h in hooks:
        h.remove()
    labels = np.concatenate(labels)

    result = {"expert": {"pairs": _finalize_pairs(acc_exp)}, "vlm": {}}
    pe = {L: np.concatenate(v) for L, v in pooled_exp.items()}
    geom, cmats = _task_geometry(pe, labels, exp_layers)
    result["expert"]["layers"] = {str(L): g for L, g in geom.items()}
    result["expert"]["task_rsa"] = _rsa(cmats, exp_pairs)
    for sl in VLM_SLICES:
        pv = {L: np.concatenate(v) for L, v in pooled_vlm[sl].items()}
        geom, cmats = _task_geometry(pv, labels, vlm_layers)
        result["vlm"][sl] = {
            "pairs": _finalize_pairs(acc_vlm[sl]),
            "layers": {str(L): g for L, g in geom.items()},
            "task_rsa": _rsa(cmats, vlm_pairs),
        }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[sc] wrote {out_path}")

    # ---- verdict table ----
    def table(pairs_dict, rsa, title):
        print(f"\n=== {title} ===")
        print(f"{'pair':>8} {'raw_cos':>8} {'cent_cos':>9} {'rel_delta':>10} {'task_rsa':>9}")
        for pair, v in pairs_dict.items():
            print(f"{pair:>8} {v['raw_cos']:8.3f} {v['cent_cos']:9.3f} {v['rel_delta']:10.3f} {rsa.get(pair, float('nan')):9.3f}")

    table(result["expert"]["pairs"], result["expert"]["task_rsa"], "EXPERT (all suffix tokens)")
    for sl in VLM_SLICES:
        table(result["vlm"][sl]["pairs"], result["vlm"][sl]["task_rsa"], f"VLM [{sl}]")

    ep = result["expert"]["pairs"]
    er = result["expert"]["task_rsa"]
    if "6-8" in ep and "10-12" in ep:
        print("\n=== E61 VALIDATION (must separate 6-8 [shareable] from 10-12 [not]) ===")
        for m in ("raw_cos", "cent_cos", "rel_delta"):
            a, b = ep["6-8"][m], ep["10-12"][m]
            print(f"  {m:>10}: (6,8)={a:.3f}  (10,12)={b:.3f}  delta={a - b:+.3f}")
        print(f"  {'task_rsa':>10}: (6,8)={er['6-8']:.3f}  (10,12)={er['10-12']:.3f}  delta={er['6-8'] - er['10-12']:+.3f}")


if __name__ == "__main__":
    main()
