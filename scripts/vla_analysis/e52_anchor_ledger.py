#!/usr/bin/env python3
"""E52 anchor-ledger probe (Josh's expert-anchor idea — the state-hidden trick applied
to the expert tower).

On the STAGE-1 checkpoint over held-out libero_10, capture matched per-sample pairs of
  - pooled LM instruction hidden (positions [3, b) of the language field) at candidate
    anchor layers, and
  - expert-tower suffix-token hiddens (per-sample mean + K-token subsample) at candidate
    expert memory layers,
then score the composite routing key  k_p = B*nrm(anchor) + (1-B)*nrm(token_p)  on a B
grid: between-task separation, basket-family cos, within-task token-level conditionality
(q_intra proxy), + the component-RMS ledger (the build's rescale constants).

Cross-tower dims differ pre-W_a (2048 vs 1024), so composite cos uses the analytic
blend  [B^2*cos_a + (1-B)^2*cos_x] / (B^2 + (1-B)^2)  (unit components, cross terms ~0
for a learned/random map at init), sanity-checked against a fixed random projection.
Probes rank, never veto (E45 arm-C rule): the trained W_a can only do better.

Env: OUT (json path). Run with the standard CLI on the stage-1 checkpoint.
"""
import json
import os

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig, _build_dataloader_for_task, _collect_task_index_to_name,
)

ANCHOR_LAYERS = [5, 7, 9, 11, 12]
EXPERT_LAYERS = [4, 6, 8, 10, 12]
TASKS = list(range(10))
FAMILY = [(4, 5), (4, 7), (5, 7)]
NB, BS, KTOK = 13, 8, 6
STATE_MARKER = 3040
BGRID = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
OUT = os.environ.get("OUT", "/home/josh/lerobot/outputs/analysis/e52/anchor_ledger.json")


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    cfg.validate()
    device = torch.device("cuda")
    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        })
    if hasattr(policy, "precompute_task_embeddings"):
        policy.precompute_task_embeddings(dataset.meta)
    policy = policy.to(device).eval()

    model = policy.model
    ctx = {}
    orig_embed = model.embed_prefix

    def embed_wrap(images, img_masks, tokens, masks):
        ctx["tokens"] = tokens.detach()
        ctx["masks"] = masks.detach()
        return orig_embed(images, img_masks, tokens, masks)
    model.embed_prefix = embed_wrap

    lm_caps, ex_caps, hooks = {}, {}, []
    lm_layers = model.paligemma_with_expert.paligemma.model.language_model.layers
    ex_layers = model.paligemma_with_expert.gemma_expert.model.layers
    for L in ANCHOR_LAYERS:
        def mk(L):
            def pre(mod, args):
                lm_caps[L] = args[0].detach()
            return pre
        hooks.append(lm_layers[L].mlp.register_forward_pre_hook(mk(L)))
    for L in EXPERT_LAYERS:
        def mk(L):
            def pre(mod, args):
                ex_caps[L] = args[0].detach()
            return pre
        hooks.append(ex_layers[L].mlp.register_forward_pre_hook(mk(L)))

    tin = _collect_task_index_to_name(dataset)
    # per task: anchors[A] = list of (2048,), exmean[E] = list of (1024,), extok[E] = list of (K,1024)
    D = {t: {"anc": {A: [] for A in ANCHOR_LAYERS},
             "exm": {E: [] for E in EXPERT_LAYERS},
             "ext": {E: [] for E in EXPERT_LAYERS}} for t in TASKS}
    rng = np.random.default_rng(0)

    for t in TASKS:
        dl = _build_dataloader_for_task(dataset, tin, t, batch_size=BS, num_workers=2, device_type=device.type)
        it = iter(dl)
        for _ in range(NB):
            try:
                b = next(it)
            except StopIteration:
                break
            for ck in dataset.meta.camera_keys:
                if ck in b and b[ck].dtype == torch.uint8:
                    b[ck] = b[ck].to(torch.float32) / 255.0
            b = preprocessor(b)
            tids = torch.full((BS,), t, dtype=torch.long, device=device)
            with torch.no_grad():
                policy.forward(b, task_emb=None, task_ids=tids)
            toks, masks = ctx["tokens"], ctx["masks"]
            v = masks.sum(dim=1).long()
            is_marker = toks == STATE_MARKER
            has = is_marker.any(dim=1)
            bnd = torch.where(has, is_marker.float().argmax(dim=1) - 1, torch.zeros_like(v))
            n_lang = toks.shape[1]
            for i in range(toks.shape[0]):
                if not has[i]:
                    continue
                bi = int(bnd[i])
                if bi <= 4:
                    continue
                ok = True
                for A in ANCHOR_LAYERS:
                    xi = lm_caps[A][i, -n_lang:].float()
                    D[t]["anc"][A].append(xi[3:bi].mean(0).cpu().numpy())
                for E in EXPERT_LAYERS:
                    xe = ex_caps[E][i].float()  # (T_suffix, 1024)
                    D[t]["exm"][E].append(xe.mean(0).cpu().numpy())
                    idx = rng.choice(xe.shape[0], size=min(KTOK, xe.shape[0]), replace=False)
                    D[t]["ext"][E].append(xe[idx].cpu().numpy())
        print(f"[task {t}] n={len(D[t]['anc'][ANCHOR_LAYERS[0]])}", flush=True)

    for h in hooks:
        h.remove()

    def unit(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    out = {"n_per_task": {t: len(D[t]["anc"][ANCHOR_LAYERS[0]]) for t in TASKS}}

    # component geometry + RMS ledger
    for A in ANCHOR_LAYERS:
        cents = {t: unit(np.stack(D[t]["anc"][A])).mean(0) for t in TASKS}
        pairs = [(a, b) for i, a in enumerate(TASKS) for b in TASKS[i + 1:]]
        inter = np.mean([cos(cents[a], cents[b]) for a, b in pairs])
        fam = np.mean([cos(cents[a], cents[b]) for a, b in FAMILY])
        intra = np.mean([np.mean(unit(np.stack(D[t]["anc"][A])) @ cents[t] / np.linalg.norm(cents[t])) for t in TASKS])
        rms = np.mean([np.linalg.norm(x) / np.sqrt(x.shape[-1]) for t in TASKS for x in D[t]["anc"][A]])
        out[f"anchor_L{A}"] = {"inter": round(float(inter), 4), "fam": round(float(fam), 4),
                               "intra": round(float(intra), 4), "rms": round(float(rms), 4)}
    for E in EXPERT_LAYERS:
        cents = {t: unit(np.stack(D[t]["exm"][E])).mean(0) for t in TASKS}
        pairs = [(a, b) for i, a in enumerate(TASKS) for b in TASKS[i + 1:]]
        inter = np.mean([cos(cents[a], cents[b]) for a, b in pairs])
        fam = np.mean([cos(cents[a], cents[b]) for a, b in FAMILY])
        # token-level within-task conditionality: mean pairwise cos among a sample's tokens
        tok_intra = np.mean([np.mean(unit(tk) @ unit(tk).T) for t in TASKS for tk in D[t]["ext"][E][:40]])
        rms = np.mean([np.linalg.norm(x) / np.sqrt(x.shape[-1]) for t in TASKS for x in D[t]["exm"][E]])
        out[f"expert_L{E}"] = {"inter": round(float(inter), 4), "fam": round(float(fam), 4),
                               "tok_intra": round(float(tok_intra), 4), "rms": round(float(rms), 4)}

    # composite grid: analytic blend on unit components
    # cos_composite(i,j) = [B^2 cos_a(i,j) + (1-B)^2 cos_x(i,j)] / (B^2 + (1-B)^2)
    for A in ANCHOR_LAYERS:
        for E in EXPERT_LAYERS:
            grid = {}
            cents_a = {t: unit(np.stack(D[t]["anc"][A])).mean(0) for t in TASKS}
            cents_x = {t: unit(np.stack(D[t]["exm"][E])).mean(0) for t in TASKS}
            pairs = [(a, b) for i, a in enumerate(TASKS) for b in TASKS[i + 1:]]
            for B in BGRID:
                w2a, w2x = B * B, (1 - B) * (1 - B)
                den = w2a + w2x
                inter = np.mean([(w2a * cos(cents_a[a], cents_a[b]) + w2x * cos(cents_x[a], cents_x[b])) / den
                                 for a, b in pairs])
                fam = np.mean([(w2a * cos(cents_a[a], cents_a[b]) + w2x * cos(cents_x[a], cents_x[b])) / den
                               for a, b in FAMILY])
                # within-task token-level cos of composite: anchor shared within sample ->
                # cos ~= [B^2 * intra_a + (1-B)^2 * tok_cos_x] / den  (intra_a between samples ~ anchor intra)
                ia = out[f"anchor_L{A}"]["intra"]
                tx = out[f"expert_L{E}"]["tok_intra"]
                intra = (w2a * ia + w2x * tx) / den
                grid[str(B)] = {"inter": round(float(inter), 4), "fam": round(float(fam), 4),
                                "intra_proxy": round(float(intra), 4)}
            out[f"grid_A{A}_E{E}"] = grid

    # random-projection sanity check at one cell (A=7, E=8, B=0.5)
    A, E = 7, 8
    R = np.random.default_rng(1).standard_normal((2048, 1024)).astype(np.float32) / np.sqrt(1024)
    cents_a = {t: unit(unit(np.stack(D[t]["anc"][A])) @ R).mean(0) for t in TASKS}
    cents_x = {t: unit(np.stack(D[t]["exm"][E])).mean(0) for t in TASKS}
    comp = {t: unit(0.5 * unit(cents_a[t]) + 0.5 * unit(cents_x[t])) for t in TASKS}
    pairs = [(a, b) for i, a in enumerate(TASKS) for b in TASKS[i + 1:]]
    out["sanity_rp_A7_E8_B0.5"] = {
        "inter": round(float(np.mean([cos(comp[a], comp[b]) for a, b in pairs])), 4),
        "fam": round(float(np.mean([cos(comp[a], comp[b]) for a, b in FAMILY])), 4),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    for A in ANCHOR_LAYERS:
        print(f"anchor L{A}: {out[f'anchor_L{A}']}")
    for E in EXPERT_LAYERS:
        print(f"expert L{E}: {out[f'expert_L{E}']}")


if __name__ == "__main__":
    main()
