#!/usr/bin/env python3
"""E45 query-statistics probe (pre-launch conditioning for the pooled-router build).

On the STAGE-1 checkpoint (= the VLM router's input features), over held-out libero_10:
per-region token stats (RMS, total/between/within variance), pooled-component geometry
(instr pool vs state pool: coherence/norm shrinkage, within-task variance, cross-task
cosine incl. basket family), and predicted router-key geometry for candidate (a:b)
anchor:state weightings (q_intra / q_inter proxies at init).

Boundary per sample: first index of token id 3040 ("▁State") minus 1 (the comma).
Instr pool = positions [3, b) (skips <bos> Task :). State pool = [b+3, v-5) (drops
", State :" markers and the ";\nAction: " tail).
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

LAYERS = [15, 16]
TASKS = list(range(10))
FAMILY = [(4, 5), (4, 7), (5, 7)]
NB, BS = 13, 8
STATE_MARKER = 3040
OUT = os.environ.get("OUT", "/home/josh/lerobot/outputs/analysis/e44/querystats_stage1.json")
GRID = [(1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0), (0.0, 1.0)]


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

    caps = {}
    hooks = []
    lm_layers = model.paligemma_with_expert.paligemma.model.language_model.layers
    for L in LAYERS:
        def mk(L):
            def pre(mod, args):
                caps[L] = args[0].detach()
            return pre
        hooks.append(lm_layers[L].mlp.register_forward_pre_hook(mk(L)))

    tin = _collect_task_index_to_name(dataset)
    # per (layer, task): lists of per-sample pools + token stats
    P = {L: {t: {"pi": [], "ps": [], "tok_rms": [], "n_i": [], "n_s": []} for t in TASKS} for L in LAYERS}
    # aligned per-position accumulators: instr aligned to field start (pos 0..24),
    # state aligned to boundary (off 0..40): per (layer, task, pos): [n, sum(2048), sumsq_scalar]
    A = {L: {t: {"instr": {}, "state": {}} for t in TASKS} for L in LAYERS}

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
            for L in LAYERS:
                x = caps[L][:, -toks.shape[1]:].float()  # language field (B, 200, d)
                for i in range(x.shape[0]):
                    if not has[i]:
                        continue
                    bi, vi = int(bnd[i]), int(v[i])
                    if bi <= 4 or vi - 5 <= bi + 3:
                        continue
                    xi = x[i]
                    instr = xi[3:bi]
                    state = xi[bi + 3:vi - 5]
                    st = P[L][t]
                    st["pi"].append(instr.mean(0).cpu().numpy())
                    st["ps"].append(state.mean(0).cpu().numpy())
                    st["tok_rms"].append(float(xi[3:vi].norm(dim=-1).mean() / np.sqrt(xi.shape[-1])))
                    st["n_i"].append(instr.shape[0]); st["n_s"].append(state.shape[0])
                    for p in range(min(bi - 3, 24)):
                        d = A[L][t]["instr"].setdefault(p, [0, None, 0.0])
                        h = instr[p].cpu().numpy()
                        d[0] += 1; d[1] = h if d[1] is None else d[1] + h; d[2] += float(h @ h)
                    for p in range(min(state.shape[0], 40)):
                        d = A[L][t]["state"].setdefault(p, [0, None, 0.0])
                        h = state[p].cpu().numpy()
                        d[0] += 1; d[1] = h if d[1] is None else d[1] + h; d[2] += float(h @ h)
        print(f"[task {t}] n={len(P[LAYERS[0]][t]['pi'])}")

    for h in hooks:
        h.remove()

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    out = {}
    for L in LAYERS:
        print(f"\n======== L{L}")
        rms = np.mean([r for t in TASKS for r in P[L][t]["tok_rms"]])
        res = {"token_rms": float(rms)}
        # region variance profile (mean over tasks): total var per aligned position
        for reg in ("instr", "state"):
            prof = []
            for p in range(40):
                tot, bet, cnt = [], [], 0
                means = []
                for t in TASKS:
                    d = A[L][t][reg].get(p)
                    if not d or d[0] < 20:
                        continue
                    mu = d[1] / d[0]
                    tot.append(d[2] / d[0] - float(mu @ mu))  # within-task var (trace)
                    means.append(mu)
                if len(means) >= 8:
                    gm = np.mean(means, axis=0)
                    bt = float(np.mean([float((m - gm) @ (m - gm)) for m in means]))
                    prof.append((p, float(np.mean(tot)), bt))
            res[f"{reg}_pos_profile"] = prof
            if prof:
                w = np.mean([x[1] for x in prof]); b_ = np.mean([x[2] for x in prof])
                print(f"  {reg:>5} per-pos var: within {w:8.1f}  between-task {b_:8.1f}  ratio b/w {b_/max(w,1e-9):.3f}")
        # pooled-component stats
        stats = {}
        for key, nm in (("pi", "instr_pool"), ("ps", "state_pool")):
            allv = {t: np.stack(P[L][t][key]) for t in TASKS}
            tmeans = {t: allv[t].mean(0) for t in TASKS}
            pool_norm = np.mean([np.linalg.norm(v) / np.sqrt(v.shape[-1]) for t in TASKS for v in allv[t]])
            wvar = np.mean([np.mean(np.sum((allv[t] - tmeans[t]) ** 2, axis=1)) for t in TASKS])
            intra = np.mean([np.mean([cos(allv[t][i], allv[t][j]) for i in range(0, 40, 4) for j in range(1, 40, 4) if i < j and j < len(allv[t])]) for t in TASKS])
            xt = [cos(tmeans[a], tmeans[b]) for i, a in enumerate(TASKS) for b in TASKS[i+1:]]
            fam = [cos(tmeans[a], tmeans[b]) for a, b in FAMILY]
            stats[nm] = {"rms": float(pool_norm), "coherence_vs_token": float((pool_norm / rms) ** 2),
                         "within_task_var": float(wvar), "intra_cos": float(intra),
                         "between_task_cos": float(np.mean(xt)), "family_cos": float(np.mean(fam))}
            print(f"  {nm:>10}: rms {pool_norm:.2f} (token {rms:.2f}, coherence {(pool_norm/rms)**2:.2f})  "
                  f"withinvar {wvar:9.1f}  intra-cos {intra:.3f}  between-cos {np.mean(xt):.3f}  family-cos {np.mean(fam):.3f}")
        res.update(stats)
        # (a:b) composite key predictions (normalize components to unit RMS first)
        print(f"  --- composite key k = a*nrm(instr_pool) + b*nrm(state_pool)")
        comp = {}
        for a, b_ in GRID:
            ks = {}
            for t in TASKS:
                pi = np.stack(P[L][t]["pi"]); ps = np.stack(P[L][t]["ps"])
                pin = pi / (np.linalg.norm(pi, axis=1, keepdims=True) + 1e-9)
                psn = ps / (np.linalg.norm(ps, axis=1, keepdims=True) + 1e-9)
                ks[t] = a * pin + b_ * psn
            tmeans = {t: ks[t].mean(0) for t in TASKS}
            intra = np.mean([np.mean([cos(ks[t][i], ks[t][j]) for i in range(0, 40, 4) for j in range(1, 40, 4) if i < j and j < len(ks[t])]) for t in TASKS])
            xt = [cos(tmeans[x], tmeans[y]) for i, x in enumerate(TASKS) for y in TASKS[i+1:]]
            fam = [cos(tmeans[x], tmeans[y]) for x, y in FAMILY]
            comp[f"a{a}_b{b_}"] = {"intra": float(intra), "inter": float(np.mean(xt)), "family": float(np.mean(fam))}
            print(f"    a={a:3} b={b_:3}: q_intra proxy {intra:.3f}   q_inter {np.mean(xt):.3f}   family {np.mean(fam):.3f}")
        res["composite"] = comp
        out[f"L{L}"] = res
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nsaved {OUT}\nQUERYSTATS-DONE")


if __name__ == "__main__":
    main()
