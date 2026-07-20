#!/usr/bin/env python3
"""E49/1A image-region query-statistics probe (pre-build conditioning for the
image-span pooled router).

On the STAGE-1 checkpoint (= the VLM router's input under frozen-route), over held-out
libero_10, for LAYERS incl. step-2 candidates below 15: per-camera image-token stats
(patch-level between/within task variance — the per-token sprawl predictor), pooled
image-region geometry at three granularities (global / 2x2 / 4x4 per camera:
coherence, within-task variance, intra/between-task/family cosine), and composite-key
predictions k = a*nrm(instr_pool) + b*nrm(region_pool) per granularity.

Image block = prefix positions before the 200-token language field, split equally per
camera (16x16 patch grid each). Cameras with ~zero within+between variance (the
empty_cameras slot) are excluded from aggregates automatically.
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

LAYERS = [7, 9, 11, 13, 15, 16]
TASKS = list(range(10))
FAMILY = [(4, 5), (4, 7), (5, 7)]
NB, BS = 13, 8
STATE_MARKER = 3040
LANG_LEN = 200
GRAIN = {"g1": 1, "g2": 2, "g4": 4}  # side of the pooling grid per camera
BGRID = [0.0, 0.5, 1.0]
N_PATCH_SAMPLE = 24  # fixed patch positions per camera for the per-token predictor
OUT = os.environ.get("OUT", "/home/josh/lerobot/outputs/analysis/e49/querystats_image_stage1.json")


def block_pool(x, side):
    """x: (256, d) one camera's patches on a 16x16 grid -> (side*side, d) block means."""
    g = x.reshape(16, 16, -1)
    step = 16 // side
    return np.stack([
        g[r * step:(r + 1) * step, c * step:(c + 1) * step].reshape(-1, g.shape[-1]).mean(0)
        for r in range(side) for c in range(side)
    ])


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
    # per (layer, task): per-sample instr pools + per-grain region pools + rms
    P = {L: {t: {"pi": [], "img_rms": [], **{g: [] for g in GRAIN}} for t in TASKS} for L in LAYERS}
    # patch-level accumulators per (layer, task, cam, patch): [n, sum_vec, sumsq]
    PATCH = {L: {t: {} for t in TASKS} for L in LAYERS}
    meta = {}

    rng = np.random.RandomState(0)
    patch_ids = None

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
            toks = ctx["tokens"]
            for L in LAYERS:
                full = caps[L].float()
                n_img = full.shape[1] - toks.shape[1]
                n_cam = n_img // 256
                assert n_cam * 256 == n_img, (full.shape, toks.shape)
                if patch_ids is None:
                    patch_ids = rng.choice(256, N_PATCH_SAMPLE, replace=False)
                    meta["n_img"] = int(n_img); meta["n_cam"] = int(n_cam)
                    print(f"[layout] prefix={full.shape[1]} img={n_img} cams={n_cam} lang={toks.shape[1]}")
                img = full[:, :n_img]                     # (B, n_cam*256, d)
                lang = full[:, n_img:]
                for i in range(img.shape[0]):
                    cams = img[i].reshape(n_cam, 256, -1).cpu().numpy()
                    st = P[L][t]
                    st["img_rms"].append(float(np.linalg.norm(cams, axis=-1).mean() / np.sqrt(cams.shape[-1])))
                    for g, side in GRAIN.items():
                        st[g].append(np.stack([block_pool(cams[c], side) for c in range(n_cam)]).astype(np.float16))
                    li = lang[i].cpu().numpy()
                    st["pi"].append(li[3:24].mean(0).astype(np.float16))  # instr pool (fixed 21-pos window)
                    for c in range(n_cam):
                        for p in patch_ids:
                            d = PATCH[L][t].setdefault((c, int(p)), [0, None, 0.0])
                            h = cams[c, p]
                            d[0] += 1; d[1] = h if d[1] is None else d[1] + h; d[2] += float(h @ h)
        print(f"[task {t}] n={len(P[LAYERS[0]][t]['pi'])}", flush=True)

    for h in hooks:
        h.remove()

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    n_cam = meta["n_cam"]
    out = {"meta": meta}
    for L in LAYERS:
        print(f"\n======== L{L}")
        res = {}
        img_rms = float(np.mean([r for t in TASKS for r in P[L][t]["img_rms"]]))
        res["img_token_rms"] = img_rms

        # per-camera activity + patch-level between/within (per-token sprawl predictor)
        cam_stats = []
        for c in range(n_cam):
            wtot, btot = [], []
            for p in set(k[1] for k in PATCH[L][TASKS[0]] if k[0] == c):
                means = []
                for t in TASKS:
                    d = PATCH[L][t].get((c, p))
                    if not d or d[0] < 20:
                        continue
                    mu = d[1] / d[0]
                    wtot.append(d[2] / d[0] - float(mu @ mu))
                    means.append(mu)
                if len(means) >= 8:
                    gm = np.mean(means, axis=0)
                    btot.append(float(np.mean([float((m - gm) @ (m - gm)) for m in means])))
            w, bt = float(np.mean(wtot)), float(np.mean(btot))
            cam_stats.append({"cam": c, "patch_within": w, "patch_between": bt,
                              "ratio": bt / max(w, 1e-9), "active": (w + bt) > 1.0})
            print(f"  cam{c}: patch within {w:9.1f} between {bt:9.1f} ratio {bt/max(w,1e-9):.3f} active={(w+bt)>1.0}")
        res["cameras"] = cam_stats
        active = [c["cam"] for c in cam_stats if c["active"]]

        # pooled-region geometry per granularity (active cameras only)
        for g, side in GRAIN.items():
            nreg = side * side
            fam_worst, fam_mean, bet_mean, intra_mean, wvar_mean, coh = [], [], [], [], [], []
            for c in active:
                for r in range(nreg):
                    V = {t: np.stack([s[c, r].astype(np.float32) for s in P[L][t][g]]) for t in TASKS}
                    tm = {t: V[t].mean(0) for t in TASKS}
                    coh.append(float((np.mean([np.linalg.norm(v) for t in TASKS for v in V[t][:20]])
                                      / np.sqrt(V[TASKS[0]].shape[-1]) / img_rms) ** 2))
                    wvar_mean.append(float(np.mean([np.mean(np.sum((V[t] - tm[t]) ** 2, axis=1)) for t in TASKS])))
                    intra_mean.append(float(np.mean([np.mean([cos(V[t][i], V[t][j])
                                     for i in range(0, 40, 8) for j in range(4, 40, 8)
                                     if i < j < len(V[t])]) for t in TASKS])))
                    xt = [cos(tm[a], tm[b]) for i, a in enumerate(TASKS) for b in TASKS[i+1:]]
                    fm = [cos(tm[a], tm[b]) for a, b in FAMILY]
                    bet_mean.append(float(np.mean(xt))); fam_mean.append(float(np.mean(fm)))
                    fam_worst.append(float(np.max(fm)))
            res[g] = {"coherence": float(np.mean(coh)), "within_var": float(np.mean(wvar_mean)),
                      "intra_cos": float(np.mean(intra_mean)), "between_cos": float(np.mean(bet_mean)),
                      "family_cos": float(np.mean(fam_mean)), "family_worst": float(np.max(fam_worst))}
            print(f"  {g} ({nreg}/cam x {len(active)} cams): coh {np.mean(coh):.2f} intra {np.mean(intra_mean):.3f} "
                  f"between {np.mean(bet_mean):.3f} family {np.mean(fam_mean):.3f} (worst {np.max(fam_worst):.3f})")

        # composite keys: a=1 fixed, b over BGRID, per granularity (region-paired)
        comp = {}
        for g, side in GRAIN.items():
            nreg = side * side
            for b_ in BGRID:
                bet, fam, intra = [], [], []
                for c in active:
                    for r in range(nreg):
                        ks = {}
                        for t in TASKS:
                            pi = np.stack([s.astype(np.float32) for s in P[L][t]["pi"]])
                            pr = np.stack([s[c, r].astype(np.float32) for s in P[L][t][g]])
                            pin = pi / (np.linalg.norm(pi, axis=1, keepdims=True) + 1e-9)
                            prn = pr / (np.linalg.norm(pr, axis=1, keepdims=True) + 1e-9)
                            ks[t] = pin + b_ * prn
                        tm = {t: ks[t].mean(0) for t in TASKS}
                        intra.append(float(np.mean([np.mean([cos(ks[t][i], ks[t][j])
                                     for i in range(0, 40, 8) for j in range(4, 40, 8)
                                     if i < j < len(ks[t])]) for t in TASKS])))
                        bet.append(float(np.mean([cos(tm[a], tm[bb]) for i, a in enumerate(TASKS) for bb in TASKS[i+1:]])))
                        fam.append(float(np.mean([cos(tm[a], tm[bb]) for a, bb in FAMILY])))
                comp[f"{g}_b{b_}"] = {"intra": float(np.mean(intra)), "inter": float(np.mean(bet)),
                                      "family": float(np.mean(fam))}
                print(f"    composite {g} b={b_}: intra {np.mean(intra):.3f} inter {np.mean(bet):.3f} family {np.mean(fam):.3f}")
        res["composite"] = comp
        out[f"L{L}"] = res

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nsaved {OUT}\nQUERYSTATS-IMAGE-DONE")


if __name__ == "__main__":
    main()
