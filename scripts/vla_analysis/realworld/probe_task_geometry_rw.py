#!/usr/bin/env python3
"""Real-world task-geometry probe (tier-1 held-out-set selection; realworld duplicate of
feat_probe.py + probe_querystats.py, extended to dump the FULL task x task matrices).

Streams every task of a LeRobot dataset (the 20-task WidowX pool) through a FROZEN pi05
checkpoint with NO memory attached — under frozen-prepass the router reads exactly these
memory-free features — and captures, per sample:

  expert_L{j}     mean over the action tokens of the expert MLP input at expert layer j
                  (the expert router's proj(x) input). Noise/time draws are seeded per
                  batch index so they are paired across tasks.
  instr_L{j}      LM MLP input at LM layer j, mean-pooled over instruction positions
                  [3, b)   (b = index of token 3040 "▁State" minus 1). This is the
                  expert-anchor source (expert layer j pools LM layer j) and the VLM anchor.
  state_L{j}      same, pooled over the state-as-text positions [b+3, v-5).
  key_L{j}        the deployed VLM palette key 1.0*nrm(instr) + 0.5*nrm(state).
  img_cam{c}_L{j} same, pooled over camera c's 256 patch positions (real cameras first,
                  empty slots last — pi05 ordering); img_L{j} = all real cameras.

Writes OUT_JSON: per space the task-centroid cosine matrix, per-task intra-sample cosine
and centroid norm, plus task names / usage counts / instruction lengths / mean episode
length; and (optional) OUT_NPZ with the per-sample pooled vectors (float16).

Env: EXPERT_LAYERS (default 4,6,8,10,14,16), LM_LAYERS (default 4,5,6,7,8,9,10,11,13,14,15,16),
NB (batches/task, 16), BS (8), NW (dataloader workers, 4), SEED (1000), OUT_JSON, OUT_NPZ.
CLI = SequentialOnlineConfig (policy.path, dataset.*, rename_map, policy.empty_cameras, ...).
"""
import json
import os

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig,
    _build_dataloader_for_task,
    _collect_task_index_to_name,
)

EXPERT_LAYERS = [int(x) for x in os.environ.get("EXPERT_LAYERS", "4,6,8,10,14,16").split(",")]
LM_LAYERS = [int(x) for x in os.environ.get("LM_LAYERS", "4,5,6,7,8,9,10,11,13,14,15,16").split(",")]
NB = int(os.environ.get("NB", "16"))
BS = int(os.environ.get("BS", "8"))
NW = int(os.environ.get("NW", "4"))
SEED = int(os.environ.get("SEED", "1000"))
OUT_JSON = os.environ["OUT_JSON"]
OUT_NPZ = os.environ.get("OUT_NPZ", "")
STATE_MARKER = 3040
KEY_AB = (1.0, 0.5)
PATCHES_PER_CAM = 256


def _cos_matrix(C: np.ndarray) -> np.ndarray:
    n = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    return n @ n.T


def _intra_cos(X: np.ndarray, max_n: int = 48) -> float:
    X = X[:max_n].astype(np.float32)
    if len(X) < 2:
        return float("nan")
    n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    G = n @ n.T
    iu = np.triu_indices(len(X), k=1)
    return float(G[iu].mean())


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    cfg.validate()
    device = torch.device("cuda")
    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    if hasattr(policy, "precompute_task_embeddings"):
        policy.precompute_task_embeddings(dataset.meta)
    policy = policy.to(device).eval()
    model = policy.model

    # capture the tokenized prompt + attention mask of the current batch
    ctx = {}
    orig_embed = model.embed_prefix

    def embed_wrap(images, img_masks, tokens, masks):
        ctx["tokens"] = tokens.detach()
        ctx["masks"] = masks.detach()
        return orig_embed(images, img_masks, tokens, masks)

    model.embed_prefix = embed_wrap

    lm_caps, ex_caps = {}, {}
    hooks = []
    lm_layers = model.paligemma_with_expert.paligemma.model.language_model.layers
    ex_layers = model.paligemma_with_expert.gemma_expert.model.layers

    def mk_lm(L):
        def pre(mod, args):
            lm_caps[L] = args[0].detach()
        return pre

    def mk_ex(L):
        def pre(mod, args):
            x = args[0]
            if x.dim() == 3 and L not in ex_caps:  # first call of this forward only
                ex_caps[L] = x.detach()
        return pre

    for L in LM_LAYERS:
        hooks.append(lm_layers[L].mlp.register_forward_pre_hook(mk_lm(L)))
    for L in EXPERT_LAYERS:
        hooks.append(ex_layers[L].mlp.register_forward_pre_hook(mk_ex(L)))

    tin = _collect_task_index_to_name(dataset)
    tasks = sorted(int(t) for t in tin.keys())
    cam_keys = list(dataset.meta.camera_keys)
    n_real = len(cam_keys)
    print(f"[geom] tasks={len(tasks)} real cameras={n_real} {cam_keys} expert={EXPERT_LAYERS} lm={LM_LAYERS} NB={NB} BS={BS}", flush=True)

    spaces: dict[str, dict[int, list]] = {}

    def add(name, t, vec):
        spaces.setdefault(name, {}).setdefault(t, []).append(np.asarray(vec, dtype=np.float16))

    used = {t: 0 for t in tasks}
    skipped = {t: 0 for t in tasks}
    instr_lens = {t: [] for t in tasks}
    layout = None

    for t in tasks:
        try:
            dl = _build_dataloader_for_task(dataset, tin, t, batch_size=BS, num_workers=NW, device_type=device.type)
        except ValueError as e:  # task index with no usable episodes
            print(f"[geom] task {t}: skipped ({e})", flush=True)
            continue
        it = iter(dl)
        for bi in range(NB):
            try:
                b = next(it)
            except StopIteration:
                break
            for ck in cam_keys:
                if ck in b and b[ck].dtype == torch.uint8:
                    b[ck] = b[ck].to(torch.float32) / 255.0
            b = preprocessor(b)
            ex_caps.clear()
            lm_caps.clear()
            torch.manual_seed(SEED + bi)  # paired noise/time draws across tasks
            with torch.no_grad():
                try:
                    policy.forward(b, task_emb=None)
                except TypeError:
                    policy.forward(b)
            toks, masks = ctx["tokens"], ctx["masks"]
            Tl = toks.shape[1]
            v = masks.sum(dim=1).long()
            is_marker = toks == STATE_MARKER
            has = is_marker.any(dim=1)
            bnd = torch.where(has, is_marker.float().argmax(dim=1) - 1, torch.zeros_like(v))
            valid = []
            for i in range(toks.shape[0]):
                bi_, vi = int(bnd[i]), int(v[i])
                ok = bool(has[i]) and bi_ > 4 and (vi - 5) > (bi_ + 3)
                if ok:
                    valid.append(i)
                else:
                    skipped[t] += 1
            for L in EXPERT_LAYERS:
                x = ex_caps[L].float().mean(dim=1).cpu().numpy()
                for i in valid:
                    add(f"expert_L{L}", t, x[i])
            for L in LM_LAYERS:
                h = lm_caps[L].float()
                T = h.shape[1]
                n_img = T - Tl
                assert n_img % PATCHES_PER_CAM == 0 and n_img >= PATCHES_PER_CAM * n_real, (T, Tl, n_real)
                if layout is None:
                    layout = {"prefix_len": int(T), "lang_len": int(Tl), "img_positions": int(n_img),
                              "cams_total": int(n_img // PATCHES_PER_CAM), "cams_real": int(n_real)}
                    print(f"[geom] prefix layout {layout}", flush=True)
                lang = h[:, -Tl:]
                img = h[:, :n_img]
                for i in valid:
                    bi_, vi = int(bnd[i]), int(v[i])
                    xi = lang[i]
                    pi = xi[3:bi_].mean(0)
                    ps = xi[bi_ + 3 : vi - 5].mean(0)
                    pin = pi / (pi.norm() + 1e-9)
                    psn = ps / (ps.norm() + 1e-9)
                    key = KEY_AB[0] * pin + KEY_AB[1] * psn
                    add(f"instr_L{L}", t, pi.cpu().numpy())
                    add(f"state_L{L}", t, ps.cpu().numpy())
                    add(f"key_L{L}", t, key.cpu().numpy())
                    for c in range(n_real):
                        add(f"img_cam{c}_L{L}", t, img[i, c * PATCHES_PER_CAM : (c + 1) * PATCHES_PER_CAM].mean(0).cpu().numpy())
                    add(f"img_L{L}", t, img[i, : n_real * PATCHES_PER_CAM].mean(0).cpu().numpy())
            for i in valid:
                instr_lens[t].append(int(bnd[i]) - 3)
                used[t] += 1
        il = float(np.mean(instr_lens[t])) if instr_lens[t] else float("nan")
        print(f"[geom] task {t:2d} ({tin[t][:48]:48s}) used {used[t]:3d} skipped {skipped[t]:3d} instr_len {il:5.1f}", flush=True)

    for h in hooks:
        h.remove()

    task_names = {int(t): tin[t] for t in tasks}
    ep_len = {}
    try:
        eps = dataset.meta.episodes
        rows = (r for _, r in eps.iterrows()) if hasattr(eps, "iterrows") else iter(eps)
        acc = {}
        for e in rows:
            ts_ = list(e["tasks"]) if e["tasks"] is not None else []
            if not ts_:
                continue
            acc.setdefault(ts_[0], []).append(int(e["length"]))
        name_to_t = {n: int(t) for t, n in task_names.items()}
        ep_len = {name_to_t[n]: float(np.mean(v)) for n, v in acc.items() if n in name_to_t}
    except Exception as ex:  # noqa: BLE001 - informational only
        print(f"[geom] ep_len unavailable: {ex}", flush=True)

    out = {
        "checkpoint": str(cfg.policy.pretrained_path),
        "dataset": str(cfg.dataset.root),
        "config": {"expert_layers": EXPERT_LAYERS, "lm_layers": LM_LAYERS, "nb": NB, "bs": BS, "seed": SEED,
                   "key_ab": KEY_AB, "layout": layout, "fps": dataset.meta.fps},
        "tasks": task_names,
        "used": used,
        "skipped": skipped,
        "instr_len_mean": {t: (float(np.mean(v)) if v else None) for t, v in instr_lens.items()},
        "ep_len_mean_frames": ep_len,
        "spaces": {},
    }
    npz = {}
    for name, per in spaces.items():
        ts = [t for t in tasks if t in per and len(per[t]) > 0]
        X = {t: np.stack(per[t]).astype(np.float32) for t in ts}
        C = np.stack([X[t].mean(0) for t in ts])
        M = _cos_matrix(C)
        out["spaces"][name] = {
            "tasks": ts,
            "cos": np.round(M, 5).tolist(),
            "intra": {t: _intra_cos(X[t]) for t in ts},
            "centroid_norm": {t: float(np.linalg.norm(C[k])) for k, t in enumerate(ts)},
        }
        if OUT_NPZ:
            for t in ts:
                npz[f"{name}__t{t}"] = X[t].astype(np.float16)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"))
    if OUT_NPZ:
        np.savez_compressed(OUT_NPZ, **npz)

    def top_pairs(name, k=8):
        d = out["spaces"][name]
        ts = d["tasks"]
        M = np.array(d["cos"])
        iu = np.triu_indices(len(ts), 1)
        order = np.argsort(-M[iu])[:k]
        return [(ts[iu[0][o]], ts[iu[1][o]], round(float(M[iu][o]), 3)) for o in order]

    for name in ([f"instr_L{L}" for L in (8, 14)] + [f"key_L{L}" for L in (7, 13)]
                 + [f"expert_L{L}" for L in (8, 14)] + [f"img_L{L}" for L in (8,)]):
        if name in out["spaces"]:
            print(f"[geom] {name:>12}: top pairs {top_pairs(name)}", flush=True)
    print(f"[geom] saved {OUT_JSON}" + (f" + {OUT_NPZ}" if OUT_NPZ else ""), flush=True)
    print("GEOM-DONE", flush=True)


if __name__ == "__main__":
    main()
