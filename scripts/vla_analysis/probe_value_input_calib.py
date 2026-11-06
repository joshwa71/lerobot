#!/usr/bin/env python3
"""Value-input noise calibration probe (E57 follow-up).

Purpose: dose the proposed value-input-noise lever from measurement instead of guess.
The off-trail campaign showed retrieval is healthy off-manifold but the VALUE content
expresses the wrong function at excursion states. The proposed fix perturbs the live
hidden state x that the LoRA slot transforms consume during sequential training (router
untouched — it reads the frozen branch). This probe measures, per memory module, HOW FAR
x actually moves between rollout-visited off-trail states and their nearest demo states —
the per-dimension displacement distribution the noise should imitate.

Mechanics:
  - Forwards run at bs=1 with ONE fixed denoise seed (CALIB_SEED) for every state, so the
    initial noise tokens are identical across states and expert-tower displacement is
    purely observation-driven.
  - A forward_pre_hook on each MLPPlusMemory captures args[0] (the live x). Pass A of the
    frozen-route dual forward is skipped via the wrapper's `_frozen_capture` flag, so only
    the pass-B value-path input is recorded. Expert modules fire once per denoise step
    (keep FIRST and LAST); VLM modules once per forward (cached prefix). VLM captures are
    sliced to the served text span (module.text_span) and near-zero-displacement rows
    (pads) are filtered.
  - Each sampled off-trail state is paired with its nearest demo state by the report's
    proprio metric (normalized 8D observation.state L2), and pairs are binned into
    near/mid/far tertiles of the harvest distance distribution.

Outputs per module x capture-point x band: median relative per-token displacement
  ||dx_t|| / median||x_demo_t||; per-dim ratio std(dx_d)/std(x_demo_d) (median/P75/P90
  over dims — the noise-scale target); top-10 SVD energy fraction of dx (structure check:
  ~isotropic -> per-dim independent noise is a fair imitation; concentrated -> a low-rank
  structured variant would be needed). Plus a suggested (p, sigma_rel) dose block matched
  to the mid band.

Env knobs: CALIB_HARVEST (harv_B dir), CALIB_OUT (json), CALIB_N (off-trail sample,
default 200), CALIB_DEMO_N (default 120), CALIB_SEED (default 12345).
CLI = SequentialOnlineConfig probe convention (COMMON_ARGS + --policy.path=<B ckpt>).
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator

from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.modules.memory_lite import MLPPlusMemory
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig,
    _build_dataloader_for_task,
    _collect_task_index_to_name,
)


# --- duplicated from probe_offtrail_score.py (kept standalone on purpose) ---
def _load_harvest(hdir):
    hdir = Path(hdir)
    index = json.load(open(hdir / "index.json"))
    eps = sorted(index["episodes"], key=lambda x: x["ep"])
    rows = []
    for meta in eps:
        d = np.load(hdir / f"ep{meta['ep']:03d}.npz")
        okeys = [k for k in d.files if k.startswith("obs__")]
        for ci in range(int(d["call_steps"].shape[0])):
            uid = f"{hdir.name}:ep{meta['ep']:03d}:c{ci:02d}"
            rows.append((uid, {k[5:]: d[k][ci] for k in okeys}, bool(meta["success"])))
    return index, rows


def _rebuild_env_obs(row_obs_list):
    pixels, robot_state = {}, {}
    for k in row_obs_list[0]:
        arr = np.stack([r[k] for r in row_obs_list])
        if k.startswith("px__"):
            pixels[k[4:]] = np.ascontiguousarray(arr).astype(np.uint8)
        else:
            parts = k[4:].split("__")
            d = robot_state
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = arr
    return {"pixels": pixels, "robot_state": robot_state}
# ---------------------------------------------------------------------------


class ValueInputTap:
    """Pre-hooks on every MLPPlusMemory; records pass-B live x per module per forward."""

    def __init__(self, policy):
        self.mods = {}   # key -> (module, text_span)
        self.buf = {}    # key -> list of captured (T, D) fp16 cpu tensors (this forward)
        for name, m in policy.named_modules():
            if isinstance(m, MLPPlusMemory):
                key = name.replace(".mlp", "")
                self.mods[key] = m
                m.register_forward_pre_hook(self._make_hook(key))

    def _make_hook(self, key):
        def hook(module, args):
            if getattr(module, "_frozen_capture", False):
                return  # pass A: memory-free routing feature, not the value input
            x = args[0]
            span = int(getattr(module, "text_span", 0) or 0)
            if span > 0:
                x = x[:, -span:, :]
            self.buf.setdefault(key, []).append(x[0].detach().to(torch.float16).cpu())
        return hook

    def start(self):
        self.buf = {}

    def collect(self):
        """Return {key: {'first': (T,D), 'last': (T,D)}} for this forward."""
        out = {}
        for key, caps in self.buf.items():
            out[key] = {"first": caps[0], "last": caps[-1], "n_fires": len(caps)}
        return out


def _tower(key):
    return "vlm" if "language_model" in key else "expert"


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    hdir = os.environ["CALIB_HARVEST"]
    out_path = os.environ["CALIB_OUT"]
    n_off = int(os.environ.get("CALIB_N", "200"))
    n_demo = int(os.environ.get("CALIB_DEMO_N", "120"))
    calib_seed = int(os.environ.get("CALIB_SEED", "12345"))
    demo_task = int(os.environ.get("CALIB_TASK", "4"))

    cfg.validate()
    accelerator = Accelerator()
    device = accelerator.device
    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    processor_kwargs = {"preprocessor_overrides": {
        "device_processor": {"device": device.type},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }}
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path, **processor_kwargs)
    env_pre, _ = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    if hasattr(policy, "precompute_task_embeddings"):
        try:
            policy.precompute_task_embeddings(dataset.meta)
        except Exception as e:
            print(f"[calib] precompute_task_embeddings skipped: {e}", flush=True)
    policy = accelerator.prepare(policy)
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    unwrapped.eval()
    tap = ValueInputTap(unwrapped)
    print(f"[calib] tapping {len(tap.mods)} memory wrappers: {list(tap.mods)}", flush=True)
    task_index_to_name = _collect_task_index_to_name(dataset)
    cam_keys = list(dataset.meta.camera_keys)

    # ---------- demo states: same seeded batches as the scorer ----------
    dl = _build_dataloader_for_task(dataset, task_index_to_name, demo_task, batch_size=12,
                                    num_workers=2, device_type=device.type, drop_n_last_frames=0)
    torch.manual_seed(4242)
    demo_rows = []  # (raw single-row batch dict, state8)
    it = iter(dl)
    for _ in range(math.ceil(n_demo / 12)):
        raw = next(it)
        for ck in cam_keys:
            if ck in raw and raw[ck].dtype == torch.uint8:
                raw[ck] = raw[ck].to(dtype=torch.float32) / 255.0
        bsz = raw["observation.state"].shape[0]
        for i in range(bsz):
            row = {}
            for k, v in raw.items():
                if torch.is_tensor(v):
                    row[k] = v[i:i + 1].clone()
                elif isinstance(v, (list, tuple)):
                    row[k] = [v[i]]
                else:
                    row[k] = v
            demo_rows.append((row, raw["observation.state"][i].float().numpy()))
    demo_rows = demo_rows[:n_demo]
    demo_states = np.stack([s for _, s in demo_rows])
    st_std = demo_states.std(axis=0) + 1e-6

    # ---------- harvest states: 8D pre-pass, tertile-stratified sample ----------
    index, rows = _load_harvest(hdir)
    task_str = index["task"]
    pre = []  # (uid, processed-obs dict pre-preprocessor, state8, d_state)
    for uid, obs, succ in rows:
        raw_env = _rebuild_env_obs([obs])
        o = preprocess_observation(raw_env)
        o["task"] = [task_str]
        o = env_pre(o)
        s8 = o["observation.state"][0].float().cpu().numpy()
        d = float(np.min(np.linalg.norm((s8 - demo_states) / st_std, axis=1)))
        pre.append((uid, o, s8, d))
    ds = np.array([p[3] for p in pre])
    t1, t2 = np.quantile(ds, [1 / 3, 2 / 3])
    rng = np.random.RandomState(97)
    sampled = []
    for lo, hi, band in ((-np.inf, t1, "near"), (t1, t2, "mid"), (t2, np.inf, "far")):
        idxs = [i for i in range(len(pre)) if lo < ds[i] <= hi]
        rng.shuffle(idxs)
        sampled += [(pre[i], band) for i in idxs[:n_off // 3]]
    print(f"[calib] tertile edges d_state: {t1:.3f}/{t2:.3f}; sampled {len(sampled)} off-trail states",
          flush=True)

    # ---------- forwards at bs=1, one fixed denoise seed everywhere ----------
    def fwd_one(batch):
        tap.start()
        torch.manual_seed(calib_seed)
        with torch.no_grad(), accelerator.autocast():
            unwrapped.predict_action_chunk(batch)
        return tap.collect()

    demo_caps = []
    for j, (row, _) in enumerate(demo_rows):
        demo_caps.append(fwd_one(preprocessor({k: (v.clone() if torch.is_tensor(v) else v)
                                               for k, v in row.items()})))
        if (j + 1) % 40 == 0:
            print(f"[calib] demo forwards {j + 1}/{len(demo_rows)}", flush=True)
    fire_note = {k: v["n_fires"] for k, v in demo_caps[0].items()}
    print(f"[calib] pass-B fires per forward: {fire_note}", flush=True)

    off_caps = []
    for j, ((uid, o, s8, d), band) in enumerate(sampled):
        nn_i = int(np.argmin(np.linalg.norm((s8 - demo_states) / st_std, axis=1)))
        caps = fwd_one(preprocessor({k: (v.clone() if torch.is_tensor(v) else v)
                                     for k, v in o.items()}))
        off_caps.append((caps, nn_i, band, d, uid))
        if (j + 1) % 40 == 0:
            print(f"[calib] off-trail forwards {j + 1}/{len(sampled)}", flush=True)

    # ---------- stats ----------
    result = {"tertile_edges": [float(t1), float(t2)], "fires": fire_note, "modules": {}}
    cap_points = {"expert": ["first", "last"], "vlm": ["last"]}
    for key in tap.mods:
        tw = _tower(key)
        mod_res = {}
        for cp in cap_points[tw]:
            xd = torch.stack([c[key][cp].float() for c in demo_caps])       # (Nd, T, D)
            x_dim_std = xd.reshape(-1, xd.shape[-1]).std(dim=0) + 1e-6      # (D,)
            tok_scale = xd.norm(dim=-1).median().item()
            cp_res = {}
            for band in ("near", "mid", "far"):
                deltas = []
                for caps, nn_i, b, d, uid in off_caps:
                    if b != band:
                        continue
                    dx = caps[key][cp].float() - demo_caps[nn_i][key][cp].float()  # (T, D)
                    if tw == "vlm":  # drop pad rows (near-zero displacement)
                        keep = dx.norm(dim=-1) > max(1e-3, 0.05 * dx.norm(dim=-1).median().item())
                        dx = dx[keep]
                    if dx.shape[0]:
                        deltas.append(dx)
                if not deltas:
                    continue
                D = torch.cat(deltas)                                        # (Ntok, dim)
                rel_tok = (D.norm(dim=-1) / tok_scale)
                ratio = (D.std(dim=0) / x_dim_std)                           # (dim,)
                sub = D[torch.randperm(D.shape[0])[:2000]]
                s = torch.linalg.svdvals(sub - sub.mean(0))
                top10 = float((s[:10] ** 2).sum() / (s ** 2).sum())
                cp_res[band] = {
                    "n_tokens": int(D.shape[0]),
                    "rel_tok_disp_median": float(rel_tok.median()),
                    "rel_tok_disp_p90": float(rel_tok.quantile(0.9)),
                    "dim_ratio_median": float(ratio.median()),
                    "dim_ratio_p75": float(ratio.quantile(0.75)),
                    "dim_ratio_p90": float(ratio.quantile(0.9)),
                    "svd_top10_energy": top10,
                }
            mod_res[cp] = cp_res
        result["modules"][key] = mod_res

    # ---------- suggested dose (mid band, action-proximal capture) ----------
    print("\n== per-module summary (mid band) ==")
    sug = {}
    for key, mod_res in sorted(result["modules"].items()):
        tw = _tower(key)
        cp = "last"
        m = mod_res.get(cp, {}).get("mid")
        if not m:
            continue
        print(f"{key.split('.')[-3]}.{key.split('.')[-1]:>2} [{tw:6}]  "
              f"rel_tok {m['rel_tok_disp_median']:.3f} (p90 {m['rel_tok_disp_p90']:.3f})  "
              f"dim_ratio {m['dim_ratio_median']:.3f}/{m['dim_ratio_p75']:.3f}/{m['dim_ratio_p90']:.3f}  "
              f"top10 {m['svd_top10_energy']:.2f}")
        sug.setdefault(tw, []).append(m["dim_ratio_median"])
    print("\n== suggested dose (variance-matched to mid band, sigma relative to per-dim x std) ==")
    result["suggested"] = {}
    for tw, rs in sug.items():
        r = float(np.median(rs))
        result["suggested"][tw] = {"dim_ratio_median_mid": r,
                                   "p0.25_sigma_rel": 2 * r, "p1.0_sigma_rel": r}
        print(f"{tw}: matched p*sigma_rel^2 = {r ** 2:.4f} -> p=0.25: sigma_rel={2 * r:.3f} | "
              f"p=1.0 (dense): sigma_rel={r:.3f}")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\n[calib] full stats -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
