#!/usr/bin/env python3
"""Off-trail probe, stage 2 (E56): score a model's denoised chunk on harvested rollout
states + demo-control states; optionally dump frozen-trunk features for the excursion
distance. Run once per model over the SAME harvest dirs; batch composition and denoise
seeds are deterministic functions of the (sorted) state list, so chunks are PAIRED across
models by construction (same noise, same batching).

Populations scored, in order:
  demo:i           demo-control states from the dataset (SCORE_TASK), via the probe-style
                   dataset path (uint8 cams -> /255 -> preprocessor). Even rows double as
                   the kNN feature reference in the report, odd rows as held-out control.
                   Ground-truth demo chunks are saved for the on-trail anchor check.
  {tag}:epE:cC     harvested states (tag = harvest dir basename), reconstructed as a raw
                   vector-env observation and pushed through preprocess_observation ->
                   env_preprocessor -> preprocessor — the exact eval-time pipeline.

Env knobs:
  SCORE_HARVESTS   comma list of harvest dirs (order matters — must match across models)
  SCORE_OUT_DIR    output dir
  SCORE_TAG        model tag for filenames (e.g. B, spec_e7)
  SCORE_SEEDS      denoise seeds per state (default 4)
  SCORE_TASK       dataset task_index for demo-control states (default 4 = e7)
  SCORE_DEMO_N     number of demo-control states (default 120)
  SCORE_FEAT_LAYER "" = off, else an LM layer index BELOW the first VLM memory layer
                   (for B: 9 — LM layers <10 are frozen AND memory-free = stage-1
                   features, so d(s) is model-independent when computed from B's pass)
  SCORE_BS         scoring batch size (default 12)

CLI = SequentialOnlineConfig (the probe convention: --policy.path + dataset + env args,
see run_e55_gpu_gradB.sh COMMON_ARGS).

Output in SCORE_OUT_DIR:
  chunks_{TAG}.npz   uids, chunks (N, K, chunk_len, 7) fp16 (normalized action space),
                     states (N, S), demo_gt (N_demo, chunk_len, 7), demo_uids
  features_{TAG}.npz uids, feats (N, hidden) fp16   [only when SCORE_FEAT_LAYER set]
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
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig,
    _build_dataloader_for_task,
    _collect_task_index_to_name,
)


def _load_harvest(hdir):
    hdir = Path(hdir)
    index = json.load(open(hdir / "index.json"))
    eps = sorted(index["episodes"], key=lambda x: x["ep"])
    rows = []  # (uid, {flat_key: np.ndarray}, success)
    for meta in eps:
        d = np.load(hdir / f"ep{meta['ep']:03d}.npz")
        okeys = [k for k in d.files if k.startswith("obs__")]
        for ci in range(int(d["call_steps"].shape[0])):
            uid = f"{hdir.name}:ep{meta['ep']:03d}:c{ci:02d}"
            rows.append((uid, {k[5:]: d[k][ci] for k in okeys}, bool(meta["success"])))
    return index, rows


def _rebuild_env_obs(row_obs_list):
    """Invert probe_rollout_harvest._flatten_raw_obs for a batch of rows."""
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


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    harvests = [h for h in os.environ["SCORE_HARVESTS"].split(",") if h]
    out_dir = Path(os.environ["SCORE_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = os.environ["SCORE_TAG"]
    n_seeds = int(os.environ.get("SCORE_SEEDS", "4"))
    demo_task = int(os.environ.get("SCORE_TASK", "4"))
    demo_n = int(os.environ.get("SCORE_DEMO_N", "120"))
    feat_layer = os.environ.get("SCORE_FEAT_LAYER", "")
    bs = int(os.environ.get("SCORE_BS", "12"))

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
        except Exception as e:  # models without the language machinery
            print(f"[score] precompute_task_embeddings skipped: {e}", flush=True)
    policy = accelerator.prepare(policy)
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    unwrapped.eval()
    task_index_to_name = _collect_task_index_to_name(dataset)
    cam_keys = list(dataset.meta.camera_keys)

    # --- feature hook (frozen trunk; fires on every prefix pass, keep the last) ---
    feat_buf = {}
    if feat_layer != "":
        target = None
        suffix = f"language_model.layers.{feat_layer}"
        for name, m in unwrapped.named_modules():
            if name.endswith(suffix):
                target = (name, m)
        assert target is not None, f"feature layer module *{suffix} not found"
        print(f"[score] feature hook on {target[0]}", flush=True)

        def fhook(module, args, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            feat_buf["feat"] = h.float().mean(dim=1).detach().cpu()  # (B, hidden)

        target[1].register_forward_hook(fhook)

    all_uids, all_chunks, all_states = [], [], []
    all_feats = []
    demo_gt, demo_uids = [], []
    batch_counter = 0

    def run_batch(batch, uids, states_np, gt=None):
        nonlocal batch_counter
        ks = []
        for s in range(n_seeds):
            torch.manual_seed(77_000 + 131 * batch_counter + s)  # paired across models
            with torch.no_grad(), accelerator.autocast():
                a = unwrapped.predict_action_chunk(batch)
            ks.append(a[:, :, :7].float().cpu())
        batch_counter += 1
        chunks = torch.stack(ks, dim=1)  # (B, K, T, 7)
        all_uids.extend(uids)
        all_chunks.append(chunks.to(torch.float16).numpy())
        all_states.append(states_np)
        if feat_layer != "":
            all_feats.append(feat_buf["feat"].to(torch.float16).numpy())
        if gt is not None:
            demo_gt.append(gt)
            demo_uids.extend(uids)

    # ---------- population 1: demo-control states ----------
    dl = _build_dataloader_for_task(dataset, task_index_to_name, demo_task, batch_size=bs,
                                    num_workers=2, device_type=device.type, drop_n_last_frames=0)
    torch.manual_seed(4242)  # reproducible batch composition across model invocations
    it = iter(dl)
    n_demo_batches = math.ceil(demo_n / bs)
    for j in range(n_demo_batches):
        raw = next(it)
        for ck in cam_keys:
            if ck in raw and raw[ck].dtype == torch.uint8:
                raw[ck] = raw[ck].to(dtype=torch.float32) / 255.0
        st = raw["observation.state"].float().cpu().numpy()
        b = preprocessor({k: (v.clone() if torch.is_tensor(v) else v) for k, v in raw.items()})
        gt = b["action"][:, :, :7].float().cpu().numpy()
        uids = [f"demo:{j * bs + i}" for i in range(st.shape[0])]
        run_batch(b, uids, st, gt=gt)
        print(f"[score:{tag}] demo batch {j + 1}/{n_demo_batches}", flush=True)

    # ---------- population 2+: harvested rollout states ----------
    for hdir in harvests:
        index, rows = _load_harvest(hdir)
        task_str = index["task"]
        for j0 in range(0, len(rows), bs):
            chunk_rows = rows[j0:j0 + bs]
            uids = [r[0] for r in chunk_rows]
            raw_env_obs = _rebuild_env_obs([r[1] for r in chunk_rows])
            o = preprocess_observation(raw_env_obs)
            o["task"] = [task_str] * len(chunk_rows)
            o = env_pre(o)
            sts = o["observation.state"].float().cpu().numpy()  # canonical 8D, post env_pre
            b = preprocessor(o)
            run_batch(b, uids, sts)
        print(f"[score:{tag}] harvest {Path(hdir).name}: {len(rows)} states done", flush=True)

    np.savez_compressed(
        out_dir / f"chunks_{tag}.npz",
        uids=np.array(all_uids),
        chunks=np.concatenate(all_chunks, axis=0),
        states=np.concatenate(all_states, axis=0),
        demo_gt=np.concatenate(demo_gt, axis=0) if demo_gt else np.zeros((0,)),
        demo_uids=np.array(demo_uids),
    )
    if feat_layer != "":
        np.savez_compressed(out_dir / f"features_{tag}.npz",
                            uids=np.array(all_uids),
                            feats=np.concatenate(all_feats, axis=0))
    meta = {"tag": tag, "n_states": len(all_uids), "n_seeds": n_seeds, "harvests": harvests,
            "demo_task": demo_task, "feat_layer": feat_layer,
            "policy_path": str(cfg.policy.pretrained_path)}
    with open(out_dir / f"meta_{tag}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[score:{tag}] DONE: {len(all_uids)} states x {n_seeds} seeds -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
