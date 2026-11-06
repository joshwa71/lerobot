#!/usr/bin/env python3
"""Jitter probe (E43): off-manifold brittleness of the denoised chunk.

The cheap proxy for the off-trail instrument (E42 addendum / E43 discussion): perturb demo
observations by a small, MODEL-INDEPENDENT amount and measure how fast the executed
(10-step-denoised) chunk degrades relative to the clean chunk. Rollout states drift off the
demo trail; a function that is locally jagged around the demo manifold converts fit into
rollouts worse than a smooth one. LoRA-FT (dense r32) converts e4 fit to 58% where our
sparse routed mixtures convert to 20-35 at BETTER on-demo chunk error — if our degradation
slopes are systematically steeper than LoRA's at matched clean error, the conversion gap is
localized to function smoothness; if slopes are equal, the gap lives elsewhere.

Channels (perturbation applied to the RAW batch, before preprocessing; RNG seeded per
(task, batch, scale) so every model sees IDENTICAL perturbed inputs):
  - state: Gaussian on observation.state, sigma in units of the per-dim dataset std
  - image: Gaussian pixel noise on both cameras (post /255 floats, clamped to [0,1])
Target stays the demo chunk: for small perturbations the demo action remains ~the right
answer, and any shared target bias cancels in the ACROSS-MODEL slope comparison (which is
the read; absolute levels are not).

Env: PROBE_RUN_DIR (run dir containing checkpoints/), PROBE_CKPTS ("t0:005000,t2:015000"),
PROBE_OUT (jsonl, appended), PROBE_SWAP_SLOTS=1/0 (1: load_slots per-ckpt slot swap —
memory runs; 0: policy loaded once from --policy.path, for LoRA/base checkpoints whose
adaptation is not in slot tensors), MINI=0/1.
Output rows: {probe: "jitter", run, task, ckpt, channel, sigma, chunk_mse, late10_mse}.
"""
import os, json
import torch
from accelerate import Accelerator
from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig, _collect_task_index_to_name, _build_dataloader_for_task,
)

MINI = os.environ.get("MINI", "0") == "1"
N_BATCHES = 2 if MINI else 4
BS = 8 if MINI else 12
SEEDS = 1 if MINI else 2
# (channel, sigma) grid; sigma=0 row is the clean anchor
CONDS = [("clean", 0.0), ("state", 0.1), ("state", 0.2), ("image", 0.05)]


def load_slots(unwrapped, run_dir, st):
    from safetensors import safe_open
    p = os.path.join(run_dir, "checkpoints", st, "pretrained_model", "model.safetensors")
    sd = {}
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if ".mlp.mem.slot_" in k:
                sd[k] = f.get_tensor(k)
    _, unexpected = unwrapped.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    print(f"[load] {st}: {len(sd)} slot tensors", flush=True)


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    run_dir = os.environ["PROBE_RUN_DIR"]
    ckpts = [c.split(":") for c in os.environ["PROBE_CKPTS"].split(",")]
    out_path = os.environ["PROBE_OUT"]
    swap_slots = os.environ.get("PROBE_SWAP_SLOTS", "1") == "1"
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
    if hasattr(policy, "precompute_task_embeddings"):
        policy.precompute_task_embeddings(dataset.meta)
    policy = accelerator.prepare(policy)
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    unwrapped.eval()
    task_index_to_name = _collect_task_index_to_name(dataset)
    cam_keys = list(dataset.meta.camera_keys)
    state_std = torch.as_tensor(
        dataset.meta.stats["observation.state"]["std"], dtype=torch.float32
    )

    def make_raw_batches(t):
        """Raw batches BEFORE preprocessing so perturbations are model-independent."""
        dl = _build_dataloader_for_task(dataset, task_index_to_name, t, batch_size=BS,
                                        num_workers=2, device_type=device.type, drop_n_last_frames=0)
        torch.manual_seed(7919 * (t + 1))  # same batches as probe_conversion
        out = []
        it = iter(dl)
        for _ in range(N_BATCHES):
            b = next(it)
            for ck in cam_keys:
                if ck in b and b[ck].dtype == torch.uint8:
                    b[ck] = b[ck].to(dtype=torch.float32) / 255.0
            out.append(b)
        return out

    def perturb(raw, channel, sigma, t, j):
        """Return a perturbed COPY; RNG seeded per (task,batch,channel-scale) — model-blind."""
        b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in raw.items()}
        if channel == "clean" or sigma == 0:
            return b
        g = torch.Generator(device="cpu")
        g.manual_seed(1_000_003 * (t + 1) + 977 * j + int(sigma * 1000) + (7 if channel == "image" else 0))
        if channel == "state":
            s = b["observation.state"]
            noise = torch.randn(s.shape, generator=g, dtype=torch.float32)
            b["observation.state"] = s + noise * state_std.to(s.dtype) * sigma
        elif channel == "image":
            for ck in cam_keys:
                if ck in b:
                    im = b[ck]
                    noise = torch.randn(im.shape, generator=g, dtype=torch.float32).to(im.dtype)
                    b[ck] = (im + sigma * noise).clamp(0.0, 1.0)
        else:
            raise ValueError(channel)
        return b

    with open(out_path, "a") as fh:
        for t_str, st in ckpts:
            t = int(t_str.lstrip("t"))
            if swap_slots:
                load_slots(unwrapped, run_dir, st)
            name = task_index_to_name[t]
            raw_batches = make_raw_batches(t)
            for channel, sigma in CONDS:
                mses, late10s = [], []
                for j, raw in enumerate(raw_batches):
                    b = preprocessor(perturb(raw, channel, sigma, t, j))
                    gt = b["action"][:, :, :7].float().cpu()
                    for s in range(SEEDS):
                        torch.manual_seed(50_000 * (t + 1) + 97 * j + s)  # same denoise seeds as probe_conversion
                        with torch.no_grad(), accelerator.autocast():
                            a = unwrapped.predict_action_chunk(b)
                        p = a.float().cpu()
                        mses.append(float(((p - gt) ** 2).mean()))
                        late10s.append(float(((p[:, -10:] - gt[:, -10:]) ** 2).mean()))
                rec = {"probe": "jitter", "run": os.path.basename(run_dir), "task": t, "ckpt": st,
                       "channel": channel, "sigma": sigma,
                       "chunk_mse": sum(mses) / len(mses), "late10_mse": sum(late10s) / len(late10s)}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"[jitter] t{t} {st} {channel}@{sigma}: chunk={rec['chunk_mse']:.4f} "
                      f"late10={rec['late10_mse']:.4f}", flush=True)


if __name__ == "__main__":
    main()
