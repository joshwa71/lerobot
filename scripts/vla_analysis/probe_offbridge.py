#!/usr/bin/env python3
"""Off-bridge generalization probe (E41).

Flow-matching training only ever queries the field ON the noise-demo interpolation bridge
(x_t = t*noise + (1-t)*a, target u = noise - a). Passing noise' = noise + sigma*xi queries it
at x_t + t*sigma*xi — off the bridge — where the exact target is still known analytically
(u' = u + sigma*xi). L(sigma) per arm measures field quality at increasing distance from the
training manifold, with paired (noise, time, xi) across arms. L(0) reproduces the MSE matrix.

Env: PROBE_RUN_DIR, PROBE_CKPTS, PROBE_OUT. Sigma grid fixed below.
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
from lerobot.policies.pi05.modeling_pi05 import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

N_BATCHES = 3
BS = 12
K = 6
SIGMAS = [0.0, 0.1, 0.3, 0.6]


def load_slots(unwrapped, model_dir):
    from safetensors import safe_open
    sd = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if ".mlp.mem.slot_" in k:
                sd[k] = f.get_tensor(k)
    _, unexpected = unwrapped.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    print(f"[load] {model_dir.split('/checkpoints/')[-1]}", flush=True)


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    run_dir = os.environ["PROBE_RUN_DIR"]
    ckpts = [c.split(":") for c in os.environ["PROBE_CKPTS"].split(",")]
    out_path = os.environ["PROBE_OUT"]
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

    def make_batches(t):
        dl = _build_dataloader_for_task(dataset, task_index_to_name, t, batch_size=BS,
                                        num_workers=2, device_type=device.type, drop_n_last_frames=0)
        torch.manual_seed(7919 * (t + 1))
        out = []
        it = iter(dl)
        for _ in range(N_BATCHES):
            b = next(it)
            for ck in cam_keys:
                if ck in b and b[ck].dtype == torch.uint8:
                    b[ck] = b[ck].to(dtype=torch.float32) / 255.0
            out.append(preprocessor(b))
        return out

    def offbridge_losses(batch, t, name):
        """dict sigma -> mean loss over K paired draws (first 7 action dims)."""
        images, img_masks = unwrapped._preprocess_images(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        B = tokens.shape[0]
        task_emb = None
        if hasattr(unwrapped, "get_task_embeddings"):
            task_emb = unwrapped.get_task_embeddings([name] * B)
            if task_emb is not None:
                task_emb = task_emb.to(device=device)
        actions = unwrapped.prepare_action(batch)
        out = {s: [] for s in SIGMAS}
        for k in range(K):
            seed = 100_000 * (t + 1) + 613 * k
            torch.manual_seed(seed)
            noise = unwrapped.model.sample_noise(actions.shape, actions.device)
            time = unwrapped.model.sample_time(actions.shape[0], actions.device)
            xi = torch.randn(actions.shape, generator=torch.Generator(device="cpu").manual_seed(seed + 7),
                             dtype=torch.float32).to(actions.device, dtype=actions.dtype)
            for s in SIGMAS:
                with torch.no_grad(), accelerator.autocast():
                    losses = unwrapped.model.forward(
                        images, img_masks, tokens, masks, actions,
                        noise + s * xi, time, task_emb=task_emb)
                out[s].append(float(losses[:, :, :7].mean()))
        return {s: sum(v) / len(v) for s, v in out.items()}

    with open(out_path, "a") as fh:
        for t_str, st in ckpts:
            t = int(t_str.lstrip("t"))
            name = task_index_to_name[t]
            load_slots(unwrapped, os.path.join(run_dir, "checkpoints", st, "pretrained_model"))
            batches = make_batches(t)
            acc = {s: [] for s in SIGMAS}
            for b in batches:
                r = offbridge_losses(b, t, name)
                for s in SIGMAS:
                    acc[s].append(r[s])
            rec = {"run": os.path.basename(run_dir), "task": t, "ckpt": st,
                   **{f"L{s}": sum(v) / len(v) for s, v in acc.items()}}
            # excess error vs the analytic floor: perturbation adds sigma^2 irreducibly IF the
            # model tracked it perfectly; excess = L(s) - L(0) - 0 (the target shift is exact,
            # so a perfect-generalizing model keeps L(s) = L(0)).
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"[offb] t{t} {st}: " + " ".join(f"L({s})={rec[f'L{s}']:.4f}" for s in SIGMAS), flush=True)


if __name__ == "__main__":
    main()
