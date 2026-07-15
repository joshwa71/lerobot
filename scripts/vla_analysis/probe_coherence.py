#!/usr/bin/env python3
"""Trajectory-error coherence probe (E41).

Steps the REAL 10-step denoise manually (replicating model.sample_actions: prefix KV once,
then v_t = denoise_step(x_t, t); x_t += dt*v_t). At the model's OWN point x_t at time t the
demo-consistent velocity is analytic: v* = (x_t - a_demo)/t (equals noise - a on the bridge).
Records e_k = v_pred - v* per step (first 7 action dims) and reports:
  - per-step RMS error profile (prediction: ~arm-invariant, like every pointwise stat)
  - adjacent-step alignment cos(e_k, e_{k+1})
  - coherence ratio ||sum_k e_k|| / sum_k ||e_k||  (1 = fully coherent accumulation,
    ~1/sqrt(10)=0.32 = orthogonal/cancelling)
  - endpoint error ||x_final - a|| (ties to the denoised-chunk probe)

Env: PROBE_RUN_DIR, PROBE_CKPTS, PROBE_OUT.
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
from lerobot.policies.pi05.modeling_pi05 import (
    OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, make_att_2d_masks,
)

N_BATCHES = 3
BS = 12
N_SEEDS = 2
NUM_STEPS = 10


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
    model = unwrapped.model
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

    @torch.no_grad()
    def traj_errors(batch, t, name, seed):
        """returns per-step errors list[(B,50,7)] and endpoint error (B,)."""
        images, img_masks = unwrapped._preprocess_images(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        bsize = tokens.shape[0]
        task_emb = None
        if hasattr(unwrapped, "get_task_embeddings"):
            task_emb = unwrapped.get_task_embeddings([name] * bsize)
            if task_emb is not None:
                task_emb = task_emb.to(device=device)
        a = unwrapped.prepare_action(batch)  # (B,50,32) normalized/padded
        torch.manual_seed(seed)
        noise = model.sample_noise(a.shape, a.device)

        with accelerator.autocast():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, tokens, masks)
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
            # sample_actions does this before the prefix KV pass (SDPA rejects the 4d
            # float mask against bf16 queries)
            model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past_kv = model.paligemma_with_expert.forward(
                attention_mask=model._prepare_attention_masks_4d(prefix_att_2d),
                position_ids=prefix_pos, past_key_values=None,
                inputs_embeds=[prefix_embs, None], use_cache=True, task_emb=task_emb)

            dt = -1.0 / NUM_STEPS
            x_t = noise.clone()
            errs = []
            for step in range(NUM_STEPS):
                tcur = 1.0 + step * dt
                tt = torch.tensor(tcur, dtype=torch.float32, device=device).expand(bsize)
                v_t = model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks, past_key_values=past_kv,
                    x_t=x_t, timestep=tt, task_emb=task_emb).float()
                v_star = (x_t.float() - a.float()) / tcur
                errs.append((v_t - v_star)[:, :, :7].cpu())
                x_t = x_t + dt * v_t.to(x_t.dtype)
            endpoint = (x_t.float() - a.float())[:, :, :7].reshape(bsize, -1).norm(dim=1).cpu()
        return errs, endpoint

    with open(out_path, "a") as fh:
        for t_str, st in ckpts:
            t = int(t_str.lstrip("t"))
            name = task_index_to_name[t]
            load_slots(unwrapped, os.path.join(run_dir, "checkpoints", st, "pretrained_model"))
            batches = make_batches(t)
            prof = torch.zeros(NUM_STEPS)
            coh, adjcos, endp, n = [], [], [], 0
            for j, b in enumerate(batches):
                for s in range(N_SEEDS):
                    errs, endpoint = traj_errors(b, t, name, seed=100_000 * (t + 1) + 31 * j + s)
                    E = torch.stack(errs)                       # (S, B, 50, 7)
                    S, B = E.shape[0], E.shape[1]
                    flat = E.reshape(S, B, -1)                  # (S,B,350)
                    norms = flat.norm(dim=2)                    # (S,B)
                    prof += (norms ** 2).mean(dim=1)
                    n += 1
                    csum = flat.sum(dim=0).norm(dim=1)          # (B,)
                    cabs = norms.sum(dim=0)
                    coh += (csum / cabs.clamp(min=1e-9)).tolist()
                    cs = torch.nn.functional.cosine_similarity(flat[:-1], flat[1:], dim=2)  # (S-1,B)
                    adjcos += cs.mean(dim=0).tolist()
                    endp += endpoint.tolist()
            prof = (prof / n).sqrt()
            rec = {"run": os.path.basename(run_dir), "task": t, "ckpt": st,
                   "step_rms": [round(float(x), 4) for x in prof],
                   "coherence": sum(coh) / len(coh),
                   "adj_cos": sum(adjcos) / len(adjcos),
                   "endpoint": sum(endp) / len(endp)}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"[coh] t{t} {st}: coherence={rec['coherence']:.3f} adj_cos={rec['adj_cos']:.3f} "
                  f"endpoint={rec['endpoint']:.3f} rms[0,4,9]={prof[0]:.3f}/{prof[4]:.3f}/{prof[9]:.3f}", flush=True)


if __name__ == "__main__":
    main()
