#!/usr/bin/env python3
"""Error-decomposition probe (E41): bias vs variance of the velocity error, per arm per task.

For paired (state, noise, tau) draws: e = v_pred - u_t (signed, first 7 dims). Over K draws
per state, decompose MSE into a per-(state,pos,dim) bias component (mean over draws, finite-K
corrected) and residual variance. Also measure the A-phase pull: cosine between the arm's
bias field and the PRE-sequential (A-checkpoint) bias field on identical draws — tests
"the bias is the shrunk residual of the A-phase content" directly.

Env: PROBE_RUN_DIR, PROBE_CKPTS ("t0:005000,..."), PROBE_A_CKPT (path to pre-sequential
pretrained_model; slot-swapped in for the reference bias), PROBE_OUT.
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

N_BATCHES = 3
BS = 12
K = 6  # (noise, tau) draws per state


def load_slots(unwrapped, model_dir):
    from safetensors import safe_open
    p = os.path.join(model_dir, "model.safetensors")
    sd = {}
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if ".mlp.mem.slot_" in k:
                sd[k] = f.get_tensor(k)
    _, unexpected = unwrapped.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    print(f"[load] {model_dir.split('/checkpoints/')[-1]}: {len(sd)} slot tensors", flush=True)


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    run_dir = os.environ["PROBE_RUN_DIR"]
    ckpts = [c.split(":") for c in os.environ["PROBE_CKPTS"].split(",")]
    a_ckpt = os.environ.get("PROBE_A_CKPT", "")
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

    v_box = {}
    unwrapped.model.action_out_proj.register_forward_hook(
        lambda m, i, o: v_box.__setitem__("v", o.detach().float()))

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

    def signed_errors(batch, t, name):
        """(K, B, 50, 7) signed velocity errors over K paired draws."""
        B = batch[next(iter(batch))].shape[0]
        task_emb = None
        if hasattr(unwrapped, "get_task_embeddings"):
            task_emb = unwrapped.get_task_embeddings([name] * B)
            if task_emb is not None:
                task_emb = task_emb.to(device=device)
        actions = unwrapped.prepare_action(batch)  # (B, 50, 32) normalized+padded
        errs = []
        for k in range(K):
            seed = 100_000 * (t + 1) + 613 * k
            # pre-draw noise/time with the same RNG stream policy.forward will use
            torch.manual_seed(seed)
            noise = unwrapped.model.sample_noise(actions.shape, actions.device)
            time = unwrapped.model.sample_time(actions.shape[0], actions.device)
            u_t = (noise - actions).float()
            torch.manual_seed(seed)
            with torch.no_grad(), accelerator.autocast():
                unwrapped.forward(batch, task_emb=task_emb)
            v_t = v_box["v"]  # (B, 50, 32)
            errs.append((v_t - u_t)[:, :, :7].cpu())
        return torch.stack(errs)  # (K, B, 50, 7)

    def decompose(E):
        """E: (K, B, 50, 7) -> dict of bias/variance stats (finite-K corrected)."""
        mse = float((E ** 2).mean())
        m = E.mean(dim=0)                      # (B,50,7) per-state bias estimate
        v = E.var(dim=0, unbiased=True)        # per-element draw variance
        bias2_state = (m ** 2 - v / K).clamp(min=0)   # unbiased bias^2 per element
        bias2_task = (E.mean(dim=(0, 1)) ** 2 - (v.mean(dim=0) / (K * E.shape[1]))).clamp(min=0)
        out = {
            "mse": mse,
            "bias_frac_state": float(bias2_state.mean() / (E ** 2).mean()),
            "bias_frac_task": float(bias2_task.mean() / (E ** 2).mean()),
            "bias_frac_late10": float(bias2_state[:, -10:].mean() / (E[:, :, -10:] ** 2).mean()),
            "bias_frac_early10": float(bias2_state[:, :10].mean() / (E[:, :, :10] ** 2).mean()),
        }
        return out, m  # m = the bias field (uncorrected mean, used for cosines)

    results = []
    with open(out_path, "a") as fh:
        for t_str, st in ckpts:
            t = int(t_str.lstrip("t"))
            name = task_index_to_name[t]
            batches = make_batches(t)
            # reference bias field from the PRE-sequential checkpoint (A phase / joint pretrain)
            biasA = None
            if a_ckpt:
                load_slots(unwrapped, a_ckpt)
                fieldsA = []
                for b in batches:
                    _, m = decompose(signed_errors(b, t, name))
                    fieldsA.append(m)
                biasA = torch.cat(fieldsA, dim=0)
            # arm state
            load_slots(unwrapped, os.path.join(run_dir, "checkpoints", st, "pretrained_model"))
            stats_acc, fields = [], []
            for b in batches:
                s, m = decompose(signed_errors(b, t, name))
                stats_acc.append(s)
                fields.append(m)
            bias_arm = torch.cat(fields, dim=0)
            rec = {"run": os.path.basename(run_dir), "task": t, "ckpt": st,
                   **{k: sum(s[k] for s in stats_acc) / len(stats_acc) for k in stats_acc[0]}}
            if biasA is not None:
                va, vb = bias_arm.flatten(), biasA.flatten()
                rec["cos_pull_A"] = float((va @ vb) / (va.norm() * vb.norm()))
                rec["bias_norm_ratio_vs_A"] = float(va.norm() / vb.norm())
            results.append(rec)
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"[bias] t{t} {st}: mse={rec['mse']:.4f} bias_state={rec['bias_frac_state']:.3f} "
                  f"bias_task={rec['bias_frac_task']:.3f} late10={rec['bias_frac_late10']:.3f} "
                  f"early10={rec['bias_frac_early10']:.3f} "
                  + (f"cosA={rec.get('cos_pull_A'):.3f} ratioA={rec.get('bias_norm_ratio_vs_A'):.3f}"
                     if biasA is not None else ""), flush=True)


if __name__ == "__main__":
    main()
