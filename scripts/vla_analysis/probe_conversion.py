#!/usr/bin/env python3
"""Conversion probes (E41): downstream-gain + denoised-chunk, per arm per task.

Probe A (downstream gain, Josh's layer-position hypothesis): at L14/L8, replace the memory
contribution delta with {zero | itself | matched-norm random | matched-norm feature-direction |
2x} under paired noise, and measure the movement of the velocity readout (action_out_proj
output). Transmission T = gain(learned)/gain(random) tells whether the frozen downstream
layers contract the learned correction directions relative to isotropic ones.

Probe B (denoised chunk): run the real 10-step denoise on demo observations and compare the
executed chunk to the demo chunk: per-step MSE, late-chunk MSE (last 10), gripper-crossing
timing error, and across-seed spread. This evaluates the integrated field (what rollouts
execute) instead of the one-step velocity regression (what the training loss sees).

Env: PROBE_RUN_DIR, PROBE_CKPTS ("t0:005000,t2:015000,t3:020000"), PROBE_OUT, MINI=0/1.
"""
import os, sys, json, types
import torch
from accelerate import Accelerator
from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig, _collect_task_index_to_name, _build_dataloader_for_task,
)

MINI = os.environ.get("MINI", "0") == "1"
N_BATCHES = 2 if MINI else 6
BS = 8 if MINI else 12
SEEDS_B = 2 if MINI else 3
LAYERS = [14, 8]


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


def get_wrapper(unwrapped, layer):
    for name, mod in unwrapped.named_modules():
        if name.endswith(f"layers.{layer}.mlp") and hasattr(mod, "mem"):
            return mod
    raise RuntimeError(f"no memory wrapper at layer {layer}")


class MemPatch:
    """Patch one wrapper's forward to capture / replace the memory contribution."""

    def __init__(self, wrapper):
        self.w = wrapper
        self.orig_forward = wrapper.forward
        self.mode = "full"
        self.captured_delta = None
        self.captured_x = None
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(1234)

    def install(self):
        patch = self

        def fwd(self, x, lang_emb=None, task_ids=None, router_x=None):
            if self._frozen_capture:
                self._frozen_stash.append(x.detach())
                return self.mlp(x)
            if router_x is None and self._frozen_stash:
                router_x = self._frozen_stash.pop(0)
            mem_out = self.mem(x, lang_emb=lang_emb, task_ids=task_ids, router_x=router_x)
            m = patch.mode
            if m == "full":
                patch.captured_delta = mem_out.detach().float().cpu()
                patch.captured_x = x.detach().float().cpu()
                inj = mem_out
            elif m == "zero":
                inj = torch.zeros_like(mem_out)
            elif m == "x2":
                inj = 2.0 * mem_out
            elif m in ("rand", "featdir"):
                tok_norm = mem_out.detach().norm(dim=-1, keepdim=True)
                if m == "rand":
                    d = torch.randn(mem_out.shape, generator=patch.rng, dtype=torch.float32).to(mem_out)
                else:
                    B = x.shape[0]
                    perm = torch.randperm(B, generator=patch.rng).to(x.device)
                    d = (x[perm] - x).detach().to(mem_out.dtype)
                d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-8) * tok_norm
                inj = d
            else:
                raise ValueError(m)
            if self.memory_only:
                return inj
            return self.mlp(x) + inj

        self.w.forward = types.MethodType(fwd, self.w)

    def restore(self):
        self.w.forward = self.orig_forward


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    run_dir = os.environ["PROBE_RUN_DIR"]
    ckpts = [c.split(":") for c in os.environ["PROBE_CKPTS"].split(",")]  # [(t0,005000),...]
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

    # velocity readout hook
    v_box = {}
    action_out = unwrapped.model.action_out_proj
    action_out.register_forward_hook(lambda m, i, o: v_box.__setitem__("v", o.detach().float().cpu()))

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

    def fwd_v(batch, task_name, seed):
        B = batch[next(iter(batch))].shape[0]
        task_emb = None
        if hasattr(unwrapped, "get_task_embeddings"):
            task_emb = unwrapped.get_task_embeddings([task_name] * B)
            if task_emb is not None:
                task_emb = task_emb.to(device=device)
        torch.manual_seed(seed)
        with torch.no_grad(), accelerator.autocast():
            unwrapped.forward(batch, task_emb=task_emb)
        return v_box["v"]

    results = []
    with open(out_path, "a") as fh:
        for t_str, st in ckpts:
            t = int(t_str.lstrip("t"))
            load_slots(unwrapped, run_dir, st)
            name = task_index_to_name[t]
            batches = make_batches(t)

            # ---------- Probe A ----------
            for L in LAYERS:
                patch = MemPatch(get_wrapper(unwrapped, L))
                patch.install()
                agg = {k: [] for k in ["dnorm", "throw", "g_learn", "g_rand", "g_feat", "g_x2"]}
                for j, b in enumerate(batches):
                    seed = 10_000 * (t + 1) + j
                    patch.mode = "full"; v_full = fwd_v(b, name, seed)
                    dn = float(patch.captured_delta.norm())
                    patch.mode = "zero"; v_zero = fwd_v(b, name, seed)
                    patch.mode = "rand"; v_rand = fwd_v(b, name, seed)
                    patch.mode = "featdir"; v_feat = fwd_v(b, name, seed)
                    patch.mode = "x2"; v_x2 = fwd_v(b, name, seed)
                    agg["dnorm"].append(dn / patch.captured_delta.shape[0] / patch.captured_delta.shape[1])
                    agg["throw"].append(float((v_full - v_zero).norm()))
                    agg["g_learn"].append(float((v_full - v_zero).norm()) / dn)
                    agg["g_rand"].append(float((v_rand - v_zero).norm()) / dn)
                    agg["g_feat"].append(float((v_feat - v_zero).norm()) / dn)
                    agg["g_x2"].append(float((v_x2 - v_full).norm()) / dn)
                patch.restore()
                rec = {"probe": "gain", "run": os.path.basename(run_dir), "task": t, "ckpt": st,
                       "layer": L, **{k: sum(v) / len(v) for k, v in agg.items()}}
                rec["T_rand"] = rec["g_learn"] / max(rec["g_rand"], 1e-9)
                rec["T_feat"] = rec["g_learn"] / max(rec["g_feat"], 1e-9)
                results.append(rec)
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"[gainA] t{t} L{L}: g_learn={rec['g_learn']:.4f} g_rand={rec['g_rand']:.4f} "
                      f"g_feat={rec['g_feat']:.4f} T_rand={rec['T_rand']:.3f} T_feat={rec['T_feat']:.3f} "
                      f"g_x2/g_learn={rec['g_x2']/max(rec['g_learn'],1e-9):.3f} throw={rec['throw']:.2f}", flush=True)

            # ---------- Probe B ----------
            m = {k: [] for k in ["chunk_mse", "late10_mse", "grip_dt", "spread"]}
            for j, b in enumerate(batches):
                gt = b["action"][:, :, :7].float().cpu()  # normalized demo chunk
                preds = []
                for s in range(SEEDS_B):
                    torch.manual_seed(50_000 * (t + 1) + 97 * j + s)
                    with torch.no_grad(), accelerator.autocast():
                        a = unwrapped.predict_action_chunk(b)
                    preds.append(a.float().cpu())
                p0 = preds[0]
                m["chunk_mse"].append(float(((p0 - gt) ** 2).mean()))
                m["late10_mse"].append(float(((p0[:, -10:] - gt[:, -10:]) ** 2).mean()))
                if len(preds) > 1:
                    ds = [float(((preds[a] - preds[c]) ** 2).mean())
                          for a in range(len(preds)) for c in range(a + 1, len(preds))]
                    m["spread"].append(sum(ds) / len(ds))
                # gripper first-crossing timing (dim 6, sign of normalized value)
                def first_cross(x):  # x: (T,)
                    sgn = x > 0
                    ch = (sgn[1:] != sgn[:-1]).nonzero()
                    return int(ch[0]) if ch.numel() else -1
                dts = []
                for i in range(gt.shape[0]):
                    a_gt, a_p = first_cross(gt[i, :, 6]), first_cross(p0[i, :, 6])
                    if a_gt >= 0 and a_p >= 0:
                        dts.append(abs(a_gt - a_p))
                if dts:
                    m["grip_dt"].append(sum(dts) / len(dts))
            rec = {"probe": "chunk", "run": os.path.basename(run_dir), "task": t, "ckpt": st,
                   **{k: (sum(v) / len(v) if v else None) for k, v in m.items()}}
            results.append(rec)
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"[chunkB] t{t}: chunk_mse={rec['chunk_mse']:.5f} late10={rec['late10_mse']:.5f} "
                  f"grip_dt={rec['grip_dt']} spread={rec['spread']:.5f}", flush=True)


if __name__ == "__main__":
    main()
