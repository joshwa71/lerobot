#!/usr/bin/env python3
"""E44 cheap-tier leverage attribution: how much of the VLM-only LoRA's win flows through
the language-token positions vs the image-token positions?

Runs the SAME fixed batches through (phase A) the adapted policy and (phase B) the base
policy, hooking each LM layer's .mlp input h (the prefix stream). Reports per layer, per
span (img = all image tokens; txt = first 16 language positions; pad = last 50 positions,
a should-be-~zero control):
  - disp:   ||h_A - h_B|| / ||h_B||  (where the adapter actually moved the stream)
  - gshare: span share of sum-over-positions ||dL/dh||  (which positions the action loss
            reads, through BOTH the deeper LM layers and the per-layer prefix-KV -> expert
            path; computed on the adapted policy)
  - taylor: sum over span of g . (h_A - h_B)  (first-order attribution of the adapter's
            loss effect to that span's stream change; per-layer views, NOT additive across
            layers)
Interpretive rule (agreed with Josh): txt-attribution ~0 kills text-only VLM memory;
moderate -> bake-off; high -> text-only is the confident build. Attribution measures the
solution LoRA FOUND, not the space of solutions.

Env: SPAN_BASE (base ckpt dir), SPAN_TASK (dataset task index), SPAN_LAYERS, SPAN_NB, MINI.
CLI --policy.path must point at the ADAPTED (LoRA) checkpoint.
"""
import os
import numpy as np
import torch
from accelerate import Accelerator
from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig, _build_dataloader_for_task, _collect_task_index_to_name,
)

MINI = os.environ.get("MINI", "0") == "1"
NB = 2 if MINI else int(os.environ.get("SPAN_NB", "4"))
BS = 8
LAYERS = [int(x) for x in os.environ.get("SPAN_LAYERS", "8,10,12,13,14,15,16,17").split(",")]
LANG_FIELD = 200  # tokenizer_max_length; text span = first 16 of it; pad control = last 50


def spans(seq_len):
    lang0 = seq_len - LANG_FIELD
    return {"img": (0, lang0), "txt": (lang0, lang0 + 32), "rest": (lang0 + 32, seq_len - 50), "pad": (seq_len - 50, seq_len)}


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    base_ckpt = os.environ["SPAN_BASE"]
    t = int(os.environ.get("SPAN_TASK", "0"))
    cfg.validate()
    accelerator = Accelerator()
    device = accelerator.device
    dataset = make_dataset(cfg)
    task_index_to_name = _collect_task_index_to_name(dataset)
    name = task_index_to_name[t]
    cam_keys = list(dataset.meta.camera_keys)

    def build(policy_path, peft=True):
        import copy
        pcfg = copy.deepcopy(cfg.policy)
        pcfg.pretrained_path = policy_path
        if not peft:
            pcfg.use_peft = False
        policy = make_policy(cfg=pcfg, ds_meta=dataset.meta, rename_map=cfg.rename_map)
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=pcfg, pretrained_path=policy_path,
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
        policy = accelerator.prepare(policy)
        unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
        unwrapped.eval()
        return unwrapped, preprocessor

    def raw_batches():
        dl = _build_dataloader_for_task(dataset, task_index_to_name, t, batch_size=BS,
                                        num_workers=2, device_type=device.type, drop_n_last_frames=0)
        torch.manual_seed(7919 * (t + 1))  # identical batches across phases (jitter-probe trick)
        out = []
        it = iter(dl)
        for _ in range(NB):
            b = next(it)
            for ck in cam_keys:
                if ck in b and b[ck].dtype == torch.uint8:
                    b[ck] = b[ck].to(torch.float32) / 255.0
            out.append(b)
        return out

    def lm_mlp(unwrapped, L):
        # robust to PEFT wrapping (base: model.paligemma...; lora: model.base_model.model.paligemma...)
        for n, m in unwrapped.named_modules():
            if n.endswith(f"language_model.layers.{L}.mlp"):
                return m
        raise RuntimeError(f"no LM mlp at layer {L}")

    def run_phase(unwrapped, preprocessor, grads: bool):
        """Return hids[L][j] (B,seq,2048 fp16 cpu) and, if grads, gnorm[L][j] (B,seq) + g[L][j]."""
        hids = {L: [] for L in LAYERS}
        gbuf = {L: [] for L in LAYERS}
        hooks = []

        def mk(L):
            def hook(module, args):
                x = args[0]
                if x.dim() != 3 or x.shape[1] < LANG_FIELD + 8:
                    return
                hids[L].append(x.detach().float().cpu())
                if grads and x.requires_grad:
                    x.register_hook(lambda g, L=L: gbuf[L].append(g.detach().float().cpu()))
            return hook

        if grads:
            # eval-path loaders freeze all params -> no autograd graph; re-enable so
            # ACTIVATION grads exist (param grads are discarded via zero_grad)
            unwrapped.requires_grad_(True)
        for L in LAYERS:
            hooks.append(lm_mlp(unwrapped, L).register_forward_pre_hook(mk(L)))
        for j, rb in enumerate(raw_batches()):
            b = preprocessor({k: (v.clone() if torch.is_tensor(v) else v) for k, v in rb.items()})
            B = b[next(iter(b))].shape[0]
            te = unwrapped.get_task_embeddings([name] * B) if hasattr(unwrapped, "get_task_embeddings") else None
            if te is not None:
                te = te.to(device)
            torch.manual_seed(50_000 * (t + 1) + 97 * j)  # fixed flow-matching noise across phases
            if grads:
                with accelerator.autocast():
                    out = unwrapped.forward(b, task_emb=te)
                loss = out["loss"] if isinstance(out, dict) else (out[0] if isinstance(out, tuple) else out)
                loss = loss.mean()
                unwrapped.zero_grad(set_to_none=True)
                loss.backward()
            else:
                with torch.no_grad(), accelerator.autocast():
                    unwrapped.forward(b, task_emb=te)
        for h in hooks:
            h.remove()
        return hids, gbuf

    print(f"[span] task {t} ({name[:50]}), layers {LAYERS}, NB={NB} BS={BS}", flush=True)
    adapted, pre_a = build(cfg.policy.pretrained_path)
    hA, gA = run_phase(adapted, pre_a, grads=True)
    del adapted
    torch.cuda.empty_cache()
    base, pre_b = build(base_ckpt, peft=False)
    hB, _ = run_phase(base, pre_b, grads=False)
    del base
    torch.cuda.empty_cache()

    print(f"\n{'L':>3} {'span':>4} {'disp':>7} {'gshare%':>8} {'taylor':>10}")
    for L in LAYERS:
        nb = min(len(hA[L]), len(hB[L]), len(gA[L]) if gA[L] else 0)
        if nb == 0:
            print(f"{L:>3}  (no captures)"); continue
        seq = hA[L][0].shape[1]
        sp = spans(seq)
        gA_L = gA[L]  # one entry per backward per layer -> already in batch order
        tot_g = sum(float(g.norm(dim=-1).sum()) for g in gA_L[:nb])
        for sname, (a, b_) in sp.items():
            disp_n = disp_d = gsum = tay = 0.0
            for j in range(nb):
                dh = (hA[L][j][:, a:b_] - hB[L][j][:, a:b_])
                disp_n += float(dh.norm())
                disp_d += float(hB[L][j][:, a:b_].norm())
                g = gA_L[j][:, a:b_]
                gsum += float(g.norm(dim=-1).sum())
                tay += float((g * dh).sum())
            print(f"{L:>3} {sname:>4} {disp_n/max(disp_d,1e-9):7.4f} {100*gsum/max(tot_g,1e-12):8.2f} {tay:10.4f}", flush=True)
        if L == LAYERS[0] or L == 14:
            prof = sum(g[:, sp["txt"][0]:sp["txt"][0]+36].norm(dim=-1).sum(0) for g in gA_L[:nb])
            prof = 100 * prof / max(tot_g, 1e-12)
            print(f"      lang-field per-position gshare% (pos 0-35): " +
                  " ".join(f"{v:.1f}" for v in prof.tolist()), flush=True)


if __name__ == "__main__":
    main()
