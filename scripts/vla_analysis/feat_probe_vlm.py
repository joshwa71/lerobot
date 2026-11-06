#!/usr/bin/env python3
"""VLM-side feature-separability probe (E44 placement study for VLM memory layers).

Same instrument as feat_probe.py but hooks the PALIGEMMA LANGUAGE MODEL layers' .mlp
inputs (the would-be router inputs for VLM-side memory). Captures per layer: all-prefix-
token mean (L{i}) and last-16-token mean (~language span, T{i}).

Streams per-task batches through a FROZEN checkpoint and captures the router-input
hidden states (the input to each expert layer's .mlp — identical quantity whether the
layer is a plain GemmaMLP or an MLPPlusMemory wrapper) at a set of layers, mean-pooled
over tokens. Saves features + task labels to an .npz for probing (task-ID logistic
probe + between/within cosine separation per layer).

Run like the audit/query_forward_standalone (parser.wrap + factories):
  FEAT_LAYERS=4,8,10,12,14,15,16,17 FEAT_NB=6 FEAT_OUT=/path/feats.npz \
  python feat_probe.py --policy.path=<ckpt> --dataset.repo_id=libero_90 ...
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from accelerate import Accelerator
from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig,
    _build_dataloader_for_task,
    _collect_task_index_to_name,
)


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    cfg.validate()
    accelerator = Accelerator()
    device = accelerator.device

    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    processor_kwargs, postprocessor_kwargs = {}, {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if hasattr(policy, "precompute_task_embeddings"):
        policy.precompute_task_embeddings(dataset.meta)
    policy = accelerator.prepare(policy)
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    unwrapped.eval()
    task_index_to_name = _collect_task_index_to_name(dataset)

    layers = [int(x) for x in os.environ.get("FEAT_LAYERS", "4,8,10,12,14,15,16,17").split(",")]
    nb = int(os.environ.get("FEAT_NB", "6"))
    out_path = os.environ["FEAT_OUT"]

    expert_layers = unwrapped.model.paligemma_with_expert.paligemma.model.language_model.layers
    captured = {L: [] for L in layers}
    captured_txt = {L: [] for L in layers}
    hooks = []

    def make_hook(L):
        def hook(module, args, kwargs=None):
            x = args[0]
            if x.dim() == 3:
                xf = x.detach().float()
                captured[L].append(xf.mean(dim=1).cpu())
                # language span = last tokenizer_max_length(200) positions; instructions
                # start at its head — first 16 of the span = real text (tail is padding)
                captured_txt[L].append(xf[:, -200:-184].mean(dim=1).cpu())
        return hook

    for L in layers:
        hooks.append(expert_layers[L].mlp.register_forward_pre_hook(make_hook(L)))

    cam_keys = list(dataset.meta.camera_keys)
    tasks = sorted(task_index_to_name.keys())
    feats = {L: [] for L in layers}
    feats_txt = {L: [] for L in layers}
    labels = []
    for t in tasks:
        name = task_index_to_name.get(t, "")
        try:
            dl = _build_dataloader_for_task(
                dataset, task_index_to_name, t, batch_size=32, num_workers=4, device_type=device.type
            )
        except ValueError as e:  # task index with no usable episodes (empty vocab entry)
            print(f"[feat] task {t}: skipped ({e})", flush=True)
            continue
        it = iter(dl)
        n = 0
        for _ in range(nb):
            try:
                batch = next(it)
            except StopIteration:
                break
            for ck in cam_keys:
                if ck in batch and batch[ck].dtype == torch.uint8:
                    batch[ck] = batch[ck].to(torch.float32) / 255.0
            batch = preprocessor(batch)
            B = batch[next(iter(batch))].shape[0]
            te = unwrapped.get_task_embeddings([name] * B) if hasattr(unwrapped, "get_task_embeddings") else None
            if te is not None:
                te = te.to(device)
            for L in layers:
                captured[L].clear()
                captured_txt[L].clear()
            with torch.no_grad(), accelerator.autocast():
                unwrapped.forward(batch, task_emb=te)
            got = min(len(captured[L]) for L in layers)
            if got == 0:
                continue
            bsz = captured[layers[0]][0].shape[0]
            for L in layers:
                feats[L].append(captured[L][0].numpy().astype(np.float16))
                feats_txt[L].append(captured_txt[L][0].numpy().astype(np.float16))
            labels.append(np.full(bsz, t, dtype=np.int32))
            n += 1
        print(f"[feat] task {t} ({name[:40]}): {n} batches", flush=True)

    for h in hooks:
        h.remove()
    np.savez_compressed(
        out_path,
        labels=np.concatenate(labels),
        **{f"L{L}": np.concatenate(feats[L]) for L in layers},
        **{f"T{L}": np.concatenate(feats_txt[L]) for L in layers},
    )
    print(f"[feat] wrote {out_path}")


if __name__ == "__main__":
    main()
