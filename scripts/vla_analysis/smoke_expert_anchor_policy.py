#!/usr/bin/env python3
"""Policy-level smokes for E52 expert-anchor routing, on the real compact A-checkpoint
(bf16, GPU). Run with the standard probe CLI + --policy.memory_layer.expert_anchor_pool=text.

P1  attach: all expert wrappers anchored, anchor_proj (expert_dim x lm_dim), capture
    hooks registered on the paired LM layers' mlps; VLM wrappers unaffected
P2  training-style forward: every expert wrapper holds a fresh (B, lm_dim) anchor with
    all-valid rows; loss finite
P3  freeze modes: checkpoint mode (train_memory_only + freeze_memory_router) leaves
    anchor_proj FROZEN; with router params re-enabled, backward reaches anchor_proj
P4  inference: predict_action_chunk runs clean (dual-pass stash discipline holds with
    the anchor active); anchors persist through the denoise
P5  stationarity: bumping expert + VLM slot values leaves the captured anchors bitwise
    unchanged and the expert retrieval indices unchanged (frozen-route + anchor)
"""
import os
import sys

import torch

from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig, _build_dataloader_for_task, _collect_task_index_to_name,
)

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)
    if not cond:
        FAILS.append(name)


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    cfg.validate()
    device = torch.device("cuda")
    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
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
    policy = policy.to(device).eval()

    pwe = policy.model.paligemma_with_expert
    exp_idx = pwe._mem_layer_indices
    lm_dim = pwe.paligemma.config.text_config.hidden_size
    ex_dim = pwe.gemma_expert.config.hidden_size
    wrappers = [pwe.gemma_expert.model.layers[j].mlp for j in exp_idx]

    # P1 attach
    check("P1a all expert wrappers anchored", all(w.expert_anchor == "text" for w in wrappers),
          f"layers {exp_idx}")
    check("P1b anchor_proj shapes", all(
        w.anchor_proj is not None and tuple(w.anchor_proj.weight.shape) == (ex_dim, lm_dim)
        for w in wrappers))
    check("P1c capture hooks on paired LM mlps", all(
        len(pwe.paligemma.model.language_model.layers[j].mlp._forward_pre_hooks) > 0
        for j in exp_idx))
    vlm_wr = [pwe.paligemma.model.language_model.layers[j].mlp
              for j in getattr(pwe, "_vlm_mem_layer_indices", [])]
    check("P1d VLM wrappers un-anchored", all(w.expert_anchor == "" for w in vlm_wr))

    # a real batch
    tin = _collect_task_index_to_name(dataset)
    dl = _build_dataloader_for_task(dataset, tin, 0, batch_size=4, num_workers=2,
                                    device_type=device.type)
    b = next(iter(dl))
    for ck in dataset.meta.camera_keys:
        if ck in b and b[ck].dtype == torch.uint8:
            b[ck] = b[ck].to(torch.float32) / 255.0
    b = preprocessor(b)
    tids = torch.zeros(4, dtype=torch.long, device=device)

    # P2 training-style forward
    with torch.no_grad():
        out = policy.forward(b, task_emb=None, task_ids=tids)
    loss = out[0] if isinstance(out, tuple) else out
    anc_ok = all(
        w._ctx_anchor is not None and w._ctx_anchor.shape == (4, lm_dim) for w in wrappers)
    val_ok = all(w._ctx_anchor_valid is not None and bool(w._ctx_anchor_valid.all())
                 for w in wrappers)
    check("P2a anchors captured (B, lm_dim) on every expert wrapper", anc_ok)
    check("P2b all rows valid (instruction span found)", val_ok)
    lv = float(loss.mean()) if torch.is_tensor(loss) else float("nan")
    check("P2c loss finite", lv == lv, f"loss {lv:.4f}")

    # P3 freeze modes
    frozen_now = all(not p.requires_grad for w in wrappers for p in w.anchor_proj.parameters())
    check("P3a anchor_proj FROZEN under checkpoint mode (freeze_memory_router)", frozen_now)
    for w in wrappers:
        for p in w.anchor_proj.parameters():
            p.requires_grad_(True)
    policy.train()
    out = policy.forward(b, task_emb=None, task_ids=tids)
    loss = out[0] if isinstance(out, tuple) else out
    loss.mean().backward()
    g = wrappers[0].anchor_proj.weight.grad
    check("P3b grads reach anchor_proj when router-enabled", g is not None and float(g.abs().max()) > 0,
          f"|g|max {float(g.abs().max()):.2e}" if g is not None else "no grad")
    policy.zero_grad(set_to_none=True)
    policy.eval()
    for w in wrappers:
        for p in w.anchor_proj.parameters():
            p.requires_grad_(False)

    # P4 inference dual pass
    with torch.no_grad():
        a = policy.predict_action_chunk(b)
    check("P4a predict_action_chunk runs with anchor active", a is not None and a.shape[0] == 4,
          f"chunk {tuple(a.shape)}")
    check("P4b anchors persisted through denoise", all(
        w._ctx_anchor is not None and w._ctx_anchor.shape[0] == 4 for w in wrappers))

    # P5 stationarity under value bumps — like-for-like training forwards with the
    # flow-matching noise/time draw PINNED (the suffix stream is noise-dependent by
    # design; the old probes seed before every forward), plus a no-bump determinism
    # control first.
    with torch.no_grad():
        torch.manual_seed(1234)
        _ = policy.forward(b, task_emb=None, task_ids=tids)
    anchors_1 = [w._ctx_anchor.clone() for w in wrappers]
    idx_1 = wrappers[0].mem.last_indices.clone()
    with torch.no_grad():
        torch.manual_seed(1234)
        _ = policy.forward(b, task_emb=None, task_ids=tids)
    anchors_2 = [w._ctx_anchor.clone() for w in wrappers]
    idx_2 = wrappers[0].mem.last_indices.clone()
    check("P5a determinism control (no bump): anchors + indices repeat bitwise",
          all(torch.equal(x, y) for x, y in zip(anchors_1, anchors_2))
          and torch.equal(idx_1, idx_2))
    with torch.no_grad():
        torch.manual_seed(999)
        for j in exp_idx:
            m = pwe.gemma_expert.model.layers[j].mlp.mem
            m.slot_up.add_(torch.randn_like(m.slot_up) * 0.02)
        for j in getattr(pwe, "_vlm_mem_layer_indices", []):
            m = pwe.paligemma.model.language_model.layers[j].mlp.mem
            m.slot_up.add_(torch.randn_like(m.slot_up) * 0.02)
        torch.manual_seed(1234)
        _ = policy.forward(b, task_emb=None, task_ids=tids)
    anchors_3 = [w._ctx_anchor for w in wrappers]
    idx_3 = wrappers[0].mem.last_indices
    check("P5b anchors bitwise stationary under value bumps", all(
        torch.equal(x, y) for x, y in zip(anchors_2, anchors_3)))
    check("P5c expert retrieval indices stationary under value bumps",
          torch.equal(idx_2, idx_3))

    print(("\nALL EXPERT-ANCHOR POLICY SMOKES PASS" if not FAILS else f"\nFAILURES: {FAILS}"),
          flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
