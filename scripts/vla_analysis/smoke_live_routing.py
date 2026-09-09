#!/usr/bin/env python3
"""E68 A1 smoke: LIVE addressing (allow_nonstationary_routing) on the paper cell's interleaved
layout, WITHOUT pre-pass and WITHOUT frozen-base routing. Policy-level on the real stage-1
checkpoint (fresh attach), float32, bs=2, small banks. Run by run_smoke_live_routing.sh.

MODE L (interleaved layout, live flags + allow switch):
  L1  attach succeeds with EVERY expert and VLM site; forward runs; loss finite
  L2  no site receives a router_x (routing reads the live mlp input; no frozen stash)
  L3  NON-stationarity is present (the property A1 gives up): bumping the LOWEST VLM bank's
      values moves the routing input of every expert site ABOVE it and of every higher VLM
      site, moves the expert anchors above it, and leaves expert sites at or below it
      unchanged; loss moves
  L4  bumping the lowest EXPERT bank moves higher expert sites' routing input and leaves every
      VLM site unchanged (the prefix never attends to the suffix); loss moves
  L5  grads reach slot values on both towers
  L6  gradient-checkpointing parity
  L7  inference: predict_action_chunk deterministic under a fixed seed; no memory-free prefix KV
      is captured (no pre-pass); actions move after a value bump; stash discipline clean
The two guard cases (same layout WITHOUT the switch -> ValueError; switch combined with
frozen_prepass=true -> ValueError) are exercised by the runner as separate invocations.
"""
import types

import torch

from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig,
    _build_dataloader_for_task,
    _collect_task_index_to_name,
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
    policy = policy.to(device)
    pwe = policy.model.paligemma_with_expert
    mem_cfg = pwe._mem_cfg
    exp_idx = list(pwe._mem_layer_indices)
    vlm_idx = list(getattr(pwe, "_vlm_mem_layer_indices", []) or [])
    want_exp = [int(x) for x in cfg.policy.memory_layer.layers]
    want_vlm = [int(x) for x in cfg.policy.memory_layer.vlm_layers]
    print(f"== live-routing smoke: expert {exp_idx} / vlm {vlm_idx}", flush=True)
    check("L1a every requested expert site attached", exp_idx == want_exp, f"{exp_idx} vs {want_exp}")
    check("L1b every requested VLM site attached", vlm_idx == want_vlm, f"{vlm_idx} vs {want_vlm}")
    check("L1c flags: live (no frozen-base routing, no pre-pass, switch on)",
          not getattr(mem_cfg, "use_frozen_base_input_features", False)
          and not getattr(mem_cfg, "frozen_prepass", False)
          and getattr(mem_cfg, "allow_nonstationary_routing", False))
    interleaved = bool(vlm_idx) and bool(exp_idx) and min(vlm_idx) <= max(exp_idx)
    check("L1d layout is interleaved (the guarded case)", interleaved)

    exp_wr = {i: pwe.gemma_expert.model.layers[i].mlp for i in exp_idx}
    vlm_wr = {i: pwe.paligemma.model.language_model.layers[i].mlp for i in vlm_idx}

    torch.manual_seed(7)
    with torch.no_grad():
        for w in list(exp_wr.values()) + list(vlm_wr.values()):
            for n, p in w.mem.named_parameters():
                if n.endswith("slot_up") or n.endswith("slot_down"):
                    p.add_(torch.randn_like(p) * 0.02)

    records = {}

    def instrument(w, name):
        orig = w.forward

        def f(self, x, lang_emb=None, task_ids=None, router_x=None):
            if not getattr(self, "_frozen_capture", False):
                records[name] = {
                    "x": x.detach().float().cpu(),
                    "rx": None if router_x is None else router_x.detach().float().cpu(),
                    "anchor": None if getattr(self, "_ctx_anchor", None) is None
                    else self._ctx_anchor.detach().float().cpu(),
                    "vm": None if getattr(self, "_ctx_valid_mask", None) is None
                    else self._ctx_valid_mask.detach().cpu(),
                    "span": int(getattr(self, "text_span", 0) or 0),
                    "stash": len(getattr(self, "_frozen_stash", []) or []),
                }
            return orig(x, lang_emb=lang_emb, task_ids=task_ids, router_x=router_x)

        w.forward = types.MethodType(f, w)

    for i, w in exp_wr.items():
        instrument(w, f"exp_{i}")
    for i, w in vlm_wr.items():
        instrument(w, f"vlm_{i}")

    tin = _collect_task_index_to_name(dataset)
    dl = _build_dataloader_for_task(dataset, tin, 0, batch_size=2, num_workers=2, device_type=device.type)
    b = next(iter(dl))
    for ck in dataset.meta.camera_keys:
        if ck in b and b[ck].dtype == torch.uint8:
            b[ck] = b[ck].to(torch.float32) / 255.0
    b = preprocessor(b)
    tids = torch.zeros(2, dtype=torch.long, device=device)
    policy.train()

    def fwd(seed=0):
        records.clear()
        torch.manual_seed(seed)
        with torch.no_grad():
            out = policy.forward(b, task_emb=None, task_ids=tids)
        loss = float(out[0].mean()) if isinstance(out, tuple) else float(out["loss"].mean())
        return {k: dict(v) for k, v in records.items()}, loss

    def routed_view(rec, name):
        t = rec[name]["x"] if rec[name]["rx"] is None else rec[name]["rx"]
        vm, span = rec[name]["vm"], rec[name]["span"]
        if name.startswith("vlm_") and vm is not None and span > 0:
            f = t[:, -span:][:, : vm.shape[1]]
            return f[vm.bool()]
        return t

    all_sites = [f"exp_{i}" for i in exp_idx] + [f"vlm_{i}" for i in vlm_idx]
    r0, l0 = fwd()
    check("L1e forward runs, loss finite, every site called",
          l0 == l0 and abs(l0) < 1e6 and all(s in r0 for s in all_sites), f"loss={l0:.4f}")
    check("L2 no site receives a router_x and no frozen stash is populated",
          all(r0[s]["rx"] is None and r0[s]["stash"] == 0 for s in all_sites))

    # L3: bump the LOWEST VLM bank ---------------------------------------------------
    v_low = vlm_idx[0]
    ups_v = [p for n, p in vlm_wr[v_low].mem.named_parameters() if n.endswith("slot_up")]
    assert ups_v, "no slot_up on the low VLM bank — value_type must be lora"
    saved_v = [p.detach().clone() for p in ups_v]
    with torch.no_grad():
        for p in ups_v:
            p.add_(torch.randn_like(p) * 0.05)
    r1, l1 = fwd()
    above = [f"exp_{i}" for i in exp_idx if i > v_low] + [f"vlm_{i}" for i in vlm_idx if i > v_low]
    below = [f"exp_{i}" for i in exp_idx if i <= v_low]
    moved = [s for s in above if not torch.equal(routed_view(r1, s), routed_view(r0, s))]
    check("L3a routing input MOVED at every site above the bumped VLM bank (non-stationary)",
          len(moved) == len(above), f"moved {len(moved)}/{len(above)}")
    still = [s for s in below if torch.equal(routed_view(r1, s), routed_view(r0, s))]
    check("L3b routing input unchanged at expert sites at/below the bumped VLM bank",
          len(still) == len(below), f"unchanged {len(still)}/{len(below)} ({below})")
    anc_moved = [i for i in exp_idx if i > v_low and r0[f"exp_{i}"]["anchor"] is not None
                 and not torch.equal(r0[f"exp_{i}"]["anchor"], r1[f"exp_{i}"]["anchor"])]
    anc_total = [i for i in exp_idx if i > v_low and r0[f"exp_{i}"]["anchor"] is not None]
    check("L3c expert anchors above the bumped VLM bank MOVED (live anchors)",
          anc_total and len(anc_moved) == len(anc_total), f"{len(anc_moved)}/{len(anc_total)}")
    check("L3d loss moved", abs(l1 - l0) > 1e-7, f"|d|={abs(l1-l0):.2e}")

    # L4: bump the LOWEST expert bank -----------------------------------------------
    e_low = exp_idx[0]
    ups_e = [p for n, p in exp_wr[e_low].mem.named_parameters() if n.endswith("slot_up")]
    with torch.no_grad():
        for p in ups_e:
            p.add_(torch.randn_like(p) * 0.05)
    r2, l2 = fwd()
    exp_above = [f"exp_{i}" for i in exp_idx if i > e_low]
    moved_e = [s for s in exp_above if not torch.equal(routed_view(r2, s), routed_view(r1, s))]
    check("L4a higher expert sites' routing input moved under an expert value bump",
          len(moved_e) == len(exp_above), f"{len(moved_e)}/{len(exp_above)}")
    vlm_same = [s for s in all_sites if s.startswith("vlm_")
                and torch.equal(routed_view(r2, s), routed_view(r1, s))]
    n_vlm = sum(1 for s in all_sites if s.startswith("vlm_"))
    check("L4b every VLM site unchanged (prefix never attends to the suffix)",
          len(vlm_same) == n_vlm, f"{len(vlm_same)}/{n_vlm}")
    check("L4c loss moved again", abs(l2 - l1) > 1e-7)

    # L5: grads ------------------------------------------------------------------------
    for p in [q for w in list(exp_wr.values()) + list(vlm_wr.values())
              for n, q in w.mem.named_parameters() if n.endswith("slot_up")]:
        p.requires_grad_(True)
    policy.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    out = policy.forward(b, task_emb=None, task_ids=tids)
    loss = out[0].mean() if isinstance(out, tuple) else out["loss"].mean()
    loss.backward()
    g_e = sum(float(p.grad.abs().sum()) for w in exp_wr.values()
              for n, p in w.mem.named_parameters() if n.endswith("slot_up") and p.grad is not None)
    g_v = sum(float(p.grad.abs().sum()) for w in vlm_wr.values()
              for n, p in w.mem.named_parameters() if n.endswith("slot_up") and p.grad is not None)
    check("L5a expert value grads nonzero", g_e > 0, f"|g|={g_e:.3e}")
    check("L5b VLM value grads nonzero", g_v > 0, f"|g|={g_v:.3e}")

    # L6: grad-ckpt parity ---------------------------------------------------------------
    torch.manual_seed(11)
    lp = float(policy.forward(b, task_emb=None, task_ids=tids)[0].mean())
    pwe.gradient_checkpointing = True
    torch.manual_seed(11)
    lc = float(policy.forward(b, task_emb=None, task_ids=tids)[0].mean())
    pwe.gradient_checkpointing = False
    check("L6 grad-ckpt parity under live routing", abs(lp - lc) < 1e-5, f"{lp:.6f} vs {lc:.6f}")

    # L7: inference -----------------------------------------------------------------------
    policy.eval()
    torch.manual_seed(123)
    records.clear()
    act1 = policy.predict_action_chunk(b)
    check("L7a inference: every site called, none with router_x",
          all(s in records and records[s]["rx"] is None for s in all_sites))
    check("L7b no memory-free prefix KV captured (no pre-pass)",
          getattr(pwe, "_frozen_prefix_kv", None) is None)
    torch.manual_seed(123)
    act1b = policy.predict_action_chunk(b)
    check("L7c inference deterministic (bitwise)", torch.equal(act1, act1b))
    with torch.no_grad():
        for p, s in zip(ups_v, saved_v):
            p.copy_(s + torch.randn_like(s) * 0.08)
    torch.manual_seed(123)
    act2 = policy.predict_action_chunk(b)
    check("L7d actions moved after a value bump", not torch.equal(act1, act2))
    clean = all(not (getattr(w, "_frozen_stash", None) or [])
                for w in list(exp_wr.values()) + list(vlm_wr.values()))
    check("L7e stash discipline clean", clean)

    print(flush=True)
    if FAILS:
        print(f"FAILED: {FAILS}")
        raise SystemExit(1)
    print("ALL LIVE-ROUTING SMOKES PASS")


if __name__ == "__main__":
    main()
