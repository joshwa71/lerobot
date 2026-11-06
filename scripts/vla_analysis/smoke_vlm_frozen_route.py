#!/usr/bin/env python3
"""Smokes for VLM-side frozen-prefix routing (E45; the flag `use_frozen_base_input_features`
now governs both towers). Policy-level on a real merged checkpoint, float32, bs=2.

F1  flag ON: L16 receives a router_x; L15 routes live (router_x None); flag OFF: both None
F2  EXACTNESS: L16's router_x == the true memory-free L16 mlp input (reference = joint
    forward with the L15 wrapper's memory bypassed, L16 input recorded)
F3  STATIONARITY: bumping L15 slot values leaves L16's router_x bitwise unchanged while
    L16's live input moves
F4  PLACEMENT: the expert tower's router_x (L14, suffix fork) is bitwise unaffected by
    VLM value bumps
F5  INFERENCE dual pass: predict_action_chunk runs clean (stash discipline holds), L16
    gets a router_x there too, and it is stationary under VLM value bumps
F6  grads reach slot values on BOTH towers with the no-grad forks present
F7  gradient-checkpointing parity: same loss with the fork active, ckpt on vs off

Run (A-phase mode so values are trainable):
  python smoke_vlm_frozen_route.py --policy.path=<merged> --policy.dtype=float32 \
    --policy.memory_layer.use_frozen_base_input_features=true \
    --policy.train_router_only=false --policy.train_memory_only=true \
    --policy.freeze_memory_router=true ... (dataset/rename args as usual)
"""
import types

import torch

from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.modules.memory_lite import MLPPlusMemory
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig, _build_dataloader_for_task, _collect_task_index_to_name,
)

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
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
    lm_layers = pwe.paligemma.model.language_model.layers
    exp_layers = pwe.gemma_expert.model.layers
    vlm_idx = pwe._vlm_mem_layer_indices
    assert len(vlm_idx) >= 2, f"need >=2 VLM memory layers, got {vlm_idx}"
    L_LO, L_HI = vlm_idx[0], vlm_idx[-1]
    EXP_HI = pwe._mem_layer_indices[-1]
    w_lo, w_hi = lm_layers[L_LO].mlp, lm_layers[L_HI].mlp
    w_exp = exp_layers[EXP_HI].mlp

    records = {}

    def instrument(w, name):
        orig = w.forward

        def f(self, x, lang_emb=None, task_ids=None, router_x=None):
            rx_eff = router_x
            if rx_eff is None and not self._frozen_capture and self._frozen_stash:
                rx_eff = self._frozen_stash[0]  # peek; orig pops it
            records[name] = {
                "x": x.detach().float().cpu(),
                "rx": None if rx_eff is None else rx_eff.detach().float().cpu(),
                "vm": None if getattr(self, "_ctx_valid_mask", None) is None
                else self._ctx_valid_mask.detach().cpu(),
            }
            return orig(x, lang_emb=lang_emb, task_ids=task_ids, router_x=router_x)

        w.forward = types.MethodType(f, w)

    def span_valid(t, vm):
        # last-200 positions = the language field; compare only its valid tokens (the
        # positions memory actually routes on). Pad rows attend nothing, so their
        # attention output legitimately differs between the joint pass (uniform over
        # prefix+suffix columns) and the prefix-only fork (uniform over prefix) — inert
        # by construction, excluded here.
        f = t[:, -vm.shape[1]:]
        return f[vm.bool()]

    instrument(w_lo, "vlm_lo")
    instrument(w_hi, "vlm_hi")
    instrument(w_exp, "exp_hi")

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
        torch.manual_seed(seed)
        with torch.no_grad():
            policy.forward(b, task_emb=None, task_ids=tids)
        return {k: dict(v) for k, v in records.items()}

    mem_cfg = pwe._mem_cfg
    assert mem_cfg.use_frozen_base_input_features, "run with the flag ON"

    # F1 -------------------------------------------------------------------
    r_on = fwd()
    check("F1a flag ON: L_hi has router_x", r_on["vlm_hi"]["rx"] is not None)
    check("F1b flag ON: L_lo routes live (router_x None)", r_on["vlm_lo"]["rx"] is None)
    mem_cfg.use_frozen_base_input_features = False
    r_off = fwd()
    check("F1c flag OFF: no VLM router_x", r_off["vlm_hi"]["rx"] is None)
    vm = r_on["vlm_hi"]["vm"]
    d1 = (span_valid(r_on["vlm_hi"]["x"], vm) - span_valid(r_off["vlm_hi"]["x"], vm)).abs().max()
    check("F1d live valid-span stream identical on/off", float(d1) < 1e-4, f"max|d|={float(d1):.2e}")
    mem_cfg.use_frozen_base_input_features = True

    # F2 -------------------------------------------------------------------
    mem_cfg.use_frozen_base_input_features = False
    w_lo._frozen_capture = True  # bypass L_lo memory -> live L_hi input IS memory-free
    r_ref = fwd()
    w_lo._frozen_capture = False
    w_lo._frozen_stash = []
    mem_cfg.use_frozen_base_input_features = True
    d_raw = (r_on["vlm_hi"]["rx"] - r_ref["vlm_hi"]["x"]).abs().max()
    d = (span_valid(r_on["vlm_hi"]["rx"], vm) - span_valid(r_ref["vlm_hi"]["x"], vm)).abs().max()
    check("F2 router_x == true memory-free L_hi input (valid span)", float(d) < 1e-3,
          f"max|d|={float(d):.2e} (raw incl. inert pad rows: {float(d_raw):.2e})")

    # F3 / F4 --------------------------------------------------------------
    ups = [p for n, p in w_lo.mem.named_parameters() if n.endswith("slot_up")]
    saved = [p.detach().clone() for p in ups]
    with torch.no_grad():
        for p in ups:
            p.add_(torch.randn_like(p) * 0.05)
    r_bump = fwd()
    check("F3a router_x stationary under L_lo value bump",
          torch.equal(r_bump["vlm_hi"]["rx"], r_on["vlm_hi"]["rx"]))
    live_moved = float((span_valid(r_bump["vlm_hi"]["x"], vm) - span_valid(r_on["vlm_hi"]["x"], vm)).abs().max())
    check("F3b live L_hi input moved", live_moved > 1e-4, f"max|d|={live_moved:.2e}")
    check("F4 expert router_x untouched by VLM values",
          torch.equal(r_bump["exp_hi"]["rx"], r_on["exp_hi"]["rx"]))

    # F5 -------------------------------------------------------------------
    policy.eval()
    records.clear()
    act1 = policy.predict_action_chunk(b)
    rx_inf_1 = records["vlm_hi"]["rx"]
    check("F5a inference pass gives L_hi a router_x", rx_inf_1 is not None)
    with torch.no_grad():
        for p, s in zip(ups, saved):
            p.copy_(s + torch.randn_like(s) * 0.05)
    records.clear()
    act2 = policy.predict_action_chunk(b)
    rx_inf_2 = records["vlm_hi"]["rx"]
    check("F5b inference router_x stationary under value bump", torch.equal(rx_inf_1, rx_inf_2))
    check("F5c actions changed (values live in the value path)",
          not torch.allclose(act1, act2))
    check("F5d stash discipline clean", not w_lo._frozen_stash and not w_hi._frozen_stash)
    with torch.no_grad():
        for p, s in zip(ups, saved):
            p.copy_(s)
    policy.train()

    # F6 -------------------------------------------------------------------
    policy.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    out = policy.forward(b, task_emb=None, task_ids=tids)
    loss = out[0] if isinstance(out, tuple) else out["loss"]
    loss.mean().backward()
    g_vlm = sum(float(p.grad.abs().sum()) for n, p in w_hi.mem.named_parameters()
                if n.endswith("slot_up") and p.grad is not None)
    g_exp = sum(float(p.grad.abs().sum()) for n, p in w_exp.mem.named_parameters()
                if n.endswith("slot_up") and p.grad is not None)
    check("F6a VLM value grads nonzero", g_vlm > 0, f"|g|={g_vlm:.3e}")
    check("F6b expert value grads nonzero", g_exp > 0, f"|g|={g_exp:.3e}")

    # F7 -------------------------------------------------------------------
    policy.zero_grad(set_to_none=True)
    torch.manual_seed(11)
    l_plain = float(policy.forward(b, task_emb=None, task_ids=tids)[0].mean())
    pwe.gradient_checkpointing = True
    torch.manual_seed(11)
    l_ckpt = float(policy.forward(b, task_emb=None, task_ids=tids)[0].mean())
    pwe.gradient_checkpointing = False
    check("F7 grad-ckpt parity with fork active", abs(l_plain - l_ckpt) < 1e-5,
          f"{l_plain:.6f} vs {l_ckpt:.6f}")

    print()
    if FAILS:
        print(f"FAILED: {FAILS}")
        raise SystemExit(1)
    print("ALL VLM-FROZEN-ROUTE SMOKES PASS")


if __name__ == "__main__":
    main()
