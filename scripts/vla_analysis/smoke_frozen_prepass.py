#!/usr/bin/env python3
"""E59 smokes: frozen_prepass — the full memory-free pre-pass that lifts the VLM
placement guard (interleaved expert/VLM memory placement).

Policy-level on the real stage-1 checkpoint (fresh attach), float32, bs=2, run twice
by run_smoke_frozen_prepass.sh:

MODE A (guard-LEGAL layout, prepass OFF at attach; equivalence by in-process flip):
  L1  fork baseline runs; sites record router_x
  L2  EQUIVALENCE: flipping frozen_prepass=true reproduces every site's routing
      features (fork-None sites compared against the fork run's live x, which is
      memory-free there by placement) — cross-implementation, tolerance-based
  L3  anchors equal across implementations
  L4  loss parity across implementations
  L5  timing ratio prepass/fork (reported, no assert)

MODE B (INTERLEAVED layout — min(vlm) <= max(expert), prepass ON at attach):
  I1  attach + forward + finite loss; EVERY memory site receives a router_x
  I2  stationarity: bump lowest-VLM-bank values -> all router_x AND anchors bitwise
      unchanged, loss changed;  I2b same for an expert bank's values
  I3  grads reach slot values on both towers through the no-grad pre-pass
  I4  gradient-checkpointing parity (router_x threads through recompute as args)
  I5  inference: predict_action_chunk deterministic under fixed seed; expert +
      VLM sites get router_x; _frozen_prefix_kv set; value bumps leave inference
      router_x bitwise unchanged while actions move; stash discipline clean

The guard-raise case (interleaved WITHOUT frozen_prepass -> ValueError) is exercised
by the runner as a third invocation that must fail with the guard message.
"""
import os
import time
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
    interleaved = bool(vlm_idx) and min(vlm_idx) <= max(exp_idx)
    mode = "B/interleaved" if getattr(mem_cfg, "frozen_prepass", False) else "A/legal"
    print(f"== mode {mode}: expert {exp_idx} / vlm {vlm_idx} (interleaved={interleaved})", flush=True)

    exp_wr = {i: pwe.gemma_expert.model.layers[i].mlp for i in exp_idx}
    vlm_wr = {i: pwe.paligemma.model.language_model.layers[i].mlp for i in vlm_idx}

    # Make memory ACTIVE (fresh attach zero-inits slot_up -> memory output 0): small
    # random values on every bank, identical across all forwards until bumped.
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
            rx_eff = router_x
            if rx_eff is None and not self._frozen_capture and self._frozen_stash:
                rx_eff = self._frozen_stash[0]  # peek; orig pops it
            if not self._frozen_capture:  # live pass only (skip pre-pass bypass calls)
                records[name] = {
                    "x": x.detach().float().cpu(),
                    "rx": None if rx_eff is None else rx_eff.detach().float().cpu(),
                    "anchor": None if getattr(self, "_ctx_anchor", None) is None
                    else self._ctx_anchor.detach().float().cpu(),
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

    def rx_of(rec, name):
        r = rec[name]["rx"]
        return rec[name]["x"] if r is None else r

    all_sites = [f"exp_{i}" for i in exp_idx] + [f"vlm_{i}" for i in vlm_idx]

    if mode == "A/legal":
        assert not interleaved, "MODE A wants a guard-legal layout"
        # L1: fork baseline ------------------------------------------------
        r_fork, l_fork = fwd()
        check("L1 fork baseline forward runs", all(s in r_fork for s in all_sites))
        # L2/L3/L4: flip to prepass in-process ------------------------------
        mem_cfg.frozen_prepass = True
        r_pre, l_pre = fwd()
        for s in all_sites:
            a, c = rx_of(r_fork, s), rx_of(r_pre, s)
            d = float((a - c).abs().max())
            check(f"L2 routing features equivalent @ {s}", d < 2e-3, f"max|d|={d:.2e}")
        anc_f = r_fork[f"exp_{exp_idx[-1]}"]["anchor"]
        anc_p = r_pre[f"exp_{exp_idx[-1]}"]["anchor"]
        if anc_f is not None or anc_p is not None:
            da = float((anc_f - anc_p).abs().max())
            check("L3 anchors equivalent across implementations", da < 2e-3, f"max|d|={da:.2e}")
        check("L4 loss parity", abs(l_fork - l_pre) < max(1e-4, 1e-3 * abs(l_fork)),
              f"{l_fork:.6f} vs {l_pre:.6f}")
        # L5: timing --------------------------------------------------------
        def clock(n=5):
            torch.cuda.synchronize(); t0 = time.monotonic()
            for _ in range(n):
                with torch.no_grad():
                    policy.forward(b, task_emb=None, task_ids=tids)
            torch.cuda.synchronize(); return (time.monotonic() - t0) / n
        t_pre = clock()
        mem_cfg.frozen_prepass = False
        t_fork = clock()
        mem_cfg.frozen_prepass = True
        print(f"[info] L5 fwd time fork={t_fork*1e3:.0f}ms prepass={t_pre*1e3:.0f}ms "
              f"ratio={t_pre/max(t_fork,1e-9):.2f}x", flush=True)
    else:
        assert interleaved and getattr(mem_cfg, "frozen_prepass", False), "MODE B wants interleaved+prepass"
        # I1 ---------------------------------------------------------------
        r0, l0 = fwd()
        check("I1a forward runs, loss finite", l0 == l0 and abs(l0) < 1e6, f"loss={l0:.4f}")
        check("I1b every memory site has router_x",
              all(r0[s]["rx"] is not None for s in all_sites))
        # I2: bump lowest VLM bank -----------------------------------------
        w_low = vlm_wr[vlm_idx[0]]
        ups_v = [p for n, p in w_low.mem.named_parameters() if n.endswith("slot_up")]
        saved_v = [p.detach().clone() for p in ups_v]
        with torch.no_grad():
            for p in ups_v:
                p.add_(torch.randn_like(p) * 0.05)
        r1, l1 = fwd()
        ok_rx = all(torch.equal(rx_of(r1, s), rx_of(r0, s)) for s in all_sites)
        check("I2a ALL router_x bitwise stationary under low-VLM value bump", ok_rx)
        anc0 = r0[f"exp_{exp_idx[-1]}"]["anchor"]
        anc1 = r1[f"exp_{exp_idx[-1]}"]["anchor"]
        if anc0 is not None:
            check("I2b anchors bitwise stationary", torch.equal(anc0, anc1))
        check("I2c loss moved (value path live)", abs(l1 - l0) > 1e-7, f"|d|={abs(l1-l0):.2e}")
        # I2d: bump an expert bank -----------------------------------------
        w_exp0 = exp_wr[exp_idx[0]]
        ups_e = [p for n, p in w_exp0.mem.named_parameters() if n.endswith("slot_up")]
        with torch.no_grad():
            for p in ups_e:
                p.add_(torch.randn_like(p) * 0.05)
        r2, l2 = fwd()
        check("I2d router_x stationary under expert value bump",
              all(torch.equal(rx_of(r2, s), rx_of(r1, s)) for s in all_sites))
        check("I2e loss moved again", abs(l2 - l1) > 1e-7)
        # I3: grads ---------------------------------------------------------
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
        check("I3a expert value grads nonzero through pre-pass", g_e > 0, f"|g|={g_e:.3e}")
        check("I3b VLM value grads nonzero through pre-pass", g_v > 0, f"|g|={g_v:.3e}")
        # I4: ckpt parity (both grad-enabled so checkpointing actually engages) ----
        torch.manual_seed(11)
        lp = float(policy.forward(b, task_emb=None, task_ids=tids)[0].mean())
        pwe.gradient_checkpointing = True
        torch.manual_seed(11)
        lc = float(policy.forward(b, task_emb=None, task_ids=tids)[0].mean())
        pwe.gradient_checkpointing = False
        check("I4 grad-ckpt parity under prepass", abs(lp - lc) < 1e-5, f"{lp:.6f} vs {lc:.6f}")
        # I5: inference -----------------------------------------------------
        policy.eval()
        torch.manual_seed(123)
        records.clear()
        act1 = policy.predict_action_chunk(b)
        rx_inf = {s: rx_of({s: records[s]}, s) for s in all_sites if s in records}
        check("I5a inference: expert+VLM sites all routed with router_x",
              all(records[s]["rx"] is not None for s in all_sites if s in records)
              and len(rx_inf) == len(all_sites))
        check("I5b memory-free prefix KV captured",
              getattr(pwe, "_frozen_prefix_kv", None) is not None)
        torch.manual_seed(123)
        records.clear()
        act1b = policy.predict_action_chunk(b)
        check("I5c inference deterministic (bitwise)", torch.equal(act1, act1b))
        with torch.no_grad():
            for p, s in zip(ups_v, saved_v):
                p.copy_(s + torch.randn_like(s) * 0.08)
        torch.manual_seed(123)
        records.clear()
        act2 = policy.predict_action_chunk(b)
        ok_inf = all(torch.equal(rx_of({s: records[s]}, s), rx_inf[s]) for s in rx_inf)
        check("I5d inference router_x bitwise stationary under value bump", ok_inf)
        check("I5e actions moved (values live)", not torch.equal(act1, act2))
        clean = all(not w._frozen_stash for w in list(exp_wr.values()) + list(vlm_wr.values()))
        check("I5f stash discipline clean", clean)

    print(flush=True)
    if FAILS:
        print(f"FAILED: {FAILS}")
        raise SystemExit(1)
    print(f"ALL FROZEN-PREPASS SMOKES PASS ({mode})")


if __name__ == "__main__":
    main()
