"""Smoke test for the SmolVLA memory port (branch smolvla-memory).

Checks that the pi05 memory configuration surface works on SmolVLA, against smolvla_base weights
and a synthetic batch (real tokenizer, so the instruction/newline layout is the real one):

  1. no-op: with zero memory output (slot_up = 0 at init; value_proj bias zeroed here, it is
     non-zero at init by design) a memory model gives the same loss as the plain model
  2. frozen routing: every wrapper's router input equals the plain model's MLP input at that layer
  3. stationarity: perturbing the values changes the output but not a single retrieved slot
  4. span: VLM memory touches only instruction + state positions (not images or pads)
  5. inference == training: one denoise step through the KV-cache path (pre-pass, anchors locked,
     memory-free prefix KV) matches the training path's velocity
  6. gradient checkpointing: same loss and grads
  7. freeze modes: router-only / memory-only / memory-only+frozen-router trainable sets + optimizer groups
  8. routing losses reach both towers; VLM banks stay in training mode under train_expert_only
  9. save / load round trip (memory attached before load, anchors re-hooked)
 10. config matrix: other pi05 configurations build, train (forward + backward) and run inference

  python scripts/vla_analysis/smoke_smolvla_memory.py [--device cuda]
"""

import argparse
import gc
import copy
import tempfile

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.modules.memory_config import MemoryLayerConfig
from lerobot.policies.modules.memory_lite import MLPPlusMemory, iter_memory_layers
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

BASE = "lerobot/smolvla_base"
TASKS = [
    "put both the alphabet soup and the tomato sauce in the basket",
    "pick up the black bowl.",
    "open the drawer",
    "turn on the stove and put the moka pot on it",
]
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def mem_cfg(**kw):
    base = dict(
        enabled=True,
        layers=[4, 6, 8, 12],
        share_groups=[[4, 6]],
        vlm_layers=[5, 7, 9],
        vlm_share_groups=[[5, 7]],
        mem_n_keys=32,
        vlm_mem_n_keys=32,
        mem_knn=8,
        vlm_mem_knn=8,
        routing_loss_topk=8,
        mem_heads=4,
        value_type="lora",
        lora_rank=2,
        vlm_lora_rank=2,
        vlm_text_span=200,
        vlm_router_pool="anchored",
        vlm_router_pool_weights=[1.0, 0.5],
        vlm_route_once=True,
        expert_anchor_pool="text",
        expert_anchor_weight=0.4,
        expert_anchor_src_dim=960,
        use_frozen_base_input_features=True,
        frozen_prepass=True,
    )
    base.update(kw)
    return MemoryLayerConfig(**base)


def make_policy(device, memory=None, **cfg_kw):
    cfg = PreTrainedConfig.from_pretrained(BASE)
    cfg.load_vlm_weights = False  # smolvla_base holds the whole model
    cfg.pretrained_path = BASE  # as lerobot-train sets it (memory attaches after the base load)
    cfg.device = device
    if memory is not None:
        cfg.memory_layers = True
        cfg.memory_layer = memory
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    policy = SmolVLAPolicy.from_pretrained(BASE, config=cfg)
    return policy.to(device)


def make_batch(policy, device, bs=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    tok = policy.model.vlm_with_expert.processor.tokenizer
    texts = [TASKS[i % len(TASKS)] + "\n" for i in range(bs)]
    enc = tok(texts, padding="max_length", max_length=policy.config.tokenizer_max_length,
              padding_side="right", return_tensors="pt")
    batch = {
        OBS_STATE: torch.randn(bs, 6, generator=g),
        ACTION: torch.randn(bs, policy.config.chunk_size, 6, generator=g),
        OBS_LANGUAGE_TOKENS: enc["input_ids"],
        OBS_LANGUAGE_ATTENTION_MASK: enc["attention_mask"].bool(),
        "task_index": torch.tensor([i % len(TASKS) for i in range(bs)]),
    }
    for k in policy.config.image_features:
        batch[k] = torch.rand(bs, 3, 256, 256, generator=g)
    return {k: v.to(device) for k, v in batch.items()}


def fixed_noise_time(policy, batch, seed=1, time=None):
    g = torch.Generator().manual_seed(seed)
    bs = batch[ACTION].shape[0]
    noise = torch.randn(bs, policy.config.chunk_size, policy.config.max_action_dim, generator=g)
    t = torch.rand(bs, generator=g) * 0.9 + 0.05 if time is None else torch.full((bs,), float(time))
    dev = batch[ACTION].device
    return noise.to(dev), t.to(dev)


def loss_of(policy, batch, seed=1):
    noise, t = fixed_noise_time(policy, batch, seed)
    loss, _ = policy.forward(dict(batch), noise=noise, time=t)
    return loss


def perturb_values(policy, seed=3, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, w in iter_memory_layers(policy):
            if getattr(w.mem, "_shared_follower", False):
                continue
            for name, p in w.mem.named_parameters():
                if "slot_up" in name:
                    p.copy_(torch.randn(p.shape, generator=g).to(p) * scale)


def mlp_inputs(policy, batch, layers):
    """MLP inputs of the plain model at {(tower, layer)} during one training forward."""
    vwe = policy.model.vlm_with_expert
    got, hooks = {}, []
    for tower, li in layers:
        mlp = (vwe.get_vlm_model().text_model.layers[li] if tower == "vlm" else vwe.lm_expert.layers[li]).mlp
        hooks.append(mlp.register_forward_pre_hook(
            lambda m, a, key=(tower, li): got.__setitem__(key, a[0].detach().clone())))
    with torch.no_grad():
        loss_of(policy, batch)
    for h in hooks:
        h.remove()
    return got


def router_inputs(policy, batch):
    """router_x each memory wrapper receives in the live training pass."""
    vwe = policy.model.vlm_with_expert
    got, originals = {}, {}
    exp_wr, vlm_wr = vwe._memory_wrappers()
    keyed = [(("expert", e), vwe.lm_expert.layers[e].mlp) for e in vwe._mem_layer_indices]
    keyed += [(("vlm", i), m) for i, m in vlm_wr]
    for key, m in keyed:
        orig = m.forward

        def fwd(x, *a, _orig=orig, _key=key, **kw):
            if not m_capture[_key]._frozen_capture:
                got[_key] = kw.get("router_x")
            return _orig(x, *a, **kw)

        originals[key] = (m, orig)
        m.forward = fwd
    m_capture = {k: m for k, m in keyed}
    with torch.no_grad():
        loss_of(policy, batch)
    for key, (m, orig) in originals.items():
        m.forward = orig
    return got


def velocity_train(policy, batch, noise):
    """v_t of the training path at time 1 (x_t = noise)."""
    out = {}
    h = policy.model.action_out_proj.register_forward_hook(lambda m, a, o: out.__setitem__("v", o.detach()))
    with torch.no_grad():
        policy.forward(dict(batch), noise=noise, time=torch.ones(noise.shape[0], device=noise.device))
    h.remove()
    return out["v"].float()


def velocity_infer(policy, batch, noise):
    """v_t of the inference path (prefix KV cache + one denoise step) at time 1."""
    p = policy
    old = p.model.config.num_steps
    p.model.config.num_steps = 1
    with torch.no_grad():
        images, img_masks = p.prepare_images(batch)
        state = p.prepare_state(batch)
        x1 = p.model.sample_actions(images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                                    batch[OBS_LANGUAGE_ATTENTION_MASK], state, noise=noise)
    p.model.config.num_steps = old
    return (noise - x1).float()  # x_0 = x_1 + dt * v, dt = -1


CONFIGS = [
    # name, memory_layer overrides, policy overrides, expected error
    ("expert only, live routing", dict(layers=[4, 8], share_groups=[], vlm_layers=[], vlm_share_groups=[],
                                       expert_anchor_pool="", use_frozen_base_input_features=False,
                                       frozen_prepass=False), {}, None),
    ("VLM only, span, frozen pre-pass", dict(layers=[], share_groups=[], vlm_layers=[5, 9], vlm_share_groups=[],
                                             expert_anchor_pool=""), {}, None),
    ("frozen routing without pre-pass (VLM above expert)",
     dict(layers=[4, 6], share_groups=[], vlm_layers=[9, 11], vlm_share_groups=[], frozen_prepass=False), {}, None),
    ("interleaved without pre-pass is refused",
     dict(layers=[4, 8], share_groups=[], vlm_layers=[5], vlm_share_groups=[], frozen_prepass=False), {}, ValueError),
    ("interleaved live addressing (allow_nonstationary_routing)",
     dict(layers=[4, 8], share_groups=[], vlm_layers=[5], vlm_share_groups=[], frozen_prepass=False,
          use_frozen_base_input_features=False, allow_nonstationary_routing=True), {}, None),
    ("VLM on all prefix positions (vlm_text_span=0)", dict(vlm_text_span=0, vlm_router_pool=""), {}, None),
    ("VLM span, per-token routing (vlm_router_pool='')", dict(vlm_router_pool=""), {}, None),
    ("VLM span, broadcast pooled key (vlm_route_once=false)", dict(vlm_route_once=False), {}, None),
    ("router warm-up fast path", dict(router_only_fast=True), dict(train_router_only=True), None),
    ("per-layer ranks", dict(layers=[4, 8], share_groups=[], layer_ranks=[2, 4]), {}, None),
    ("anchor without pre-pass needs matching src dim", dict(expert_anchor_src_dim=2048), {}, ValueError),
]


def config_matrix(dev, batch):
    for name, mkw, pkw, err in CONFIGS:
        try:
            p = make_policy(dev, memory=mem_cfg(**mkw), **pkw)
        except Exception as e:  # noqa: BLE001
            check(f"config: {name}", err is not None and isinstance(e, err), f"{type(e).__name__}: {e}"[:160])
            continue
        if err is not None:
            check(f"config: {name}", False, "expected an error at build time")
            continue
        try:
            p.train()
            p.zero_grad(set_to_none=True)
            loss = loss_of(p, batch)
            loss.backward()
            n_grad = sum(q.grad is not None for q in p.parameters())
            p.eval()
            with torch.no_grad():
                noise, _ = fixed_noise_time(p, batch, seed=7)
                v = velocity_infer(p, batch, noise)
            ok = bool(torch.isfinite(loss)) and n_grad > 0 and bool(torch.isfinite(v).all())
            check(f"config: {name}", ok, f"loss {loss.item():.4f}, {n_grad} grads")
        except Exception as e:  # noqa: BLE001
            check(f"config: {name}", False, f"{type(e).__name__}: {e}"[:200])
        del p
        gc.collect()  # anchor hooks hold reference cycles: refcounting alone never frees a policy
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    plain = make_policy(dev).eval()
    batch = make_batch(plain, dev)
    mem = make_policy(dev, memory=mem_cfg()).eval()
    vwe = mem.model.vlm_with_expert
    check("attach: layers", vwe._mem_layer_indices == [4, 6, 8, 12] and vwe._vlm_mem_layer_indices == [5, 7, 9],
          f"expert {vwe._mem_layer_indices} vlm {vwe._vlm_mem_layer_indices}")
    w4, w6 = vwe.lm_expert.layers[4].mlp.mem, vwe.lm_expert.layers[6].mlp.mem
    check("attach: share group aliases one table", w4.keys.data_ptr() == w6.keys.data_ptr())
    check("attach: base MLP weights loaded under the wrapper", torch.equal(
        vwe.lm_expert.layers[4].mlp.mlp.gate_proj.weight,
        plain.model.vlm_with_expert.lm_expert.layers[4].mlp.gate_proj.weight))
    check("attach: anchor hooks", len(vwe._anchor_hooks) == 4)

    def indices(p):
        with torch.no_grad():
            loss_of(p, batch)
        return {n: w.mem.last_indices.clone() for n, w in iter_memory_layers(p)}

    # 1. no-op with zero memory output
    with torch.no_grad():
        for _, w in iter_memory_layers(mem):
            if getattr(w.mem, "value_proj", None) is not None and w.mem.value_proj.bias is not None:
                w.mem.value_proj.bias.zero_()
    l0, l1 = loss_of(plain, batch), loss_of(mem, batch)
    check("no-op with zero memory output (loss)", torch.allclose(l0, l1, rtol=0, atol=1e-6), f"{l0.item():.8f} vs {l1.item():.8f}")
    idx_init = indices(mem)

    # 2. frozen routing inputs == plain model's MLP inputs (values perturbed so live != frozen)
    perturb_values(mem)
    l2 = loss_of(mem, batch)
    check("values perturbed change the loss", abs(l2.item() - l0.item()) > 1e-4, f"{l0.item():.6f} -> {l2.item():.6f}")
    keys = [("expert", e) for e in vwe._mem_layer_indices] + [("vlm", i) for i in vwe._vlm_mem_layer_indices]
    ref = mlp_inputs(plain, batch, keys)
    rx = router_inputs(mem, batch)
    worst = max((rx[k].float() - ref[k].float()).abs().max().item() for k in keys)
    check("frozen routing: router_x == memory-free MLP input at every wrapper",
          all(rx.get(k) is not None for k in keys) and worst < 1e-5, f"max abs diff {worst:.2e}")

    # 3. stationarity: retrieved slots identical with values at init vs perturbed
    ia, ib = idx_init, indices(mem)
    same = all(torch.equal(ia[n], ib[n]) for n in ia)
    check("stationarity: same slots retrieved after changing values", same and len(ia) == 7, f"{len(ia)} wrappers")

    # 4. span: VLM memory output differs from the base MLP only at instruction + state positions
    wv = vwe.get_vlm_model().text_model.layers[7].mlp
    cap = {}
    h = wv.register_forward_hook(lambda m, a, o: cap.__setitem__("io", (a[0].detach().clone(), o.detach().clone())))
    with torch.no_grad():
        loss_of(mem, batch)
    h.remove()
    x, y = cap["io"]
    delta = (y.float() - wv.mlp(x).float()).abs().amax(-1)  # (B, S)
    S = x.shape[1]
    lang_start, n_state = mem.model._prefix_layout
    L = batch[OBS_LANGUAGE_TOKENS].shape[1]
    nv = batch[OBS_LANGUAGE_ATTENTION_MASK].sum(1)
    expect = torch.zeros_like(delta, dtype=torch.bool)
    for b in range(x.shape[0]):
        expect[b, lang_start:lang_start + int(nv[b])] = True
        expect[b, lang_start + L:lang_start + L + n_state] = True
    touched = delta > 0
    check("span: memory only at instruction + state positions",
          bool((touched & ~expect).sum() == 0) and bool(touched[expect].float().mean() > 0.9),
          f"outside {int((touched & ~expect).sum())}, inside {touched[expect].float().mean():.2f}, S={S}")

    # 5. inference path == training path (one denoise step at t=1), in fp32 so the two paths of
    # the plain model agree to rounding and the memory model's agreement means something
    print("param dtypes:", sorted({str(q.dtype) for q in mem.parameters()}))
    plain.float()
    mem.float()
    noise, _ = fixed_noise_time(mem, batch, seed=5)
    vt_p, vi_p = velocity_train(plain, batch, noise), velocity_infer(plain, batch, noise)
    vt_m, vi_m = velocity_train(mem, batch, noise), velocity_infer(mem, batch, noise)
    d_plain = (vt_p - vi_p).abs().max().item()
    d_mem = (vt_m - vi_m).abs().max().item()
    check("inference == training (memory model, frozen pre-pass)", d_mem <= max(10 * d_plain, 1e-4),
          f"memory {d_mem:.2e} (plain model {d_plain:.2e}; memory effect {(vt_m - vt_p).abs().max().item():.2e})")

    # 5b. routing at inference reads memory-free features too: VLM wrappers in the prefix pass and
    # expert wrappers in the (last) denoise step see the plain model's MLP inputs
    def infer_inputs_plain():
        got, hooks = {}, []
        for tower, li in keys:
            m = (plain.model.vlm_with_expert.get_vlm_model().text_model.layers[li] if tower == "vlm"
                 else plain.model.vlm_with_expert.lm_expert.layers[li]).mlp
            hooks.append(m.register_forward_pre_hook(
                lambda mod, a, key=(tower, li): got.__setitem__(key, a[0].detach().clone())))
        velocity_infer(plain, batch, noise)
        for h in hooks:
            h.remove()
        return got

    def infer_inputs_mem():
        got, saved = {}, []
        for tower, li in keys:
            m = (vwe.get_vlm_model().text_model.layers[li] if tower == "vlm" else vwe.lm_expert.layers[li]).mlp
            orig = m.forward

            def fwd(x, *a, _m=m, _orig=orig, _key=(tower, li), **kw):
                if not _m._frozen_capture:
                    rx = kw.get("router_x")
                    if rx is None and _m._frozen_stash:
                        rx = _m._frozen_stash[0]
                    got[_key] = None if rx is None else rx.detach().clone()
                return _orig(x, *a, **kw)

            m.forward = fwd
            saved.append((m, orig))
        velocity_infer(mem, batch, noise)
        for m, orig in saved:
            m.forward = orig
        return got

    ref_i, got_i = infer_inputs_plain(), infer_inputs_mem()
    worst_i = max(((got_i[k].float() - ref_i[k].float()).abs().max().item() if got_i.get(k) is not None
                   else float("inf")) for k in keys)
    check("frozen routing at inference: router_x == memory-free MLP input", worst_i < 1e-4,
          f"max abs diff {worst_i:.2e}")

    # 6. gradient checkpointing: same loss + grads
    mem.train()
    mem.zero_grad(set_to_none=True)
    la = loss_of(mem, batch)
    la.backward()
    ga = {n: p.grad.clone() for n, p in mem.named_parameters() if p.grad is not None}
    mem.zero_grad(set_to_none=True)
    vwe.gradient_checkpointing_enable()
    lb = loss_of(mem, batch)
    lb.backward()
    gb = {n: p.grad.clone() for n, p in mem.named_parameters() if p.grad is not None}
    vwe.gradient_checkpointing_disable()
    mem.zero_grad(set_to_none=True)
    gdiff = max((ga[n] - gb[n]).abs().max().item() for n in ga) if ga.keys() == gb.keys() else float("inf")
    check("gradient checkpointing: same loss and grads", abs(la.item() - lb.item()) < 1e-6 and gdiff < 1e-5,
          f"loss {la.item():.6f}/{lb.item():.6f}, max grad diff {gdiff:.2e}, {len(ga)} grads")
    vlm_mem_grads = [n for n in ga if ".text_model." in n and ".mlp.mem." in n]
    check("grads reach VLM memory banks", len(vlm_mem_grads) > 0, f"{len(vlm_mem_grads)} tensors")

    # 8. VLM banks in training mode; routing losses on both towers
    check("VLM banks follow train() under train_expert_only",
          all(w.training for _, w in vwe._memory_wrappers()[1]) and not vwe.vlm.training)
    mem.config.memory_layer.contrastive_loss_weight = 0.05
    mem.config.memory_layer.routing_inter_task_separation_weight = 8.0
    for _, w in iter_memory_layers(mem):
        w.mem.contrastive_loss_weight = 0.05
        w.mem.routing_inter_task_separation_weight = 8.0
    _, ld = mem.forward(dict(batch))
    has_v = any(k.startswith("vlm_routing_inter_task") for k in ld)
    has_e = any(k.startswith("routing_inter_task_similarity_L") for k in ld)
    check("routing losses on both towers", has_v and has_e and "contrastive_loss_mean" in ld,
          f"vlm {has_v} expert {has_e}")
    # 8b. the VLM routing losses see EVERY valid span position (all instruction tokens and the state
    # token), not the batch's shortest span: cutting to the shortest dropped SmolVLA's state token
    # (it ends each variable-length span) in ~94% of samples until 11 Nov 26.
    # Broadcast routing (the warm-up setting) has one row per valid position: every language token
    # (newline included) + the state token. Route-once (audit / A-phase / sequential) folds the state
    # region into ONE pooled-key row: every instruction token + 1.
    nv = batch[OBS_LANGUAGE_ATTENTION_MASK].sum(1).cpu()
    n_state = mem.model._prefix_layout[1]
    vlm_wr = vwe._memory_wrappers()[1]
    for route_once in (False, True):
        seen, origs = {}, []
        for li, w in vlm_wr:
            w.route_once = route_once
            o = w.mem._compute_routing_losses
            def spy(*a, _o=o, _li=li, **kw):
                seen[_li] = kw.get("row_mask")
                return _o(*a, **kw)
            w.mem._compute_routing_losses = spy
            origs.append((w.mem, o))
        mem.forward(dict(batch))
        for m_, o in origs:
            m_._compute_routing_losses = o
        want = nv - 1 + 1 if route_once else nv + n_state  # batch texts end in "\n" (newline in the state region)
        rows = {li: (None if r is None else r.sum(1).cpu()) for li, r in seen.items()}
        check(f"VLM routing losses cover every valid row ({'route-once' if route_once else 'broadcast'})",
              len(rows) == len(vlm_wr) and all(r is not None and torch.equal(r, want) for r in rows.values())
              and len(set(want.tolist())) > 1,
              f"want {want.tolist()}, loss rows {[None if r is None else r.tolist() for r in rows.values()][:1]}")
    for _, w in vlm_wr:
        w.route_once = True
    del mem
    torch.cuda.empty_cache()

    # 7. freeze modes
    for mode, kw in [("router_only", dict(train_router_only=True)),
                     ("memory_only", dict(train_memory_only=True)),
                     ("memory_only+frozen_router", dict(train_memory_only=True, freeze_memory_router=True))]:
        p = make_policy(dev, memory=mem_cfg(), **kw)
        tr = [n for n, q in p.named_parameters() if q.requires_grad]
        groups = p.get_optim_params()
        n_opt = sum(len(g["params"]) for g in groups)
        router = all(".mlp.mem.keys" in n or ".mlp.mem.query_proj." in n for n in tr)
        mem_only = all(".mlp.mem." in n for n in tr)
        has_vals = any("slot_" in n for n in tr)
        has_router = any(".mlp.mem.keys" in n or ".mlp.mem.query_proj." in n for n in tr)
        anchor_tr = any("anchor_proj" in n for n in tr)
        ok = {
            "router_only": router and not has_vals and len(groups) == 1,
            "memory_only": mem_only and has_vals and has_router and len(groups) == 2,
            "memory_only+frozen_router": mem_only and has_vals and not has_router and len(groups) == 2,
        }[mode]
        check(f"freeze mode {mode}", ok and n_opt == len(tr) and not anchor_tr,
              f"{len(tr)} trainable, {len(groups)} optimizer groups, anchor_proj trainable={anchor_tr}")
        del p
        torch.cuda.empty_cache()

    # 9. save / load round trip
    p = make_policy(dev, memory=mem_cfg()).eval()
    perturb_values(p)
    ref_loss = loss_of(p, batch)
    with tempfile.TemporaryDirectory() as d:
        p.save_pretrained(d)
        cfg = copy.deepcopy(p.config)
        cfg.pretrained_path = d
        q = SmolVLAPolicy.from_pretrained(d, config=cfg).to(dev).eval()
    l_q = loss_of(q, batch)
    check("save/load round trip", torch.allclose(ref_loss, l_q, rtol=0, atol=1e-6) and len(q.model.vlm_with_expert._anchor_hooks) == 4,
          f"{ref_loss.item():.8f} vs {l_q.item():.8f}")

    del p, q, plain  # free the reference policy too: the matrix builds one policy per config
    gc.collect()
    torch.cuda.empty_cache()
    config_matrix(dev, batch)

    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
