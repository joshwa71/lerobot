#!/usr/bin/env python3
"""E61 shared-pair memory tables — smoke suite.

Modes (env SMOKE_MODE):
  legacy - no share flags: every module owns its storage; forward+backward runs;
           merge-helper unit tests (identity for first writer, union/min-compose after).
  shared - share_groups on (+ frozen_prepass, interleaved layout = the production
           composition): aliasing identity, state_dict dedupe contract, grad flow
           into the shared tables from a real forward+backward, mask union-merge on
           the real value_params, protection-store group sync, strict in-memory
           state_dict round-trip with bitwise forward parity.
  guards - config-level validation raises (no model load).

Convention: probe harness (parser.wrap + SequentialOnlineConfig), stage-1 base
checkpoint, fp32, small banks — mirrors run_smoke_frozen_prepass.sh.
"""
import os
import sys

import torch

from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.modules.memory_config import MemoryLayerConfig
from lerobot.policies.modules.memory_lite import MLPPlusMemory
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig,
    _build_dataloader_for_task,
    _collect_task_index_to_name,
    _iter_memory_modules,
    _merge_allowed_rows,
    _merge_protect_scale,
    _protect_usefulness_by_module,
    _sync_shared_protection_stores,
)

MODE = os.environ.get("SMOKE_MODE", "shared")


def _fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def _ok(msg):
    print(f"[ok] {msg}")


def _unit_test_merge_helpers():
    p1 = torch.nn.Parameter(torch.zeros(100, 4))
    p2 = torch.nn.Parameter(torch.zeros(100, 4))
    d = {}
    _merge_allowed_rows(d, p1, torch.tensor([1, 2, 3]))
    _merge_allowed_rows(d, p2, torch.tensor([7]))
    if set(d[p1].tolist()) != {1, 2, 3} or set(d[p2].tolist()) != {7}:
        _fail("merge helper: first-writer identity broken")
    _merge_allowed_rows(d, p1, torch.tensor([3, 4, 5]))
    if set(d[p1].tolist()) != {1, 2, 3, 4, 5}:
        _fail("merge helper: union broken")
    s = {}
    a = torch.ones(10)
    a[0] = 0.5
    a[1] = 2.0
    b = torch.ones(10)
    b[0] = 0.25
    b[2] = 0.8
    _merge_protect_scale(s, p1, a)
    _merge_protect_scale(s, p1, b)
    m = s[p1]
    if not (abs(m[0].item() - 0.25) < 1e-6 and abs(m[1].item() - 2.0) < 1e-6
            and abs(m[2].item() - 0.8) < 1e-6 and abs(m[3].item() - 1.0) < 1e-6):
        _fail(f"scale merge semantics broken: {m[:4].tolist()}")
    _ok("merge helpers: first-writer identity, union, min-compose w/ neutral-defer")


def _test_guards():
    base = dict(enabled=True, layers=[4, 6, 8, 10], vlm_layers=[5, 7, 9, 11],
                use_frozen_base_input_features=True, frozen_prepass=True)
    cases = [
        ("singleton group", dict(share_groups=[[4]])),
        ("non-member layer", dict(share_groups=[[4, 12]])),
        ("overlapping groups", dict(share_groups=[[4, 6], [6, 8]])),
        ("unsorted group", dict(share_groups=[[6, 4]])),
        ("vlm non-member", dict(vlm_share_groups=[[5, 13]])),
        ("rank mismatch in group", dict(share_groups=[[4, 6]], layer_ranks=[2, 4, 2, 2])),
    ]
    for name, kw in cases:
        try:
            MemoryLayerConfig(**base, **kw)
            _fail(f"guard '{name}' did not raise")
        except ValueError as e:
            _ok(f"guard '{name}' raised: {str(e)[:80]}")
    MemoryLayerConfig(**base, share_groups=[[4, 6], [8, 10]], vlm_share_groups=[[5, 7], [9, 11]])
    _ok("valid share config accepted")
    print("GUARDS-PASS")


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    if MODE == "guards":
        _test_guards()
        return
    _unit_test_merge_helpers()

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

    mems = _iter_memory_modules(policy)
    by_key = {jk: mem for _, mem, _, jk in mems}
    n_shared = sum(1 for _, mem, _, _ in mems if getattr(mem, "_storage_shared_from", None) is not None)

    if MODE == "legacy":
        if n_shared != 0:
            _fail(f"legacy mode has {n_shared} shared modules")
        _ok(f"legacy: {len(mems)} modules all own their storage")
    else:
        # ---- structural checks against the configured groups ----
        share_cfg = cfg.policy.memory_layer
        exp_groups = [list(g) for g in (share_cfg.share_groups or [])]
        vlm_groups = [list(g) for g in (share_cfg.vlm_share_groups or [])]
        exp_keys = {int(k.split("layers.")[-1]): k for k in by_key if "gemma_expert" in k}
        vlm_keys = {int(k.split("layers.")[-1]): k for k in by_key if "gemma_expert" not in k}
        n_follow_expected = sum(len(g) - 1 for g in exp_groups + vlm_groups)
        if n_shared != n_follow_expected:
            _fail(f"expected {n_follow_expected} followers, found {n_shared}")
        for groups, keymap, tower in ((exp_groups, exp_keys, "expert"), (vlm_groups, vlm_keys, "vlm")):
            for g in groups:
                lead = by_key[keymap[g[0]]]
                for li in g[1:]:
                    fol = by_key[keymap[li]]
                    for name in fol.storage_param_names():
                        if getattr(fol, name) is not getattr(lead, name):
                            _fail(f"{tower} L{li}.{name} is not aliased to L{g[0]}")
                _ok(f"{tower} group {g}: storage identity verified ({len(lead.storage_param_names())} tensors)")

        # ---- state_dict dedupe contract ----
        sd = policy.state_dict()
        for groups, keymap, tower in ((exp_groups, exp_keys, "expert"), (vlm_groups, vlm_keys, "vlm")):
            for g in groups:
                lead_prefix = keymap[g[0]]
                if f"{lead_prefix}.mlp.mem.keys" not in sd:
                    _fail(f"{tower} L{g[0]}: leader storage missing from state_dict (prefix bug?)")
                for li in g[1:]:
                    fol_prefix = keymap[li]
                    fol = by_key[keymap[li]]
                    bad = [n for n in fol.storage_param_names() if f"{fol_prefix}.mlp.mem.{n}" in sd]
                    if bad:
                        _fail(f"{tower} L{li}: follower storage {bad} present in state_dict (not deduped)")
                    if not any(k.startswith(f"{fol_prefix}.mlp.mem.query_proj") for k in sd):
                        _fail(f"{tower} L{li}: per-site query_proj missing from state_dict")
        n_val = sum(1 for _, p in policy.named_parameters() if getattr(p, "pk_value_param", False))
        n_tables = len(exp_groups) + len(vlm_groups) + (len(exp_keys) + len(vlm_keys) - sum(len(g) for g in exp_groups + vlm_groups))
        per_table = len(next(iter(by_key.values()))._slot_param_names())
        if n_val != n_tables * per_table:
            _fail(f"pk_value_param count {n_val} != tables {n_tables} x {per_table}")
        _ok(f"state_dict dedupe: {n_val} value params across {n_tables} tables (per-site heads retained)")

    # ---- forward + backward on a real batch (values trainable) ----
    for p in policy.parameters():
        p.requires_grad = bool(getattr(p, "pk_value_param", False))
    tin = _collect_task_index_to_name(dataset)
    loader = _build_dataloader_for_task(
        dataset, tin, dataset_task_id=0, batch_size=cfg.batch_size,
        num_workers=0, device_type=device.type,
    )
    batch = next(iter(loader))
    batch = preprocessor(batch)
    policy.train()
    loss, _ = policy.forward(batch)
    loss.backward()
    _ok(f"forward+backward ok, loss {float(loss):.4f}")

    grads_seen = 0
    for _, mem, value_params, jk in mems:
        if getattr(mem, "last_indices", None) is None:
            _fail(f"{jk}: no retrieval recorded (site did not route)")
        if getattr(mem, "_storage_shared_from", None) is None:
            for vp in value_params:
                if vp.grad is not None and float(vp.grad.abs().sum()) > 0:
                    grads_seen += 1
    if grads_seen == 0:
        _fail("no gradient reached any slot table")
    _ok(f"all {len(mems)} sites routed; {grads_seen} owned slot tensors carry grad")

    if MODE == "shared":
        # ---- mask union-merge on the REAL shared value_params ----
        allowed = {}
        share_members = [(mem, vps, jk) for _, mem, vps, jk in mems]
        seen_param_entries = set()
        for mem, vps, jk in share_members:
            rows = torch.unique(mem.last_indices.reshape(-1).to(torch.long))[:8]
            for vp in vps:
                _merge_allowed_rows(allowed, vp, rows)
                seen_param_entries.add(id(vp))
        if len(allowed) != len(seen_param_entries):
            _fail("allowed_by_param has duplicate-object entries")
        n_tables_masked = len({id(k) for k in allowed})
        _ok(f"mask union-merge: {len(share_members)} sites -> {n_tables_masked} table-param entries")

        # ---- protection-store group sync ----
        _protect_usefulness_by_module.clear()
        for i, (mem, vps, jk) in enumerate(share_members):
            u = torch.zeros(mem.size)
            u[i % mem.size] = 1.0
            _protect_usefulness_by_module[jk] = u
        _sync_shared_protection_stores(policy)
        groups_by_table = {}
        for mem, vps, jk in share_members:
            groups_by_table.setdefault(id(vps[0]), []).append(jk)
        for tbl, jks in groups_by_table.items():
            if len(jks) < 2:
                continue
            ref = _protect_usefulness_by_module[jks[0]]
            for jk in jks[1:]:
                if not torch.equal(_protect_usefulness_by_module[jk], ref):
                    _fail(f"protection store not synced across {jks}")
            expected_mass = float(sum(1.0 for _ in jks))
            if abs(float(ref.sum()) - expected_mass) > 1e-6:
                _fail(f"protection sync mass wrong: {float(ref.sum())} != {expected_mass}")
        _protect_usefulness_by_module.clear()
        _ok("protection-store sync: elementwise max across group members")

        # ---- strict in-memory round trip + bitwise forward parity ----
        policy2 = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
        if hasattr(policy2, "precompute_task_embeddings"):
            policy2.precompute_task_embeddings(dataset.meta)
        policy2 = policy2.to(device)
        missing, unexpected = policy2.load_state_dict(policy.state_dict(), strict=True), None
        policy2.eval(); policy.eval()
        with torch.no_grad():
            l1, _ = policy.forward(batch)
            torch.manual_seed(0)
            l1b, _ = policy.forward(batch)
            torch.manual_seed(0)
            l2, _ = policy2.forward(batch)
        if float((l1b - l2).abs()) != 0.0:
            _fail(f"round-trip forward parity broken: {float(l1b)} vs {float(l2)}")
        for _, mem2, _, jk in _iter_memory_modules(policy2):
            pass
        n_shared2 = sum(1 for _, m, _, _ in _iter_memory_modules(policy2)
                        if getattr(m, "_storage_shared_from", None) is not None)
        if n_shared2 != n_shared:
            _fail(f"aliasing not reconstructed on load: {n_shared2} != {n_shared}")
        _ok("strict state_dict round-trip: loads clean, aliasing reconstructed, forward bitwise-identical")

    print(f"{MODE.upper()}-PASS")


if __name__ == "__main__":
    main()
