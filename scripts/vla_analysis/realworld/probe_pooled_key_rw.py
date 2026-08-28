#!/usr/bin/env python3
"""E65 gate-fail diagnostic: how much does the VLM route-once STATE KEY vary across samples
of one task?  In the anchored pooled mode every sample routes one composite key
k = nrm(a*nrm(instr_pool) + b*nrm(state_pool)) (memory_lite._pooled_components) plus its
constant instruction tokens; a flat per-task effnum of ~knn*heads means k is ~constant.

Per VLM memory layer, over NB batches x BS samples per task, reports the mean off-diagonal
cosine similarity across samples of the instruction pool, the state pool and the composite k,
the mean cosine between the two pools, the instruction/state boundary (il) and state-span
length actually seen by the router, and the number of DISTINCT knn slots hit by the state-key
rows vs the instruction-token rows (from mem.last_indices, route-once row layout).

Env: TASKS (csv), NB, BS, OUT (json). CLI: the audit args with --policy.path=<warm-up ckpt>.
Works on any dataset (LIBERO for the sim comparison); no --env needed.
"""
import json
import os
from collections import defaultdict

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.modules.memory_lite import MLPPlusMemory
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig, _build_dataloader_for_task, _collect_task_index_to_name,
)

TASKS = [int(x) for x in os.environ.get("TASKS", "0,1,3").split(",")]
NB = int(os.environ.get("NB", "2"))
BS = int(os.environ.get("BS", "16"))
OUT = os.environ["OUT"]


def _offdiag_cos(u):
    u = torch.nn.functional.normalize(u.float(), dim=-1)
    c = u @ u.T
    n = c.shape[0]
    return float((c.sum() - torch.diagonal(c).sum()) / max(n * (n - 1), 1))


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

    wrappers, cap = {}, defaultdict(list)
    for n, m in policy.named_modules():
        if isinstance(m, MLPPlusMemory) and "language_model" in n:
            L = int(n.split("layers.")[1].split(".")[0])
            wrappers[L] = m
            m.mem.EVAL_MEMORY = True

            def make(L, m, orig):
                def patched(base, vm2):
                    comp = orig(base, vm2)
                    rec = {"router_pool": m.router_pool, "w": list(m.router_pool_w)}
                    if comp is None:
                        rec["none"] = True
                    else:
                        k, il, v, row_ok = comp
                        bf = base.float()
                        T = base.shape[1]
                        pos = torch.arange(T, device=base.device).unsqueeze(0)
                        bnd = il.to(base.device).unsqueeze(1)
                        imask = (pos >= 3) & (pos < bnd) & vm2
                        smask = (pos >= bnd + 3) & (pos < (v - 5).unsqueeze(1)) & vm2

                        def pool(mask):
                            mm = mask.unsqueeze(-1).float()
                            return (bf * mm).sum(1) / mm.sum(1).clamp_min(1.0)

                        def nrm(u):
                            return u / u.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)

                        ip, sp = nrm(pool(imask)), nrm(pool(smask))
                        rec.update(
                            il=il.tolist(), v=v.tolist(), n_state_tokens=smask.sum(1).tolist(),
                            row_ok=int(row_ok.sum()), cos_instr=_offdiag_cos(ip),
                            cos_state=_offdiag_cos(sp), cos_k=_offdiag_cos(k),
                            cos_instr_vs_state=float(torch.nn.functional.cosine_similarity(ip, sp, dim=-1).mean()),
                        )
                    cap[L].append(rec)
                    m._last_comp = comp
                    return comp
                return patched
            m._pooled_components = make(L, m, m._pooled_components)
    print(f"[setup] VLM memory layers {sorted(wrappers)}; pool={wrappers[min(wrappers)].router_pool} w={wrappers[min(wrappers)].router_pool_w}")

    tin = _collect_task_index_to_name(dataset)
    out = {"policy": str(cfg.policy.pretrained_path), "dataset": str(cfg.dataset.root), "tasks": {}}
    for t in TASKS:
        dl = _build_dataloader_for_task(dataset, tin, t, batch_size=BS, num_workers=2, device_type="cuda")
        it = iter(dl)
        for L in wrappers:
            cap[L].clear()
        slots = {L: {"state": set(), "instr": set(), "state_rows": 0, "layout": None} for L in wrappers}
        for _ in range(NB):
            b = next(it)
            for ck in dataset.meta.camera_keys:
                if ck in b and b[ck].dtype == torch.uint8:
                    b[ck] = b[ck].to(torch.float32) / 255.0
            b = preprocessor(b)
            tids = torch.full((BS,), t, dtype=torch.long, device=device)
            with torch.no_grad():
                policy.forward(b, task_emb=None, task_ids=tids)
            for L, m in wrappers.items():
                li, comp = m.mem.last_indices, getattr(m, "_last_comp", None)
                if li is None or comp is None:
                    continue
                _, il, v, _ = comp
                il, v = il.cpu().numpy().astype(int), v.cpu().numpy().astype(int)
                flat = li.reshape(li.shape[0], -1).cpu().numpy()
                rows = flat.shape[0]
                if rows == int(v.sum()):            # [state x n_state, instr x il] per sample
                    slots[L]["layout"] = "state_x_nstate+instr"
                    r = 0
                    for i in range(len(v)):
                        ns = int(v[i] - il[i])
                        slots[L]["state"].update(flat[r].tolist()); slots[L]["state_rows"] += 1
                        for rr in range(r + ns, r + int(v[i])):
                            slots[L]["instr"].update(flat[rr].tolist())
                        r += int(v[i])
                elif rows == int((1 + il).sum()):   # [state key, instr x il] per sample
                    slots[L]["layout"] = "statekey+instr"
                    r = 0
                    for i in range(len(v)):
                        slots[L]["state"].update(flat[r].tolist()); slots[L]["state_rows"] += 1
                        for rr in range(r + 1, r + 1 + int(il[i])):
                            slots[L]["instr"].update(flat[rr].tolist())
                        r += 1 + int(il[i])
                else:
                    slots[L]["layout"] = f"unknown rows={rows} sum_v={int(v.sum())}"
        rec = {}
        for L in sorted(wrappers):
            ok = [x for x in cap[L] if not x.get("none")]
            mean = lambda key: float(np.mean([x[key] for x in ok])) if ok else None  # noqa: E731
            li = wrappers[L].mem.last_indices
            rec[f"L{L}"] = {
                "batches": len(cap[L]), "fallback_none": len(cap[L]) - len(ok),
                "cos_instr": mean("cos_instr"), "cos_state": mean("cos_state"), "cos_k": mean("cos_k"),
                "cos_instr_vs_state": mean("cos_instr_vs_state"),
                "il_first4": ok[0]["il"][:4] if ok else None, "n_state_tokens_first4": ok[0]["n_state_tokens"][:4] if ok else None,
                "row_ok": ok[0]["row_ok"] if ok else None, "layout": slots[L]["layout"],
                "state_rows": slots[L]["state_rows"], "distinct_slots_state_rows": len(slots[L]["state"]),
                "distinct_slots_instr_rows": len(slots[L]["instr"]),
                "knn_x_heads": int(li.shape[1] * li.shape[2]) if li is not None and li.dim() == 3 else None,
            }
            print(f"[t{t} L{L}] " + "  ".join(f"{k}={v}" for k, v in rec[f"L{L}"].items()), flush=True)
        out["tasks"][str(t)] = rec
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
