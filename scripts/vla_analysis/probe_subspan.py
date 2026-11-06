#!/usr/bin/env python3
"""E44 sub-span routing probe: which language-field positions carry the cross-task
shared routing mass in the VLM text-span memory?

Per-position reconstruction: eval-path last_indices is mask-filtered but ORDER-
PRESERVING over the (B, T) flatten, so with the per-sample valid counts v_i the
(sample, position) of every row is exact. Accumulates per-(task, layer, position)
slot histograms over the held-out libero_10 demos, then reports cross-task IoU and
footprint size per position and per region (instr=[0:16), mid=[16:28), state=[28:...)).

Usage: python probe_subspan.py --policy.path=<ckpt> ... (audit CLI args) ARM=<tag> via env.
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

ARM = os.environ.get("ARM", "arm")
NB = int(os.environ.get("NB", "13"))
BS = int(os.environ.get("BS", "8"))
OUT = os.environ.get("OUT", f"/home/josh/lerobot/outputs/analysis/e44/subspan_{ARM}.json")
TASKS = list(range(10))
FAMILY = [(4, 5), (4, 7), (5, 7)]
REGIONS = {"instr[0:16)": (0, 16), "mid[16:28)": (16, 28), "state[28:]": (28, 200),
           "img[300:340)": (300, 340)}


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
    policy.eval()

    wrappers = {}  # layer_idx -> module
    for n, m in policy.named_modules():
        if isinstance(m, MLPPlusMemory) and "language_model" in n:
            li = int(n.split("layers.")[1].split(".")[0])
            wrappers[li] = m
            m.mem.EVAL_MEMORY = True
    print(f"[setup] wrapped LM layers: {sorted(wrappers)}")
    assert wrappers, "no VLM memory modules found"

    tin = _collect_task_index_to_name(dataset)
    # counts[layer][task] = dict pos -> np.array bincount over slots (sparse dict slot->cnt)
    counts = {L: {t: defaultdict(lambda: defaultdict(int)) for t in TASKS} for L in wrappers}
    span_valid = []

    for t in TASKS:
        dl = _build_dataloader_for_task(dataset, tin, t, batch_size=BS, num_workers=2,
                                        device_type=device.type)
        it = iter(dl)
        for bi in range(NB):
            try:
                b = next(it)
            except StopIteration:
                break
            for ck in dataset.meta.camera_keys:
                if ck in b and b[ck].dtype == torch.uint8:
                    b[ck] = b[ck].to(torch.float32) / 255.0
            b = preprocessor(b)
            tids = torch.full((BS,), t, dtype=torch.long, device=device)
            with torch.no_grad():
                policy.forward(b, task_emb=None, task_ids=tids)
            for L, m in wrappers.items():
                li = m.mem.last_indices  # (rows, heads, knn) order-preserving
                vm = getattr(m, "_ctx_valid_mask", None)
                if li is None or vm is None:
                    continue
                v = vm.sum(dim=1).cpu().numpy().astype(int)  # per-sample valid counts
                # E49 route-once-aware row mapping. Three layouts, detected by row count:
                #   legacy broadcast (text span):    rows = sum(v_i), per-position order
                #   compact route-once (text span):  rows = sum(v_i): [state x n_state, instr x il]
                #   compact route-once (image span): rows = sum(K*64 + v_i):
                #       [region_j x 64 ...(j<K), state x n_state, instr x il]
                #   literal broadcast (image span):  rows = sum(n_act*256 + v_i), per-position
                il_t = getattr(m, "_ctx_instr_len", None)
                ia = getattr(m, "_ctx_img_active", None)
                K = 0
                n_ia = 0
                if getattr(m, "image_regions", 0) > 0 and ia is not None:
                    s2 = int(m.image_regions) ** 2
                    K = int(ia[0].long().sum().item()) * s2
                    n_ia = int(ia[0].long().sum().item()) * 256
                rows = li.shape[0]
                pooled = bool(getattr(m, "router_pool", "")) and il_t is not None
                if K > 0 and rows == int(v.sum()) + K * (256 // (int(m.image_regions) ** 2)) * len(v):
                    rsize = 256 // (int(m.image_regions) ** 2)
                    pos_list = []
                    for i, vi in enumerate(v):
                        ili = int(il_t[i])
                        img = np.concatenate([np.full(rsize, 300 + j) for j in range(K)])
                        st = ili + np.arange(max(vi - ili, 1))[: vi - ili] if vi > ili else np.empty(0, int)
                        pos_list.append(np.concatenate([img, st, np.arange(ili)]))
                    pos = np.concatenate(pos_list).astype(int)
                elif K > 0 and rows == n_ia * len(v) + int(v.sum()):
                    # literal broadcast image layout: rows are per-POSITION over
                    # [n_ia image positions (camera-major, 16x16 row-major), language].
                    s = int(m.image_regions)
                    step = 16 // s
                    q = np.arange(256)
                    reg_in_cam = (q // 16 // step) * s + (q % 16) // step
                    img_pos = np.concatenate([
                        300 + c * s * s + reg_in_cam for c in range(n_ia // 256)
                    ])
                    pos_list = [np.concatenate([img_pos, np.arange(vi)]) for vi in v]
                    pos = np.concatenate(pos_list).astype(int)
                elif pooled and getattr(m, "route_once", False) and rows == int(v.sum()):
                    # compact text-span layout (the deferred E47 mapping): state rows first
                    pos_list = []
                    for i, vi in enumerate(v):
                        ili = int(il_t[i])
                        pos_list.append(np.concatenate([ili + np.arange(vi - ili), np.arange(ili)]))
                    pos = np.concatenate(pos_list).astype(int)
                else:
                    assert rows == int(v.sum()), (li.shape, v.sum(), K)
                    pos = np.concatenate([np.arange(vi) for vi in v])
                flat = li.reshape(li.shape[0], -1).numpy()  # rows x (heads*knn)
                cdict = counts[L][t]
                for r in range(flat.shape[0]):
                    d = cdict[int(pos[r])]
                    for s in flat[r]:
                        d[int(s)] += 1
                if L == sorted(wrappers)[0]:
                    span_valid.extend(v.tolist())
        print(f"[task {t}] done ({tin.get(t,'?')[:50]})")

    vmax = int(max(span_valid))
    print(f"[valid] span lengths: min {min(span_valid)} max {vmax} mean {np.mean(span_valid):.1f}")

    def iou(a, b):
        mn = np.minimum(a, b).sum(); mx = np.maximum(a, b).sum()
        return float(mn / max(mx, 1e-12))

    out = {"arm": ARM, "valid_mean": float(np.mean(span_valid)), "valid_max": vmax}
    for L in sorted(wrappers):
        print(f"\n===== L{L} per-position profile (mean pairwise IoU over all 45 pairs | family mean | mean uniq slots/task)")
        prof = []
        for p in range(vmax):
            w = {}
            for t in TASKS:
                d = counts[L][t].get(p)
                if not d:
                    continue
                ks = np.fromiter(d.keys(), dtype=np.int64)
                vs = np.fromiter(d.values(), dtype=np.float64)
                a = np.zeros(65536); a[ks] = vs; w[t] = a / vs.sum()
            if len(w) < 8:
                continue
            pairs = [iou(w[a_], w[b_]) for i, a_ in enumerate(sorted(w)) for b_ in sorted(w)[i+1:]]
            fam = [iou(w[a_], w[b_]) for a_, b_ in FAMILY if a_ in w and b_ in w]
            uniq = np.mean([np.count_nonzero(w[t]) for t in w])
            prof.append((p, float(np.mean(pairs)), float(np.mean(fam)) if fam else None, float(uniq)))
            if p % 4 == 0 or p == vmax - 1:
                print(f"  pos {p:>3}: IoU {np.mean(pairs):.3f}  fam {np.mean(fam) if fam else float('nan'):.3f}  uniq {uniq:6.0f}")
        out[f"L{L}_profile"] = prof
        # region aggregates (pool histograms within region)
        print(f"----- L{L} region aggregates")
        for rname, (lo, hi) in REGIONS.items():
            w = {}
            for t in TASKS:
                agg = defaultdict(int)
                hi_eff = min(hi, vmax) if hi <= 200 else hi  # synthetic img codes live at 300+
                for p in range(lo, hi_eff):
                    for s, c in counts[L][t].get(p, {}).items():
                        agg[s] += c
                if not agg:
                    continue
                ks = np.fromiter(agg.keys(), dtype=np.int64)
                vs = np.fromiter(agg.values(), dtype=np.float64)
                a = np.zeros(65536); a[ks] = vs; w[t] = a / vs.sum()
            pairs = [iou(w[a_], w[b_]) for i, a_ in enumerate(sorted(w)) for b_ in sorted(w)[i+1:]]
            fam = [iou(w[a_], w[b_]) for a_, b_ in FAMILY if a_ in w and b_ in w]
            bg = [v for v, (i, j) in zip(pairs, [(a_, b_) for i2, a_ in enumerate(sorted(w)) for b_ in sorted(w)[i2+1:]]) if (i, j) not in FAMILY]
            effs, uniqs, c50s = [], [], []
            for t in w:
                nz = w[t][w[t] > 0]
                effs.append(1.0 / (w[t] ** 2).sum()); uniqs.append(len(nz))
                order = np.sort(w[t])[::-1]; c50s.append(int(np.searchsorted(np.cumsum(order), 0.5)) + 1)
            print(f"  {rname:>13}: famIoU {np.mean(fam):.3f}  bgIoU {np.mean(bg):.3f}  allIoU {np.mean(pairs):.3f}  "
                  f"core50 {np.mean(c50s):7.0f}  effnum {np.mean(effs):7.0f}  uniq {np.mean(uniqs):7.0f}")
            out[f"L{L}_{rname}"] = {"famIoU": float(np.mean(fam)), "bgIoU": float(np.mean(bg)),
                                    "core50": float(np.mean(c50s)), "effnum": float(np.mean(effs))}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"))
    print(f"\nsaved {OUT}\nSUBSPAN-PROBE-DONE")


if __name__ == "__main__":
    main()
