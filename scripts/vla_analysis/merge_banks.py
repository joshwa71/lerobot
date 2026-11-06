#!/usr/bin/env python3
"""Merge the warmed EXPERT memory bank into a warmed VLM-memory checkpoint (E45).

Pure safetensors surgery — no model instantiation, no dataset needed:
  out/model.safetensors = VLM warm ckpt tensors (stage-1 backbone + VLM mem, disjoint)
                          UNION the expert bank file (60 tensors from extract_router_bank)
  out/config.json       = EXPERT warm ckpt config (expert memory geometry incl. n256)
                          + the VLM fields (vlm_layers / vlm_* / vlm_router_pool*)
Training-mode flags (train_router_only etc.) ride along in the config and MUST be
overridden explicitly at each downstream stage's CLI (E37 gotcha).

Safety: placement guard (min vlm_layer > max expert layer) means the two towers'
routing certificates provably survive the union — expert routing reads prefix KV <= L14
which VLM memory at 15/16 cannot touch; VLM routing input is memory-free below by
construction; both value sets are zero-output at merge time.

Usage: merge_banks.py <vlm_warm_ckpt_dir> <expert_config.json> <expert_bank.safetensors> <out_dir>
"""
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

VLM_FIELDS = ["vlm_layers", "vlm_mem_n_keys", "vlm_lora_rank", "vlm_mem_knn",
              "vlm_text_span", "vlm_router_pool", "vlm_router_pool_weights"]


def main():
    vlm_dir, exp_cfg_path, bank, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)

    cfg = json.load(open(exp_cfg_path))
    vcfg = json.load(open(f"{vlm_dir}/config.json"))
    ml, vml = cfg["memory_layer"], vcfg["memory_layer"]
    assert ml["layers"] == [8, 10, 12, 14] and ml["mem_n_keys"] == 256, (ml["layers"], ml["mem_n_keys"])
    assert vml["vlm_layers"] == [15, 16], vml["vlm_layers"]
    defaults = {"vlm_router_pool": "", "vlm_router_pool_weights": [1.0, 1.0]}
    for k in VLM_FIELDS:
        if k in vml:
            ml[k] = vml[k]
        elif k not in ml:
            if k not in defaults:
                raise KeyError(f"{k} missing from both configs")
            ml[k] = defaults[k]

    tensors = {}
    with safe_open(f"{vlm_dir}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    n_vlm_mem = sum(1 for k in tensors if ".mlp.mem." in k and "language_model" in k)
    with safe_open(bank, framework="pt") as f:
        bkeys = list(f.keys())
        for k in bkeys:
            assert k not in tensors, f"overlapping key: {k}"
            tensors[k] = f.get_tensor(k)
    assert n_vlm_mem >= 12, f"VLM memory tensors missing from {vlm_dir} ({n_vlm_mem})"
    assert len(bkeys) == 60, f"expert bank expected 60 tensors, got {len(bkeys)}"

    for fn in os.listdir(vlm_dir):
        if fn not in ("model.safetensors", "config.json"):
            src = os.path.join(vlm_dir, fn)
            (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(out, fn))
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=2)
    save_file(tensors, f"{out}/model.safetensors")
    print(f"merged: {len(tensors)} tensors ({n_vlm_mem} VLM mem + {len(bkeys)} expert mem) -> {out}")
    print("MERGE-BANKS-DONE")


if __name__ == "__main__":
    main()
