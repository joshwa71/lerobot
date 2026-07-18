#!/usr/bin/env python3
"""Extract a tower's memory tensors from a warm-up checkpoint into a standalone
safetensors file (E45 merge tooling).

Default selection = the EXPERT tower's full memory state (router + gate/projections +
values: slot_down random-init / slot_up zero from the router-only warm-up) so the merge
is a bit-exact hand-off with no re-init on the receiving box.

Usage: python extract_router_bank.py <pretrained_model_dir> <out.safetensors>
"""
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MATCH, TOWER = ".mlp.mem.", "gemma_expert"


def main():
    src, dst = sys.argv[1], sys.argv[2]
    out = {}
    with safe_open(f"{src}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            if MATCH in k and TOWER in k:
                out[k] = f.get_tensor(k)
    assert out, f"no {TOWER} memory tensors found in {src}"
    layers = sorted({k.split(".layers.")[1].split(".")[0] for k in out})
    n = sum(v.numel() for v in out.values())
    print(f"extracted {len(out)} tensors, layers {layers}, {n/1e9:.2f}B params")
    save_file(out, dst)
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
