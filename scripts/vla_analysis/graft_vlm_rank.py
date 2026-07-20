#!/usr/bin/env python3
"""Graft a certified joint router warm-up checkpoint onto a different VLM LoRA rank.

The warm-up (train_router_only) trains only PQ keys + query projection/FiLM; the
value tensors are untrained init (slot_down random / slot_up zeros => memory output 0),
so a rank change loses nothing — but torch's load_state_dict raises on shape-mismatched
tensors even at strict=False, so the r2-shaped VLM slot tensors must be REMOVED from the
checkpoint (missing keys are tolerated; mismatched ones are not). This script copies a
pretrained_model dir, drops the VLM tower's slot_down/slot_up tensors, and rewrites
config.json with the new vlm_lora_rank so the artifact is self-describing (E37 rule).
Expert tower and every router/gate tensor pass through bit-exact.

Usage: graft_vlm_rank.py --src <pretrained_model dir> --dst <pretrained_model dir> --vlm-rank 4
"""
import argparse
import json
import os
import shutil

from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--vlm-rank", type=int, required=True)
    args = ap.parse_args()

    cfg_path = os.path.join(args.src, "config.json")
    cfg = json.load(open(cfg_path))
    mem = cfg.get("memory_layer") or {}
    if not mem.get("vlm_layers"):
        raise SystemExit("src config has no vlm_layers — nothing to graft")
    old_rank = mem.get("vlm_lora_rank")
    if old_rank == args.vlm_rank:
        raise SystemExit(f"src already at vlm_lora_rank={old_rank}")

    sd = load_file(os.path.join(args.src, "model.safetensors"))
    dropped = [
        k for k in sd
        if ".language_model." in k and ".mlp.mem.slot_" in k
    ]
    expect = 2 * len(mem["vlm_layers"])  # slot_down + slot_up per VLM layer
    assert len(dropped) == expect, f"expected {expect} VLM slot tensors, found {len(dropped)}: {dropped}"
    for k in dropped:
        del sd[k]

    os.makedirs(args.dst, exist_ok=True)
    for f in os.listdir(args.src):
        if f not in ("model.safetensors", "config.json"):
            shutil.copy2(os.path.join(args.src, f), os.path.join(args.dst, f))
    mem["vlm_lora_rank"] = args.vlm_rank
    cfg["memory_layer"] = mem
    with open(os.path.join(args.dst, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    save_file(sd, os.path.join(args.dst, "model.safetensors"), metadata={"format": "pt"})
    print(f"grafted {args.src} -> {args.dst}: vlm_lora_rank {old_rank}->{args.vlm_rank}, "
          f"dropped {len(dropped)} tensors:")
    for k in dropped:
        print(f"  - {k}")


if __name__ == "__main__":
    main()
