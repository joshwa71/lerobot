#!/usr/bin/env python3
"""Rebuild of the E39 MSE forgetting-matrix instrument (scratchpad copy was lost).
Loads each per-task checkpoint of a sequential run and computes the paired-noise
flow-matching MSE for all 5 tasks via the trainer's own _eval_loss_on_seen_tasks
(seed=0, n_batches=16 == E39 settings). Env: MSEMAT_RUN_DIR, MSEMAT_STEPS, MSEMAT_OUT."""
import os, sys, json, glob
import torch
from safetensors.torch import load_file
from accelerate import Accelerator
from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig, _collect_task_index_to_name, _eval_loss_on_seen_tasks,
)

@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    run_dir = os.environ["MSEMAT_RUN_DIR"]
    steps = os.environ["MSEMAT_STEPS"].split(",")
    out_path = os.environ["MSEMAT_OUT"]
    cfg.validate()
    accelerator = Accelerator()
    device = accelerator.device
    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    processor_kwargs = {}
    processor_kwargs["preprocessor_overrides"] = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path, **processor_kwargs,
    )
    if hasattr(policy, "precompute_task_embeddings"):
        policy.precompute_task_embeddings(dataset.meta)
    policy = accelerator.prepare(policy)
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    task_index_to_name = _collect_task_index_to_name(dataset)
    results = {}
    with open(out_path, "a") as fh:
        for st in steps:
            sd_path = os.path.join(run_dir, "checkpoints", st, "pretrained_model", "model.safetensors")
            from safetensors import safe_open as _so
            sd = {}
            with _so(sd_path, framework="pt") as f:
                for k in f.keys():
                    if ".mlp.mem.slot_" in k:
                        sd[k] = f.get_tensor(k)
            print(f"[load] {st}: {len(sd)} slot tensors", flush=True)
            _, unexpected = unwrapped.load_state_dict(sd, strict=False)
            if unexpected:
                print(f"[warn] {st}: unexpected={unexpected[:5]}")
            del sd
            per_task = _eval_loss_on_seen_tasks(
                policy, accelerator, dataset, task_index_to_name, [0,1,2,3,4],
                batch_size=32, num_workers=4, device=device, n_batches=16,
                preprocessor=preprocessor, seed=0,
            )
            rec = {"run": os.path.basename(run_dir), "ckpt": st,
                   "per_task": {str(k): v for k, v in per_task.items()}}
            results[st] = per_task
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"[done] {st}: {per_task}", flush=True)

if __name__ == "__main__":
    main()
