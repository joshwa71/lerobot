#!/usr/bin/env python3
"""Paired-noise MSE forgetting matrix for DENSE per-task checkpoints (RETAIN, E67).

Sibling of scripts/vla_analysis/mse_matrix2.py (which deliberately partial-loads only the memory
slot tensors) and mse_matrix_peft.py (adapter swap): here every boundary's full
`model.safetensors` is loaded strictly into a policy built once from the first boundary, then
scored with the trainer's own `_eval_loss_on_seen_tasks` (seed=0, n_batches=16 == E39/E65).
Guard: after each load the state's L1 must differ from the previous row's (a silent no-op load
would otherwise produce a fake flat matrix).
Invoke like mse_matrix2.py: --policy.path at the FIRST boundary's pretrained_model, env
MSEMAT_RUN_DIR, MSEMAT_STEPS (comma list of 6-digit ids), MSEMAT_OUT, MSEMAT_TASKS (default all 10).
"""
import json
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from safetensors.torch import load_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_dataset_policy_processors, eval_loss_matrix_row  # noqa: E402

from lerobot.configs import parser  # noqa: E402
from lerobot.scripts.lerobot_sequential_train import SequentialOnlineConfig  # noqa: E402


def _l1(model) -> float:
    return sum(p.detach().float().abs().sum().item() for p in model.parameters())


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    run_dir = os.environ["MSEMAT_RUN_DIR"]
    steps = os.environ["MSEMAT_STEPS"].split(",")
    out_path = os.environ["MSEMAT_OUT"]
    tasks = [int(x) for x in os.environ.get("MSEMAT_TASKS", "0,1,2,3,4,5,6,7,8,9").split(",")]
    n_batches = int(os.environ.get("MSEMAT_NBATCHES", "16"))
    cfg.validate()
    accelerator = Accelerator()
    device = accelerator.device
    dataset, policy, preprocessor, _, t2n = build_dataset_policy_processors(cfg, device)
    policy = accelerator.prepare(policy)
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    prev_l1 = None
    with open(out_path, "a") as fh:
        for st in steps:
            sd_path = os.path.join(run_dir, "checkpoints", st, "pretrained_model", "model.safetensors")
            load_model(unwrapped, sd_path, strict=True, device=str(device))
            l1 = _l1(unwrapped)
            if prev_l1 is not None and abs(l1 - prev_l1) <= 1e-9 * max(1.0, prev_l1):
                raise RuntimeError(f"{st}: state identical to the previous row (L1 {l1}); load is a no-op")
            prev_l1 = l1
            print(f"[load] {st}: full state loaded, L1={l1:.6e}", flush=True)
            per_task = eval_loss_matrix_row(policy, accelerator, dataset, t2n, tasks, device, preprocessor,
                                            n_batches=n_batches, batch_size=32, num_workers=4, seed=0)
            rec = {"run": os.path.basename(run_dir.rstrip("/")), "ckpt": st,
                   "per_task": {str(k): v for k, v in per_task.items()}}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[done] {st}: {per_task}", flush=True)


if __name__ == "__main__":
    main()
