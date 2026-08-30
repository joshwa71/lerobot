#!/usr/bin/env python3
"""REAL-WORLD duplicate of mse_matrix2.py (the E39 MSE forgetting-matrix instrument).

Loads each per-task checkpoint of a sequential run (slot tensors only, swapped into the policy
built from --policy.path) and computes the paired-noise flow-matching MSE for every task in
MSEMAT_TASKS via the trainer's own _eval_loss_on_seen_tasks (seed=0). IDENTICAL numerics to
mse_matrix2.py at the defaults (n_batches=16, batch_size=32, num_workers=4); the only delta is
that those three are env-overridable (MSEMAT_NB / MSEMAT_BS / MSEMAT_NW) so the SMOKE=1 battery
can run at 2 x bs4 beside a training job. No simulator: call with the RW dataset args and no
--env.* (SequentialOnlineConfig tolerates env=None, as the RW sequential stage itself does).

Env: MSEMAT_RUN_DIR (run dir containing checkpoints/), MSEMAT_STEPS (csv of checkpoint dirs,
e.g. 005000,010000,...), MSEMAT_OUT (jsonl, appended), MSEMAT_TASKS (default 0,1,2,3,4).
Output rows: {"run", "ckpt", "per_task": {task: mse}}.
"""
import json
import os

from accelerate import Accelerator
from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig,
    _collect_task_index_to_name,
    _eval_loss_on_seen_tasks,
)

N_BATCHES = int(os.environ.get("MSEMAT_NB", "16"))
BATCH_SIZE = int(os.environ.get("MSEMAT_BS", "32"))
NUM_WORKERS = int(os.environ.get("MSEMAT_NW", "4"))


@parser.wrap()
def main(cfg: SequentialOnlineConfig):
    run_dir = os.environ["MSEMAT_RUN_DIR"]
    steps = os.environ["MSEMAT_STEPS"].split(",")
    out_path = os.environ["MSEMAT_OUT"]
    tasks = [int(x) for x in os.environ.get("MSEMAT_TASKS", "0,1,2,3,4").split(",")]
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
    print(f"[msemat-rw] run={os.path.basename(run_dir)} steps={steps} tasks={tasks} "
          f"n_batches={N_BATCHES} bs={BATCH_SIZE} nw={NUM_WORKERS}", flush=True)
    results = {}
    with open(out_path, "a") as fh:
        for st in steps:
            sd_path = os.path.join(run_dir, "checkpoints", st, "pretrained_model", "model.safetensors")
            from safetensors import safe_open as _so
            sd = {}
            with _so(sd_path, framework="pt") as f:
                for k in f.keys():
                    # E65 add-16: shared tables (E61) save 2 of the 7 storages under
                    # `<layer>.mlp.mem._storage_shared_from.slot_*`, which the old
                    # `".mlp.mem.slot_" in k` filter MISSED -> those tables stayed at the first
                    # checkpoint's values in every row (10 of 14 tensors loaded). Match both.
                    if ".mlp.mem." in k and (".slot_down" in k or ".slot_up" in k):
                        sd[k] = f.get_tensor(k)
            print(f"[load] {st}: {len(sd)} slot tensors", flush=True)
            _, unexpected = unwrapped.load_state_dict(sd, strict=False)
            if unexpected:
                print(f"[warn] {st}: unexpected={unexpected[:5]}")
            del sd
            per_task = _eval_loss_on_seen_tasks(
                policy, accelerator, dataset, task_index_to_name, tasks,
                batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, device=device, n_batches=N_BATCHES,
                preprocessor=preprocessor, seed=0,
            )
            rec = {"run": os.path.basename(run_dir), "ckpt": st,
                   "per_task": {str(k): v for k, v in per_task.items()}}
            results[st] = per_task
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"[done] {st}: {per_task}", flush=True)


if __name__ == "__main__":
    main()
