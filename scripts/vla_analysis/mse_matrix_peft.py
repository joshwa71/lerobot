#!/usr/bin/env python3
"""MSE forgetting matrix for PEFT (naive sequential LoRA) runs — E58 addendum 5.

Adapter-swap sibling of mse_matrix2.py (which partial-loads memory slot tensors):
the policy is built ONCE from the first per-task checkpoint via the factory's
use_peft path (base + adapters; frozen is fine, eval only), then each checkpoint's
adapter_model.safetensors is swapped in with peft.set_peft_model_state_dict (handles
the saved-key -> live-key ".default" adapter-name mapping) and scored with the
trainer's own paired-noise _eval_loss_on_seen_tasks (seed=0, n_batches=16 == E39).

Guard: after each swap, L1(lora_B) of the live model must match the file's within
2% (bf16-cast slack). A silent no-op swap would otherwise produce a fake flat
matrix — the one failure mode this instrument must never emit.

Invoke like mse_matrix2.py but with --policy.use_peft=true and --policy.path at the
FIRST per-task checkpoint. Env: MSEMAT_RUN_DIR, MSEMAT_STEPS, MSEMAT_OUT.
"""
import json
import os

from accelerate import Accelerator
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_sequential_train import (
    SequentialOnlineConfig,
    _collect_task_index_to_name,
    _eval_loss_on_seen_tasks,
)


def _l1_lora_b_file(sd):
    return sum(t.float().abs().sum().item() for k, t in sd.items() if "lora_B" in k)


def _l1_lora_b_model(model):
    return sum(
        p.detach().float().abs().sum().item()
        for n, p in model.named_parameters()
        if ".lora_B." in n
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
    with open(out_path, "a") as fh:
        for st in steps:
            sd_path = os.path.join(run_dir, "checkpoints", st, "pretrained_model", "adapter_model.safetensors")
            sd = load_file(sd_path)
            l1_file = _l1_lora_b_file(sd)
            load_result = set_peft_model_state_dict(unwrapped, sd)
            unexpected = getattr(load_result, "unexpected_keys", [])
            if unexpected:
                raise RuntimeError(f"{st}: unexpected keys in adapter swap: {list(unexpected)[:5]}")
            l1_model = _l1_lora_b_model(unwrapped)
            rel = abs(l1_model - l1_file) / max(l1_file, 1e-9)
            print(f"[swap] {st}: {len(sd)} tensors, L1(lora_B) file={l1_file:.4e} model={l1_model:.4e} rel={rel:.2%}", flush=True)
            if rel > 0.02:
                raise RuntimeError(f"{st}: adapter swap L1 mismatch {rel:.2%} — swap did not land, refusing to score")
            del sd
            per_task = _eval_loss_on_seen_tasks(
                policy, accelerator, dataset, task_index_to_name, [0, 1, 2, 3, 4],
                batch_size=32, num_workers=4, device=device, n_batches=16,
                preprocessor=preprocessor, seed=0,
            )
            rec = {"run": os.path.basename(run_dir), "ckpt": st,
                   "per_task": {str(k): v for k, v in per_task.items()}}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[done] {st}: {per_task}", flush=True)


if __name__ == "__main__":
    main()
