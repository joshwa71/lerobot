#!/usr/bin/env python
"""
Test how quickly a meta-trained (or base) policy can adapt to a target task.

This script repeatedly adapts the policy on a task and evaluates it,
continuing until the policy reaches a target success rate threshold.
This helps understand how "far off" a meta-prior is from being able to learn a task.

Usage:
    lerobot-meta-prior-test \
        --policy.path=outputs/train/meta_checkpoint/pretrained_model \
        --dataset.repo_id=outputs/libero \
        --task_ids=[0,1,2] \
        --target_success_rate=0.1 \
        --inner_steps_per_eval=10 \
        --max_total_steps=1000 \
        --env.type=libero \
        --output_dir=outputs/prior_test
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import re
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.multiprocessing as mp
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors import safe_open

from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.lora import LoraAttachConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import make_dataset
from lerobot.meta.configs import InnerOptConfig
from lerobot.meta.tasks import build_task_dataloader, cycle, get_episode_indices_for_task
from lerobot.policies.adapters.lora import attach_lora
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import init_logging

mp.set_start_method("spawn", force=True)


def infer_lora_config_from_checkpoint(model_file: Path) -> LoraAttachConfig | None:
    """Infer LoRA configuration from a saved checkpoint by inspecting the weights."""
    lora_params = {}
    try:
        with safe_open(str(model_file), framework="pt") as f:
            for key in f.keys():
                if ".lora_A" in key or ".lora_B" in key:
                    lora_params[key] = f.get_tensor(key).shape
    except Exception as e:
        logging.warning("Could not read model file to infer LoRA config: %s", e)
        logging.warning("Falling back to no LoRA. If the model has LoRA, specify --lora.* manually.")
        return None

    if not lora_params:
        logging.info("No LoRA parameters found in checkpoint - not using LoRA")
        return None

    ranks = set()
    target_modules = set()
    for key, shape in lora_params.items():
        if ".lora_A" in key:
            ranks.add(shape[1])
            module_path = key.rsplit(".lora_A", 1)[0]
            target_modules.add(module_path)

    if len(ranks) > 1:
        logging.warning("Multiple LoRA ranks detected: %s. Using the most common.", ranks)
    r = max(ranks, key=lambda x: sum(1 for k, s in lora_params.items() if ".lora_A" in k and s[1] == x))

    regex_patterns = []
    for module in target_modules:
        parts = module.split(".")
        if len(parts) >= 2:
            pattern = re.escape(parts[-2]) + r"\." + re.escape(parts[-1]) + "$"
        else:
            pattern = re.escape(parts[-1]) + "$"
        regex_patterns.append(pattern)

    unique_patterns = list(set(regex_patterns))

    logging.info("Inferred LoRA config: r=%d, target_modules=%s", r, unique_patterns[:5])

    return LoraAttachConfig(
        enable=True,
        r=r,
        alpha=float(r * 2),
        dropout=0.0,
        train_lora_only=True,
        target_modules_regex=unique_patterns,
    )


@dataclass
class MetaPriorTestConfig:
    dataset: DatasetConfig
    policy: PreTrainedConfig | None = None
    env: envs.EnvConfig | None = None
    output_dir: Path | None = None

    task_ids: list[int] = field(default_factory=lambda: [0])
    target_success_rate: float = 0.1
    inner_steps_per_eval: int = 10
    max_total_steps: int = 1000
    batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int | None = None
    frames_per_task: int = 50000

    dataset_to_env_task_mapping: dict[int, int] | None = None

    inner_opt: InnerOptConfig = field(default_factory=InnerOptConfig)
    lora: LoraAttachConfig | None = None

    eval: EvalConfig = field(default_factory=lambda: EvalConfig(batch_size=1, n_episodes=10))
    wandb: WandBConfig = field(default_factory=WandBConfig)

    def validate(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError(
                "Policy configuration is required. Please provide --policy.path "
                "to load a pretrained policy."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def run_prior_test(cfg: MetaPriorTestConfig):
    """Run the prior test for each task."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    policy_weight_dir = cfg.policy.pretrained_path
    if not policy_weight_dir:
        raise ValueError("--policy.path must point to a pretrained_model directory.")
    policy_weight_dir = Path(policy_weight_dir)
    model_file = policy_weight_dir / SAFETENSORS_SINGLE_FILE
    if not model_file.is_file():
        raise FileNotFoundError(f"Expected model weights at {model_file}, but file does not exist.")

    lora_cfg = cfg.lora
    if lora_cfg is None:
        lora_cfg = infer_lora_config_from_checkpoint(model_file)
    if lora_cfg is None:
        lora_cfg = LoraAttachConfig(enable=False)

    logging.info("Loading dataset from %s", cfg.dataset.repo_id)

    class _DSCfg:
        def __init__(self, dataset, policy, num_workers):
            self.dataset = dataset
            self.policy = policy
            self.num_workers = num_workers

    ds = make_dataset(_DSCfg(cfg.dataset, cfg.policy, cfg.num_workers))
    logging.info(
        "Dataset ready: frames=%s episodes=%s tasks=%s",
        ds.num_frames,
        ds.num_episodes,
        ds.meta.total_tasks,
    )

    cfg.policy.pretrained_path = None
    logging.info("Creating policy from base weights (LoRA will be attached)")
    policy = make_policy(cfg.policy, ds_meta=ds.meta)
    attach_lora(policy, lora_cfg)

    policy_cls: type[PreTrainedPolicy] = type(policy)
    map_location = getattr(cfg.policy, "device", None) or "cpu"
    policy_cls._load_as_safetensor(
        model=policy,
        model_file=str(model_file),
        map_location=map_location,
        strict=False,
    )
    logging.info("Loaded meta-trained weights from %s", model_file)

    device = cfg.policy.device
    pre_overrides = {
        "device_processor": {"device": device},
        "normalizer_processor": {
            "stats": ds.meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
    }
    post_overrides = {
        "unnormalizer_processor": {
            "stats": ds.meta.stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        },
    }
    preproc, postproc = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=policy_weight_dir,
        preprocessor_overrides=pre_overrides,
        postprocessor_overrides=post_overrides,
    )
    logging.info("Processors ready")

    results = {}

    for task_id in cfg.task_ids:
        logging.info("=" * 60)
        logging.info("Testing task_id=%s", task_id)
        logging.info("=" * 60)

        ep_idxs = get_episode_indices_for_task(ds, task_id)
        if len(ep_idxs) == 0:
            logging.warning("No episodes found for task_id=%s. Skipping.", task_id)
            results[task_id] = {"error": "no episodes found"}
            continue

        theta_init = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}

        loader = build_task_dataloader(
            ds,
            task_index=task_id,
            frames_per_task=cfg.frames_per_task,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
        )
        support_iter = cycle(loader)

        task_result = {
            "task_id": task_id,
            "target_success_rate": cfg.target_success_rate,
            "inner_steps_per_eval": cfg.inner_steps_per_eval,
            "max_total_steps": cfg.max_total_steps,
            "steps_to_target": None,
            "final_success_rate": 0.0,
            "reached_target": False,
            "eval_history": [],
        }

        total_steps = 0
        current_success_rate = 0.0

        current_success_rate = evaluate_task(
            cfg, policy, preproc, postproc, task_id, total_steps
        )
        task_result["eval_history"].append({
            "steps": total_steps,
            "success_rate": current_success_rate,
        })
        logging.info(
            "[task=%s] Initial eval: success_rate=%.2f%%",
            task_id, current_success_rate * 100
        )

        if current_success_rate >= cfg.target_success_rate:
            task_result["reached_target"] = True
            task_result["steps_to_target"] = 0
            task_result["final_success_rate"] = current_success_rate
            results[task_id] = task_result
            logging.info("[task=%s] Already at target!", task_id)
            with torch.no_grad():
                for n, p in policy.named_parameters():
                    if p.requires_grad and n in theta_init:
                        p.copy_(theta_init[n])
            continue

        params = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.inner_opt.lr,
            weight_decay=cfg.inner_opt.weight_decay,
            foreach=True,
        )

        if params:
            param_device = params[0].device
        else:
            param_device = torch.device("cpu")
        device_type = "cuda" if str(param_device).startswith("cuda") else "cpu"
        use_amp = device_type == "cuda" and bool(
            getattr(getattr(policy, "config", None), "use_amp", False)
        )
        amp_dtype = torch.bfloat16 if use_amp else None

        policy.train()

        while total_steps < cfg.max_total_steps and current_success_rate < cfg.target_success_rate:
            loss_sum = 0.0
            for _ in range(cfg.inner_steps_per_eval):
                batch = next(support_iter)
                batch = preproc(batch)
                autocast_cm = (
                    torch.autocast(device_type=device_type, dtype=amp_dtype)
                    if use_amp
                    else nullcontext()
                )
                with autocast_cm:
                    loss, _ = policy.forward(batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.inner_opt.grad_clip_norm and cfg.inner_opt.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        params, cfg.inner_opt.grad_clip_norm, error_if_nonfinite=False
                    )
                optimizer.step()
                loss_sum += loss.detach().item()

            total_steps += cfg.inner_steps_per_eval
            inner_loss = loss_sum / cfg.inner_steps_per_eval

            logging.info(
                "[task=%s] steps=%d | inner_loss=%.6f",
                task_id, total_steps, inner_loss
            )

            current_success_rate = evaluate_task(
                cfg, policy, preproc, postproc, task_id, total_steps
            )
            task_result["eval_history"].append({
                "steps": total_steps,
                "success_rate": current_success_rate,
                "inner_loss": inner_loss,
            })

            logging.info(
                "[task=%s] steps=%d | inner_loss=%.6f | success_rate=%.2f%%",
                task_id, total_steps, inner_loss, current_success_rate * 100
            )

            if current_success_rate >= cfg.target_success_rate:
                task_result["reached_target"] = True
                task_result["steps_to_target"] = total_steps
                break

        task_result["final_success_rate"] = current_success_rate
        if not task_result["reached_target"]:
            task_result["steps_to_target"] = None
        results[task_id] = task_result

        if task_result["reached_target"]:
            logging.info(
                "[task=%s] REACHED target %.0f%% after %d steps",
                task_id, cfg.target_success_rate * 100, task_result["steps_to_target"]
            )
        else:
            logging.info(
                "[task=%s] DID NOT REACH target %.0f%% after %d steps (final=%.2f%%)",
                task_id, cfg.target_success_rate * 100, cfg.max_total_steps,
                current_success_rate * 100
            )

        with torch.no_grad():
            for n, p in policy.named_parameters():
                if p.requires_grad and n in theta_init:
                    p.copy_(theta_init[n])

    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    for task_id, res in results.items():
        if "error" in res:
            logging.info("Task %s: ERROR - %s", task_id, res["error"])
        elif res["reached_target"]:
            logging.info(
                "Task %s: REACHED target in %d steps (final=%.2f%%)",
                task_id, res["steps_to_target"], res["final_success_rate"] * 100
            )
        else:
            logging.info(
                "Task %s: DID NOT REACH target after %d steps (final=%.2f%%)",
                task_id, cfg.max_total_steps, res["final_success_rate"] * 100
            )

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "prior_test_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logging.info("Results saved to %s", results_file)

    return results


def evaluate_task(
    cfg: MetaPriorTestConfig,
    policy,
    preproc,
    postproc,
    task_id: int,
    total_steps: int,
) -> float:
    """Evaluate the policy on a single task and return success rate."""
    from lerobot.scripts.lerobot_eval import eval_policy_all

    if cfg.env is None or cfg.env.type != "libero":
        logging.warning("No LIBERO env configured; returning 0.0 for success rate.")
        return 0.0

    import gymnasium as gym

    from lerobot.envs.libero import _get_suite, create_libero_envs
    from lerobot.envs.utils import close_envs

    env_task_id = task_id
    if cfg.dataset_to_env_task_mapping is not None:
        if task_id not in cfg.dataset_to_env_task_mapping:
            logging.warning(
                "Dataset task_id %s not found in dataset_to_env_task_mapping; skipping eval.",
                task_id
            )
            return 0.0
        env_task_id = cfg.dataset_to_env_task_mapping[task_id]
        logging.info("Mapped dataset task_id %s -> env task_id %s", task_id, env_task_id)

    env_cfg = cfg.env
    try:
        suite = _get_suite(env_cfg.task)
        lib_task = suite.tasks[env_task_id]
        logging.debug(
            "[eval] dataset_task_id=%s env_task_id=%s env_task_name=%r",
            task_id, env_task_id, getattr(lib_task, "name", None)
        )
    except Exception as e:
        logging.warning("Failed to resolve LIBERO task for env_task_id=%s: %s", env_task_id, e)
        return 0.0

    envs = create_libero_envs(
        task=env_cfg.task,
        n_envs=1,
        camera_name=env_cfg.camera_name,
        init_states=env_cfg.init_states,
        gym_kwargs={**env_cfg.gym_kwargs, "task_ids": [env_task_id]},
        env_cls=gym.vector.SyncVectorEnv,
    )

    videos_dir = None
    max_episodes_rendered = 0
    if cfg.output_dir:
        videos_dir = Path(cfg.output_dir) / "videos" / f"task_{task_id}" / f"step_{total_steps:06d}"
        videos_dir.mkdir(parents=True, exist_ok=True)
        max_episodes_rendered = min(4, cfg.eval.n_episodes)

    try:
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            preprocessor=preproc,
            postprocessor=postproc,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=max_episodes_rendered,
            videos_dir=videos_dir,
            start_seed=42,
            max_parallel_tasks=1,
        )
        success_rate = info["overall"].get("pc_success", 0.0) / 100.0
        if videos_dir and max_episodes_rendered > 0:
            logging.info("[task=%s] Videos saved to %s", task_id, videos_dir)
    except Exception as e:
        logging.error("Evaluation failed for task_id=%s: %s", task_id, e)
        success_rate = 0.0
    finally:
        close_envs(envs)

    return success_rate


@parser.wrap()
def meta_prior_test(cfg: MetaPriorTestConfig):
    cfg.validate()
    run_prior_test(cfg)


def main():
    init_logging()
    meta_prior_test()


if __name__ == "__main__":
    main()

