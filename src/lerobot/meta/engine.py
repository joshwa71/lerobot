#!/usr/bin/env python

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Iterator

import torch

from lerobot.datasets.factory import make_dataset
from lerobot.meta.algorithms.reptile import Reptile
from lerobot.meta.configs import MetaTrainConfig, ReptileConfig
from lerobot.meta.tasks import (
    build_task_dataloader,
    default_task_split,
    cycle,
    get_episode_indices_for_task,
)
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.adapters.lora import attach_lora
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.train_utils import get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint
from lerobot.rl.wandb_utils import WandBLogger


@dataclass
class MetaEngine:
    cfg: MetaTrainConfig

    def setup(self):
        logging.info("Creating dataset")
        # Build a small shim to reuse datasets.factory.make_dataset
        class _DSCfg:
            def __init__(self, dataset, policy):
                self.dataset = dataset
                self.policy = policy
                self.num_workers = 8
        ds = make_dataset(_DSCfg(self.cfg.dataset, self.cfg.policy))
        logging.info(
            "Dataset ready: frames=%s episodes=%s tasks=%s cameras=%s",
            ds.num_frames,
            ds.num_episodes,
            ds.meta.total_tasks,
            ds.meta.camera_keys,
        )
        self.dataset = ds

        # policy + processors
        logging.info("Creating policy")
        policy = make_policy(self.cfg.policy, ds_meta=ds.meta)
        attach_lora(policy, self.cfg.lora)
        self.policy = policy

        # Always build fresh processors from dataset stats unless resuming
        logging.info("Building processors (pre/post)...")
        # Mirror standard training overrides: ensure device placement and dataset stats are used
        device = self.cfg.policy.device
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
            policy_cfg=self.cfg.policy,
            pretrained_path=self.cfg.policy.pretrained_path,
            preprocessor_overrides=pre_overrides,
            postprocessor_overrides=post_overrides,
        )
        logging.info("Processors ready")
        self.preproc = preproc
        self.postproc = postproc

        # Meta algorithm
        if isinstance(self.cfg.algo, ReptileConfig):
            self.algo = Reptile(self.cfg.algo)
            # Bridge verbose flag into algo for inner-step logging
            setattr(self.algo, "_verbose_log", bool(self.cfg.verbose_log))
        else:
            raise NotImplementedError("Only Reptile is implemented for now")

        # Optim for outer loop: reuse policy presets but on trainable params only
        # Reuse optimizer/scheduler factory by making a lightweight cfg shim
        class _Shim:
            def __init__(self, steps, optimizer, scheduler, use_policy_training_preset=False):
                self.steps = steps
                self.optimizer = optimizer
                self.scheduler = scheduler
                self.use_policy_training_preset = use_policy_training_preset

        self.outer_cfg = _Shim(
            steps=10**9,
            optimizer=self.cfg.policy.get_optimizer_preset(),
            scheduler=self.cfg.policy.get_scheduler_preset(),
            use_policy_training_preset=False,
        )
        logging.info("Creating outer optimizer/scheduler...")
        self.optimizer, self.lr_scheduler = make_optimizer_and_scheduler(self.outer_cfg, self.policy)
        # Outer optimizer is not used by Reptile; keep for future algos
        logging.info("Optimizer ready: %s param groups", len(self.optimizer.param_groups))

        # WandB logger (follow existing pattern)
        if self.cfg.wandb.enable and self.cfg.wandb.project:
            # Reuse TrainPipelineConfig-like shim for logger
            class _TrainShim:
                def __init__(self, meta_cfg: MetaTrainConfig):
                    self.dataset = meta_cfg.dataset
                    self.env = meta_cfg.env
                    self.policy = meta_cfg.policy
                    self.output_dir = meta_cfg.output_dir
                    self.job_name = meta_cfg.job_name or f"meta_{meta_cfg.policy.type}"
                    self.resume = False
                    self.seed = 1000
                    self.wandb = meta_cfg.wandb
                def to_dict(self):
                    return {
                        "meta": {
                            "algo": type(self).__name__,
                        }
                    }
            self.wandb_logger = WandBLogger(_TrainShim(self.cfg))
            logging.info("WandB initialized and logging enabled.")
        else:
            self.wandb_logger = None

        # Task split
        if self.cfg.train_tasks is None or self.cfg.eval_tasks is None:
            split = default_task_split(self.dataset)
            self.train_tasks = split.train
            self.eval_tasks = split.eval
        else:
            self.train_tasks = self.cfg.train_tasks
            self.eval_tasks = self.cfg.eval_tasks
        logging.info("Task split -> train=%s eval=%s", len(self.train_tasks), len(self.eval_tasks))

    def build_task_iters(self, tasks: list[int], frames_per_task: int, batch_size: int, shuffle: bool) -> dict[int, Iterator]:
        iters = {}
        for t in tasks:
            ep_idxs = get_episode_indices_for_task(self.dataset, t)
            logging.info(
                "Building DataLoader for task=%s | episodes=%s | frames_per_task=%s | batch_size=%s | num_workers=%s",
                t,
                len(ep_idxs),
                frames_per_task,
                batch_size,
                4,
            )
            loader = build_task_dataloader(
                self.dataset,
                task_index=t,
                frames_per_task=frames_per_task,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
            )
            # Create an infinite iterator to avoid StopIteration during inner adaptation
            iters[t] = cycle(loader)
        return iters

    def train(self, total_outer_steps: int, batch_size: int = 8, log_freq: int = 100, save_freq: int = 1000):
        self.setup()
        policy = self.policy
        preproc = self.preproc

        support_iters = self.build_task_iters(self.train_tasks, self.cfg.support_frames_per_task, batch_size, True)
        # Defer query loader creation to when needed (FOMAML) to avoid memory pressure

        meters = {
            "meta_update_s": AverageMeter("updt_s", ":.3f"),
        }
        tracker = MetricsTracker(batch_size, self.dataset.num_frames, self.dataset.num_episodes, meters, initial_step=0)

        for step in range(1, total_outer_steps + 1):
            # sample tasks
            tasks = random.sample(self.train_tasks, k=min(self.cfg.tasks_per_outer_step, len(self.train_tasks)))
            if self.cfg.verbose_log and (step <= 3 or step % max(1, self.cfg.log_freq) == 0):
                logging.info("Outer step %s: sampled tasks=%s", step, tasks)
            task_results = []
            inner_losses = []
            for t in tasks:
                if self.cfg.verbose_log and step <= 3:
                    logging.info("Adapting on task=%s for %s inner steps", t, self.cfg.inner_steps)
                res = self.algo.adapt(
                    model=policy,
                    support_iter=support_iters[t],
                    steps=self.cfg.inner_steps,
                    inner_cfg=self.cfg.inner_opt,
                    preprocessor=preproc,
                )
                task_results.append(res)
                if res.metrics and "inner_avg_loss" in res.metrics:
                    inner_losses.append(res.metrics["inner_avg_loss"])

            self.algo.outer_step(policy, task_results)

            is_log_step = log_freq > 0 and step % log_freq == 0
            if is_log_step:
                logging.info(tracker)
                avg_inner = float(sum(inner_losses) / len(inner_losses)) if inner_losses else None
                tracker.reset_averages()
                if self.wandb_logger:
                    log_payload = {"outer_step": step}
                    if avg_inner is not None:
                        log_payload["inner_avg_loss"] = avg_inner
                    self.wandb_logger.log_dict(log_payload, step=step)
            # Lightweight heartbeat each step
            if self.cfg.verbose_log and step % 10 == 0:
                logging.info("Heartbeat outer_step=%s", step)

            if step % save_freq == 0:
                checkpoint_dir = get_step_checkpoint_dir(self.cfg.output_dir, total_outer_steps, step)
                save_checkpoint(
                    checkpoint_dir,
                    step,
                    self._to_train_like_cfg(total_outer_steps),
                    policy,
                    self.optimizer,
                    self.lr_scheduler,
                    self.preproc,
                    self.postproc,
                )
                update_last_checkpoint(checkpoint_dir)

            # Periodic meta-eval (disabled when eval_freq == 0)
            if self.cfg.eval_freq > 0 and (step % self.cfg.eval_freq == 0):
                self._run_meta_eval(step, total_outer_steps)

            tracker.step()

    def _run_meta_eval(self, step: int, total_outer_steps: int):
        from lerobot.scripts.lerobot_eval import eval_policy_all  # Lazy import to avoid gym deps when eval disabled
        # 1) Save checkpoint to return after eval; already saved above in train every save_freq.
        # For safety, save a temporary eval checkpoint too.
        ckpt_dir = get_step_checkpoint_dir(self.cfg.output_dir, total_outer_steps, step)
        save_checkpoint(
            ckpt_dir,
            step,
            self._to_train_like_cfg(total_outer_steps),
            self.policy,
            self.optimizer,
            self.lr_scheduler,
            self.preproc,
            self.postproc,
        )
        update_last_checkpoint(ckpt_dir)

        # 2) and 3) Evaluate on held-out tasks: adapt on support, then roll out in LIBERO envs
        # Build support loaders for eval tasks
        support_iters_eval = self.build_task_iters(self.eval_tasks, self.cfg.support_frames_per_task, batch_size=self.cfg.eval.batch_size, shuffle=True)

        # Per-task: clone meta-weights -> adapt -> build env -> eval
        per_task_results = {}
        for t in self.eval_tasks:
            theta = {n: p.detach().clone() for n, p in self.policy.named_parameters() if p.requires_grad}

            # Inner adaptation on held-out task
            res = self.algo.adapt(
                model=self.policy,
                support_iter=support_iters_eval[t],
                steps=self.cfg.inner_steps,
                inner_cfg=self.cfg.inner_opt,
                preprocessor=self.preproc,
            )
            # Apply adapted deltas
            with torch.no_grad():
                for n, p in self.policy.named_parameters():
                    if p.requires_grad and n in res.delta:
                        p.add_(res.delta[n].to(p.device))

            # Build a single-task LIBERO env for this task id
            # Restrict creation to the requested task_id to avoid unnecessary envs
            if self.cfg.env is not None and self.cfg.env.type == "libero":
                from lerobot.envs.libero import create_libero_envs
                import gymnasium as gym
                env_cfg = self.cfg.env
                envs_subset = create_libero_envs(
                    task=env_cfg.task,
                    n_envs=1,
                    camera_name=env_cfg.camera_name,
                    init_states=env_cfg.init_states,
                    gym_kwargs={**env_cfg.gym_kwargs, "task_ids": [t]},
                    env_cls=gym.vector.SyncVectorEnv,
                )
                try:
                    info = eval_policy_all(
                        envs=envs_subset,
                        policy=self.policy,
                        preprocessor=self.preproc,
                        postprocessor=self.postproc,
                        n_episodes=self.cfg.eval.n_episodes,
                        max_episodes_rendered=4,
                        videos_dir=self.cfg.output_dir / "eval" / f"videos_step_{step:06d}",
                        start_seed=self.cfg.wandb.seed if hasattr(self.cfg.wandb, "seed") else None,
                        max_parallel_tasks=1,
                    )
                    per_task_results[t] = info["overall"]
                finally:
                    from lerobot.envs.utils import close_envs
                    close_envs(envs_subset)
            else:
                per_task_results[t] = {"note": "Env not provided; skipped real sim eval."}

            # Restore meta-weights for next eval task
            with torch.no_grad():
                for n, p in self.policy.named_parameters():
                    if p.requires_grad and n in theta:
                        p.copy_(theta[n])

        # 5) Log results (stdout for now; could be wandb if configured)
        import json
        print(f"[META-EVAL step={step}] results:\n" + json.dumps(per_task_results, indent=2))
        if self.wandb_logger:
            # Log aggregate over eval tasks if available
            agg = {}
            try:
                avg_sum = [v.get("avg_sum_reward") for v in per_task_results.values() if isinstance(v, dict) and "avg_sum_reward" in v]
                pc_succ = [v.get("pc_success") for v in per_task_results.values() if isinstance(v, dict) and "pc_success" in v]
                if len(avg_sum) > 0:
                    agg["meta_eval/avg_sum_reward"] = float(sum(avg_sum) / len(avg_sum))
                if len(pc_succ) > 0:
                    agg["meta_eval/pc_success"] = float(sum(pc_succ) / len(pc_succ))
            except Exception:
                pass
            # Always log the step for eval timeline
            agg["outer_step"] = step
            self.wandb_logger.log_dict(agg, step=step, mode="eval")

        # 6) Reload checkpoint: already restored meta-weights in loop, but ensure exact checkpoint state is preserved
        # Optimizer/scheduler states are already preserved in training loop; no mutation here

    def _to_train_like_cfg(self, total_steps: int):
        # Build a pseudo TrainPipelineConfig for checkpoint compatibility
        from lerobot.configs.train import TrainPipelineConfig
        return TrainPipelineConfig(
            dataset=self.cfg.dataset,
            env=None,
            policy=self.cfg.policy,
            output_dir=self.cfg.output_dir,
            job_name=self.cfg.job_name or f"meta_{self.cfg.policy.type}",
            resume=False,
            seed=1000,
            num_workers=4,
            batch_size=8,
            steps=total_steps,
            eval_freq=0,
            log_freq=0,
            save_checkpoint=True,
            save_freq=1000,
            use_policy_training_preset=True,
            optimizer=self.cfg.policy.get_optimizer_preset(),
            scheduler=self.cfg.policy.get_scheduler_preset(),
            eval=self.cfg.eval,
            wandb=self.cfg.wandb,
            lora=self.cfg.lora,
        )


