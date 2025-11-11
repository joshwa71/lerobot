#!/usr/bin/env python

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Iterator, List

import torch
import copy
from concurrent.futures import ThreadPoolExecutor

from lerobot.datasets.factory import make_dataset
from lerobot.meta.algorithms.reptile import Reptile
from lerobot.meta.algorithms.base import TaskResult
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
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("Creating dataset")
        # Build a small shim to reuse datasets.factory.make_dataset
        class _DSCfg:
            def __init__(self, dataset, policy, num_workers):
                self.dataset = dataset
                self.policy = policy
                self.num_workers = num_workers
        ds = make_dataset(_DSCfg(self.cfg.dataset, self.cfg.policy, self.cfg.num_workers))
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

        # Parallel adaptation setup
        try:
            first_param = next(self.policy.parameters())
            self.master_device = str(first_param.device)
        except StopIteration:
            self.master_device = str(self.cfg.policy.device)
        self.devices: List[str] = self._detect_devices()
        self.parallel_enabled = self._is_parallel_enabled()
        self.preproc_by_device: dict[str, object] = {self.master_device: self.preproc}
        self.replicas: dict[str, torch.nn.Module] = {}
        if self.parallel_enabled and len(self.devices) > 1:
            self._build_replicas_and_preprocessors()

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
            if len(ep_idxs) == 0:
                logging.warning("No episodes found for task_id=%s. Skipping this task.", t)
                continue
            logging.info(
                "Building DataLoader for task=%s | episodes=%s | frames_per_task=%s | batch_size=%s | num_workers=%s",
                t,
                len(ep_idxs),
                frames_per_task,
                batch_size,
                self.cfg.num_workers,
            )
            loader = build_task_dataloader(
                self.dataset,
                task_index=t,
                frames_per_task=frames_per_task,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.cfg.num_workers,
                prefetch_factor=self.cfg.prefetch_factor,
            )
            iters[t] = cycle(loader)
        return iters

    def _detect_devices(self) -> List[str]:
        # Respect user-specified device ids when provided
        if self.cfg.parallel.device_ids is not None:
            ids = [int(i) for i in self.cfg.parallel.device_ids]
            if len(ids) == 0:
                return [self.master_device]
            return [f"cuda:{i}" for i in ids]
        # Fallback to all visible CUDA devices when available
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if n > 0:
                return [f"cuda:{i}" for i in range(n)]
        return [self.master_device]

    def _is_parallel_enabled(self) -> bool:
        mode = (self.cfg.parallel.enable or "auto").lower()
        if mode == "off":
            return False
        if mode == "on":
            return True
        # auto
        return torch.cuda.is_available() and torch.cuda.device_count() > 1

    def _copy_trainable_parameters(self, src: torch.nn.Module, dst: torch.nn.Module) -> None:
        with torch.no_grad():
            src_named = dict((n, p) for n, p in src.named_parameters() if p.requires_grad)
            for n, p in dst.named_parameters():
                if p.requires_grad and n in src_named:
                    p.copy_(src_named[n].to(p.device, non_blocking=True))

    def _build_replicas_and_preprocessors(self) -> None:
        # Build per-device preprocessors and policy replicas (excluding master, which uses self.policy/self.preproc)
        logging.info("Setting up %s devices for parallel adaptation: %s", len(self.devices), self.devices)
        # Build preprocessors for non-master devices
        for dev in self.devices:
            pre_overrides = {
                "device_processor": {"device": dev},
                "normalizer_processor": {
                    "stats": self.dataset.meta.stats,
                    "features": {**self.policy.config.input_features, **self.policy.config.output_features},
                    "norm_map": self.policy.config.normalization_mapping,
                },
            }
            post_overrides = {
                "unnormalizer_processor": {
                    "stats": self.dataset.meta.stats,
                    "features": self.policy.config.output_features,
                    "norm_map": self.policy.config.normalization_mapping,
                },
            }
            pre_dev, _ = make_pre_post_processors(
                policy_cfg=self.cfg.policy,
                pretrained_path=self.cfg.policy.pretrained_path,
                preprocessor_overrides=pre_overrides,
                postprocessor_overrides=post_overrides,
            )
            self.preproc_by_device[dev] = pre_dev

        # Create replicas for all devices (including master) by deep-copying the master policy and moving to device
        for dev in self.devices:
            replica = copy.deepcopy(self.policy)
            replica.to(dev)
            self._copy_trainable_parameters(self.policy, replica)
            self.replicas[dev] = replica

    def _adapt_on_device(self, task_id: int, device: str, support_iter: Iterator) -> tuple[dict[str, torch.Tensor], float]:
        # Choose model and preprocessor for the target device
        model = self.replicas[device]
        self._copy_trainable_parameters(self.policy, model)
        preproc = self.preproc_by_device[device]
        res = self.algo.adapt(
            model=model,
            support_iter=support_iter,
            steps=self.cfg.inner_steps,
            inner_cfg=self.cfg.inner_opt,
            preprocessor=preproc,
        )
        avg_loss = res.metrics["inner_avg_loss"] if res.metrics and "inner_avg_loss" in res.metrics else None
        return res.delta, float(avg_loss) if avg_loss is not None else float("nan")

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
            if self.parallel_enabled and len(self.devices) > 1 and len(tasks) > 1:
                # Determine concurrency
                max_conc = self.cfg.parallel.max_concurrent if self.cfg.parallel.max_concurrent and self.cfg.parallel.max_concurrent > 0 else None
                device_pool = self.devices
                if max_conc is not None:
                    device_pool = device_pool[: max(1, min(max_conc, len(device_pool)))]
                # Execute in waves to avoid assigning multiple tasks to the same device concurrently
                start = 0
                while start < len(tasks):
                    wave_tasks = tasks[start : start + len(device_pool)]
                    with ThreadPoolExecutor(max_workers=len(wave_tasks)) as executor:
                        futures = []
                        for i, t in enumerate(wave_tasks):
                            device = device_pool[i]
                            futures.append(executor.submit(self._adapt_on_device, t, device, support_iters[t]))
                        for fut in futures:
                            delta, avg_loss = fut.result()
                            metrics = {"inner_avg_loss": avg_loss} if not (avg_loss != avg_loss) else None
                            task_results.append(TaskResult(delta=delta, metrics=metrics))
                            if not (avg_loss != avg_loss):  # not NaN
                                inner_losses.append(avg_loss)
                    start += len(device_pool)
            else:
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
        inner_losses_eval: list[float] = []
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
            # Track inner-loop average loss during eval adaptation when available
            if res.metrics and "inner_avg_loss" in res.metrics:
                inner_losses_eval.append(float(res.metrics["inner_avg_loss"]))
            # Apply adapted deltas
            with torch.no_grad():
                for n, p in self.policy.named_parameters():
                    if p.requires_grad and n in res.delta:
                        p.add_(res.delta[n].to(p.device))

            # Map dataset task_id to env task_id if mapping is provided
            env_task_id = t
            if self.cfg.dataset_to_env_task_mapping is not None:
                if t not in self.cfg.dataset_to_env_task_mapping:
                    logging.warning(f"Dataset task_id {t} not found in dataset_to_env_task_mapping; skipping eval.")
                    # Restore meta-weights and continue
                    with torch.no_grad():
                        for n, p in self.policy.named_parameters():
                            if p.requires_grad and n in theta:
                                p.copy_(theta[n])
                    continue
                env_task_id = self.cfg.dataset_to_env_task_mapping[t]
                logging.info(f"Mapped dataset task_id {t} to env task_id {env_task_id}")

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
                    gym_kwargs={**env_cfg.gym_kwargs, "task_ids": [env_task_id]},
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
            avg_sum = [v.get("avg_sum_reward") for v in per_task_results.values() if isinstance(v, dict) and "avg_sum_reward" in v]
            pc_succ = [v.get("pc_success") for v in per_task_results.values() if isinstance(v, dict) and "pc_success" in v]
            if len(avg_sum) > 0:
                agg["meta_eval/avg_sum_reward"] = float(sum(avg_sum) / len(avg_sum))
            if len(pc_succ) > 0:
                agg["meta_eval/pc_success"] = float(sum(pc_succ) / len(pc_succ))
            if len(inner_losses_eval) > 0:
                agg["meta_eval/inner_avg_loss"] = float(sum(inner_losses_eval) / len(inner_losses_eval))

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
            num_workers=self.cfg.num_workers,
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


