#!/usr/bin/env python3
"""RETAIN baseline (Yadav, Zhou, Wagenmaker, Pertsch, Levine; ICLR 2026, arXiv 2512.08333),
run under OUR continual protocol (E67).

Method, Eq. (4) of the paper (continual task adaptation):
    theta~_n = (1 - alpha) * theta~_{n-1} + alpha * theta_ft,n
where theta_ft,n is a FULL fine-tune of theta~_{n-1} on task n only (their task-FT variant; the
co-FT variant replays pretraining data, which our no-replay constraint forbids), and alpha is
fixed at 0.5 at every boundary (the paper's continual setting, App. A.8.3: "the merging weight is
always fixed at 50% task-FT, 50% base model"). Interpolation is uniform over every parameter
(their Eq. 2; the modality-specific variant was not used in their continual run).

Protocol (identical to every other sequential row): stage-1 LIBERO-90 base, 10 LIBERO-10 tasks
in dataset order, 5,000 steps/task, effective batch 32, no replay, no task identity, optimizer
re-initialised and LR schedule reset each task. The per-task fine-tune recipe is the full-FT
recipe of the paper's joint full-FT rows (pi0.5 preset: AdamW, peak 2.5e-5, wd 0.01, betas
(0.9,0.95), clip 1.0, cosine decay to 2.5e-6) compressed to 5k steps with a 500-step warm-up
(set via --policy.optimizer_lr / --policy.scheduler_*; the preset is read from the policy config
exactly as lerobot-train reads it).

Outputs (`--output_dir`):
  checkpoints/<k*5000:06d>/pretrained_model/   the MERGED model after task k  (what gets evaluated;
                                               the next task's initialisation)
  checkpoints/<k*5000:06d>/ft_pretrained_model/ the un-merged fine-tune (audit; --keep_ft_weights)
  checkpoints/<k*5000:06d>/retain_boundary.json alpha, task, merge statistics
  retain_state/progress.json                   {"completed_tasks": k}
  retain_state/current/                        in-progress task state (model + optimizer +
                                               scheduler + RNG), rewritten every --ckpt_every steps
                                               and on SIGTERM; deleted at the boundary.
Resume is the default: an existing output_dir is continued from the last boundary and, if
present, the in-progress state. `--fresh=true` wipes both. `--stop_after_steps=N` exits like a
preemption after N global steps (smoke test of the resume path).
"""

import gc
import logging
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_model

# scripts/baselines on sys.path so `common` imports regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retain.retain_merge import cpu_copy as _cpu_copy, merge_in_place as _merge_in_place  # noqa: E402
from common import (  # noqa: E402
    task_episode_indices,
    RunningMeans,
    atomic_replace_dir,
    build_accelerator,
    build_dataset_policy_processors,
    install_sigterm_handler,
    log_gpu_mem,
    prep_batch,
    read_json,
    resolve_state_dir,
    task_dataloader,
    timed,
    train_step,
    write_json_atomic,
)

from lerobot.common.train_utils import (  # noqa: E402
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger  # noqa: E402
from lerobot.configs import parser  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402


@dataclass
class RetainConfig(TrainPipelineConfig):
    online_task_ids: list[int] = field(default_factory=lambda: list(range(10)))
    online_steps_per_task: int = 5000
    retain_alpha: float = 0.5
    ckpt_every: int = 1000
    keep_ft_weights: bool = True
    fresh: bool = False
    stop_after_steps: int = 0
    # E69 replay variant (defaults OFF => the RETAIN / naive paths are byte-identical): at each
    # boundary `replay_episodes_per_task` episodes of the finished task (seeded choice, recorded in
    # progress.json + the boundary json) join a buffer; every later task's sampler draws each frame
    # from the buffer with probability `replay_fraction` (uniform over buffered frames).
    replay_episodes_per_task: int = 0
    replay_fraction: float = 0.25
    replay_seed: int = 0
    # TrainPipelineConfig.validate() refuses an existing output_dir unless resuming; this baseline
    # resumes by default (CLAUDE.md 9.4.5), so the flag the parent looks for is always on.
    resume_sequential: bool = True

    def validate(self) -> None:
        super().validate()
        if not (0.0 < self.retain_alpha <= 1.0):
            raise ValueError("retain_alpha must be in (0, 1]")
        if self.peft is not None:
            raise ValueError("RETAIN is a full fine-tune baseline; do not pass --peft")
        if self.replay_episodes_per_task < 0 or not (0.0 <= self.replay_fraction < 1.0):
            raise ValueError("replay_episodes_per_task must be >= 0 and replay_fraction in [0, 1)")
        # step ids zero-padded like the other 10-task runs (005000 ... 050000)
        self.steps = self.online_steps_per_task * len(self.online_task_ids)


def _save_current_state(state_root: Path, task_pos: int, step_in_task: int, global_step: int,
                        cfg, unwrapped, optimizer, scheduler, pre, post) -> float:
    tmp = state_root / "current.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with timed() as t:
        save_checkpoint(checkpoint_dir=tmp, step=global_step, cfg=cfg, policy=unwrapped,
                        optimizer=optimizer, scheduler=scheduler, preprocessor=pre, postprocessor=post)
        write_json_atomic(tmp / "meta.json", {"task_pos": task_pos, "step_in_task": step_in_task,
                                              "global_step": global_step, "wall": time.time()})
        atomic_replace_dir(tmp, state_root / "current")
    return t.s


@parser.wrap()
def main(cfg: RetainConfig):
    cfg.validate()
    accelerator = build_accelerator(cfg)
    init_logging(accelerator=accelerator)
    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    out = Path(cfg.output_dir)
    state_root = out / "retain_state"
    ckpt_root = out / "checkpoints"
    if cfg.fresh:
        for d in (state_root, ckpt_root):
            if d.exists():
                logging.warning(f"--fresh: removing {d}")
                shutil.rmtree(d)
    state_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    S = cfg.online_steps_per_task
    n_tasks = len(cfg.online_task_ids)
    alpha = cfg.retain_alpha
    logging.info(f"RETAIN chain: alpha={alpha} tasks={cfg.online_task_ids} steps/task={S} "
                 f"batch={cfg.batch_size}x{cfg.gradient_accumulation_steps} ckpt_every={cfg.ckpt_every}")
    logging.info(f"optimizer preset: {cfg.optimizer}")
    logging.info(f"scheduler preset: {cfg.scheduler}")

    dataset, policy, pre, post, t2n = build_dataset_policy_processors(cfg, device)
    policy.train()
    n_total = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f"trainable {n_train/1e9:.3f}B / total {n_total/1e9:.3f}B params (full fine-tune)")
    if n_train != n_total:
        raise RuntimeError("RETAIN requires every parameter trainable; check freeze flags on the policy config")
    policy = accelerator.prepare(policy)
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)

    # ---- resume state ---------------------------------------------------------------------
    progress = read_json(state_root / "progress.json") if (state_root / "progress.json").exists() \
        else {"completed_tasks": 0, "global_step": 0}
    k = int(progress["completed_tasks"])
    replay_buffer = [int(e) for e in progress.get("replay_episodes", [])]   # E69: episodes banked at past boundaries
    if replay_buffer:
        logging.info(f"resume: replay buffer {replay_buffer}")
    if k > 0:
        bdir = get_step_checkpoint_dir(out, cfg.steps, k * S) / "pretrained_model"
        logging.info(f"resume: loading merged boundary {k} weights from {bdir}")
        load_model(unwrapped, str(bdir / "model.safetensors"), strict=True, device=str(device))
    theta_prev = _cpu_copy(unwrapped)
    resume_step = 0
    current = resolve_state_dir(state_root / "current")
    if current is not None:
        meta = read_json(current / "meta.json")
        if int(meta["task_pos"]) == k and 0 < int(meta["step_in_task"]) < S:
            logging.info(f"resume: in-progress task {k} at step {meta['step_in_task']} from {current}")
            load_model(unwrapped, str(current / "pretrained_model" / "model.safetensors"), strict=True, device=str(device))
            resume_step = int(meta["step_in_task"])
        else:
            logging.warning(f"stale in-progress state ignored: {meta} (completed_tasks={k})")
            current = None
    global_step = k * S + resume_step
    if k >= n_tasks:
        logging.info("all tasks complete; nothing to do")
        print("RETAIN-CHAIN-DONE", flush=True)
        return

    preempt = install_sigterm_handler()
    wandb_logger = WandBLogger(cfg) if (cfg.wandb.enable and cfg.wandb.project and accelerator.is_main_process) else None
    log_gpu_mem("after-setup")

    for task_pos, task_id in enumerate(cfg.online_task_ids):
        if task_pos < k:
            continue
        logging.info(f"=== RETAIN task {task_pos+1}/{n_tasks} | dataset_task_id={task_id} | {t2n.get(int(task_id), '')}")
        use_replay = cfg.replay_episodes_per_task > 0 and len(replay_buffer) > 0 and cfg.replay_fraction > 0.0
        if use_replay:
            logging.info(f"replay buffer: {len(replay_buffer)} episode(s) {replay_buffer} mixed at fraction {cfg.replay_fraction}")
        dl, it = task_dataloader(dataset, t2n, task_id, cfg, device,
                                 replay_episodes=replay_buffer if use_replay else None,
                                 replay_fraction=cfg.replay_fraction)
        optimizer = cfg.optimizer.build(unwrapped.get_optim_params())
        lr_scheduler = cfg.scheduler.build(optimizer, S) if cfg.scheduler is not None else None
        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
        step0 = 0
        if task_pos == k and resume_step > 0 and current is not None:
            saved_step, optimizer, lr_scheduler = load_training_state(current, optimizer, lr_scheduler)
            step0 = resume_step
            logging.info(f"resume: optimizer/scheduler/RNG restored (saved global step {saved_step}), continuing at step {step0}")
        resume_step = 0
        means = RunningMeans()
        t_last = time.perf_counter()
        for step in range(step0, S):
            for _micro in range(cfg.gradient_accumulation_steps):
                with accelerator.accumulate(policy):
                    batch = prep_batch(next(it), dataset, pre)
                    loss, _pen, gn, lr, synced, _ = train_step(
                        policy, batch, optimizer, lr_scheduler, accelerator, cfg.optimizer.grad_clip_norm)
            global_step += 1
            means.add(loss=loss, grad_norm=gn)
            if (step + 1) % cfg.log_freq == 0 or step + 1 == S:
                now = time.perf_counter()
                sps = (now - t_last) / cfg.log_freq
                t_last = now
                logging.info(f"task {task_pos+1} step {step+1}/{S} (global {global_step}) loss {means.mean('loss'):.4f} "
                             f"grdn {means.mean('grad_norm') or 0:.3f} lr {lr:.2e} {sps:.2f}s/step")
                if wandb_logger:
                    wandb_logger.log_dict({"loss": means.mean("loss"), "grad_norm": means.mean("grad_norm") or 0.0,
                                           "lr": lr, "s_per_step": sps, "task_pos": task_pos}, global_step)
                means.reset()
            stop = bool(cfg.stop_after_steps) and global_step >= cfg.stop_after_steps
            periodic = cfg.ckpt_every > 0 and (step + 1) % cfg.ckpt_every == 0 and (step + 1) < S
            if periodic or preempt.flag or stop:
                s = _save_current_state(state_root, task_pos, step + 1, global_step, cfg, unwrapped, optimizer, lr_scheduler, pre, post)
                logging.info(f"[ckpt] in-progress state written in {s:.1f}s (task {task_pos+1} step {step+1})")
                if preempt.flag or stop:
                    logging.info("exiting cleanly after checkpoint (preemption / stop_after_steps)")
                    print("RETAIN-PREEMPT-EXIT" if preempt.flag else "RETAIN-STOP-AFTER-STEPS", flush=True)
                    return
        # ---- boundary: (optionally) keep the fine-tuned weights, merge, save the merged model ----
        bdir = get_step_checkpoint_dir(out, cfg.steps, global_step)
        tmp_b = bdir.with_name(bdir.name + ".tmp")
        if tmp_b.exists():
            shutil.rmtree(tmp_b)
        tmp_b.mkdir(parents=True)
        with timed() as t_ft:
            if cfg.keep_ft_weights:
                unwrapped.save_pretrained(tmp_b / "ft_pretrained_model")
        stats = _merge_in_place(unwrapped, theta_prev, alpha)
        theta_prev = _cpu_copy(unwrapped)
        replay_added: list[int] = []
        if cfg.replay_episodes_per_task > 0:   # E69: bank episodes of the task just finished
            eps = task_episode_indices(dataset, t2n, int(task_id))
            rng = random.Random(cfg.replay_seed * 1000 + task_pos)
            replay_added = sorted(rng.sample(eps, min(cfg.replay_episodes_per_task, len(eps))))
            replay_buffer = replay_buffer + replay_added
            logging.info(f"[boundary {task_pos+1}] replay: banked episodes {replay_added} of task {int(task_id)}; buffer now {replay_buffer}")
        with timed() as t_b:
            save_checkpoint(checkpoint_dir=tmp_b, step=global_step, cfg=cfg, policy=unwrapped,
                            optimizer=None, scheduler=None, preprocessor=pre, postprocessor=post)
        stats.update({"alpha": alpha, "task_pos": task_pos, "dataset_task_id": int(task_id),
                      "task_name": t2n.get(int(task_id), ""), "global_step": global_step,
                      "save_ft_s": t_ft.s, "save_merged_s": t_b.s,
                      "replay_added": replay_added, "replay_buffer": list(replay_buffer)})
        write_json_atomic(tmp_b / "retain_boundary.json", stats)
        atomic_replace_dir(tmp_b, bdir)
        update_last_checkpoint(bdir)
        write_json_atomic(state_root / "progress.json", {"completed_tasks": task_pos + 1, "global_step": global_step,
                                                          "replay_episodes": list(replay_buffer)})
        for d in (state_root / "current", state_root / "current.old", state_root / "current.tmp"):
            if d.exists():
                shutil.rmtree(d)
        current = None
        r = stats["largest_tensor"]["ratio_fp32"]
        logging.info(f"[boundary {task_pos+1}] merged alpha={alpha}: mean|ft-prev|={stats['mean_abs_ft_minus_prev']:.3e} "
                     f"mean|merged-prev|={stats['mean_abs_merged_minus_prev']:.3e} ratio={stats['ratio_merged_over_ft']:.4f} "
                     f"(largest tensor ratio {r:.4f}); saved in {t_b.s:.1f}s -> {bdir}")
        if r != r or abs(r - alpha) > 1e-3:   # NaN (nothing moved) or off-alpha
            raise RuntimeError(f"merge exactness witness failed: ratio {r} != alpha {alpha}")
        print(f"RETAIN-BOUNDARY-{task_pos+1}", flush=True)
        # Free the task's optimizer state for real. `del optimizer` alone leaks it: accelerator.prepare() keeps
        # its own reference (accelerator._optimizers / _schedulers), so every task left one AdamW state
        # (~14.1 GiB) on the GPU - allocated 22.9 -> 37.0 GiB across boundaries 1 -> 2 in the E67 run,
        # i.e. an OOM by task 4-5 and a permanently closed eval VRAM gate (E67 addendum 6).
        optimizer.state.clear()
        unwrapped.zero_grad(set_to_none=True)
        for _reg in ("_optimizers", "_schedulers"):
            getattr(accelerator, _reg, []).clear()
        del optimizer, lr_scheduler, dl, it
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log_gpu_mem(f"boundary-{task_pos+1}")
    print("RETAIN-CHAIN-DONE", flush=True)


if __name__ == "__main__":
    main()
