#!/usr/bin/env python3
"""O-LoRA baseline (Wang et al., EMNLP Findings 2023, arXiv 2310.14152) under OUR continual
protocol (E67). See olora_common.py for the method and the checkpoint format.

Recipe = the rank-64 specialist / naive-sequential LoRA recipe verbatim (same target set, alpha/r
= 0.25, AdamW betas (0.9, 0.999) eps 1e-8 wd 0, LR 1e-4 -> 1e-5 linear over the 5,000 steps of
each task, grad-clip 1.0, effective batch 32, optimizer re-initialised per task), plus:
  * a NEW adapter per task (`olora_t<k>`), the earlier adapters frozen but active;
  * the orthogonality penalty lambda_1 * sum_{i<t} |A_i A_t^T|_1 (official-code form), lambda_1 =
    0.5, skipped on modules matching --penalty_exclude_regex (the two 32-input projections).

Outputs (`--output_dir`):
  checkpoints/<k*5000:06d>/pretrained_model/  rank-concatenated single adapter (padded to
                                              --export_rank) + policy config + processors; loads via
                                              the factory's use_peft path exactly like every other
                                              LoRA row.
  checkpoints/<k*5000:06d>/olora_boundary.json penalty statistics, export + exactness checks
  olora_state/adapters/olora_t<k>.safetensors  the raw per-task adapters (resume + audit)
  olora_state/progress.json, olora_state/current/  (in-progress adapter + optimizer + sched + RNG)
Resume is the default; --fresh=true wipes; --stop_after_steps=N simulates a preemption.
"""

import copy
import logging
import gc
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from peft import get_peft_model
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
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
from olora.olora_common import (  # noqa: E402
    OrthPenalty,
    adapter_key,
    adapter_tensors,
    check_key_format,
    concat_adapters,
    l1_of_adapter_B,
    load_adapter_tensors,
    lora_layers,
    set_trainable_adapter,
    write_export_checkpoint,
)

from lerobot.common.train_utils import get_step_checkpoint_dir, update_last_checkpoint  # noqa: E402
from lerobot.common.wandb_utils import WandBLogger  # noqa: E402
from lerobot.configs import parser  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402


@dataclass
class OLoraConfig(TrainPipelineConfig):
    online_task_ids: list[int] = field(default_factory=lambda: list(range(10)))
    online_steps_per_task: int = 5000
    lr_start: float = 1e-4
    lr_end: float = 1e-5
    lambda1: float = 0.5
    penalty_exclude_regex: str | None = r"(^|\.)(state_proj|action_in_proj)$"
    export_rank: int = 0            # 0 -> r * n_tasks
    export_check: bool = True
    ckpt_every: int = 1000
    fresh: bool = False
    stop_after_steps: int = 0
    resume_sequential: bool = True  # TrainPipelineConfig.validate(): allow the existing output_dir

    def validate(self) -> None:
        super().validate()
        if self.peft is None or str(self.peft.method_type).upper() != "LORA":
            raise ValueError("O-LoRA needs --peft.method_type=LORA with r / lora_alpha / target_modules")
        if self.export_rank <= 0:
            self.export_rank = int(self.peft.r) * len(self.online_task_ids)
        self.steps = self.online_steps_per_task * len(self.online_task_ids)


def adapter_name(k: int) -> str:
    return f"olora_t{k}"


def _linear_lambda(start: float, end: float, total: int):
    def f(step: int) -> float:  # identical to lerobot_sequential_train._build_memory_scheduler (linear)
        if total <= 1:
            return 1.0
        progress = min(step / max(total - 1, 1), 1.0)
        return 1.0 - progress * (1.0 - end / start)
    return f


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _load_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def _save_current(state_root: Path, peft_model, cur: str, optimizer, sched, task_pos: int, step_in_task: int,
                  global_step: int) -> float:
    tmp = state_root / "current.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with timed() as t:
        save_file(adapter_tensors(peft_model, cur), str(tmp / "adapter.safetensors"), metadata={"format": "pt"})
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": sched.state_dict() if sched is not None else None,
                    "rng": _rng_state()}, tmp / "train_state.pt")
        write_json_atomic(tmp / "meta.json", {"task_pos": task_pos, "adapter": cur, "step_in_task": step_in_task,
                                              "global_step": global_step, "wall": time.time()})
        atomic_replace_dir(tmp, state_root / "current")
    return t.s


def _save_adapter_atomic(path: Path, tensors: dict) -> None:
    """temp file in the same directory, fsync, os.replace, fsync the directory (CLAUDE.md 9.4.3)."""
    tmp = path.with_suffix(".safetensors.tmp")
    save_file(tensors, str(tmp), metadata={"format": "pt"})
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


@torch.no_grad()
def _export_check(peft_model, active: list[str], tensors_unpadded: dict, r_tot: int, scaling: float,
                  batch, accelerator) -> dict:
    """Load the concatenated adapter as a temporary adapter, compare (a) its tensors bitwise with
    the export and (b) the forward loss against the multi-adapter forward on the same batch/seed."""
    name = "_exportcheck"
    cfg = copy.deepcopy(peft_model.peft_config[active[0]])
    cfg.r = int(r_tot)
    cfg.lora_alpha = scaling * r_tot
    cfg.rank_pattern = {}
    cfg.alpha_pattern = {}
    peft_model.add_adapter(name, cfg)
    try:
        load_adapter_tensors(peft_model, name, tensors_unpadded)
        # (a) bitwise: what sits in the layers is what the file holds
        back = adapter_tensors(peft_model, name)
        for k, v in tensors_unpadded.items():
            if not torch.equal(back[k].float(), v.float()):
                raise RuntimeError(f"export-check: tensor mismatch after load at {k}")
        # scaling must coincide with the multi-adapter one
        for _, mod in lora_layers(peft_model):
            if name in mod.scaling and abs(float(mod.scaling[name]) - scaling) > 1e-9:
                raise RuntimeError(f"export-check: scaling {mod.scaling[name]} != {scaling}")
        peft_model.eval()
        peft_model.base_model.set_adapter(active, inference_mode=True)
        torch.manual_seed(4321)
        with accelerator.autocast():
            l_multi, out_multi = peft_model.forward(batch)
        peft_model.base_model.set_adapter([name], inference_mode=True)
        torch.manual_seed(4321)
        with accelerator.autocast():
            l_one, out_one = peft_model.forward(batch)
        l_multi, l_one = float(l_multi), float(l_one)
        rel = abs(l_one - l_multi) / (abs(l_multi) + 1e-12)
        return {"loss_multi_adapter": l_multi, "loss_concat_adapter": l_one, "rel_diff": rel}
    finally:
        peft_model.base_model.set_adapter(active, inference_mode=False)
        peft_model.delete_adapter(name)
        peft_model.train()


@parser.wrap()
def main(cfg: OLoraConfig):
    cfg.validate()
    accelerator = build_accelerator(cfg)
    init_logging(accelerator=accelerator)
    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    out = Path(cfg.output_dir)
    state_root = out / "olora_state"
    adapters_dir = state_root / "adapters"
    ckpt_root = out / "checkpoints"
    if cfg.fresh:
        for d in (state_root, ckpt_root):
            if d.exists():
                logging.warning(f"--fresh: removing {d}")
                shutil.rmtree(d)
    adapters_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    S = cfg.online_steps_per_task
    n_tasks = len(cfg.online_task_ids)
    logging.info(f"O-LoRA chain: r={cfg.peft.r} alpha={cfg.peft.lora_alpha} lambda1={cfg.lambda1} "
                 f"exclude={cfg.penalty_exclude_regex!r} export_rank={cfg.export_rank} tasks={cfg.online_task_ids} "
                 f"steps/task={S} lr {cfg.lr_start}->{cfg.lr_end} batch={cfg.batch_size}x{cfg.gradient_accumulation_steps}")

    dataset, policy, pre, post, t2n = build_dataset_policy_processors(cfg, device)
    base_model_path = str(cfg.policy.pretrained_path)

    # ---- first adapter: the same wrap as PreTrainedPolicy.wrap_with_peft, adapter named olora_t0 ----
    lora_cfg = policy._build_peft_config(asdict(cfg.peft))
    policy._validate_peft_config(lora_cfg)
    for p in policy.parameters():
        p.requires_grad_(False)
    if policy.config.pretrained_path:
        policy.name_or_path = str(policy.config.pretrained_path)
    peft_model = get_peft_model(policy, lora_cfg, adapter_name=adapter_name(0))
    peft_model.config.use_peft = True
    peft_model.train()
    peft_model = accelerator.prepare(peft_model)
    unwrapped = accelerator.unwrap_model(peft_model, keep_fp32_wrapper=True)
    check_key_format(unwrapped, adapter_name(0))
    n_layers = len(lora_layers(unwrapped))
    per_adapter = sum(p.numel() for n, p in unwrapped.named_parameters() if adapter_name(0) in n.split("."))
    logging.info(f"LoRA on {n_layers} modules; {per_adapter/1e6:.1f}M params per adapter; "
                 f"adapter dtype {next(m.lora_A[adapter_name(0)].weight.dtype for _, m in lora_layers(unwrapped))}")

    # ---- resume: completed adapters ----
    progress = read_json(state_root / "progress.json") if (state_root / "progress.json").exists() \
        else {"completed_tasks": 0, "global_step": 0}
    k = int(progress["completed_tasks"])
    for i in range(k):
        a = adapter_name(i)
        if i > 0:
            unwrapped.add_adapter(a, copy.deepcopy(lora_cfg))
        n_loaded = load_adapter_tensors(unwrapped, a, load_file(str(adapters_dir / f"{a}.safetensors")))
        logging.info(f"resume: adapter {a} restored ({n_loaded} tensors, L1(B)={l1_of_adapter_B(unwrapped, a):.4e})")
    resume_step = 0
    current = resolve_state_dir(state_root / "current")
    if current is not None:
        meta = read_json(current / "meta.json")
        if int(meta["task_pos"]) == k and 0 < int(meta["step_in_task"]) < S:
            resume_step = int(meta["step_in_task"])
            logging.info(f"resume: in-progress task {k} at step {resume_step} from {current}")
        else:
            logging.warning(f"stale in-progress state ignored: {meta} (completed_tasks={k})")
            current = None
    global_step = k * S + resume_step
    if k >= n_tasks:
        logging.info("all tasks complete; nothing to do")
        print("OLORA-CHAIN-DONE", flush=True)
        return

    preempt = install_sigterm_handler()
    wandb_logger = WandBLogger(cfg) if (cfg.wandb.enable and cfg.wandb.project and accelerator.is_main_process) else None
    log_gpu_mem("after-setup")

    for task_pos, task_id in enumerate(cfg.online_task_ids):
        if task_pos < k:
            continue
        cur = adapter_name(task_pos)
        prev = [adapter_name(i) for i in range(task_pos)]
        active = prev + [cur]
        if task_pos > 0 and cur not in unwrapped.peft_config:
            unwrapped.add_adapter(cur, copy.deepcopy(lora_cfg))
        unwrapped.base_model.set_adapter(active)          # all active in the forward pass ...
        n_train = set_trainable_adapter(unwrapped, cur)   # ... only the newest one trains
        if n_train != per_adapter:
            raise RuntimeError(f"trainable {n_train} != one adapter {per_adapter}")
        d0 = next(m.lora_A[cur].weight.dtype for _, m in lora_layers(unwrapped))
        logging.info(f"=== O-LoRA task {task_pos+1}/{n_tasks} | dataset_task_id={task_id} | {t2n.get(int(task_id), '')} | "
                     f"active adapters {active} | trainable {n_train/1e6:.1f}M ({d0})")
        penalty = OrthPenalty(unwrapped, cur, prev, cfg.penalty_exclude_regex) if prev else None
        if penalty is not None:
            st = penalty.stats()
            logging.info(f"penalty over {st['n_modules']} modules; excluded {penalty.excluded}; "
                         f"unsatisfiable {penalty.unsatisfiable}; initial L1 {st['total_l1']:.3e}")
        lam = float(cfg.lambda1)
        extra = (lambda: lam * penalty()) if (penalty is not None and lam > 0) else None

        dl, it = task_dataloader(dataset, t2n, task_id, cfg, device)
        trainable = [p for p in unwrapped.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([{"params": trainable, "lr": cfg.lr_start, "weight_decay": 0.0}],
                                      betas=(0.9, 0.999), eps=1e-8)
        for g in optimizer.param_groups:
            g["initial_lr"] = cfg.lr_start
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, _linear_lambda(cfg.lr_start, cfg.lr_end, S))
        optimizer, sched = accelerator.prepare(optimizer, sched)
        step0 = 0
        if task_pos == k and resume_step > 0 and current is not None:
            load_adapter_tensors(unwrapped, cur, load_file(str(current / "adapter.safetensors")))
            ts = torch.load(current / "train_state.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(ts["optimizer"])
            if ts.get("scheduler") is not None:
                sched.load_state_dict(ts["scheduler"])
            _load_rng_state(ts["rng"])
            step0 = resume_step
            logging.info(f"resume: adapter/optimizer/scheduler/RNG restored, continuing at step {step0} "
                         f"(lr now {optimizer.param_groups[0]['lr']:.2e})")
        resume_step = 0
        frozen_l1 = {a: l1_of_adapter_B(unwrapped, a) for a in prev}
        means = RunningMeans()
        t_last = time.perf_counter()
        last_batch = None
        for step in range(step0, S):
            for _micro in range(cfg.gradient_accumulation_steps):
                with accelerator.accumulate(peft_model):
                    batch = prep_batch(next(it), dataset, pre)
                    loss, pen, gn, lr, synced, _ = train_step(
                        peft_model, batch, optimizer, sched, accelerator, cfg.optimizer.grad_clip_norm, extra_loss_fn=extra)
            last_batch = batch
            global_step += 1
            means.add(loss=loss, penalty=pen, grad_norm=gn)
            if (step + 1) % cfg.log_freq == 0 or step + 1 == S:
                now = time.perf_counter()
                sps = (now - t_last) / cfg.log_freq
                t_last = now
                pm = means.mean("penalty")
                logging.info(f"task {task_pos+1} step {step+1}/{S} (global {global_step}) loss {means.mean('loss'):.4f} "
                             f"orth {pm if pm is not None else 0:.3e} grdn {means.mean('grad_norm') or 0:.3f} lr {lr:.2e} {sps:.2f}s/step")
                if wandb_logger:
                    wandb_logger.log_dict({"loss": means.mean("loss"), "orth_penalty": pm or 0.0,
                                           "grad_norm": means.mean("grad_norm") or 0.0, "lr": lr, "s_per_step": sps,
                                           "task_pos": task_pos}, global_step)
                means.reset()
            stop = bool(cfg.stop_after_steps) and global_step >= cfg.stop_after_steps
            periodic = cfg.ckpt_every > 0 and (step + 1) % cfg.ckpt_every == 0 and (step + 1) < S
            if periodic or preempt.flag or stop:
                s = _save_current(state_root, unwrapped, cur, optimizer, sched, task_pos, step + 1, global_step)
                logging.info(f"[ckpt] in-progress state written in {s:.1f}s (task {task_pos+1} step {step+1})")
                if preempt.flag or stop:
                    logging.info("exiting cleanly after checkpoint (preemption / stop_after_steps)")
                    print("OLORA-PREEMPT-EXIT" if preempt.flag else "OLORA-STOP-AFTER-STEPS", flush=True)
                    return
        # ---- boundary ----
        for a in prev:  # frozen adapters must not have moved
            l1 = l1_of_adapter_B(unwrapped, a)
            if abs(l1 - frozen_l1[a]) > 1e-6 * max(1.0, frozen_l1[a]):
                raise RuntimeError(f"frozen adapter {a} changed during task {task_pos+1}: {frozen_l1[a]} -> {l1}")
        _save_adapter_atomic(adapters_dir / f"{cur}.safetensors", adapter_tensors(unwrapped, cur))
        bdir = get_step_checkpoint_dir(out, cfg.steps, global_step)
        tmp_b = bdir.with_name(bdir.name + ".tmp")
        if tmp_b.exists():
            shutil.rmtree(tmp_b)
        tmp_b.mkdir(parents=True)
        with timed() as t_b:
            info = write_export_checkpoint(tmp_b / "pretrained_model", unwrapped, active, cfg.export_rank,
                                           base_model_path, cfg, pre, post)
        info.update({"task_pos": task_pos, "dataset_task_id": int(task_id), "task_name": t2n.get(int(task_id), ""),
                     "global_step": global_step, "active_adapters": active, "lambda1": lam, "save_s": t_b.s,
                     "trainable_params": n_train, "l1_B_current": l1_of_adapter_B(unwrapped, cur),
                     "penalty_final": penalty.stats() if penalty is not None else None})
        if cfg.export_check and last_batch is not None:
            tensors_unpadded, r_tot, scaling = concat_adapters(unwrapped, active, None)
            chk = _export_check(unwrapped, active, tensors_unpadded, r_tot, scaling, last_batch, accelerator)
            set_trainable_adapter(unwrapped, cur)
            info["export_check"] = chk
            logging.info(f"[export-check] loss multi {chk['loss_multi_adapter']:.6f} vs concat {chk['loss_concat_adapter']:.6f} "
                         f"(rel {chk['rel_diff']:.2e}); algebra max rel err {info['algebra_max_rel_err']:.2e}")
            if chk["rel_diff"] > 2e-2 or info["algebra_max_rel_err"] > 1e-4:
                raise RuntimeError(f"export exactness check failed: {chk} / {info['algebra_max_rel_err']}")
            print("OLORA-EXPORT-CHECK-OK", flush=True)
        write_json_atomic(tmp_b / "olora_boundary.json", info)
        atomic_replace_dir(tmp_b, bdir)
        update_last_checkpoint(bdir)
        write_json_atomic(state_root / "progress.json", {"completed_tasks": task_pos + 1, "global_step": global_step})
        for d in (state_root / "current", state_root / "current.old", state_root / "current.tmp"):
            if d.exists():
                shutil.rmtree(d)
        current = None
        logging.info(f"[boundary {task_pos+1}] exported rank {info['rank_concat']} (padded {info['export_rank']}) adapter, "
                     f"{info['n_export_params']/1e9:.3f}B params, in {t_b.s:.1f}s -> {bdir}")
        print(f"OLORA-BOUNDARY-{task_pos+1}", flush=True)
        # see retain_sequential_train.py: accelerator.prepare() keeps optimizer/scheduler references, so clear
        # the state and the registry explicitly (E67 addendum 6)
        optimizer.state.clear()
        for _reg in ("_optimizers", "_schedulers"):
            getattr(accelerator, _reg, []).clear()
        del optimizer, sched, dl, it, penalty, extra
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log_gpu_mem(f"boundary-{task_pos+1}")
    print("OLORA-CHAIN-DONE", flush=True)


if __name__ == "__main__":
    main()
