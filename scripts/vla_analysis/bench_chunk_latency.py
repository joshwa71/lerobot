#!/usr/bin/env python3
"""Action-chunk latency: base pi0.5 vs the memory-augmented model, on one GPU.

Times policy.predict_action_chunk() only — the per-chunk GPU work a robot waits on.
Data loading and video decode are deliberately excluded. Real LIBERO instruction
strings are used (token length affects VLM sequence length, hence timing); image and
state tensors are random, which does not change the FLOPs.
"""
import argparse, json, statistics, time
from pathlib import Path

import torch


def stat_dims(pre):
    """Raw per-key dims the preprocessor's normalizer actually expects (config input_features
    reports the PADDED state dim, which the normalizer would reject)."""
    dims = {}
    for step in getattr(pre, "steps", []):
        stats = getattr(step, "stats", None)
        if not isinstance(stats, dict):
            continue
        for key, st in stats.items():
            for field in ("mean", "std", "min", "max"):
                v = st.get(field) if isinstance(st, dict) else None
                if v is not None:
                    try:
                        dims[key] = tuple(torch.as_tensor(v).shape)
                    except Exception:
                        pass
                    break
    return dims


def build_obs(cfg, instruction, gen, dims):
    """One policy-space observation dict, shapes from the normalizer where it has an opinion."""
    obs = {}
    for key, ft in cfg.input_features.items():
        t = str(ft.type.value if hasattr(ft.type, "value") else ft.type).upper()
        shape = dims.get(key, tuple(ft.shape))
        if "VISUAL" in t:
            shape = dims.get(key) if dims.get(key) and len(dims.get(key)) == 3 else tuple(ft.shape)
            obs[key] = torch.rand(shape, generator=gen, dtype=torch.float32)
        else:
            obs[key] = torch.rand(shape, generator=gen, dtype=torch.float32) * 2 - 1
    obs["task"] = instruction
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--tasks_parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_prepass", action="store_true",
                    help="latency attribution: disable the frozen prepass (changes numerics, timing only)")
    ap.add_argument("--bf16", action="store_true",
                    help="cast the policy to bf16 on CPU before the GPU move; paired with autocast, "
                         "this is the precision the triangle evals run (--policy.dtype=bfloat16)")
    args = ap.parse_args()

    import pandas as pd
    from lerobot.policies.factory import get_policy_class
    from lerobot.configs.policies import PreTrainedConfig

    df = pd.read_parquet(args.tasks_parquet)
    instructions = list(df.index.astype(str))[:10] if df.index.dtype == object else list(df.iloc[:, 0].astype(str))[:10]
    print(f"[{args.label}] {len(instructions)} instructions, e.g. {instructions[0][:60]!r}", flush=True)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.device = "cpu"   # load on CPU; cast, then move (a fp32 backbone would not fit otherwise)
    if args.no_prepass and getattr(cfg, "memory_layer", None) is not None:
        # latency attribution only: how much of the gap is the extra frozen forward pass?
        for attr in ("frozen_prepass", "use_frozen_base_input_features"):
            if hasattr(cfg.memory_layer, attr):
                setattr(cfg.memory_layer, attr, False)
    cls = get_policy_class(cfg.type)
    policy = cls.from_pretrained(args.ckpt, config=cfg)
    print(f"[{args.label}] loaded on cpu; gpu alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
    # Match the two checkpoints' precision layouts before the GPU move. The base checkpoint stores its
    # backbone in bf16; ours stores the same backbone in fp32 (19 GB on disk vs 8.8 GB), which OOMs a
    # 24 GB card. Casting ONLY the backbone makes the layouts identical and leaves the small fp32
    # projection/time-MLP heads alone, which the sampler's fp32 noise requires.
    cast_report = "none"
    if args.bf16:
        policy = policy.to(dtype=torch.bfloat16)
        cast_report = "whole policy -> bfloat16"
    print(f"[{args.label}] cast: {cast_report}", flush=True)
    cpu_bytes = sum(p.numel() * p.element_size() for p in policy.parameters())
    print(f"[{args.label}] param bytes after cast: {cpu_bytes/2**30:.2f} GiB", flush=True)
    cfg.device = "cuda"
    policy.to(device)
    policy.eval()

    n_params = sum(p.numel() for p in policy.parameters())
    mem_on = bool(getattr(cfg, "memory_layers", False))
    print(f"[{args.label}] params {n_params/1e9:.3f}B  memory_layers={mem_on}  dtype={cfg.dtype}", flush=True)

    from lerobot.policies.factory import make_pre_post_processors
    pre, _post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    dims = stat_dims(pre)
    print(f"[{args.label}] normalizer dims: " + ", ".join(f"{k.split('.')[-1]}={v}" for k, v in sorted(dims.items())), flush=True)
    gen = torch.Generator().manual_seed(0)
    samples = [build_obs(cfg, instructions[i % len(instructions)], gen, dims) for i in range(args.n + args.warmup)]

    times = []
    with torch.no_grad():
        for i, obs in enumerate(samples):
            batch = pre(dict(obs))
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.ndim >= 1 and k != "task":
                    if v.shape[0] != 1:
                        batch[k] = v.unsqueeze(0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= args.warmup:
                times.append(dt)
            if i == args.warmup - 1:
                print(f"[{args.label}] warm-up done ({args.warmup} iters)", flush=True)
            policy.reset() if hasattr(policy, "reset") else None

    times.sort()
    res = {
        "label": args.label, "ckpt": args.ckpt, "n": len(times), "bf16": args.bf16, "no_prepass": args.no_prepass,
        "params_B": round(n_params / 1e9, 4), "memory_layers": mem_on,
        "chunk_size": getattr(cfg, "chunk_size", None),
        "mean_ms": round(statistics.mean(times), 2),
        "std_ms": round(statistics.pstdev(times), 2),
        "median_ms": round(statistics.median(times), 2),
        "p10_ms": round(times[int(0.10 * len(times))], 2),
        "p90_ms": round(times[int(0.90 * len(times))], 2),
        "min_ms": round(times[0], 2), "max_ms": round(times[-1], 2),
        "peak_vram_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    print(json.dumps(res, indent=2), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
