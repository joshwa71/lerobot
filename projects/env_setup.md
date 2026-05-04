# `lerobot-memory-updated` environment setup

This note records the local environment that worked on `vla-memory-updated`, plus the cluster steps needed to reproduce it.

Do not copy the repo-local absolute paths from this machine. In the commands below, first set:

```bash
export ROOT_DIR={your_path_to_lerobot}
```

and replace `{your_path_to_lerobot}` with the path to your own LeRobot checkout.

## What changed versus the old branch

The main CLI change that matters for your old scripts is the dataset path handling:

- old style:
  - `--dataset.repo_id=/full/path/to/dataset`
- new style:
  - `--dataset.repo_id=<logical_name>`
  - `--dataset.root=/full/path/to/dataset`

For the local LIBERO tree I used:

```bash
--dataset.repo_id=libero
--dataset.root="$ROOT_DIR/outputs/libero"
```

The memory/VLA setup also still needs:

```bash
--rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
--policy.empty_cameras=1
```

The rename map must point at the policy's expected feature keys (`observation.images.camera{1,2,3}`), not at `observation.image*`. With 2 dataset cameras renamed to `camera1`/`camera2`, `empty_cameras=1` fills the remaining slot with zeros.

## Local environment that worked

```bash
conda create -y -n lerobot-memory-updated python=3.12
conda install -y -n lerobot-memory-updated -c conda-forge ffmpeg=7.1.1
conda activate lerobot-memory-updated

cd "$ROOT_DIR"
pip install -e ".[core_scripts,training,smolvla,libero]"
pip install sentence-transformers==5.2.0
```

Notes:

- `ffmpeg=7.1.1` matches the upstream install note for `torchcodec`.
- `sentence-transformers` is required by the memory code path when `lang_to_query=true`. It is not installed by the upstream extras.
- LIBERO assets were downloaded automatically on first eval into `~/.cache/libero/assets`.

## Cluster setup

Use the same base install on the cluster:

```bash
source /share/apps/miniconda3/etc/profile.d/conda.sh
export ROOT_DIR={your_path_to_lerobot}
conda create -y -n lerobot-memory-updated python=3.12
conda install -y -n lerobot-memory-updated -c conda-forge ffmpeg=7.1.1
conda activate lerobot-memory-updated

cd "$ROOT_DIR"
pip install -e ".[core_scripts,training,smolvla,libero]"
pip install sentence-transformers==5.2.0
```

For headless LIBERO eval/train jobs:

```bash
export MUJOCO_GL=egl
unset DISPLAY
```

If EGL discovery is flaky on the cluster, keep the same fallback you already use in your job scripts:

```bash
if [ -e /usr/lib/x86_64-linux-gnu/libEGL.so.1 ]; then
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi
if [ -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
fi
```

On clusters that ship Mesa-only EGL (no `libnvidia-gl-*`, no `10_nvidia.json` vendor file), `MUJOCO_GL=egl` will fail with `Cannot initialize a EGL device display`. As long as `libosmesa6` is installed, fall back to:

```bash
export MUJOCO_GL=osmesa
unset DISPLAY
```

This was needed for the Nebius-style instance with H200 + Mesa-only userspace (driver 570.195.03).

## Porting the old training scripts

The main edit is replacing path-valued `dataset.repo_id` with `dataset.root` + a short repo id.

Example:

```bash
# old
--dataset.repo_id="$DATASET_SCRATCH"

# new
--dataset.repo_id=libero_95
--dataset.root="$DATASET_SCRATCH"
```

Likewise for sequential:

```bash
--dataset.repo_id=libero_10
--dataset.root="$DATASET_SCRATCH"
```

Keep the memory flags themselves as-is unless you intentionally want to retune them.

## Minimal smoke commands

These are the commands I validated locally in the new env.

### 1. Pretraining smoke

This uses a reduced memory size because the local machine only had a 24 GB GPU available. The purpose was env validation, not reproducing the full cluster-scale run.

```bash
lerobot-train \
  --policy.path="$ROOT_DIR/outputs/smolvla_base" \
  --policy.empty_cameras=1 \
  --dataset.repo_id=libero \
  --dataset.root="$ROOT_DIR/outputs/libero" \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --output_dir="$ROOT_DIR/outputs/smoke/lerobot-memory-updated/train-smoke-smallmem" \
  --job_name=smoke_pretrain_mem_small \
  --steps=1 \
  --batch_size=2 \
  --num_workers=0 \
  --prefetch_factor=2 \
  --persistent_workers=false \
  --eval_freq=0 \
  --save_freq=1 \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.memory_layers=true \
  --policy.memory_layer.enabled=true \
  --policy.memory_layer.layers='[8,10,12,14]' \
  --policy.memory_layer.log_usage=true \
  --policy.memory_layer.aggregate_usage=true \
  --policy.memory_layer.mem_n_keys=32 \
  --policy.memory_layer.mem_heads=2 \
  --policy.memory_layer.mem_knn=4 \
  --policy.memory_layer.mem_k_dim=128 \
  --policy.memory_layer.value_fixed_lr=0.001 \
  --policy.memory_layer.memory_lr=0.001 \
  --policy.memory_layer.lang_to_query=true \
  --policy.memory_layer.fuse_method=film \
  --policy.memory_layer.embedding_model=all-mpnet-base-v2 \
  --policy.memory_layer.value_type=lora \
  --policy.memory_layer.lora_rank=2 \
  --policy.memory_layer.contrastive_method=sample \
  --policy.memory_layer.contrastive_loss_weight=1.0 \
  --policy.memory_layer.contrastive_margin=0.0 \
  --policy.memory_layer.contrastive_query_queue=8 \
  --policy.memory_layer.routing_loss_topk=4 \
  --policy.memory_layer.routing_intra_task_locality_weight=0.25 \
  --policy.memory_layer.routing_intra_task_min_support=4 \
  --policy.memory_layer.routing_intra_task_max_support=32 \
  --policy.memory_layer.routing_inter_task_separation_weight=0.25
```

### 2. Sequential training smoke

```bash
python -m lerobot.scripts.lerobot_sequential_train \
  --policy.path="$ROOT_DIR/outputs/smoke/lerobot-memory-updated/train-smoke-smallmem/checkpoints/000001/pretrained_model" \
  --dataset.repo_id=libero \
  --dataset.root="$ROOT_DIR/outputs/libero" \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --output_dir="$ROOT_DIR/outputs/smoke/lerobot-memory-updated/sequential-smoke-smallmem" \
  --steps=2 \
  --batch_size=2 \
  --num_workers=0 \
  --prefetch_factor=2 \
  --persistent_workers=false \
  --log_freq=1 \
  --wandb.enable=false \
  --online_task_ids='[0,1]' \
  --online_steps_per_task=1 \
  --save_after_each_task=true \
  --reinit_optimizer_each_task=true \
  --tfidf_enable=true \
  --tfidf_top_t=16 \
  --use_online_idf_stats=true \
  --idf_exponent=1 \
  --memory_value_lr=0.001 \
  --memory_value_lr_end=0.0001 \
  --memory_value_scheduler_type=linear \
  --log_full_memory_usage_viz=false
```

### 3. LIBERO eval smoke

```bash
export MUJOCO_GL=egl
unset DISPLAY

lerobot-eval \
  --policy.path="$ROOT_DIR/outputs/smoke/lerobot-memory-updated/train-smoke-smallmem/checkpoints/000001/pretrained_model" \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env.max_parallel_tasks=1 \
  --output_dir="$ROOT_DIR/outputs/smoke/lerobot-memory-updated/eval-smoke-smallmem"
```

## Local validation results

Validated on `vla-memory-updated` in `lerobot-memory-updated`:

- pretraining smoke passed
  - output: `$ROOT_DIR/outputs/smoke/lerobot-memory-updated/train-smoke-smallmem`
- sequential smoke passed
  - output: `$ROOT_DIR/outputs/smoke/lerobot-memory-updated/sequential-smoke-smallmem`
- LIBERO eval smoke passed
  - output: `$ROOT_DIR/outputs/smoke/lerobot-memory-updated/eval-smoke-smallmem`
  - 1 episode on `libero_10` task `0`
  - success rate was `0.0`, which is fine for a smoke run

## Practical guidance for the cluster run

For the real cluster experiment:

1. Keep this new Python 3.12 env and the updated dataset CLI format.
2. Use your real memory settings again, not the reduced local smoke settings.
3. Keep `MUJOCO_GL=egl` for eval.
4. Keep `rename_map` and `policy.empty_cameras=1` for the current local LIBERO dataset/model pairing.

The local reduced-memory smoke config was only to validate the merged branch and env on a smaller GPU. Your actual A100/H100 training jobs should continue using the larger memory settings from your research scripts.
