#!/bin/bash
# WEEKEND BASELINES (Josh, 13 Aug): 10-task multitask LoRA — one adapter, ALL
# 379 libero_10 episodes (task_index 0-9), for the 10-task table's
# multitask-adapter row. BUDGET CONVENTION (flagged to Josh): the front-5
# multitask cell used 5k total steps for 5 tasks = 1k/task; this scales the
# same convention to 10 tasks = 10000 total steps (decay 10000, schedule
# honored per the E20 gotcha). Otherwise byte-identical recipe to
# loraft_multitask5 (frozen stage-1 base, r=32, same targets, bs16xacc2
# no-ckpt, lr 1e-4). Eval = the 4-seed campaign, not in-run.
set -eo pipefail
echo "Weekend multitask-LoRA-10 started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
BASE_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model"
RUN_DIR="$ROOT_DIR/outputs/train/loraft_multitask10"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$BASE_CKPT" ] || { echo "ERROR: stage-1 base checkpoint missing"; exit 1; }

TARGETS='(.*\.gemma_expert\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|.*\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'

# stub-dir guard (E55/E60 lesson)
if [ -d "$RUN_DIR" ] && [ ! -d "$RUN_DIR/checkpoints" ]; then
  echo "[mt10] wiping stub output dir (no checkpoints): $RUN_DIR"
  rm -rf "$RUN_DIR"
fi
if [ -d "$RUN_DIR/checkpoints/010000" ]; then
  echo "[mt10] final checkpoint exists - skipping train."
else
  lerobot-train \
    --policy.path="$BASE_CKPT" \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --gradient_accumulation_steps=2 \
    --policy.optimizer_lr=1e-4 \
    --policy.scheduler_warmup_steps=200 \
    --policy.scheduler_decay_steps=10000 \
    --policy.scheduler_decay_lr=1e-5 \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --peft.method_type=LORA \
    --peft.r=32 \
    --peft.target_modules="$TARGETS" \
    --peft.full_training_modules='[]' \
    --dataset.repo_id=libero_10 \
    --dataset.root="$ROOT_DIR/outputs/libero_10" \
    --rename_map="$RENAME" \
    --output_dir="$RUN_DIR" \
    --steps=10000 \
    --batch_size=16 \
    --num_workers=8 \
    --log_freq=200 \
    --save_freq=10000 \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --job_name="loraft_multitask10"
fi
[ -d "$RUN_DIR/checkpoints/010000" ] || { echo "[mt10] FATAL: 010000 missing"; exit 1; }
echo "Weekend multitask-LoRA-10 completed at $(date)"
