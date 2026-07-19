#!/bin/bash
# Held-out routing audit (Entry 19 review instrument).
#
# Streams the libero_10 demos through a FROZEN pretrained checkpoint using the
# normal sequential-train loop with memory_value_lr=1e-12 (numerically inert:
# total value drift over 1000 steps is ~1e-9 in Adam-normalized units), no env
# (=> no rollout eval), no checkpoint saving. The per-task usage JSONs are
# flushed after every task regardless of checkpointing, so this produces
# <output_dir>/memory_by_task/memory_usage_task_{0..9}.json in exactly the
# format the slot-analysis pipeline already reads — but measuring the PRISTINE
# PRIOR's footprints (no value adaptation, no writes).
#
# Usage: audit_heldout_routing.sh <pretrained_model_dir> <audit_run_name>

set -eo pipefail

CKPT="$1"
RUN="$2"
if [ -z "$CKPT" ] || [ -z "$RUN" ]; then
  echo "usage: $0 <pretrained_model_dir> <audit_run_name>"; exit 1
fi
if [ ! -d "$CKPT" ]; then
  echo "ERROR: checkpoint dir not found: $CKPT"; exit 1
fi

echo "Audit $RUN started on $(hostname) at $(date)"
echo "  checkpoint: $CKPT"

ROOT_DIR=/home/josh/lerobot
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
OUTPUT_DIR="$ROOT_DIR/outputs/train/$RUN"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated

cd "$ROOT_DIR"

lerobot-sequential-train \
  --policy.path="$CKPT" \
  --policy.empty_cameras=1 \
  --policy.dtype=bfloat16 \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --dataset.repo_id=libero_10 \
  --dataset.root="$SEQ_DATASET_ROOT" \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
  --output_dir="$OUTPUT_DIR" \
  --steps=200000 \
  --batch_size=32 \
  --gradient_accumulation_steps=1 \
  --num_workers=8 \
  --log_freq=100 \
  --wandb.enable=false \
  --job_name="$RUN" \
  --online_task_ids='[0,1,2,3,4,5,6,7,8,9]' \
  --online_steps_per_task=100 \
  --policy.memory_layer.aggregate_usage=false \
  --policy.memory_layer.vlm_route_once=true \
  --save_checkpoint=false \
  --save_after_each_task=false \
  --reinit_optimizer_each_task=true \
  --tfidf_enable=true \
  --tfidf_top_t=1536 \
  --use_online_idf_stats=true \
  --idf_exponent=1 \
  --memory_value_lr=1e-12 \
  --memory_value_lr_end=1e-12 \
  --memory_value_scheduler_type=linear

echo "Audit $RUN completed at $(date)"
