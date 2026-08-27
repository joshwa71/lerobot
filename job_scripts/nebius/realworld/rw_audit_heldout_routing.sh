#!/bin/bash
# Held-out routing audit — REAL-WORLD duplicate of libero_90/probes/audit_heldout_routing.sh
# (Entry 19 instrument). Streams the held-out SEQ split through a FROZEN checkpoint with
# memory_value_lr=1e-12 (numerically inert), no env, no checkpoint saving; the per-task usage
# JSONs are flushed after every task -> <output_dir>/memory_by_task/memory_usage_task_{t}.json
# for t in RW_SEQ_TASK_IDS, measuring the PRISTINE prior's footprints.
#
# Usage: rw_audit_heldout_routing.sh <pretrained_model_dir> <audit_run_name>
# Env (from rw_env.sh, or set explicitly): RW_SEQ_ROOT RW_SEQ_ID RW_SEQ_TASK_IDS RW_RENAME_MAP
# RW_NORM_MAP AUDIT_BS (8) AUDIT_STEPS (400: at bs8 = 3200 samples/task, the E48 matched-coverage
# convention) ROOT_DIR
set -eo pipefail
CKPT="$1"; RUN="$2"
if [ -z "$CKPT" ] || [ -z "$RUN" ]; then echo "usage: $0 <pretrained_model_dir> <audit_run_name>"; exit 1; fi
[ -d "$CKPT" ] || { echo "ERROR: checkpoint dir not found: $CKPT"; exit 1; }
ROOT_DIR=${ROOT_DIR:-/home/josh/lerobot}
RW_SEQ_ROOT=${RW_SEQ_ROOT:?set RW_SEQ_ROOT}
RW_SEQ_ID=${RW_SEQ_ID:-$(basename "$RW_SEQ_ROOT")}
RW_SEQ_TASK_IDS=${RW_SEQ_TASK_IDS:-[0,1,2,3,4]}
RW_RENAME_MAP=${RW_RENAME_MAP:-'{"observation.images.cam_high":"observation.images.base_0_rgb","observation.images.cam_wrist":"observation.images.left_wrist_0_rgb"}'}
RW_NORM_MAP=${RW_NORM_MAP:-'{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'}
OUTPUT_DIR="$ROOT_DIR/outputs/train/$RUN"
echo "Audit $RUN started on $(hostname) at $(date)  ckpt=$CKPT  seq=$RW_SEQ_ROOT ids=$RW_SEQ_TASK_IDS bs=${AUDIT_BS:-8} steps=${AUDIT_STEPS:-400}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"

lerobot-sequential-train \
  --policy.path="$CKPT" \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --policy.dtype=bfloat16 \
  --policy.normalization_mapping="$RW_NORM_MAP" \
  --dataset.repo_id="$RW_SEQ_ID" \
  --dataset.root="$RW_SEQ_ROOT" \
  --rename_map="$RW_RENAME_MAP" \
  --output_dir="$OUTPUT_DIR" \
  --steps=200000 \
  --batch_size=${AUDIT_BS:-8} \
  --gradient_accumulation_steps=1 \
  --num_workers=8 \
  --log_freq=100 \
  --wandb.enable=false \
  --job_name="$RUN" \
  --online_task_ids="$RW_SEQ_TASK_IDS" \
  --online_steps_per_task=${AUDIT_STEPS:-400} \
  --eval.type=none \
  --policy.memory_layer.aggregate_usage=false \
  --policy.memory_layer.vlm_route_once=true \
  --policy.memory_layer.router_only_fast=false \
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
