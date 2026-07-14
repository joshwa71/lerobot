#!/bin/bash
# Full pipeline for the validated probe-C recipe (research_log Entry 20):
#   stage 1: FRESH 40k pretrain of the recipe (clean uncompressed schedule)  (~44h)
#   stage 2: held-out routing audit of the 40k ckpt                          (~35 min)
#   stage 3: sequential adaptation on libero_10, top_t=512                   (~45h)
#
# v2 (11 Jun): stage 1 changed from resume-probe-C to a fresh 40k run.
# lerobot auto-scales the LR schedule when steps < scheduler_decay_steps, so
# the 10k probe ran a compressed cosine and resuming produced an LR sawtooth
# (floor -> 0.85*peak at step 10001) that is not comparable to the control.
#
# Stages are idempotent: each is skipped if its output already exists, so the
# pipeline can be re-run safely after a crash and continues where it stopped.
# Deliberately NOT `set -e` at the top level; each stage is checked explicitly.
#
# top_t=512 rationale (pre-committed, not audit-gated): protection now comes
# from the separated/compacted footprints, not the write mask. Per-batch
# accessed slots shrink ~7x with this prior, so 512 is RELATIVELY more
# generous than 1536 was in the old regime; libero_goal showed 512 safe at
# far worse IoU (0.17) than this prior's held-out IoU (0.05 bg / 0.19 family).

ROOT_DIR=/home/josh/lerobot
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/outputs/probe_logs"
mkdir -p "$LOG_DIR"

PRETRAIN_OUT="$ROOT_DIR/outputs/train/libero_90_pi05_8_10_12_14_contrastive_0.05_negonly_q512_40k"
PRETRAIN_40K="$PRETRAIN_OUT/checkpoints/040000/pretrained_model"

AUDIT_RUN=audit_heldout_c005_40k
AUDIT_OUT="$ROOT_DIR/outputs/train/$AUDIT_RUN"

SEQ_RUN=libero_10_sequential_pi05_8_10_12_14_contrastive_0.05_negonly_q512_40k_top_t_512
SEQ_OUTPUT_DIR="$ROOT_DIR/outputs/train/$SEQ_RUN"
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"

log() { echo "[pipeline] $(date '+%F %T') $*" | tee -a "$LOG_DIR/pipeline.log"; }

source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated

###############################################################################
# Stage 1 - fresh 40k pretrain of the probe-C recipe
###############################################################################
if [ -d "$PRETRAIN_40K" ]; then
  log "stage 1: 40k checkpoint already exists - skipping pretrain."
else
  log "stage 1: fresh 40k pretrain (contrastive 0.05, negatives_only, queue 512)"
  bash "$DIR/pretrain_c_0.05_negonly_q512_40k.sh" > "$LOG_DIR/pipeline_pretrain40k.log" 2>&1
  log "stage 1: pretrain exited with code $?"
  if [ ! -d "$PRETRAIN_40K" ]; then
    log "stage 1 FAILED: $PRETRAIN_40K missing. Aborting pipeline."
    exit 1
  fi
fi

###############################################################################
# Stage 2 - held-out routing audit of the 40k checkpoint (informational)
###############################################################################
if [ -d "$AUDIT_OUT/memory_by_task" ] && [ "$(ls "$AUDIT_OUT/memory_by_task" | wc -l)" -ge 10 ]; then
  log "stage 2: audit already complete - skipping."
else
  log "stage 2: auditing 40k checkpoint"
  bash "$DIR/audit_heldout_routing.sh" "$PRETRAIN_40K" "$AUDIT_RUN" > "$LOG_DIR/pipeline_audit_40k.log" 2>&1
  log "stage 2: audit exited with code $? (informational - not a gate)"
fi

###############################################################################
# Stage 3 - sequential adaptation on libero_10 (top_t=512)
###############################################################################
if [ -d "$SEQ_OUTPUT_DIR" ]; then
  log "stage 3: $SEQ_OUTPUT_DIR already exists - skipping (delete to re-run)."
else
  log "stage 3: sequential on libero_10 from 40k checkpoint, top_t=512"
  (
    export MUJOCO_GL=osmesa; unset DISPLAY
    export TOKENIZERS_PARALLELISM=false
    export TORCH_NCCL_BLOCKING_WAIT=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_P2P_DISABLE=1
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    cd "$ROOT_DIR"

    lerobot-sequential-train \
      --policy.path="$PRETRAIN_40K" \
      --policy.empty_cameras=1 \
      --policy.dtype=bfloat16 \
      --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
      --dataset.repo_id=libero_10 \
      --dataset.root="$SEQ_DATASET_ROOT" \
      --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
      --env.type=libero \
      --env.task=libero_10 \
      --output_dir="$SEQ_OUTPUT_DIR" \
      --steps=200000 \
      --batch_size=32 \
      --gradient_accumulation_steps=1 \
      --num_workers=8 \
      --eval.batch_size=1 \
      --eval.n_episodes=50 \
      --log_freq=200 \
      --wandb.enable=true \
      --wandb.project=vla-memory \
      --job_name="$SEQ_RUN" \
      --online_task_ids='[0,1,2,3,4,5,6,7,8,9]' \
      --online_steps_per_task=3000 \
      --policy.memory_layer.aggregate_usage=false \
      --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
      --save_after_each_task=true \
      --reinit_optimizer_each_task=true \
      --tfidf_enable=true \
      --tfidf_top_t=512 \
      --use_online_idf_stats=true \
      --idf_exponent=1 \
      --memory_value_lr=0.001 \
      --memory_value_lr_end=0.0001 \
      --memory_value_scheduler_type=linear
  ) > "$LOG_DIR/pipeline_sequential.log" 2>&1
  log "stage 3: sequential exited with code $?"
fi

log "=== pipeline done ==="
