#!/bin/bash
# Non-memory pi05 baselines (research_log Entry 30) — the standard multi-task
# finetuning CEILING we never measured. Plain pi05 (NO memory layers, no routing/
# contrastive/protection), finetuned with the SAME base train args as the memory
# runs (same scheduler/warmup, pi05's default base LR, bs32, grad-ckpt, bf16,
# empty_cameras, rename_map, normalization). The ONLY deltas vs the memory pretrain
# stage-1 command are: memory flags removed, steps 40k->50k (decay matched), and
# eval/save at the END ONLY (50k) at 50 eps/task on libero_10.
# NB: gradient_checkpointing MUST stay true here. Tested false (29 Jun) -> CUDA OOM on
# step 0 at bs32 (pi05's full-backbone activations exceed the 140GB H200 without it).
# It's a base-model requirement, not a memory-module one: the memory PRETRAIN used it for
# the same reason; the sequential stage could disable it only because it FREEZES the
# backbone (values-only -> no backbone activations stored for backward). These baselines
# train the full backbone, so they need it -- and grads are identical with/without anyway.
#
#   B1  libero_90_and_long  : 50k steps on libero_90 (3959 eps) + libero_10/Long (379 eps)
#                             -> the joint multi-task ceiling WITH the pretrain data
#   B2  libero_10           : 50k steps on just libero_10/Long (10 tasks, 379 eps)
#                             -> the 10-task-only finetune ceiling (isolates pretrain-data value)
#
# Both EVAL on libero_10 @ 50 eps/task (per-task + aggregate pc_success via eval_policy_all),
# eval+save at 50k only (code stable -> don't spend ages on eval / blow up storage).
# Runs B1 -> B2 sequentially; one failure does NOT abort the batch; skip-if-final-ckpt-exists.

set -uo pipefail
echo "BASELINES started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot

# pi05_base: same pinned revision as every working memory run (processor config loads
# with the current code). See combined/pi05_libero_10_..._sep5_...topt1536.sh.
PI05_BASE="/home/josh/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30"

LOGDIR="$ROOT_DIR/outputs/baseline_logs"; mkdir -p "$LOGDIR"

export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"

if [ ! -d "$PI05_BASE" ]; then echo "ERROR: pi05_base missing at $PI05_BASE"; exit 1; fi

# run_base <run_name> <dataset_repo_id> <dataset_root>
run_base () {
  local RUN="$1" DS_REPO="$2" DS_ROOT="$3"
  local OUT="$ROOT_DIR/outputs/train/$RUN"
  local LOG="$LOGDIR/${RUN}.log"
  echo "=============================================================="
  echo "[$(date)] BASELINE $RUN"
  echo "  dataset: $DS_REPO ($DS_ROOT)   eval: libero_10 @ 50 eps/task"
  echo "  out: $OUT"; echo "  log: $LOG"
  if [ ! -d "$DS_ROOT" ]; then echo "  -> ERROR dataset root missing: $DS_ROOT ; SKIP"; return 1; fi
  if [ -d "$OUT/checkpoints/050000" ]; then
    echo "  -> final checkpoint 050000 already exists; SKIP"; return 0
  fi
  lerobot-train \
    --policy.path="$PI05_BASE" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id="$DS_REPO" \
    --dataset.root="$DS_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_10 \
    --output_dir="$OUT" \
    --save_freq=50000 \
    --steps=50000 \
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=50 \
    --eval_freq=50000 \
    --log_freq=200 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=50000 \
    --job_name="$RUN" \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --wandb.disable_artifact=true \
    --policy.gradient_checkpointing=true \
    > "$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "  -> DONE ok"; else echo "  -> FAILED rc=$rc (continuing)"; fi
}

# B1 first (the headline joint ceiling), then B2 (10-task-only).
run_base "libero_90_and_long_pi05_base_50k" "libero_90_and_long" "$ROOT_DIR/outputs/libero_90_and_long"
run_base "libero_10_pi05_base_50k"          "libero_10"          "$ROOT_DIR/outputs/libero_10"

echo "BASELINES completed at $(date)"
