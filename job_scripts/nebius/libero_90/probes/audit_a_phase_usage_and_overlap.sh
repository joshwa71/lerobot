#!/bin/bash
# E42 addendum: regenerate the A-phase (libero-90) read profile + generalist-slot overlap
# analysis. The A-phase run never dumped memory_usage.json; under frozen-base routing the
# read profile is VALUE-INDEPENDENT (router + backbone only), so a forward-only inert-LR
# sweep of the libero-90 demos through the A-checkpoint reproduces it exactly.
#
# Stage 1: audit sweep — 90 tasks x 40 batches through the stageB A checkpoint (same
#          instrument as audit_heldout_routing.sh, pointed at libero_90). ~1.5h.
# Stage 2: analysis (scripts/vla_analysis/generalist_overlap.py) — aggregate per-slot
#          A-phase read mass per layer; intersect with each sequential task's read mass
#          and update-event mass at top-{50,20,10,5}%-mass thresholds. Feeds the
#          "freeze the generalist slots" decision (E42 addendum).
set -eo pipefail
ROOT_DIR=/home/josh/lerobot
CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_8_10_12_14_frozenroute_rwarmupB_values10k_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model"
RUN=audit_libero90_usage_rwarmupB_A
OUTPUT_DIR="$ROOT_DIR/outputs/train/$RUN"

export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$CKPT" ] || { echo "ERROR: A checkpoint missing"; exit 1; }

if [ -f "$OUTPUT_DIR/memory_by_task/memory_usage_task_89.json" ]; then
  echo "[audit] all 90 task JSONs exist - skipping sweep."
else
  lerobot-sequential-train \
    --policy.path="$CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_90 \
    --dataset.root="$ROOT_DIR/outputs/libero_90" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --output_dir="$OUTPUT_DIR" \
    --steps=200000 \
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --num_workers=8 \
    --log_freq=200 \
    --wandb.enable=false \
    --job_name="$RUN" \
    --online_task_ids="[$(seq -s, 0 89)]" \
    --online_steps_per_task=40 \
    --policy.memory_layer.aggregate_usage=false \
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
fi

python scripts/vla_analysis/generalist_overlap.py
echo "A-phase usage audit + overlap analysis completed at $(date)"
