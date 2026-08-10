#!/bin/bash
# FULL-FINETUNE BASELINE 1 (Josh, 7 Aug): raw pi05 base -> full finetune on
# libero_10 (multitask, no memory) 50K -> 4-seed eval (E60 campaign instrument).
# NOTE vs the deleted "72.6": that was E31's B1 = libero_90+libero_10 JOINT
# finetune; the libero_10-only cell (THIS one) was B2, killed before completion —
# this is a NEW cell, not a redo. Train args mirror the E31/stage-1 convention
# verbatim (bs32, grad-ckpt REQUIRED for full-backbone, warmup 4K/decay 50K,
# pi05 default LR); in-run eval disabled (eval_freq > steps) — the 4-seed
# campaign is the instrument. Preemption-safe: save_freq=10000 + auto-resume.
# Gated on the `baseline-seeds` unit (multitask/naive 4-seed evals) exiting.
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/fullft_baselines.log
exec >> "$LOG" 2>&1

while true; do
  st=$(systemctl is-active baseline-seeds 2>/dev/null)
  [ "$st" = "active" ] || [ "$st" = "activating" ] || break
  sleep 300
done
echo "=== fullft-l10: baseline-seeds exited (state=$st) — starting $(date -u) ==="

source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

PI05_BASE="/home/josh/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30"
RUN=libero_10_pi05_fullft_frombase_nomem_50k
OUT=$ROOT/outputs/train/$RUN
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'

# stub-dir guard (E55/E60 lesson): an aborted start leaves an output dir with no
# checkpoints, which blocks lerobot-train's validate() on the next launch
if [ -d "$OUT" ] && [ ! -d "$OUT/checkpoints" ]; then
  echo "[ft] wiping stub output dir (no checkpoints): $OUT"
  rm -rf "$OUT"
fi

if [ -d "$OUT/checkpoints/050000" ]; then
  echo "[ft] final checkpoint exists - skipping train."
elif [ -d "$OUT/checkpoints/last/pretrained_model" ]; then
  echo "[ft] RESUMING from $(readlink -f $OUT/checkpoints/last)"
  lerobot-train --resume=true \
    --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" \
    --batch_size=8 --gradient_accumulation_steps=4 --policy.gradient_checkpointing=false
else
  lerobot-train \
    --policy.path="$PI05_BASE" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_10 \
    --dataset.root="$ROOT/outputs/libero_10" \
    --rename_map="$RENAME" \
    --env.type=libero \
    --env.task=libero_10 \
    --output_dir="$OUT" \
    --save_freq=10000 \
    --steps=50000 \
    --batch_size=8 \
    --gradient_accumulation_steps=4 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=50 \
    --eval_freq=60000 \
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
    --policy.gradient_checkpointing=false
fi
[ -d "$OUT/checkpoints/050000/pretrained_model" ] || { echo "[ft] FATAL: 050000 missing"; exit 1; }

# prune optimizer states of non-final checkpoints (full-backbone Adam ~40G each)
for d in "$OUT"/checkpoints/0*/training_state; do
  case "$d" in *"/050000/"*) ;; *) [ -d "$d" ] && rm -rf "$d" && echo "[prune] $d" ;; esac
done

export CAMP_SEEDS="1000,2000,3000,4000"
export CAMP_TAG=fullft_l10
export CAMP_OUT=$ROOT/outputs/analysis/e60/seeds_fullft_l10.json
if [ ! -f "$CAMP_OUT" ]; then
  python scripts/vla_analysis/eval_seeds_campaign.py \
    --policy.path="$OUT/checkpoints/050000/pretrained_model" \
    --policy.dtype=bfloat16 \
    --env.type=libero --env.task=libero_10 --env.task_ids="[4,6,9,2,7]" \
    --rename_map="$RENAME" \
    --eval.batch_size=13 --eval.n_episodes=25 \
    --seed=1000 \
    --output_dir=/tmp/camp_fullft_l10 \
    || echo "[FAIL] fullft_l10 campaign"
fi
echo "=== FULLFT-L10 COMPLETE $(date -u) ==="
