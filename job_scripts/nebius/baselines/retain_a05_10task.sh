#!/bin/bash
# E67 (Josh, 7 Sep 26): RETAIN baseline (Yadav et al., ICLR 2026) under our continual protocol.
# Full fine-tune of the running merged model on each task, then theta <- 0.5*theta_prev + 0.5*theta_ft
# (their continual setting: alpha fixed at 0.5 at every boundary; task-FT variant, no replay).
#
# RECIPE = the paper's joint full-FT rows (pi0.5 preset: AdamW peak 2.5e-5, wd 0.01, betas
# (0.9,0.95), clip 1.0, cosine to 2.5e-6) compressed to 5,000 steps/task with a 500-step warm-up;
# bs8 x acc4 (effective 32; the measured no-grad-ckpt rung, E60 add-7: 2.20 s/step, 92G).
# Everything else = the sequential rows: stage-1 libero_90 base, bf16, 10 LIBERO-10 tasks in
# dataset order, optimizer reinit + LR reset per task, no task identity, seed 1000.
# Trainer: scripts/baselines/retain/retain_sequential_train.py (resumes by default; SIGTERM-safe).
# SMOKE=1: 2 tasks x 20 steps in a throwaway dir, with a forced stop/resume and the dense loss
# matrix on both boundaries (end-to-end instrument check). Prints E67-RETAIN-SMOKE-OK on success.
set -eo pipefail
SMOKE=${SMOKE:-0}
ALPHA=${ALPHA:-0.5}
ROOT=/home/josh/lerobot
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
BASE_CKPT=$ROOT/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
[ -d "$BASE_CKPT" ] || { echo "ERROR: stage-1 base checkpoint missing: $BASE_CKPT"; exit 1; }
ATAG=$(python3 -c "print(('%g' % $ALPHA).replace('0.','0').replace('.',''))")   # 0.5 -> a05
RUN=libero_10_seq10_retain_a${ATAG}_fullft_steps5k
TASKS='[0,1,2,3,4,5,6,7,8,9]'; STEPS=5000; CKPT_EVERY=1000; WANDB=true; LOGF=100; EXTRA=()
if [ "$SMOKE" = "1" ]; then
  RUN=smoke_retain_a${ATAG}; TASKS='[0,1]'; STEPS=20; CKPT_EVERY=10; WANDB=false; LOGF=5
  rm -rf $ROOT/outputs/train/$RUN
fi
RUN_DIR=$ROOT/outputs/train/$RUN
run_retain () {  # <bs> <accum> [extra args...]
  local bs=$1 acc=$2; shift 2
  echo "=== RETAIN alpha=$ALPHA — tasks $TASKS x $STEPS steps, bs$bs x acc$acc ($(date -u)) ==="
  python scripts/baselines/retain/retain_sequential_train.py \
    --policy.path="$BASE_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.optimizer_lr=2.5e-5 \
    --policy.scheduler_warmup_steps=500 \
    --policy.scheduler_decay_steps=$STEPS \
    --policy.scheduler_decay_lr=2.5e-6 \
    --policy.push_to_hub=false \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_10 \
    --dataset.root="$ROOT/outputs/libero_10" \
    --rename_map="$RENAME" \
    --output_dir="$RUN_DIR" \
    --batch_size=$bs \
    --gradient_accumulation_steps=$acc \
    --num_workers=8 \
    --log_freq=$LOGF \
    --seed=1000 \
    --wandb.enable=$WANDB \
    --wandb.project=vla-memory \
    --wandb.disable_artifact=true \
    --job_name="$RUN" \
    --online_task_ids="$TASKS" \
    --online_steps_per_task=$STEPS \
    --retain_alpha=$ALPHA \
    --ckpt_every=$CKPT_EVERY \
    "$@"
}
if [ "$SMOKE" = "1" ]; then
  # 1) forced stop after 25 global steps (task 1 done, task 2 at step 5) -> 2) resume to completion
  run_retain 8 4 --stop_after_steps=25 | tee /tmp/retain_smoke_1.log
  grep -q "RETAIN-STOP-AFTER-STEPS" /tmp/retain_smoke_1.log || { echo "E67-RETAIN-SMOKE-FAIL (no stop marker)"; exit 1; }
  [ -f "$RUN_DIR/retain_state/current/meta.json" ] || { echo "E67-RETAIN-SMOKE-FAIL (no in-progress state)"; exit 1; }
  run_retain 8 4 | tee /tmp/retain_smoke_2.log
  grep -q "resume: in-progress task 1 at step 5" /tmp/retain_smoke_2.log || { echo "E67-RETAIN-SMOKE-FAIL (did not resume at step 5)"; exit 1; }
  grep -q "RETAIN-CHAIN-DONE" /tmp/retain_smoke_2.log || { echo "E67-RETAIN-SMOKE-FAIL (no done marker)"; exit 1; }
  for b in 000020 000040; do
    [ -f "$RUN_DIR/checkpoints/$b/pretrained_model/model.safetensors" ] || { echo "E67-RETAIN-SMOKE-FAIL (boundary $b missing)"; exit 1; }
    [ -f "$RUN_DIR/checkpoints/$b/retain_boundary.json" ] || { echo "E67-RETAIN-SMOKE-FAIL (boundary json $b missing)"; exit 1; }
  done
  # 3) the dense loss-matrix instrument on both boundaries (the RETAIN drift instrument)
  M=$RUN_DIR/mse_matrix_smoke.jsonl; rm -f "$M"
  MSEMAT_RUN_DIR=$RUN_DIR MSEMAT_STEPS=000020,000040 MSEMAT_TASKS=0,1 MSEMAT_NBATCHES=4 MSEMAT_OUT=$M \
  python scripts/baselines/mse_matrix_dense.py \
    --policy.path=$RUN_DIR/checkpoints/000020/pretrained_model --policy.empty_cameras=1 --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_10 --dataset.root=$ROOT/outputs/libero_10 --rename_map="$RENAME" \
    --output_dir=$RUN_DIR/msemat_out --steps=40 --batch_size=32 --num_workers=4 \
    --online_task_ids='[0,1]' --online_steps_per_task=20 --wandb.enable=false --job_name=smoke_retain_msemat
  [ "$(grep -c '^{' $M)" = "2" ] || { echo "E67-RETAIN-SMOKE-FAIL (matrix rows $(grep -c '^{' $M) != 2)"; exit 1; }
  cat "$M"
  echo "E67-RETAIN-SMOKE-OK"; exit 0
fi
if [ -f "$RUN_DIR/retain_state/progress.json" ] && [ "$(python3 -c "import json;print(json.load(open('$RUN_DIR/retain_state/progress.json'))['completed_tasks'])")" = "10" ]; then
  echo "[retain] all 10 boundaries exist - nothing to do."; echo "RETAIN-CHAIN-DONE"; exit 0
fi
# VRAM ladder (effective batch 32 held): the trainer resumes from its own state on each rung
LADDER=${LADDER:-"8:4,4:8"}
ok=0
for rung in ${LADDER//,/ }; do
  IFS=: read -r rb ra <<< "$rung"
  if run_retain "$rb" "$ra" 2>&1 | tee /tmp/retain_last.log | grep -v "^$"; then ok=1; break; fi
  if grep -q "OutOfMemoryError" /tmp/retain_last.log; then echo "[retain] rung bs=$rb OOM - next rung"; continue; fi
  echo "[retain] rung bs=$rb failed for a non-VRAM reason - aborting (state on disk is resumable)"; exit 1
done
[ "$ok" = 1 ] || { echo "ERROR: all LADDER rungs failed"; exit 1; }
echo "=== retain_a${ATAG}_10task (SMOKE=$SMOKE) EXIT OK $(date -u) ==="
