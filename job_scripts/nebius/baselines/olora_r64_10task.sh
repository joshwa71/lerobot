#!/bin/bash
# E67 (Josh, 7 Sep 26): O-LoRA baseline (Wang et al., EMNLP Findings 2023) under our continual
# protocol — the r=64 specialist/naive LoRA recipe verbatim (same target set, alpha/r=0.25, lr
# 1e-4 -> 1e-5 linear per task, bs16 x acc2, AdamW betas (0.9,0.999) wd 0, clip 1.0, 5,000
# steps/task, optimizer reinit per task; micro-batch 8 x accum 4, see LADDER) + one NEW adapter per task with the earlier adapters
# frozen-but-active + the orthogonality penalty lambda1 * sum_{i<t} |A_i A_t^T|_1 (official-code
# form), lambda1 = 0.5, skipped on the two 32-input projections (state_proj, action_in_proj).
# Every boundary is exported as ONE rank-concatenated PEFT adapter (padded to 640) so the
# standard eval campaign (--policy.use_peft=true) and mse_matrix_peft.py run unchanged.
# Trainer: scripts/baselines/olora/olora_sequential_train.py (resumes by default; SIGTERM-safe).
# SMOKE=1: 2 tasks x 20 steps, forced stop/resume, export exactness checks, mse_matrix_peft on
# both boundaries. Prints E67-OLORA-SMOKE-OK on success.
set -eo pipefail
SMOKE=${SMOKE:-0}
ROOT=/home/josh/lerobot
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
BASE_CKPT=$ROOT/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model
# the oracle / naive r512 / specialist target set (no vision tower)
TARGETS='(.*\.gemma_expert\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|.*\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
LORA_R=${LORA_R:-64}; LORA_ALPHA=${LORA_ALPHA:-16}; LAMBDA1=${LAMBDA1:-0.5}
[ -d "$BASE_CKPT" ] || { echo "ERROR: stage-1 base checkpoint missing: $BASE_CKPT"; exit 1; }
python -c "import peft" || { echo "ERROR: peft not installed"; exit 1; }
LTAG=$(python3 -c "print(('%g' % $LAMBDA1).replace('0.','0').replace('.',''))")   # 0.5 -> lam05
RUN=libero_10_seq10_olora_r${LORA_R}_a${LORA_ALPHA}_lam${LTAG}_steps5k
TASKS='[0,1,2,3,4,5,6,7,8,9]'; STEPS=5000; CKPT_EVERY=1000; WANDB=true; LOGF=100; NTASK=10
if [ "$SMOKE" = "1" ]; then
  RUN=smoke_olora_r${LORA_R}; TASKS='[0,1]'; STEPS=20; CKPT_EVERY=10; WANDB=false; LOGF=5; NTASK=2
  rm -rf $ROOT/outputs/train/$RUN
fi
RUN_DIR=$ROOT/outputs/train/$RUN
EXPORT_RANK=$((LORA_R*NTASK))
run_olora () {  # <bs> <accum> [extra args...]
  local bs=$1 acc=$2; shift 2
  echo "=== O-LoRA r$LORA_R/a$LORA_ALPHA lambda1=$LAMBDA1 — tasks $TASKS x $STEPS steps, bs$bs x acc$acc ($(date -u)) ==="
  python scripts/baselines/olora/olora_sequential_train.py \
    --policy.path="$BASE_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --policy.push_to_hub=false \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --peft.method_type=LORA \
    --peft.r=$LORA_R \
    --peft.lora_alpha=$LORA_ALPHA \
    --peft.target_modules="$TARGETS" \
    --peft.full_training_modules='[]' \
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
    --lr_start=1e-4 --lr_end=1e-5 \
    --lambda1=$LAMBDA1 \
    --export_rank=$EXPORT_RANK \
    --ckpt_every=$CKPT_EVERY \
    "$@"
}
if [ "$SMOKE" = "1" ]; then
  run_olora 8 4 --stop_after_steps=25 2>&1 | tee /tmp/olora_smoke_1.log
  grep -q "OLORA-STOP-AFTER-STEPS" /tmp/olora_smoke_1.log || { echo "E67-OLORA-SMOKE-FAIL (no stop marker)"; exit 1; }
  grep -q "OLORA-EXPORT-CHECK-OK" /tmp/olora_smoke_1.log || { echo "E67-OLORA-SMOKE-FAIL (boundary-1 export check)"; exit 1; }
  run_olora 8 4 2>&1 | tee /tmp/olora_smoke_2.log
  grep -q "resume: in-progress task 1 at step 5" /tmp/olora_smoke_2.log || { echo "E67-OLORA-SMOKE-FAIL (did not resume at step 5)"; exit 1; }
  grep -q "OLORA-CHAIN-DONE" /tmp/olora_smoke_2.log || { echo "E67-OLORA-SMOKE-FAIL (no done marker)"; exit 1; }
  grep -q "OLORA-EXPORT-CHECK-OK" /tmp/olora_smoke_2.log || { echo "E67-OLORA-SMOKE-FAIL (boundary-2 export check)"; exit 1; }
  for b in 000020 000040; do
    [ -f "$RUN_DIR/checkpoints/$b/pretrained_model/adapter_model.safetensors" ] || { echo "E67-OLORA-SMOKE-FAIL (boundary $b missing)"; exit 1; }
  done
  # the PEFT adapter-swap loss matrix (E58 add-5 instrument) on both boundaries, through the factory's use_peft path
  M=$RUN_DIR/mse_matrix_smoke.jsonl; rm -f "$M"
  MSEMAT_RUN_DIR=$RUN_DIR MSEMAT_STEPS=000020,000040 MSEMAT_TASKS=0,1 MSEMAT_OUT=$M \
  python scripts/vla_analysis/mse_matrix_peft.py \
    --policy.path=$RUN_DIR/checkpoints/000020/pretrained_model --policy.use_peft=true \
    --policy.empty_cameras=1 --policy.dtype=bfloat16 --policy.gradient_checkpointing=false \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_10 --dataset.root=$ROOT/outputs/libero_10 --rename_map="$RENAME" \
    --output_dir=$RUN_DIR/msemat_out --steps=40 --batch_size=32 --num_workers=4 \
    --online_task_ids='[0,1]' --online_steps_per_task=20 --tfidf_enable=false --wandb.enable=false --job_name=smoke_olora_msemat
  [ "$(grep -c '^{' $M)" = "2" ] || { echo "E67-OLORA-SMOKE-FAIL (matrix rows $(grep -c '^{' $M) != 2)"; exit 1; }
  cat "$M"
  echo "E67-OLORA-SMOKE-OK"; exit 0
fi
if [ -f "$RUN_DIR/olora_state/progress.json" ] && [ "$(python3 -c "import json;print(json.load(open('$RUN_DIR/olora_state/progress.json'))['completed_tasks'])")" = "10" ]; then
  echo "[olora] all 10 boundaries exist - nothing to do."; echo "OLORA-CHAIN-DONE"; exit 0
fi
LADDER=${LADDER:-"8:4,8:4,4:8"}   # bs8 x acc4: the smoke peaked at 118G at bs16 x acc2, which would block the eval unit from overlapping; retry once before demoting
ok=0
for rung in ${LADDER//,/ }; do
  IFS=: read -r rb ra <<< "$rung"
  if run_olora "$rb" "$ra" 2>&1 | tee /tmp/olora_last.log | grep --line-buffered -v "^$"; then ok=1; break; fi
  if grep -q "OutOfMemoryError" /tmp/olora_last.log; then echo "[olora] rung bs=$rb OOM - next rung"; continue; fi
  echo "[olora] rung bs=$rb failed for a non-VRAM reason - aborting (state on disk is resumable)"; exit 1
done
[ "$ok" = 1 ] || { echo "ERROR: all LADDER rungs failed"; exit 1; }
echo "=== olora_r${LORA_R}_10task (SMOKE=$SMOKE) EXIT OK $(date -u) ==="
