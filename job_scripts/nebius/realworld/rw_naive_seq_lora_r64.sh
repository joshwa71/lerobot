#!/bin/bash
# E65 REAL-WORLD NAIVE SEQUENTIAL LoRA r64 — realworld duplicate of baselines/naive_seq_lora_r512_10task.sh
# at the r64/a16 rung (alpha/r 0.25 = the RW specialists' rank). "The r64 specialist recipe applied
# sequentially": ONE dense adapter trained task after task over the SAME tasks / order / budget as our
# chain (RW SEQ split, 5k steps/task, per-task LR reset 1e-4 -> 1e-5 linear + optimizer reinit) with NO
# protection, NO memory, NO anti-forgetting machinery — the headline catastrophic-forgetting foil.
# Deltas vs sim: RW datasets/maps (rw_env.sh); --eval.type=loss + --eval_loss_n_batches (the RW in-run
# forgetting triangle, the SAME instrument the chain's sequential stage uses) instead of env/none; no
# --env.* / --ds_to_env_map_json / episode-count args. save_after_each_task=true: the per-task
# checkpoints are the input to the post-hoc adapter-swap MSE matrix (mse_matrix_peft.py).
# SELF-RESUMING (E58-add-5 PEFT resume branch): if checkpoints/last/sequential_state.pt exists and the
# final does not, a relaunch continues from the last completed task boundary with the TRAINED adapters
# (--policy.use_peft=true --resume_sequential=true; hard-fails on L1(lora_B)=0).
# SMOKE=1: rw_env's 6 steps x 2 tasks, 2 loss-eval batches, throwaway _smoke_ dir.  DRYRUN=1: print only.
set -eo pipefail
source /home/josh/lerobot/job_scripts/nebius/realworld/rw_env.sh
DRYRUN=${DRYRUN:-0}
LORA_R=${LORA_R:-64}; LORA_ALPHA=${LORA_ALPHA:-16}
TARGETS='(.*\.gemma_expert\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|.*\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
# the REAL stage-1 checkpoint even under SMOKE=1 (rw_env prefixes STAGE1_CKPT with _smoke_; no smoke stage-1 exists)
BASE_CKPT=${BASE_CKPT:-$ROOT_DIR/outputs/train/${STAGE1_RUN}/checkpoints/last/pretrained_model}
TASKS=$RW_SEQ_TASK_IDS; N=$RW_N_SEQ; STEPS=$SEQ_STEPS
if [ "$SMOKE" = "1" ]; then TASKS='[0,1]'; N=2; fi
STEPS_TAG=$([ "$STEPS" = 5000 ] && echo 5k || echo "$STEPS")
RUN=${RUN_PREFIX}realworld_${RW_TAG}_seq${N}_naive_lora_r${LORA_R}_a${LORA_ALPHA}_steps${STEPS_TAG}
RUN_DIR=$ROOT_DIR/outputs/train/$RUN
FINAL=$(printf '%06d' $((N * STEPS)))
echo "RW naive sequential LoRA r${LORA_R}/a${LORA_ALPHA} [$RUN] tasks=$TASKS steps/task=$STEPS final=$FINAL (smoke=$SMOKE dryrun=$DRYRUN) on $(hostname) at $(date -u)"
[ -d "$BASE_CKPT" ] || { echo "ERROR: stage-1 checkpoint missing: $BASE_CKPT"; exit 1; }
python -c "import peft" || { echo "ERROR: peft not installed"; exit 1; }
if [ "$SMOKE" = "1" ] && [ "$DRYRUN" != "1" ]; then rm -rf "$RUN_DIR"; fi
if [ -d "$RUN_DIR/checkpoints/$FINAL" ]; then
  echo "[naive-r$LORA_R] final checkpoint $FINAL exists - nothing to do."; exit 0
fi
# stub-dir guard (E55/E60 lesson): a dir with no checkpoints blocks validate()
if [ -d "$RUN_DIR" ] && [ ! -d "$RUN_DIR/checkpoints" ] && [ "$DRYRUN" != "1" ]; then
  echo "[naive-r$LORA_R] wiping stub output dir (no checkpoints): $RUN_DIR"; rm -rf "$RUN_DIR"
fi
if [ -f "$RUN_DIR/checkpoints/last/sequential_state.pt" ]; then
  echo "[naive-r$LORA_R] RESUMING from $(readlink -f "$RUN_DIR/checkpoints/last") (PEFT sequential resume)"
  POLICY_ARGS=(--policy.path="$RUN_DIR/checkpoints/last/pretrained_model" --policy.use_peft=true --resume_sequential=true)
else
  echo "[naive-r$LORA_R] FRESH start from the stage-1 base"
  POLICY_ARGS=(--policy.path="$BASE_CKPT")
fi
CMD=(lerobot-sequential-train
  "${POLICY_ARGS[@]}"
  --policy.empty_cameras=1
  --policy.dtype=bfloat16
  --policy.gradient_checkpointing=false
  --policy.normalization_mapping="$RW_NORM_MAP"
  --peft.method_type=LORA
  --peft.r=$LORA_R
  --peft.lora_alpha=$LORA_ALPHA
  --peft.target_modules="$TARGETS"
  --peft.full_training_modules='[]'
  --dataset.repo_id="$RW_SEQ_ID"
  --dataset.root="$RW_SEQ_ROOT"
  --rename_map="$RW_RENAME_MAP"
  --output_dir="$RUN_DIR"
  --steps=200000
  --batch_size=16
  --gradient_accumulation_steps=2
  --num_workers=8
  --eval.type=loss
  --eval_loss_n_batches=$EVAL_LOSS_NB
  --log_freq=200
  --wandb.enable=$WANDB
  --wandb.project=vla-memory
  --job_name="$RUN"
  --online_task_ids="$TASKS"
  --online_steps_per_task=$STEPS
  --tfidf_enable=false
  --use_online_idf_stats=false
  --protect_prior_slots=false
  --memory_value_lr=1e-4
  --memory_value_lr_end=1e-5
  --memory_value_scheduler_type=linear
  --save_after_each_task=true
  --reinit_optimizer_each_task=true)
if [ "$DRYRUN" = "1" ]; then printf '  %q' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"
[ -d "$RUN_DIR/checkpoints/$FINAL" ] || { echo "[naive-r$LORA_R] ERROR: final $FINAL missing after training"; exit 1; }
echo "=== RW naive_seq_lora_r${LORA_R} [$RUN] (SMOKE=$SMOKE) COMPLETE $(date -u) ==="
