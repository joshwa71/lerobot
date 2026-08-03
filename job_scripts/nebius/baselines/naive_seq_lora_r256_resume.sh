#!/bin/bash
# E58 NAIVE SEQUENTIAL LoRA r256 — PREEMPTION-RESUME wrapper (3 Aug 26)
# =====================================================================================
# The 2 Aug run (wandb bncprsuz) was spot-preempted at 22:12 UTC, step ~21.4K — 1.4K
# steps into the FINAL block (task 4 / e7). Tasks 0-3 are fully banked (checkpoints
# 005000-020000 + boundary evals + sequential_state.pt). This wrapper relaunches from
# checkpoints/last (=020000) and re-runs task 4 only.
#
# Deltas vs naive_seq_lora_r256.sh (the fresh-launch wrapper, kept canonical):
#   --policy.path         -> the per-task checkpoint (not the stage-1 base)
#   --policy.use_peft=true   (factory rebuilds base + TRAINED adapters from the ckpt)
#   --resume_sequential=true (skips tasks 0-3; restores task index + eval histories)
# Requires the trainer's PEFT-resume branch (same commit as this file): it re-enables
# the loaded adapters instead of fresh-wrapping, and hard-fails if the adapters look
# freshly initialized (L1(lora_B)=0) — i.e. it refuses to silently run a different
# method.
#
# Method-exactness notes: reinit_optimizer_each_task=true means no optimizer/scheduler
# state crosses task boundaries by design, and this baseline has no protection/IDF
# stores — the adapter weights in the checkpoint are the complete cross-task state.
# The only divergence vs an uninterrupted run is dataloader RNG within the e7 block
# (batch order), which is not a controlled variable across arms.
# SMOKE=1: 25-step task-4 resume into a throwaway output dir, eval none, wandb off —
# exercises the load path + resume branch end-to-end without touching the real run dir.
# =====================================================================================
set -eo pipefail
SMOKE=${SMOKE:-0}
ROOT=/home/josh/lerobot
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

RUN=libero_10_seq5_naive_lora_r256_a64_steps5k
RUN_DIR=$ROOT/outputs/train/$RUN
CKPT=$RUN_DIR/checkpoints/last
[ -f "$CKPT/sequential_state.pt" ] || { echo "FATAL: $CKPT/sequential_state.pt missing — refusing to resume"; exit 1; }

TARGETS='(.*\.gemma_expert\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|.*\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'

OUT_DIR=$RUN_DIR; STEPS=5000; EVAL_TYPE=env; WANDB=true
if [ "$SMOKE" = "1" ]; then
  OUT_DIR=$ROOT/outputs/train/smoke_naive_lora_r256_resume; STEPS=25; EVAL_TYPE=none; WANDB=false
  rm -rf $OUT_DIR
fi

lerobot-sequential-train \
  --policy.path="$CKPT/pretrained_model" \
  --policy.use_peft=true \
  --policy.empty_cameras=1 \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=false \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --peft.method_type=LORA \
  --peft.r=256 \
  --peft.lora_alpha=64 \
  --peft.target_modules="$TARGETS" \
  --peft.full_training_modules='[]' \
  --dataset.repo_id=libero_10 \
  --dataset.root="$ROOT/outputs/libero_10" \
  --rename_map="$RENAME" \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir="$OUT_DIR" \
  --steps=200000 \
  --batch_size=16 \
  --gradient_accumulation_steps=2 \
  --num_workers=8 \
  --eval.type=$EVAL_TYPE \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --eval_final_episodes=50 \
  --log_freq=200 \
  --wandb.enable=$WANDB \
  --wandb.project=vla-memory \
  --job_name="$RUN" \
  --online_task_ids='[0,1,2,3,4]' \
  --online_steps_per_task=$STEPS \
  --tfidf_enable=false \
  --use_online_idf_stats=false \
  --protect_prior_slots=false \
  --memory_value_lr=2.5e-5 \
  --memory_value_lr_end=2.5e-6 \
  --memory_value_scheduler_type=linear \
  --save_after_each_task=true \
  --reinit_optimizer_each_task=true \
  --resume_sequential=true \
  --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'
echo "=== naive_seq_lora_r256 RESUME (SMOKE=$SMOKE) COMPLETE $(date -u) ==="
