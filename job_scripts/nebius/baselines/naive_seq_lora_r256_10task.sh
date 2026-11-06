#!/bin/bash
# 10-TASK VARIANT (Josh, 17 Aug): the catastrophic-forgetting foil at full suite
# length, to sit beside the 10-task memory row (67.8). Deltas vs the 5-task cell
# (18.0 at 4 seeds): online_task_ids [0..9], final ckpt 050000. Everything else
# byte-identical — same base, same targets, r=256/alpha=64, 5k steps/task,
# per-task optimizer reinit, no protection/tfidf/memory.
# E58 NAIVE SEQUENTIAL LoRA — r=256 (capacity-sweep point 2 of 3; the headline
# forgetting baseline, never previously run)
# =====================================================================================
# One dense LoRA adapter trained SEQUENTIALLY over the same 5 tasks / order / budget as
# our method (t0-t4, 5k steps/task, per-task LR reset + optimizer reinit, identical
# eval protocol via lerobot-sequential-train) with NO protection, NO memory, NO
# anti-forgetting machinery of any kind. The "is it just parameters" control (E58
# discussion): the sweep shows naive adaptation's forgetting is capacity-INSENSITIVE —
# r=32 (53M) / r=256 (426M, this arm) / big point (TBD with supervisor: full-FT 6.6B or
# r~2000 adapter) — so no parameter count rescues the naive method.
#   - alpha/r held at the specialists' 0.25 (r32@a8 -> r256@a64): capacity varies,
#     effective adapter gain does not.
#   - LR = the pi05 preset the specialists trained under (2.5e-5 -> 2.5e-6, per-block
#     linear like our arms). AdamW wd=0.
#   - Same base (stage-1 libero_90 finetune), same seeds/eval protocol as B ->
#     retention matrices directly comparable.
# Comparators: B 53.2 / dose05x 54.4 (flat matrices, give-back <= +3.8%); specialists
# 63.2 (no CL constraint); multitask-LoRA 49.2. Pre-registered expectation: strong
# diagonal fits (>= specialist-level early-task inits) followed by the classic
# catastrophic-forgetting collapse of early tasks; the MSE matrix (run after) is the
# headline figure alongside ours.
# SMOKE=1: 2 tasks x 20 steps, eval none, throwaway dir — exercises wrap/optimizer/
# step/save end-to-end in ~10 min.
# =====================================================================================
set -eo pipefail
SMOKE=${SMOKE:-0}
ROOT=/home/josh/lerobot
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

BASE_CKPT=$ROOT/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model
TARGETS='(.*\.gemma_expert\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|.*\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'

RUN=libero_10_seq10_naive_lora_r256_a64_steps5k
TASKS='[0,1,2,3,4,5,6,7,8,9]'; STEPS=5000; EVAL_TYPE=env; WANDB=true
if [ "$SMOKE" = "1" ]; then
  RUN=smoke_naive_lora_r256; TASKS='[0,1]'; STEPS=20; EVAL_TYPE=none; WANDB=false
  rm -rf $ROOT/outputs/train/$RUN
fi

lerobot-sequential-train \
  --policy.path="$BASE_CKPT" \
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
  --output_dir="$ROOT/outputs/train/$RUN" \
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
  --online_task_ids="$TASKS" \
  --online_steps_per_task=$STEPS \
  --tfidf_enable=false \
  --use_online_idf_stats=false \
  --protect_prior_slots=false \
  --memory_value_lr=2.5e-5 \
  --memory_value_lr_end=2.5e-6 \
  --memory_value_scheduler_type=linear \
  --save_after_each_task=true \
  --reinit_optimizer_each_task=true \
  --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'
echo "=== naive_seq_lora_r256 (SMOKE=$SMOKE) COMPLETE $(date -u) ==="
