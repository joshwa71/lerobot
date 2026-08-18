#!/bin/bash
# E64 (Josh, 18 Aug): NAIVE SEQUENTIAL LoRA at r=512 — the catastrophic-forgetting
# foil at full suite length, re-provisioned to the uniform rank of every LoRA row
# in the paper table (multitask-10 r512/50k, specialists r512/5k). This is "the
# r512 specialist recipe applied sequentially": one dense adapter trained task
# after task over the SAME 10 tasks / order / budget as our method (5k steps/task,
# per-task LR reset + optimizer reinit, identical eval protocol via
# lerobot-sequential-train), with NO protection, NO memory, NO anti-forgetting
# machinery. Replaces the r256 10-task run killed at block 5 (its four boundaries
# reproduced the 5-task collapse bit-for-bit: 35 / 0,60 / 0,0,60 / 0,0,0,90).
#   r=512 / lora_alpha=128 : alpha/r = 0.25 held (r32@a8 -> r512@a128), 852M
#                            trainable — above our per-site bottleneck (288/128),
#                            per-token active (~240x) and per-step budget (~3.7x).
#   lr 1e-4 -> 1e-5 per-block linear : NOW RECIPE-IDENTICAL to the specialists
#     (the r256 foil ran 2.5e-5 -> 2.5e-6; its header called that "the preset the
#     specialists trained under", which was wrong — the specialist scripts use
#     optimizer_lr=1e-4). Same base (stage-1 libero_90 finetune), same targets,
#     bs16 x acc2 no-ckpt, 20-ep boundary evals + 50-ep final (the memory runs'
#     protocol); the 4-seed campaign is the headline instrument.
# PREEMPTION: this wrapper is SELF-RESUMING. If checkpoints/last/sequential_state.pt
# exists and the final 050000 does not, it relaunches via the E58-add-5 PEFT
# resume branch (--policy.path=<last ckpt> --policy.use_peft=true
# --resume_sequential=true): the trainer re-enables the TRAINED adapters (hard-fails
# if L1(lora_B)=0, i.e. refuses to silently run a fresh adapter) and skips the
# completed tasks. Method-exact: reinit_optimizer_each_task=true + no cross-task
# stores => the adapter weights ARE the complete cross-task state.
# SMOKE=1: 2 tasks x 20 steps, eval none, throwaway dir.
set -eo pipefail
SMOKE=${SMOKE:-0}
ROOT=/home/josh/lerobot
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

BASE_CKPT=$ROOT/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model
TARGETS='(.*\.gemma_expert\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|.*\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
[ -d "$BASE_CKPT" ] || { echo "ERROR: stage-1 base checkpoint missing"; exit 1; }

RUN=libero_10_seq10_naive_lora_r512_a128_steps5k
TASKS='[0,1,2,3,4,5,6,7,8,9]'; STEPS=5000; EVAL_TYPE=env; WANDB=true; FINAL=050000
if [ "$SMOKE" = "1" ]; then
  RUN=smoke_naive_lora_r512; TASKS='[0,1]'; STEPS=20; EVAL_TYPE=none; WANDB=false; FINAL=000040
  rm -rf $ROOT/outputs/train/$RUN
fi
RUN_DIR=$ROOT/outputs/train/$RUN

if [ -d "$RUN_DIR/checkpoints/$FINAL" ]; then
  echo "[naive-r512] final checkpoint $FINAL exists - nothing to do."; exit 0
fi
# stub-dir guard (E55/E60 lesson): a dir with no checkpoints blocks validate()
if [ -d "$RUN_DIR" ] && [ ! -d "$RUN_DIR/checkpoints" ]; then
  echo "[naive-r512] wiping stub output dir (no checkpoints): $RUN_DIR"; rm -rf "$RUN_DIR"
fi

if [ -f "$RUN_DIR/checkpoints/last/sequential_state.pt" ]; then
  echo "[naive-r512] RESUMING from $(readlink -f $RUN_DIR/checkpoints/last) (PEFT sequential resume)"
  POLICY_ARGS=(--policy.path="$RUN_DIR/checkpoints/last/pretrained_model" --policy.use_peft=true --resume_sequential=true)
else
  echo "[naive-r512] FRESH start from the stage-1 base"
  POLICY_ARGS=(--policy.path="$BASE_CKPT")
fi

lerobot-sequential-train \
  "${POLICY_ARGS[@]}" \
  --policy.empty_cameras=1 \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=false \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --peft.method_type=LORA \
  --peft.r=512 \
  --peft.lora_alpha=128 \
  --peft.target_modules="$TARGETS" \
  --peft.full_training_modules='[]' \
  --dataset.repo_id=libero_10 \
  --dataset.root="$ROOT/outputs/libero_10" \
  --rename_map="$RENAME" \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir="$RUN_DIR" \
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
  --memory_value_lr=1e-4 \
  --memory_value_lr_end=1e-5 \
  --memory_value_scheduler_type=linear \
  --save_after_each_task=true \
  --reinit_optimizer_each_task=true \
  --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'
echo "=== naive_seq_lora_r512_10task (SMOKE=$SMOKE) COMPLETE $(date -u) ==="
