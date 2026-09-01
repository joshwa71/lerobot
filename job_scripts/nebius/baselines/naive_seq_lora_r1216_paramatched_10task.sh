#!/bin/bash
# E66 (Josh, 1 Sep 26): PARAMETER-MATCHED naive sequential LoRA — the reviewer control for
# "your method just does better because you add loads of parameters".
#
# THE MATCH. Our merged-6x2 memory adds 2.684B parameters (measured from the checkpoint: 7 shared
# tables x 65,536 slots; expert d=1024 -> 268.4M/table, VLM d=2048 -> 536.9M/table). To ADD the same
# count with a dense adapter we need rank 1216 over a target set that includes the vision tower:
#   target set = attn q/k/v/o + MLP gate/up/down on BOTH towers, all layers,
#                + action/state projections,
#                + vision tower self_attn q/k/v/out_proj + mlp fc1/fc2 (27 encoder layers)
#   -> 416 matrices, 2.2044M params per rank unit
#   -> r=1216 gives 2.681B added, -0.1% vs our 2.684B.  alpha=304 holds alpha/r=0.25.
#
# HONEST CAVEAT TO CARRY INTO THE PAPER: those 416 matrices contain only 2.704B parameters in TOTAL,
# so a dense adapter matching our count is over-complete on 362/416 of them (true at ANY rank near
# the match — it is structural, not a rank choice). I.e. matching our added-parameter count with a
# dense adapter is only possible by roughly equalling full fine-tuning of everything it touches,
# while paying dense compute at EVERY token; our memory is a rank-2 adapter per token (~0.1% of the
# slots active per forward). We run the baseline anyway — that is the point of the control.
#
# EVERYTHING ELSE IS THE ORACLE/NAIVE RECIPE, UNCHANGED (Josh: "make sure all other training params
# align with the oracle cells"): stage-1 libero_90 base, bf16, no grad-ckpt, bs16 x accum2
# (effective 32), 5,000 steps/task over the same 10 tasks in the same order, per-task LR reset
# 1e-4 -> 1e-5 linear, optimizer reinit each task, alpha/r = 0.25, no protection / no memory /
# no TF-IDF, save_after_each_task, --eval.type=none (the 4-seed campaign is the instrument).
# Deltas vs naive_seq_lora_r512_10task.sh are EXACTLY: r 512->1216, alpha 128->304, + vision targets.
#
# Comparators (all 4-seed, 25 eps, same instrument): ours merged6x2 10-task 65.1 /
# specialists 63.7 / multitask-LoRA-10 53.2 / naive seq-LoRA r512 (852M) 9.7 / naive r32, r256.
#
# PREEMPTION: self-resuming via the E58-add-5 PEFT branch (--policy.use_peft=true
# --resume_sequential=true); the trainer hard-fails if L1(lora_B)=0 rather than silently running a
# fresh adapter. SMOKE=1: 2 tasks x 20 steps, throwaway dir.
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
# base target set (identical to the oracles / naive r512) + the vision tower
TARGETS='(.*\.gemma_expert\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|.*\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|.*\.vision_tower\.vision_model\.encoder\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|mlp\.fc(1|2))|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
LORA_R=${LORA_R:-1216}
LORA_ALPHA=${LORA_ALPHA:-304}

[ -d "$BASE_CKPT" ] || { echo "ERROR: stage-1 base checkpoint missing: $BASE_CKPT"; exit 1; }
python -c "import peft" || { echo "ERROR: peft not installed"; exit 1; }

RUN=libero_10_seq10_naive_lora_r${LORA_R}_a${LORA_ALPHA}_paramatched_steps5k
TASKS='[0,1,2,3,4,5,6,7,8,9]'; STEPS=5000; EVAL_TYPE=none; WANDB=true; FINAL=050000
if [ "$SMOKE" = "1" ]; then
  RUN=smoke_naive_lora_r${LORA_R}_paramatched; TASKS='[0,1]'; STEPS=20; EVAL_TYPE=none; WANDB=false; FINAL=000040
  rm -rf $ROOT/outputs/train/$RUN
fi
RUN_DIR=$ROOT/outputs/train/$RUN

if [ -d "$RUN_DIR/checkpoints/$FINAL" ]; then
  echo "[naive-r$LORA_R] final checkpoint $FINAL exists - nothing to do."; exit 0
fi
# stub-dir guard (E55/E60 lesson): a dir with no checkpoints blocks validate()
if [ -d "$RUN_DIR" ] && [ ! -d "$RUN_DIR/checkpoints" ]; then
  echo "[naive-r$LORA_R] wiping stub output dir (no checkpoints): $RUN_DIR"; rm -rf "$RUN_DIR"
fi
if [ -f "$RUN_DIR/checkpoints/last/sequential_state.pt" ]; then
  echo "[naive-r$LORA_R] RESUMING from $(readlink -f $RUN_DIR/checkpoints/last) (PEFT sequential resume)"
  POLICY_ARGS=(--policy.path="$RUN_DIR/checkpoints/last/pretrained_model" --policy.use_peft=true --resume_sequential=true)
else
  echo "[naive-r$LORA_R] FRESH start from the stage-1 base"
  POLICY_ARGS=(--policy.path="$BASE_CKPT")
fi

# VRAM ladder (E66, 1 Sep): bs16 x acc2 OOMed in the smoke — r=1216 LoRA on the vision tower
# materialises a [B, 1024, 1216] intermediate at each of 162 image-side modules. Rungs preserve
# EFFECTIVE BATCH 32 exactly (the oracle/naive recipe); only the microbatch changes.
LADDER=${LADDER:-"8:4,4:8,16:2"}
run_seq () {  # <bs> <accum>
echo "=== naive seq-LoRA r$LORA_R/a$LORA_ALPHA (parameter-matched to 2.684B memory) — 10 tasks, bs$1 x acc$2 ==="
lerobot-sequential-train \
  "${POLICY_ARGS[@]}" \
  --policy.empty_cameras=1 \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=false \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --peft.method_type=LORA \
  --peft.r=$LORA_R \
  --peft.lora_alpha=$LORA_ALPHA \
  --peft.target_modules="$TARGETS" \
  --peft.full_training_modules='[]' \
  --dataset.repo_id=libero_10 \
  --dataset.root="$ROOT/outputs/libero_10" \
  --rename_map="$RENAME" \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir="$RUN_DIR" \
  --steps=200000 \
  --batch_size=$1 \
  --gradient_accumulation_steps=$2 \
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
}
ok=0
for rung in ${LADDER//,/ }; do
  IFS=: read -r rb ra <<< "$rung"
  if run_seq "$rb" "$ra"; then ok=1; break; fi
  if ls -d "$RUN_DIR"/checkpoints/[0-9]* >/dev/null 2>&1; then
    echo "[naive-r$LORA_R] rung bs=$rb failed AFTER a checkpoint - not VRAM; aborting."; exit 1
  fi
  echo "[naive-r$LORA_R] rung bs=$rb acc=$ra failed before any checkpoint (treating as VRAM) - next rung"
  rm -rf "$RUN_DIR"
done
[ "$ok" = 1 ] || { echo "ERROR: all LADDER rungs failed"; exit 1; }
echo "=== naive_seq_lora_r${LORA_R}_paramatched_10task (SMOKE=$SMOKE) COMPLETE $(date -u) ==="
