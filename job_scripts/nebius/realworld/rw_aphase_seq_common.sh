#!/bin/bash
# A-phase + sequential — REAL-WORLD duplicate of libero_90/staged/joint_aphase_seq5_common.sh
# (E47/E48 graduation body). Sourced by rw_merged6x2_full_chain.sh with WARM_RUN / GRAD_TAG /
# SEQ_* exported.
#   stage A: joint A-phase, values-only on the RW PRETRAIN split, BOTH towers, routers frozen
#            (train_memory_only + freeze_memory_router; E37 overrides for the flags the warm-up
#            ckpt carries: train_router_only=false, vlm_route_once=true, router_only_fast=false)
#   stage B: sequential on the RW SEQ split, C-config as exported by the chain (corefrac beta4,
#            top_t 3072, lr 2e-3 -> 2e-4, 5000 steps/task), per-task checkpoints + resume plumbing.
# Deltas vs the LIBERO body: datasets/rename map; no --env.* anywhere (the LIBERO A-phase built
# 90 gym envs; the LIBERO sequential ran a rollout eval at every boundary); A-phase --eval_freq=0;
# sequential --eval.type=loss --eval_loss_n_batches (in-run paired-noise forgetting matrix ->
# eval/loss_results.jsonl + forgetting/task_* in wandb); no eval episode counts, no
# --ds_to_env_map_json; task ids / final-checkpoint step derived from RW_SEQ_TASK_IDS x SEQ_STEPS.
set -eo pipefail
echo "RW graduation chain [$GRAD_TAG] from $WARM_RUN started on $(hostname) at $(date)"
WARM_CKPT="$ROOT_DIR/outputs/train/$WARM_RUN/checkpoints/last/pretrained_model"
A_RUN=${RUN_PREFIX}realworld_${RW_TAG}_pi05_jointA10k_${GRAD_TAG}
A_OUT="$ROOT_DIR/outputs/train/$A_RUN"
A_CKPT="$A_OUT/checkpoints/last/pretrained_model"
SEQ_TOP_T=${SEQ_TOP_T:-1536}
SEQ_VALUE_LR=${SEQ_VALUE_LR:-0.001}
SEQ_VALUE_LR_END=${SEQ_VALUE_LR_END:-0.0001}
SEQ_BS=${SEQ_BS:-32}
SEQ_ACCUM=${SEQ_ACCUM:-1}
SEQ_TOP_P=${SEQ_TOP_P:-0}
SEQ_TOP_P_CAP=${SEQ_TOP_P_CAP:-16384}
SEQ_PROTECT_MODE=${SEQ_PROTECT_MODE:-rank}
SEQ_PROTECT_UNORM=${SEQ_PROTECT_UNORM:-peak}
SEQ_TASK_IDS=${SEQ_TASK_IDS:-$RW_SEQ_TASK_IDS}
SEQ_N=$(echo "$SEQ_TASK_IDS" | tr -d '[] ' | awk -F, '{print NF}')
SEQ_FINAL_CKPT=$(printf '%06d' $((SEQ_N * SEQ_STEPS)))
SEQ_FIRST_CKPT=$(printf '%06d' "$SEQ_STEPS")
SEQ_RUN=${SEQ_RUN:-${RUN_PREFIX}realworld_${RW_TAG}_seq${SEQ_N}_jw_${GRAD_TAG}_beta4_topt1536_steps5k}
SEQ_OUT="$ROOT_DIR/outputs/train/$SEQ_RUN"
A_LADDER=${A_LADDER:-"32:1:false,16:2:false,16:2:true"}
if [ ! -d "$A_CKPT" ]; then
  [ -d "$WARM_CKPT" ] || { echo "ERROR: warm-up checkpoint missing: $WARM_CKPT"; exit 1; }
fi

# ---------- stage A: joint A-phase (values both towers, routers frozen) ----------
a_phase () {
  lerobot-train \
    --policy.path="$WARM_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$A_RUN" \
    --policy.push_to_hub=false \
    --policy.normalization_mapping="$RW_NORM_MAP" \
    --dataset.repo_id="$RW_PRETRAIN_ID" \
    --dataset.root="$RW_PRETRAIN_ROOT" \
    --rename_map="$RW_RENAME_MAP" \
    --output_dir="$A_OUT" \
    --save_freq=10000 \
    --steps=$A_STEPS \
    --batch_size=$1 \
    --gradient_accumulation_steps=$2 \
    --num_workers=8 \
    --eval_freq=0 \
    --log_freq=200 \
    --policy.train_router_only=false \
    --policy.train_memory_only=true \
    --policy.freeze_memory_router=true \
    --policy.memory_layer.use_frozen_base_input_features=true \
    --policy.memory_layer.vlm_route_once=true \
    --policy.memory_layer.router_only_fast=false \
    --policy.optimizer_lr=2.5e-5 \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=40000 \
    --job_name="$A_RUN" \
    --wandb.enable=$WANDB \
    --wandb.project=vla-memory \
    --wandb.disable_artifact=true \
    --policy.gradient_checkpointing=${3:-false}
}
if [ -d "$A_CKPT" ]; then
  echo "[A-phase] checkpoint exists - skipping."
else
  ok=0
  for rung in ${A_LADDER//,/ }; do
    IFS=: read -r rb ra rc <<< "$rung"
    echo "[A-phase] rung bs=$rb accum=$ra grad_ckpt=$rc"
    if a_phase "$rb" "$ra" "$rc"; then ok=1; break; fi
    echo "[A-phase] rung failed - wiping and trying next rung"; rm -rf "$A_OUT"
  done
  [ "$ok" = 1 ] || { echo "ERROR: all A_LADDER rungs failed"; exit 1; }
fi
[ -d "$A_CKPT" ] || { echo "ERROR: A-phase finished but checkpoint missing"; exit 1; }

# ---------- stage B: sequential (C-config; --eval.type=loss) ----------
seq_stage () {
  lerobot-sequential-train \
    --policy.path="$SEQ_POLICY_PATH" $SEQ_RESUME_FLAG \
    --policy.push_to_hub=false \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=${3:-false} \
    --policy.normalization_mapping="$RW_NORM_MAP" \
    --dataset.repo_id="$RW_SEQ_ID" \
    --dataset.root="$RW_SEQ_ROOT" \
    --rename_map="$RW_RENAME_MAP" \
    --output_dir="$SEQ_OUT" \
    --steps=200000 \
    --batch_size=$1 \
    --gradient_accumulation_steps=$2 \
    --num_workers=8 \
    --eval.type=loss \
    --eval_loss_n_batches=$EVAL_LOSS_NB \
    --log_freq=200 \
    --wandb.enable=$WANDB \
    --wandb.project=vla-memory \
    --job_name="$SEQ_RUN" \
    --online_task_ids="$SEQ_TASK_IDS" \
    --online_steps_per_task=$SEQ_STEPS \
    --policy.train_router_only=false \
    --policy.memory_layer.vlm_route_once=true \
    --policy.memory_layer.router_only_fast=false \
    --policy.memory_layer.aggregate_usage=false \
    --policy.memory_layer.use_frozen_base_input_features=true \
    --save_after_each_task=true \
    --reinit_optimizer_each_task=true \
    --tfidf_enable=true \
    --tfidf_top_t=$SEQ_TOP_T \
    --tfidf_top_p=$SEQ_TOP_P \
    --tfidf_top_p_cap=$SEQ_TOP_P_CAP \
    --use_online_idf_stats=true \
    --idf_exponent=1 \
    --protect_prior_slots=true \
    --protect_beta=4 \
    --protect_mode=$SEQ_PROTECT_MODE \
    --protect_u_norm=$SEQ_PROTECT_UNORM \
    --memory_value_lr=$SEQ_VALUE_LR \
    --memory_value_lr_end=$SEQ_VALUE_LR_END \
    --memory_value_scheduler_type=linear \
    $SEQ_EXTRA_ARGS
}
# Auto-resume from a completed task boundary (sequential_state.pt carries the protection store /
# online-IDF accumulators / eval histories; the trainer refuses to resume without it).
SEQ_POLICY_PATH="$A_CKPT"
SEQ_RESUME_FLAG=""
if [ -f "$SEQ_OUT/checkpoints/last/sequential_state.pt" ] && [ ! -d "$SEQ_OUT/checkpoints/$SEQ_FINAL_CKPT" ]; then
  SEQ_POLICY_PATH="$SEQ_OUT/checkpoints/last/pretrained_model"
  SEQ_RESUME_FLAG="--resume_sequential=true"
  echo "[seq] RESUMING from $(readlink -f "$SEQ_OUT/checkpoints/last" 2>/dev/null)"
fi
if [ -d "$SEQ_OUT/checkpoints/$SEQ_FINAL_CKPT" ]; then
  echo "[seq] final checkpoint $SEQ_FINAL_CKPT exists - skipping."
elif [ -z "$SEQ_LADDER" ]; then
  seq_stage $SEQ_BS $SEQ_ACCUM ${SEQ_GRAD_CKPT:-false}
else
  ok=0
  for rung in ${SEQ_LADDER//,/ }; do
    IFS=: read -r rb ra rc <<< "$rung"
    echo "[seq] ladder rung: bs=$rb accum=$ra grad_ckpt=$rc"
    if seq_stage "$rb" "$ra" "$rc"; then ok=1; break; fi
    if [ -d "$SEQ_OUT/checkpoints/$SEQ_FIRST_CKPT" ]; then
      echo "[seq] rung failed AFTER the task-0 checkpoint - not a VRAM failure; aborting."; exit 1
    fi
    echo "[seq] rung failed before $SEQ_FIRST_CKPT (treating as VRAM) - wiping and trying next rung"
    rm -rf "$SEQ_OUT"
  done
  [ "$ok" = 1 ] || { echo "ERROR: all SEQ_LADDER rungs failed"; exit 1; }
fi
echo "RW graduation chain [$GRAD_TAG] COMPLETE at $(date) -> $SEQ_OUT"
