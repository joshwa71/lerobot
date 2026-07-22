#!/bin/bash
# E47 graduation chain — COMMON BODY (sourced by wrappers setting WARM_RUN / GRAD_TAG).
# For a certified joint router warm-up checkpoint (both towers' routers in ONE ckpt —
# no merge step, unlike the E45 pool chain):
#   stage A: joint A-phase, 10k values-only on libero_90, BOTH towers, routers frozen
#            (train_memory_only + freeze_memory_router; explicit E37 overrides for the
#            flags the warm-up ckpt carries: train_router_only=false, and
#            vlm_route_once=true — the E47 bcast warm-ups save False, but with the
#            router frozen the compact path is numerically identical and saves 6-10GB);
#   stage B: 5-task sequential, C-config (beta4 protection, top_t 1536, 5000 steps/task,
#            value lr 1e-3 -> 1e-4, 20-ep intermediates + 50-ep FINAL AT 50 EPS),
#            comparable to stageB (32.0 final / 35.0 init) and the E45 e4 anchors.
# GATE 2 (central, not automated here): the sequential's t0 block IS the e4 probe
# (same C-config 5k steps) — run the chunk probe on checkpoints/005000 while t1+ train;
# kill the run at chunk >= ~0.12 (anchors 0.153 staged-best / 0.0994 poolB / 0.020 LoRA).
# Per-task checkpoints are kept (save_after_each_task) for the probe battery.
set -eo pipefail
echo "E47 graduation chain [$GRAD_TAG] from $WARM_RUN started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
WARM_CKPT="$ROOT_DIR/outputs/train/$WARM_RUN/checkpoints/last/pretrained_model"
A_RUN=libero_90_pi05_jointA10k_${GRAD_TAG}
A_OUT="$ROOT_DIR/outputs/train/$A_RUN"
A_CKPT="$A_OUT/checkpoints/last/pretrained_model"
# E48 parametrization (defaults = the E47 graduation config, byte-identical commands):
# wrappers may override the sequential's write budget / value LR / micro-batching and
# must then set SEQ_RUN so the run name reflects the actual config.
SEQ_TOP_T=${SEQ_TOP_T:-1536}
SEQ_VALUE_LR=${SEQ_VALUE_LR:-0.001}
SEQ_VALUE_LR_END=${SEQ_VALUE_LR_END:-0.0001}
SEQ_BS=${SEQ_BS:-32}
SEQ_ACCUM=${SEQ_ACCUM:-1}
SEQ_TOP_P=${SEQ_TOP_P:-0}
SEQ_PROTECT_MODE=${SEQ_PROTECT_MODE:-rank}
SEQ_PROTECT_UNORM=${SEQ_PROTECT_UNORM:-peak}
SEQ_RUN=${SEQ_RUN:-libero_10_seq5_jw_${GRAD_TAG}_beta4_topt1536_steps5k}
SEQ_OUT="$ROOT_DIR/outputs/train/$SEQ_RUN"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
# The warm-up checkpoint is only needed when stage A actually runs (a seq-only reuse of
# an existing A checkpoint on a box without the warm-up dir is legitimate — E48).
if [ ! -d "$A_CKPT" ]; then
  [ -d "$WARM_CKPT" ] || { echo "ERROR: warm-up checkpoint missing: $WARM_CKPT (rsync it to this box first)"; exit 1; }
fi

# ---------- stage A: joint A-phase (values both towers, routers frozen) ----------
a_phase () {
  lerobot-train \
    --policy.path="$WARM_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$A_RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_90 \
    --dataset.root="$DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_90 \
    --output_dir="$A_OUT" \
    --save_freq=10000 \
    --steps=10000 \
    --batch_size=$1 \
    --gradient_accumulation_steps=$2 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=4 \
    --eval_freq=20000 \
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
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --wandb.disable_artifact=true \
    --policy.gradient_checkpointing=false
}
if [ -d "$A_CKPT" ]; then
  echo "[A-phase] checkpoint exists - skipping."
else
  echo "[A-phase] launching at bs32 (fallback bs16 x accum2 on failure)"
  a_phase 32 1 || { echo "[A-phase] bs32 failed - retrying bs16 x accum2"; rm -rf "$A_OUT"; a_phase 16 2; }
fi
[ -d "$A_CKPT" ] || { echo "ERROR: A-phase finished but checkpoint missing"; exit 1; }

# ---------- stage B: 5-task sequential (C-config; t0 block == the e4 probe) ----------
# train_memory_only + freeze_memory_router + frozen-route ride in the A-ckpt config
# (desired downstream); train_router_only + vlm_route_once are explicitly overridden.
if [ -d "$SEQ_OUT/checkpoints/025000" ]; then
  echo "[seq5] final checkpoint exists - skipping."
else
  lerobot-sequential-train \
    --policy.path="$A_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_10 \
    --dataset.root="$SEQ_DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_10 \
    --output_dir="$SEQ_OUT" \
    --steps=200000 \
    --batch_size=$SEQ_BS \
    --gradient_accumulation_steps=$SEQ_ACCUM \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=20 \
    --eval_final_episodes=50 \
    --log_freq=200 \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --job_name="$SEQ_RUN" \
    --online_task_ids='[0,1,2,3,4]' \
    --online_steps_per_task=5000 \
    --policy.train_router_only=false \
    --policy.memory_layer.vlm_route_once=true \
    --policy.memory_layer.router_only_fast=false \
    --policy.memory_layer.aggregate_usage=false \
    --policy.memory_layer.use_frozen_base_input_features=true \
    --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
    --save_after_each_task=true \
    --reinit_optimizer_each_task=true \
    --tfidf_enable=true \
    --tfidf_top_t=$SEQ_TOP_T \
    --tfidf_top_p=$SEQ_TOP_P \
    --use_online_idf_stats=true \
    --idf_exponent=1 \
    --protect_prior_slots=true \
    --protect_beta=4 \
    --protect_mode=$SEQ_PROTECT_MODE \
    --protect_u_norm=$SEQ_PROTECT_UNORM \
    --memory_value_lr=$SEQ_VALUE_LR \
    --memory_value_lr_end=$SEQ_VALUE_LR_END \
    --memory_value_scheduler_type=linear
fi
echo "E47 graduation chain [$GRAD_TAG] COMPLETE at $(date)"
