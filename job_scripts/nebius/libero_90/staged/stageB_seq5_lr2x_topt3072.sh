#!/bin/bash
# E42 arm 1 (VM1): lr2x + top_t 3072 — compose the two validated fit levers.
#
# Deltas vs stageB sequential (both individually validated in E41/E42):
#   --memory_value_lr=0.002 / end 0.0002   (amplitude: the only init mover, +12pp;
#                                           chunk error -6..-19% on every task)
#   --tfidf_top_t=3072                     (coverage: self-adapted read mass 62-86% -> 74-95%;
#                                           lowest family loss 0.1086; give-back +2.8)
#
# Why composition is safe here (E42): the mechanisms are independent (per-slot displacement
# x fraction of read mass adapted); frozen-base routing kills the E30-era drift coupling; and
# topt3072 measured 2-3x bleed with ZERO rollout/function retention cost (e6 chunk flat in all
# 6 arms), so the extra write mass from 2xLR-on-2x-slots is not expected to cost retention.
# Protection stays the stageB legacy default (rank/peak beta4, effectively inert) for lineage.
#
# Pre-registered reads: inits >= ~47 (lr2x-class) with give-back >= ~0 (top3k-class) => final
# >= ~42-45 at 50 eps = new staged frontier. e9 init >= 20 (amplitude carries it at 3072 too).
# Failure signature: give-back <= -8 => amplitude x coverage interact badly after all.
set -eo pipefail
echo "E42 lr2x+topt3072 started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
A_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_8_10_12_14_frozenroute_rwarmupB_values10k_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model"
SEQ_RUN=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_3072_protect_beta4_lr2x_steps5k_tasks5
SEQ_OUT="$ROOT_DIR/outputs/train/$SEQ_RUN"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$A_CKPT" ] || { echo "ERROR: stageB A checkpoint missing"; exit 1; }

# 5 tasks x 5000 steps -> final checkpoint 025000
if [ -d "$SEQ_OUT/checkpoints/025000" ]; then
  echo "[seq] final checkpoint exists - skipping."
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
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
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
    --policy.memory_layer.aggregate_usage=false \
    --policy.memory_layer.use_frozen_base_input_features=true \
    --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
    --save_after_each_task=true \
    --reinit_optimizer_each_task=true \
    --tfidf_enable=true \
    --tfidf_top_t=3072 \
    --use_online_idf_stats=true \
    --idf_exponent=1 \
    --protect_prior_slots=true \
    --protect_beta=4 \
    --memory_value_lr=0.002 \
    --memory_value_lr_end=0.0002 \
    --memory_value_scheduler_type=linear
fi
echo "E42 lr2x+topt3072 completed at $(date)"
