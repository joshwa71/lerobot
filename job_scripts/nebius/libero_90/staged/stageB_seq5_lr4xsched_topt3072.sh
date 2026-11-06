#!/bin/bash
# E42 arm 2 (VM2): lr 4e-3 -> 2e-4 + top_t 3072 — the amplitude-response arm.
#
# Single delta vs E42 arm 1 (lr2x+topt3072): peak value LR 0.002 -> 0.004, floor kept at
# 0.0002 (a steeper schedule, NOT a uniform 4x — aggressive early traversal, hard late decay).
#
# Rationale (E42 discussion): the LR schedule SHAPE is unexplored (every run to date was
# linear peak -> peak/10). 2x LR is the only lever that ever moved inits (+12pp); within-block
# curves at 2e-3 converge by ~2.5-3k steps, so a higher peak buys early traversal, and the
# hard decay kills the late-block wander that is 4x's main risk. Known headwinds to test
# against: L14 amplitude saturation (probe A: 2x delta -> only +60-70% output) and grad-norm
# headroom (max 0.022 at 2x, clip 1.0 — 45x margin, so instability unlikely).
#
# Pre-registered reads (vs arm 1, its exact twin at 2e-3): e9 init/chunk is THE cell — the one
# amplitude-limited task (only mover 5 -> 25-30 at 2x; budget/coverage/concentration all did
# nothing to it). If inits ~= arm 1 with worse block-END MSEs => saturation is binding and 2x
# is the amplitude plateau — axis closed with evidence. If e9 init > arm 1 by >=10pp => the
# schedule lever is real; consider 4e-3 -> 1e-4 next.
set -eo pipefail
echo "E42 lr4xsched+topt3072 started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
A_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_8_10_12_14_frozenroute_rwarmupB_values10k_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model"
SEQ_RUN=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_3072_protect_beta4_lr4xsched_steps5k_tasks5
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
    --memory_value_lr=0.004 \
    --memory_value_lr_end=0.0002 \
    --memory_value_scheduler_type=linear
fi
echo "E42 lr4xsched+topt3072 completed at $(date)"
