#!/bin/bash
# E43 arm: FIXED soft protection (momentum-aware blend) on the lr4xsched+topt3072 config.
#
# Single delta vs the E42 lr4xsched+topt3072 arm (VM2, 50-ep final 42.4): protect_mode
# rank -> grad_scale + protect_u_norm peak -> corefrac. Everything else byte-identical
# (value_lr 4e-3 -> 2e-4, top_t 3072, beta 4, frozen-base routing, seed 1000).
#
# Why now (E43): at 4x amplitude the e9-directed bleed (e2-block 7.55% + e7-block 8.43% of
# e9's read-mass-weighted field at L14) produced the first REAL function-space give-back on
# the staged track: e9 own-block chunk 0.2792 -> final 0.3197 (+14.5%; the 2x twin +7.7%,
# 1x arms -3%). The E42 softprotect arm never tested the mechanism (Adam momentum leak,
# ~90% passthrough); this arm runs the FIXED blend (exp_avg scaled with theta,
# commit edb79239) whose smoke reproduces designed attenuation exactly under mask churn.
#
# Pre-registered reads (vs VM2, its exact twin):
#   - e9 own-block chunk ~= 0.28 (writer fit preserved; protection must not tax the diagonal
#     -- watch e2/e7 own-block chunks and block-min MSEs, the E42 softp writer-cost channel)
#   - e9 own->final chunk give-back <= +4% (vs +14.5% unprotected) => mechanism works
#   - e9/e4 50-ep finals up vs 14/20; others ~unchanged
#   - FAIL: e2/e7 block-min MSE up >10% vs VM2 => the corefrac gate is starving writers at 4x
set -eo pipefail
echo "E43 lr4xsched+topt3072+softprotect-fixed started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
A_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_8_10_12_14_frozenroute_rwarmupB_values10k_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model"
SEQ_RUN=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_3072_softprotect_fix_cf_beta4_lr4xsched_steps5k_tasks5
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
    --protect_mode=grad_scale \
    --protect_u_norm=corefrac \
    --memory_value_lr=0.004 \
    --memory_value_lr_end=0.0002 \
    --memory_value_scheduler_type=linear
fi
echo "E43 lr4xsched+topt3072+softprotect-fixed completed at $(date)"
