#!/bin/bash
# E40 4-way batch, ARM 3: steps/task 5000 -> 7000 (optimization budget on stationary targets).
#
# Sequential-only from the EXISTING stageB A checkpoint
# (frozenroute_rwarmupB_values10k — warmed router, values-only fill, frozen-base routing).
# Config = stageB sequential stage verbatim (C's config + frozen-base routing + beta4,
# first 5 tasks) with the single delta online_steps_per_task=7000, plus the batch-wide
# 50-ep final eval (--eval_final_episodes=50; intermediate evals stay 20 eps).
# Key read: MSE floors + inits; forgetting must stay flat (matrix if needed).
#
# NB (15 Jul): reconstruction — the original arm scripts were written on the source box
# inside gitignored job_scripts/ and never force-added, so they did not propagate to the
# cloned VMs. Recreated from the stageB template + E40 spec and force-added to git.
set -eo pipefail
echo "E40 arm 3 (steps7k) started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
A_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_8_10_12_14_frozenroute_rwarmupB_values10k_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model"
SEQ_RUN=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps7k_tasks5
SEQ_OUT="$ROOT_DIR/outputs/train/$SEQ_RUN"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$A_CKPT" ] || { echo "ERROR: stageB A checkpoint missing"; exit 1; }

# 5 tasks x 7000 steps -> final checkpoint 035000
if [ -d "$SEQ_OUT/checkpoints/035000" ]; then
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
    --online_steps_per_task=7000 \
    --policy.memory_layer.aggregate_usage=false \
    --policy.memory_layer.use_frozen_base_input_features=true \
    --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
    --save_after_each_task=true \
    --reinit_optimizer_each_task=true \
    --tfidf_enable=true \
    --tfidf_top_t=1536 \
    --use_online_idf_stats=true \
    --idf_exponent=1 \
    --protect_prior_slots=true \
    --protect_beta=4 \
    --memory_value_lr=0.001 \
    --memory_value_lr_end=0.0001 \
    --memory_value_scheduler_type=linear
fi
echo "E40 arm 3 (steps7k) completed at $(date)"
