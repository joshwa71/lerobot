#!/bin/bash
# Sequential-only: prior-usefulness WRITE PROTECTION on top of the sep=5 prior.
# (research_log Entry 26 follow-up; the IDF/protection discussion.)
#
# Reuses the EXISTING sep=5 40k pretrain checkpoint (no new pretrain) and runs the
# libero_10 sequential adaptation with the new task-identity-aware write gate enabled.
# Only delta vs the sep5 sequential baseline (..._top_t_1536):
#   --protect_prior_slots=true   (master toggle; default off = legacy TF-IDF mask)
#   --protect_beta=4             (gate sharpness; (1 - u(s))**beta on the write score,
#                                 u(s) = max over prior tasks of peak-normalized read profile)
#   --eval.n_episodes 50 -> 20   (faster eval)
#
# Mechanism: at each task boundary the just-finished task's read profile is folded into a
# per-module usefulness store u(s); during later tasks the per-batch TF-IDF write score is
# multiplied by (1-u(s))**beta, pushing slots that earlier tasks relied on out of the top-t
# update set. Offline beta-sweep (static footprints): net-positive across the 10-task run,
# env7 read-through-overwrite ~81% -> ~53% at beta=4, writer cost ~13-16% on env0/env1.
# NB: implementation beta acts on the top-t RANKING (reselection), so its scale differs from
# the offline soft-suppression model; treat beta=4 as "moderate" and sweep {2,4,8} if needed.
#
# WHAT TO WATCH vs the sep5 baseline (..._top_t_1536, final 34.0%):
#   - env7 (soup+cheese, ord t4) retention: baseline 18->0; predicted partial rescue.
#   - env0 / env1 (ord t5 / t7) fit cost: the basket writers pay for the protection.
#   - memory_iou/all_modules_mean and read-through (should drop further than 0.052).
#   - diagonal/init unchanged (this attacks interference, not the plasticity ceiling).

set -eo pipefail
echo "Job started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"

PRETRAIN_RUN=libero_90_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k
SEQ_RUN=libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536_protect_beta4

PRETRAIN_CHECKPOINT="$ROOT_DIR/outputs/train/$PRETRAIN_RUN/checkpoints/last/pretrained_model"
SEQ_OUTPUT_DIR="$ROOT_DIR/outputs/train/$SEQ_RUN"

# Headless rendering (libero in-training eval).
export MUJOCO_GL=osmesa
unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
cd "$ROOT_DIR"

if [ ! -d "$PRETRAIN_CHECKPOINT" ]; then
  echo "ERROR: sep5 pretrain checkpoint not found at $PRETRAIN_CHECKPOINT"
  exit 1
fi

echo "=============================================================="
echo "[sequential] libero_10 (10 tasks: dataset 0..9)  +  PRIOR-USEFULNESS PROTECTION (beta=4)"
echo "  loading from: $PRETRAIN_CHECKPOINT"
echo "  output: $SEQ_OUTPUT_DIR"
echo "=============================================================="

lerobot-sequential-train \
  --policy.path="$PRETRAIN_CHECKPOINT" \
  --policy.empty_cameras=1 \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=false \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --dataset.repo_id=libero_10 \
  --dataset.root="$SEQ_DATASET_ROOT" \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir="$SEQ_OUTPUT_DIR" \
  --steps=200000 \
  --batch_size=32 \
  --gradient_accumulation_steps=1 \
  --num_workers=8 \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --log_freq=200 \
  --wandb.enable=true \
  --wandb.project=vla-memory \
  --job_name="$SEQ_RUN" \
  --online_task_ids='[0,1,2,3,4,5,6,7,8,9]' \
  --online_steps_per_task=3000 \
  --policy.memory_layer.aggregate_usage=false \
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

echo "Job completed at $(date)"
