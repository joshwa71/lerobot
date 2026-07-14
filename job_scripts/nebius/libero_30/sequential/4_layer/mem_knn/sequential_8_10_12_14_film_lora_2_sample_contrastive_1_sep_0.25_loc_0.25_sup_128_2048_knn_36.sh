#!/bin/bash
# Nebius port of:
#   job_scripts/smolvla-memory/sequential/4_layer/mem_knn/
#     sequential_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_36.sh
#
# Differences from the cluster (qsub) version:
#  - No SGE directives, no scratch staging, no periodic backup.
#  - conda lives at /home/josh/miniforge3 (not /share/apps/miniconda3).
#  - MUJOCO_GL=osmesa (Mesa-only EGL on this image).
#  - rename_map + empty_cameras=1 to map dataset image keys to the policy's
#    camera{1,2,3} expectations.
#  - PRETRAIN_RUN points at the pretrain output dir on this machine.
#  - Outputs go directly to $ROOT_DIR/outputs/train/$RUN_NAME.
#
# Run AFTER pretraining finishes and a checkpoint exists at
# $ROOT_DIR/outputs/train/$PRETRAIN_RUN/checkpoints/last/pretrained_model.

set -eo pipefail

echo "Job started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot
PRETRAIN_RUN=libero_95_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_36
RUN_NAME=sequential_libero_95_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_36
DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
PRETRAIN_DIR="$ROOT_DIR/outputs/train/$PRETRAIN_RUN"
POLICY_PATH="$PRETRAIN_DIR/checkpoints/last/pretrained_model"
OUTPUT_DIR="$ROOT_DIR/outputs/train/$RUN_NAME"

if [ ! -d "$POLICY_PATH" ]; then
  echo "Pretrained policy not found at $POLICY_PATH — has pretraining finished?"
  exit 1
fi

# Headless rendering (libero env eval).
export MUJOCO_GL=osmesa
unset DISPLAY

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1

source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated

echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

cd "$ROOT_DIR"

python -m lerobot.scripts.lerobot_sequential_train \
  --policy.path="$POLICY_PATH" \
  --policy.empty_cameras=1 \
  --dataset.repo_id=libero_10 \
  --dataset.root="$DATASET_ROOT" \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir="$OUTPUT_DIR" \
  --steps=200000 \
  --batch_size=64 \
  --num_workers=8 \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --log_freq=200 \
  --wandb.enable=true \
  --wandb.project=vla-memory \
  --job_name="$RUN_NAME" \
  --online_task_ids='[6,7,8,9]' \
  --online_steps_per_task=3000 \
  --policy.memory_layer.aggregate_usage=false \
  --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
  --save_after_each_task=true \
  --reinit_optimizer_each_task=true \
  --tfidf_enable=true \
  --tfidf_top_t=512 \
  --use_online_idf_stats=true \
  --idf_exponent=1 \
  --memory_value_lr=0.001 \
  --memory_value_lr_end=0.0001 \
  --memory_value_scheduler_type=linear

echo "Job completed at $(date)"
