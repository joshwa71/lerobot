#!/bin/bash
# Nebius port of:
#   job_scripts/smolvla-memory/pretrain/4_layer/mem_knn/
#     pretrain_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_36.sh
#
# Differences from the cluster (qsub) version:
#  - No SGE directives, no scratch staging, no periodic backup.
#  - conda lives at /home/josh/miniforge3 (not /share/apps/miniconda3).
#  - MUJOCO_GL=osmesa: this Nebius image ships Mesa-only EGL (no NVIDIA EGL
#    vendor file), so robosuite EGL fails. OSMesa is software-rendered but
#    works headlessly.
#  - rename_map + empty_cameras=1: the local libero_95 has
#    observation.images.{image,image2}; smolvla_base expects camera{1,2,3}.
#  - Outputs go directly to $ROOT_DIR/outputs/train/$RUN_NAME.

set -eo pipefail

echo "Job started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot
RUN_NAME=libero_95_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_36
DATASET_ROOT="$ROOT_DIR/outputs/libero_95"
MODEL_PATH="$ROOT_DIR/outputs/smolvla_base"
OUTPUT_DIR="$ROOT_DIR/outputs/train/$RUN_NAME"

# Do NOT mkdir $OUTPUT_DIR — lerobot-train's validate() refuses to write into
# an existing directory unless --resume=true.

# Headless rendering (libero in-training eval).
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

lerobot-train \
  --policy.path="$MODEL_PATH" \
  --policy.empty_cameras=1 \
  --policy.repo_id="outputs/train/$RUN_NAME" \
  --dataset.repo_id=libero_95 \
  --dataset.root="$DATASET_ROOT" \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --env.type=libero \
  --env.task=libero_spatial \
  --output_dir="$OUTPUT_DIR" \
  --save_freq=20000 \
  --steps=100000 \
  --batch_size=32 \
  --num_workers=12 \
  --eval.batch_size=1 \
  --eval.n_episodes=4 \
  --eval_freq=20000 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true \
  --policy.scheduler_warmup_steps=10000 \
  --policy.scheduler_decay_steps=80000 \
  --job_name="$RUN_NAME" \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.project=vla-memory \
  --wandb.disable_artifact=true \
  --policy.gradient_checkpointing=false \
  --policy.memory_layers=true \
  --policy.memory_layer.memory_only=false \
  --policy.memory_layer.layers="[8,10,12,14]" \
  --policy.memory_layer.log_usage=true \
  --policy.memory_layer.enabled=true \
  --policy.memory_layer.aggregate_usage=true \
  --policy.memory_layer.mem_n_keys=384 \
  --policy.memory_layer.mem_heads=4 \
  --policy.memory_layer.mem_knn=36 \
  --policy.memory_layer.mem_k_dim=512 \
  --policy.memory_layer.value_fixed_lr=0.001 \
  --policy.memory_layer.memory_lr=0.001 \
  --policy.memory_layer.lang_to_query=true \
  --policy.memory_layer.fuse_method=film \
  --policy.memory_layer.embedding_model=all-mpnet-base-v2 \
  --policy.memory_layer.value_type=lora \
  --policy.memory_layer.lora_rank=2 \
  --policy.memory_layer.contrastive_method=sample \
  --policy.memory_layer.contrastive_loss_weight=1.0 \
  --policy.memory_layer.contrastive_margin=0.0 \
  --policy.memory_layer.contrastive_query_queue=128 \
  --policy.memory_layer.routing_loss_topk=36 \
  --policy.memory_layer.routing_intra_task_locality_weight=0.25 \
  --policy.memory_layer.routing_intra_task_min_support=128 \
  --policy.memory_layer.routing_intra_task_max_support=2048 \
  --policy.memory_layer.routing_inter_task_separation_weight=0.25

echo "Job completed at $(date)"
