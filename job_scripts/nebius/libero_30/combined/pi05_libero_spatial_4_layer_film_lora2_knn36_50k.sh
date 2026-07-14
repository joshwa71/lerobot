#!/bin/bash
# Combined pi05 + memory: pretrain on libero_minus_spatial (eval libero_goal),
# then sequential adapt on libero_spatial (eval libero_spatial).
#
# Memory configuration mirrors the smolvla best acid-test:
#   job_scripts/nebius/libero/{pretrain,sequential}/4_layer/mem_knn/
#     pretrain_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_36.sh
# but on the pi05 backbone (gemma_2b VLM + gemma_300m action expert), following the
# pi05 conventions from job_scripts/nebius/combined/realworld_pi05_4_layer_film_lora2_sample_contrastive_50k.sh
# (bfloat16, normalization_mapping, gradient_checkpointing).
#
# Datasets:
#   pretrain  -> outputs/libero_split/libero_minus_spatial  (task indices 0..29, 30 tasks)
#   sequential -> outputs/libero_split/libero_spatial       (task indices 30..39, 10 tasks)
#
# Eval mapping (dataset task_index -> libero_spatial env task_id), based on the
# libero suite task ordering vs the dataset tasks.parquet:
#   30 next-to-cookie-box -> 6
#   31 top-drawer         -> 4
#   32 on-the-ramekin     -> 5
#   33 on-the-stove       -> 7
#   34 between-plate-ramekin -> 0
#   35 on-cookie-box      -> 3
#   36 next-to-plate      -> 8
#   37 next-to-ramekin    -> 1
#   38 from-table-center  -> 2
#   39 on-wooden-cabinet  -> 9

set -eo pipefail

echo "Job started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot

PRETRAIN_DATASET_ROOT="$ROOT_DIR/outputs/libero_split/libero_minus_spatial"
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_split/libero_spatial"

PRETRAIN_RUN=libero_minus_spatial_pi05_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_36_50k
SEQ_RUN=libero_spatial_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048_knn_36_50k

PRETRAIN_OUTPUT_DIR="$ROOT_DIR/outputs/train/$PRETRAIN_RUN"
SEQ_OUTPUT_DIR="$ROOT_DIR/outputs/train/$SEQ_RUN"
PRETRAIN_CHECKPOINT="$PRETRAIN_OUTPUT_DIR/checkpoints/last/pretrained_model"

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

###############################################################################
# Stage 1 - pretrain on libero_minus_spatial (30 tasks), eval on libero_goal
###############################################################################
echo "=============================================================="
echo "[stage 1] PRETRAIN on libero_minus_spatial (30 tasks)"
echo "  eval env: libero_goal"
echo "  output: $PRETRAIN_OUTPUT_DIR"
echo "=============================================================="

# Skip pretrain if a checkpoint already exists (allows safe re-runs).
if [ -d "$PRETRAIN_CHECKPOINT" ]; then
  echo "[stage 1] Pretrained checkpoint already exists at $PRETRAIN_CHECKPOINT - skipping pretrain."
else
  lerobot-train \
    --policy.path=lerobot/pi05_base \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$PRETRAIN_RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_minus_spatial \
    --dataset.root="$PRETRAIN_DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_goal \
    --output_dir="$PRETRAIN_OUTPUT_DIR" \
    --save_freq=10000 \
    --steps=50000 \
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=4 \
    --eval_freq=10000 \
    --log_freq=200 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=50000 \
    --job_name="$PRETRAIN_RUN" \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --wandb.disable_artifact=true \
    --policy.gradient_checkpointing=true \
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
fi

if [ ! -d "$PRETRAIN_CHECKPOINT" ]; then
  echo "ERROR: pretrain finished but $PRETRAIN_CHECKPOINT does not exist"
  exit 1
fi

###############################################################################
# Stage 2 - sequential adaptation on libero_spatial (10 tasks: dataset 30..39)
###############################################################################
echo "=============================================================="
echo "[stage 2] SEQUENTIAL on libero_spatial (10 tasks: dataset 30..39)"
echo "  eval env: libero_spatial"
echo "  loading from: $PRETRAIN_CHECKPOINT"
echo "  output: $SEQ_OUTPUT_DIR"
echo "=============================================================="

lerobot-sequential-train \
  --policy.path="$PRETRAIN_CHECKPOINT" \
  --policy.empty_cameras=1 \
  --policy.dtype=bfloat16 \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --dataset.repo_id=libero_spatial \
  --dataset.root="$SEQ_DATASET_ROOT" \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
  --env.type=libero \
  --env.task=libero_spatial \
  --output_dir="$SEQ_OUTPUT_DIR" \
  --steps=200000 \
  --batch_size=32 \
  --gradient_accumulation_steps=1 \
  --num_workers=8 \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --log_freq=200 \
  --wandb.enable=true \
  --wandb.project=vla-memory \
  --job_name="$SEQ_RUN" \
  --online_task_ids='[30,31,32,33,34,35,36,37,38,39]' \
  --online_steps_per_task=3000 \
  --policy.memory_layer.aggregate_usage=false \
  --ds_to_env_map_json='{"30":6,"31":4,"32":5,"33":7,"34":0,"35":3,"36":8,"37":1,"38":2,"39":9}' \
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
