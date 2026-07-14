#!/bin/bash
# Combined pi05 + memory: pretrain on libero_minus_goal (eval libero_spatial),
# then sequential adapt on libero_goal (eval libero_goal).
#
# This is the libero_spatial follow-up that applies the realworld Entry 1 fix
# (contrastive_loss_weight=0.01) and avoids the c=1 collapse seen in
#   libero_minus_spatial_pi05_..._sup_128_2048_knn_36_50k
# Pretrain checkpoint cadence halved (20k vs 10k) to keep disk usage manageable.
#
# Datasets:
#   pretrain   -> outputs/libero_split/libero_minus_goal  (30 tasks: long 0-9, object 20-29, spatial 30-39)
#   sequential -> outputs/libero_split/libero_goal        (10 tasks: 10-19)
#
# Eval mapping (dataset task_index -> libero_goal env task_id), based on the
# libero suite task ordering vs the dataset tasks.parquet:
#   10 put-the-bowl-on-the-plate            -> 8
#   11 put-the-wine-bottle-on-the-rack      -> 9
#   12 open-top-drawer-and-put-the-bowl     -> 3
#   13 put-the-cream-cheese-in-the-bowl     -> 6
#   14 put-wine-bottle-on-top-of-cabinet    -> 2
#   15 push-the-plate-to-front-of-stove     -> 5
#   16 turn-on-the-stove                    -> 7
#   17 put-the-bowl-on-the-stove            -> 1
#   18 put-the-bowl-on-top-of-cabinet       -> 4
#   19 open-the-middle-drawer-of-cabinet    -> 0

set -eo pipefail

echo "Job started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot

PRETRAIN_DATASET_ROOT="$ROOT_DIR/outputs/libero_split/libero_minus_goal"
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_split/libero_goal"

PRETRAIN_RUN=libero_minus_goal_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_30k
SEQ_RUN=libero_goal_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_30k

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
export PYTORCH_ALLOC_CONF=expandable_segments:True

source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated

echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

cd "$ROOT_DIR"

###############################################################################
# Stage 1 - pretrain on libero_minus_goal (30 tasks), eval on libero_spatial
###############################################################################
echo "=============================================================="
echo "[stage 1] PRETRAIN on libero_minus_goal (30 tasks)"
echo "  eval env: libero_spatial"
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
    --dataset.repo_id=libero_minus_goal \
    --dataset.root="$PRETRAIN_DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_spatial \
    --output_dir="$PRETRAIN_OUTPUT_DIR" \
    --save_freq=20000 \
    --steps=30000 \
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=4 \
    --eval_freq=20000 \
    --log_freq=200 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=30000 \
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
    --policy.memory_layer.contrastive_loss_weight=0.01 \
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
# Stage 2 - sequential adaptation on libero_goal (10 tasks: dataset 10..19)
###############################################################################
echo "=============================================================="
echo "[stage 2] SEQUENTIAL on libero_goal (10 tasks: dataset 10..19)"
echo "  eval env: libero_goal"
echo "  loading from: $PRETRAIN_CHECKPOINT"
echo "  output: $SEQ_OUTPUT_DIR"
echo "=============================================================="

lerobot-sequential-train \
  --policy.path="$PRETRAIN_CHECKPOINT" \
  --policy.empty_cameras=1 \
  --policy.dtype=bfloat16 \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --dataset.repo_id=libero_goal \
  --dataset.root="$SEQ_DATASET_ROOT" \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
  --env.type=libero \
  --env.task=libero_goal \
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
  --online_task_ids='[10,11,12,13,14,15,16,17,18,19]' \
  --online_steps_per_task=3000 \
  --policy.memory_layer.aggregate_usage=false \
  --ds_to_env_map_json='{"10":8,"11":9,"12":3,"13":6,"14":2,"15":5,"16":7,"17":1,"18":4,"19":0}' \
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
