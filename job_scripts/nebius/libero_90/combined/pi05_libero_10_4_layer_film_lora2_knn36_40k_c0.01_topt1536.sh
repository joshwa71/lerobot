#!/bin/bash
# Combined pi05 + memory: pretrain on libero_90 (eval libero_90),
# then sequential adapt on libero_10 (eval libero_10).
#
# Mirrors libero_30/combined/pi05_libero_goal_4_layer_film_lora2_knn36_30k_c0.01.sh
# (contrastive_loss_weight=0.01) but pretrains on the libero_90 suite and adapts
# sequentially on the libero_10 (LIBERO-Long) suite. Only deltas vs that script:
#   - pretrain dataset libero_90 (40k steps; scheduler decay 40k to match)
#   - sequential dataset libero_10; tfidf_top_t=1536
#   - names + the libero_10 online_task_ids / ds_to_env map
#
# Datasets:
#   pretrain   -> outputs/libero_90   (90 tasks)
#   sequential -> outputs/libero_10   (10 tasks: dataset task_index 0-9, the LIBERO-Long tasks)
#
# Eval mapping (dataset task_index -> libero_10 env task_id), based on the
# libero suite task ordering vs the dataset tasks.parquet:
#   0 put white mug left plate + yellow/white mug right   -> 4
#   1 put white mug on plate + choc pudding right          -> 6
#   2 put yellow and white mug in microwave and close it   -> 9
#   3 turn on the stove and put the moka pot on it         -> 2
#   4 put both alphabet soup and cream cheese box in basket-> 7
#   5 put both alphabet soup and tomato sauce in basket    -> 0
#   6 put both moka pots on the stove                      -> 8
#   7 put both cream cheese box and butter in basket       -> 1
#   8 put black bowl in bottom drawer of cabinet and close -> 3
#   9 pick up the book and place it in back of the caddy   -> 5

set -eo pipefail

echo "Job started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot

# pi05_base: pin to the cached revision whose processor config loads with the
# current code. HF's lerobot/pi05_base 'main' moved to a revision that uses a
# 'relative_actions_processor' step which this codebase (PR #2970) registers as
# 'delta_actions_processor', so the bare repo id 'lerobot/pi05_base' no longer
# loads. 9e55186 is the revision used by the working May-29 runs (identical model
# config; processor steps all present in the registry).
PI05_BASE="/home/josh/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30"

PRETRAIN_DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"

PRETRAIN_RUN=libero_90_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_40k
SEQ_RUN=libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_40k_top_t_1536

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
# Stage 1 - pretrain on libero_90 (90 tasks), eval on libero_90
###############################################################################
echo "=============================================================="
echo "[stage 1] PRETRAIN on libero_90 (90 tasks)"
echo "  eval env: libero_90"
echo "  output: $PRETRAIN_OUTPUT_DIR"
echo "=============================================================="

# Skip pretrain if a checkpoint already exists (allows safe re-runs).
if [ -d "$PRETRAIN_CHECKPOINT" ]; then
  echo "[stage 1] Pretrained checkpoint already exists at $PRETRAIN_CHECKPOINT - skipping pretrain."
else
  lerobot-train \
    --policy.path="$PI05_BASE" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$PRETRAIN_RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_90 \
    --dataset.root="$PRETRAIN_DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_90 \
    --output_dir="$PRETRAIN_OUTPUT_DIR" \
    --save_freq=20000 \
    --steps=40000 \
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
    --policy.scheduler_decay_steps=40000 \
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
# Stage 2 - sequential adaptation on libero_10 (10 tasks: dataset 0..9)
###############################################################################
echo "=============================================================="
echo "[stage 2] SEQUENTIAL on libero_10 (10 tasks: dataset 0..9)"
echo "  eval env: libero_10"
echo "  loading from: $PRETRAIN_CHECKPOINT"
echo "  output: $SEQ_OUTPUT_DIR"
echo "=============================================================="

lerobot-sequential-train \
  --policy.path="$PRETRAIN_CHECKPOINT" \
  --policy.empty_cameras=1 \
  --policy.dtype=bfloat16 \
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
  --eval.n_episodes=50 \
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
  --memory_value_lr=0.001 \
  --memory_value_lr_end=0.0001 \
  --memory_value_scheduler_type=linear

echo "Job completed at $(date)"
