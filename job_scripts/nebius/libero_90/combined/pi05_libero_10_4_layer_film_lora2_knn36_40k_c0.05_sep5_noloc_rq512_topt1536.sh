#!/bin/bash
# Combined pi05 + memory: pretrain on libero_90 (eval libero_90),
# then sequential adapt on libero_10 (eval libero_10).
#
# GRADUATION RUN for the sep=5 prior (research_log Entry 25/26). Base = the
# c0.01/top_t1536 combined script; deltas are EXACTLY the validated probe-9 recipe
# plus the routing queue:
#   - contrastive_loss_weight   0.01 -> 0.05   (load-bearing compaction; goal A / P7)
#   - contrastive_query_queue   128  -> 512    (match the probe's SupCon coverage)
#   - routing_inter_task_separation_weight 0.25 -> 5.0  (sep curve winner; goal B / P9)
#   - routing_intra_task_locality_weight    0.25 -> 0   (locality inert; goal C / P10)
#       + support-band flags dropped (inert with locality off)
#   - routing_query_queue       (absent) -> 512  (CRITICAL: the cross-batch routing
#       queue that makes separation decouple; this 40k reference predates it, Entry 23)
# Held-out audit of this recipe at 10k: famIoU 0.264 / core50 2679 (clears the gate,
# capacity >= control), via translation not shrinkage.
#
# Sequential tfidf_top_t stays 1536: the prior is broad-but-separated (core50 ~2679,
# effnum ~14881 ~ control), so ~1536 is the right write budget per Entry 22(d), and
# it's safer than the Entry-19 cliff because overlap is lower (famIoU 0.264 vs 0.349).
# WATCH task9-reads-task8-updates early; re-derive top_t from per-batch L14 effnum if
# overwrite climbs.
#
# Datasets:
#   pretrain   -> outputs/libero_90   (90 tasks)
#   sequential -> outputs/libero_10   (10 tasks: dataset task_index 0-9, LIBERO-Long)
#
# Eval mapping (dataset task_index -> libero_10 env task_id):
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
# current code (see c0.01 combined script for the full rationale; 9e55186 is the
# revision used by the working runs).
PI05_BASE="/home/josh/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30"

PRETRAIN_DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"

PRETRAIN_RUN=libero_90_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k
SEQ_RUN=libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536

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
echo "[stage 1] PRETRAIN on libero_90 (90 tasks)  [sep=5 / c=0.05 / noloc / rq512]"
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
    --policy.memory_layer.contrastive_loss_weight=0.05 \
    --policy.memory_layer.contrastive_margin=0.0 \
    --policy.memory_layer.contrastive_query_queue=512 \
    --policy.memory_layer.routing_loss_topk=36 \
    --policy.memory_layer.routing_intra_task_locality_weight=0 \
    --policy.memory_layer.routing_inter_task_separation_weight=5.0 \
    --policy.memory_layer.routing_query_queue=512
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
