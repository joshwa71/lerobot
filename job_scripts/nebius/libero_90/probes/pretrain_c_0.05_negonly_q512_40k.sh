#!/bin/bash
# FULL 40k pretrain of the validated probe-C recipe (research_log Entry 20):
#   contrastive_loss_weight=0.05 + contrastive_negatives_only=true + queue 512
#   (locality stays 0.25; all else = the 40k control recipe)
#
# Fresh run, NOT a resume of the 10k probe: lerobot auto-scales the LR schedule
# when steps < scheduler_decay_steps (schedulers.py:111), so the probe ran a
# COMPRESSED cosine (warmup 1000 / decay 10000). Resuming it to 40k rebuilds an
# unscaled scheduler -> LR sawtooth at step 10001, not comparable to control.
# Here steps == scheduler_decay_steps == 40000, so warmup 4000 / decay 40000
# are honored exactly as in the control run -> clean recipe attribution.
#
# Compare against control 7wu3dyax
# (libero_90_pi05_..._contrastive_0.01_..._40k):
#   - train/mse_loss: 0.261 @10k, 0.196 @20k, 0.133 @40k
#   - eval/pc_success: 76.4 @20k, 81.1 @40k
#   - routing_intra_task_support / query sims: does the SupCon
#     compaction/separation hold under the uncompressed schedule?

set -eo pipefail

echo "Pretrain c=0.05 negonly q512 40k started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot

# Pinned pi05_base revision (see combined script for rationale).
PI05_BASE="/home/josh/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30"

PRETRAIN_DATASET_ROOT="$ROOT_DIR/outputs/libero_90"

RUN=libero_90_pi05_8_10_12_14_contrastive_0.05_negonly_q512_40k
OUTPUT_DIR="$ROOT_DIR/outputs/train/$RUN"

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

lerobot-train \
  --policy.path="$PI05_BASE" \
  --policy.empty_cameras=1 \
  --policy.dtype=bfloat16 \
  --policy.repo_id="outputs/train/$RUN" \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --dataset.repo_id=libero_90 \
  --dataset.root="$PRETRAIN_DATASET_ROOT" \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
  --env.type=libero \
  --env.task=libero_90 \
  --output_dir="$OUTPUT_DIR" \
  --save_freq=10000 \
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
  --job_name="$RUN" \
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
  --policy.memory_layer.contrastive_negatives_only=true \
  --policy.memory_layer.contrastive_margin=0.0 \
  --policy.memory_layer.contrastive_query_queue=512 \
  --policy.memory_layer.routing_loss_topk=36 \
  --policy.memory_layer.routing_intra_task_locality_weight=0.25 \
  --policy.memory_layer.routing_intra_task_min_support=128 \
  --policy.memory_layer.routing_intra_task_max_support=2048 \
  --policy.memory_layer.routing_inter_task_separation_weight=0.25

echo "Pretrain c=0.05 negonly q512 40k completed at $(date)"
