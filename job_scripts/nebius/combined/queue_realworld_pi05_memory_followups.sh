#!/bin/bash
# Queue two real-world pi0.5 memory follow-up experiments after the currently
# running contrastive=0.01 / support=4096 combined run completes.

set -eo pipefail

ROOT_DIR=/home/josh/lerobot

PRETRAIN_DATASET_ROOT="$ROOT_DIR/outputs/realworld_pretrain"
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/realworld_seq"

CURRENT_SESSION=${CURRENT_SESSION:-rw_pi05_c001_50k}
CURRENT_SEQ_RUN=realworld_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_4096_knn_36_50k
CURRENT_SEQ_DONE="$ROOT_DIR/outputs/train/$CURRENT_SEQ_RUN/checkpoints/015000/pretrained_model/model.safetensors"

export MUJOCO_GL=osmesa
unset DISPLAY

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1

source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated

cd "$ROOT_DIR"

echo "Queue started on $(hostname) at $(date)"
echo "Waiting for current run completion marker:"
echo "  $CURRENT_SEQ_DONE"

wait_for_current_pair() {
  if [ -f "$CURRENT_SEQ_DONE" ]; then
    echo "Current pair already complete."
    return
  fi

  while [ ! -f "$CURRENT_SEQ_DONE" ]; do
    if ! tmux has-session -t "$CURRENT_SESSION" 2>/dev/null; then
      echo "ERROR: tmux session '$CURRENT_SESSION' ended before current sequential checkpoint existed."
      echo "Missing: $CURRENT_SEQ_DONE"
      exit 1
    fi
    echo "Current pair still running at $(date). Checking again in 5 minutes."
    sleep 300
  done

  echo "Current sequential checkpoint found at $(date). Waiting for tmux session to exit cleanly."
  while tmux has-session -t "$CURRENT_SESSION" 2>/dev/null; do
    sleep 60
  done
  echo "Current tmux session has exited."
}

run_pair() {
  local contrastive_weight="$1"
  local max_support="$2"

  local pretrain_run="realworld_pretrain_pi05_8_10_12_14_film_lora_2_sample_contrastive_${contrastive_weight}_sep_0.25_loc_0.25_sup_128_${max_support}_knn_36_50k"
  local seq_run="realworld_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_${contrastive_weight}_sep_0.25_loc_0.25_sup_128_${max_support}_knn_36_50k"

  local pretrain_output_dir="$ROOT_DIR/outputs/train/$pretrain_run"
  local seq_output_dir="$ROOT_DIR/outputs/train/$seq_run"
  local pretrain_checkpoint="$pretrain_output_dir/checkpoints/last/pretrained_model"
  local pretrain_done="$pretrain_checkpoint/model.safetensors"
  local seq_done="$seq_output_dir/checkpoints/015000/pretrained_model/model.safetensors"

  echo "=============================================================="
  echo "[pair] contrastive=$contrastive_weight max_support=$max_support"
  echo "  pretrain output:   $pretrain_output_dir"
  echo "  sequential output: $seq_output_dir"
  echo "=============================================================="

  if [ -f "$pretrain_done" ]; then
    echo "[pretrain] Checkpoint already exists at $pretrain_done - skipping pretrain."
  else
    lerobot-train \
      --policy.path=lerobot/pi05_base \
      --policy.empty_cameras=1 \
      --policy.dtype=bfloat16 \
      --policy.repo_id="outputs/train/$pretrain_run" \
      --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
      --dataset.repo_id=realworld_pretrain \
      --dataset.root="$PRETRAIN_DATASET_ROOT" \
      --rename_map='{"observation.images.cam_high":"observation.images.base_0_rgb","observation.images.cam_wrist":"observation.images.left_wrist_0_rgb"}' \
      --output_dir="$pretrain_output_dir" \
      --save_freq=20000 \
      --steps=50000 \
      --batch_size=32 \
      --gradient_accumulation_steps=1 \
      --num_workers=8 \
      --eval_freq=0 \
      --log_freq=200 \
      --policy.freeze_vision_encoder=false \
      --policy.train_expert_only=false \
      --policy.scheduler_warmup_steps=4000 \
      --policy.scheduler_decay_steps=50000 \
      --job_name="$pretrain_run" \
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
      --policy.memory_layer.contrastive_loss_weight="$contrastive_weight" \
      --policy.memory_layer.contrastive_margin=0.0 \
      --policy.memory_layer.contrastive_query_queue=128 \
      --policy.memory_layer.routing_loss_topk=36 \
      --policy.memory_layer.routing_intra_task_locality_weight=0.25 \
      --policy.memory_layer.routing_intra_task_min_support=128 \
      --policy.memory_layer.routing_intra_task_max_support="$max_support" \
      --policy.memory_layer.routing_inter_task_separation_weight=0.25
  fi

  if [ ! -f "$pretrain_done" ]; then
    echo "ERROR: pretrain finished but $pretrain_done does not exist"
    exit 1
  fi

  if [ -f "$seq_done" ]; then
    echo "[sequential] Final checkpoint already exists at $seq_done - skipping sequential."
  else
    lerobot-sequential-train \
      --policy.path="$pretrain_checkpoint" \
      --policy.empty_cameras=1 \
      --policy.dtype=bfloat16 \
      --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
      --dataset.repo_id=realworld_seq \
      --dataset.root="$SEQ_DATASET_ROOT" \
      --rename_map='{"observation.images.cam_high":"observation.images.base_0_rgb","observation.images.cam_wrist":"observation.images.left_wrist_0_rgb"}' \
      --output_dir="$seq_output_dir" \
      --steps=200000 \
      --batch_size=32 \
      --gradient_accumulation_steps=1 \
      --num_workers=8 \
      --log_freq=200 \
      --wandb.enable=true \
      --wandb.project=vla-memory \
      --job_name="$seq_run" \
      --online_task_ids='[0,1,2,3,4]' \
      --online_steps_per_task=3000 \
      --policy.memory_layer.aggregate_usage=false \
      --save_after_each_task=true \
      --reinit_optimizer_each_task=true \
      --tfidf_enable=true \
      --tfidf_top_t=512 \
      --use_online_idf_stats=true \
      --idf_exponent=1 \
      --memory_value_lr=0.001 \
      --memory_value_lr_end=0.0001 \
      --memory_value_scheduler_type=linear
  fi

  if [ ! -f "$seq_done" ]; then
    echo "ERROR: sequential finished but $seq_done does not exist"
    exit 1
  fi

  echo "[pair complete] contrastive=$contrastive_weight max_support=$max_support at $(date)"
}

wait_for_current_pair
run_pair "0.05" "4096"
run_pair "0.01" "8000"

echo "All queued follow-up runs completed at $(date)"
