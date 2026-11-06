cat > pretrain_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048.sh << 'INNER_EOF'
#!/bin/bash
#$ -S /bin/bash
#$ -l tmem=64G
#$ -l h_rt=72:00:00
#$ -l gpu=true,gpu_type=(a100_80|h100)
#$ -pe gpu 1
#$ -R y
#$ -l tscratch=200G
#$ -N libero_95_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048
#$ -wd /SAN/vision/jo71_vla_wd/lerobot_memory
#$ -j y
#$ -o /SAN/vision/jo71_vla_wd/lerobot_memory/outputs/train/job_output_$JOB_ID.log

set -eo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $JOB_ID"

function finish {
    set +e
    echo "Stopping periodic backup process..."
    if [ ! -z "$BACKUP_PID" ]; then
        kill $BACKUP_PID 2>/dev/null || true
        wait $BACKUP_PID 2>/dev/null || true
    fi
    echo "Syncing outputs from scratch before cleanup..."
    if [ -n "$OUTPUT_SCRATCH" ] && [ -d "$OUTPUT_SCRATCH" ]; then
        mkdir -p "$FINAL_OUTPUT_DIR"
        cp -r "$OUTPUT_SCRATCH"/* "$FINAL_OUTPUT_DIR/" || true
    fi
    echo "Cleaning up scratch space..."
    rm -rf "$SCRATCH_DIR"
    echo "Cleanup completed at $(date)"
}
trap finish EXIT ERR INT TERM

SCRATCH_DIR="/scratch0/johara/$JOB_ID"
mkdir -p "$SCRATCH_DIR"/{cache,data,outputs}

export MUJOCO_GL=egl
unset DISPLAY

if [ -e /usr/lib/x86_64-linux-gnu/libEGL.so.1 ]; then
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi
if [ -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
fi

echo "Created scratch directory: $SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR/tmp"
export HF_DATASETS_CACHE="$SCRATCH_DIR/cache/hf_datasets"
export HUGGINGFACE_HUB_CACHE="$SCRATCH_DIR/cache/hf_hub"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/cache/transformers"
export TORCH_HOME="$SCRATCH_DIR/cache/torch_home"
export WANDB_DIR="$SCRATCH_DIR/wandb"
export WANDB_CACHE_DIR="$SCRATCH_DIR/wandb/cache"
export WANDB_DISABLE_GPU=false
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR"

export PATH=/share/apps/miniconda3/bin:$PATH
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate lerobot-memory

echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo "Copying dataset to scratch space..."
DATASET_SOURCE="/SAN/vision/jo71_vla_wd/lerobot/outputs/libero_95"
DATASET_SCRATCH="$SCRATCH_DIR/data/libero_95"
cp -r "$DATASET_SOURCE" "$DATASET_SCRATCH"
echo "Dataset copied to $DATASET_SCRATCH"

echo "Copying pretrained model to scratch space..."
MODEL_SOURCE="/SAN/vision/jo71_vla_wd/lerobot/outputs/smolvla_base"
MODEL_SCRATCH="$SCRATCH_DIR/smolvla_base"
cp -r "$MODEL_SOURCE" "$MODEL_SCRATCH"
echo "Model copied to $MODEL_SCRATCH"

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false

OUTPUT_SCRATCH="$SCRATCH_DIR/outputs/train/libero_95_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048"
FINAL_OUTPUT_DIR="/SAN/vision/jo71_vla_wd/lerobot_memory/outputs/train/libero_95_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048"

function periodic_backup {
    local scratch_dir="$1"
    local final_dir="$2"
    while true; do
        sleep 21600
        if [ -d "$scratch_dir" ]; then
            echo "[$(date)] Performing periodic backup of training outputs..."
            mkdir -p "$final_dir"
            if command -v rsync &> /dev/null; then
                rsync -av --delete "$scratch_dir/" "$final_dir/" 2>&1 | head -20
            else
                cp -r "$scratch_dir"/* "$final_dir/"
            fi
            echo "[$(date)] Periodic backup completed"
        fi
    done
}

echo "Starting periodic backup process (every 6 hours)..."
periodic_backup "$OUTPUT_SCRATCH" "$FINAL_OUTPUT_DIR" &
BACKUP_PID=$!
echo "Periodic backup process started with PID: $BACKUP_PID"

cd /SAN/vision/jo71_vla_wd/lerobot_memory

lerobot-train \
  --policy.path="$MODEL_SCRATCH" \
  --policy.repo_id=outputs/train/libero_95_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048 \
  --dataset.repo_id="$DATASET_SCRATCH" \
  --env.type=libero \
  --env.task=libero_spatial \
  --output_dir="$OUTPUT_SCRATCH" \
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
  --job_name=libero_95_8_10_12_14_film_lora_2_sample_contrastive_1_sep_0.25_loc_0.25_sup_128_2048 \
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
  --policy.memory_layer.mem_knn=16 \
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
  --policy.memory_layer.routing_intra_task_locality_weight=0.25 \
  --policy.memory_layer.routing_intra_task_min_support=128 \
  --policy.memory_layer.routing_intra_task_max_support=2048 \
  --policy.memory_layer.routing_inter_task_separation_weight=0.25

echo "Job completed at $(date)"
INNER_EOF
