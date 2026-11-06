cat > smolvla_sequential_train_6_12_wide_film_dropout_lora_4.sh << 'EOF'
#!/bin/bash
#$ -S /bin/bash
#$ -l tmem=64G
#$ -l h_rt=72:00:00
#$ -l gpu=true,gpu_type=(a100_80|h100)
#$ -pe gpu 1
#$ -R y
#$ -l tscratch=200G
#$ -N smolvla_sequential_train_6_12_wide_film_dropout_lora_4
#$ -wd /SAN/vision/jo71_vla_wd/lerobot_memory
#$ -j y
#$ -o /SAN/vision/jo71_vla_wd/lerobot_memory/outputs/train/job_output_$JOB_ID.log

set -eo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $JOB_ID"

# Setup finish trap: sync outputs then cleanup scratch
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

# Create scratch directory
SCRATCH_DIR="/scratch0/johara/$JOB_ID"
mkdir -p "$SCRATCH_DIR"/{cache,data,outputs}

export MUJOCO_GL=egl
unset DISPLAY

# Force NVIDIA EGL (avoid conda/Mesa libEGL)
if [ -e /usr/lib/x86_64-linux-gnu/libEGL.so.1 ]; then
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi
if [ -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
fi

echo "Created scratch directory: $SCRATCH_DIR"
# Set cache directories to scratch space
export TMPDIR="$SCRATCH_DIR/tmp"
export HF_DATASETS_CACHE="$SCRATCH_DIR/cache/hf_datasets"
export HUGGINGFACE_HUB_CACHE="$SCRATCH_DIR/cache/hf_hub"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/cache/transformers"
export TORCH_HOME="$SCRATCH_DIR/cache/torch_home"
export WANDB_DIR="$SCRATCH_DIR/wandb"
export WANDB_CACHE_DIR="$SCRATCH_DIR/wandb/cache"
export WANDB_DISABLE_GPU=false
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR"

# Setup conda
export PATH=/share/apps/miniconda3/bin:$PATH
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate lerobot-memory


# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Copy dataset to scratch
echo "Copying dataset to scratch space..."
DATASET_SOURCE="/SAN/vision/jo71_vla_wd/lerobot/outputs/libero_10"
DATASET_SCRATCH="$SCRATCH_DIR/data/libero_10"
cp -r "$DATASET_SOURCE" "$DATASET_SCRATCH"
echo "Dataset copied to $DATASET_SCRATCH"


# Copy pretrained model to scratch
echo "Copying pretrained model to scratch space..."
MODEL_SOURCE="/SAN/vision/jo71_vla_wd/lerobot_memory/outputs/train/libero_95_6_12_wide_film_dropout_lora_4"
MODEL_SCRATCH="$SCRATCH_DIR/libero_95_6_12_wide_film_dropout_lora_4"
cp -r "$MODEL_SOURCE" "$MODEL_SCRATCH"
echo "Model copied to $MODEL_SCRATCH"

# Configure PyTorch NCCL for long-running operations (eval with torch.compile)
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false


# Output directory in scratch
OUTPUT_SCRATCH="$SCRATCH_DIR/outputs/train/sequential_libero_95_6_12_wide_film_dropout_lora_4"
# Final output target (used by trap for sync-back)
FINAL_OUTPUT_DIR="/SAN/vision/jo71_vla_wd/lerobot_memory/outputs/train/sequential_libero_95_6_12_wide_film_dropout_lora_4"

# Periodic backup function (every 6 hours)
function periodic_backup {
    local scratch_dir="$1"
    local final_dir="$2"
    while true; do
        sleep 21600  # 6 hours in seconds
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

# Start periodic backup in background
echo "Starting periodic backup process (every 6 hours)..."
periodic_backup "$OUTPUT_SCRATCH" "$FINAL_OUTPUT_DIR" &
BACKUP_PID=$!
echo "Periodic backup process started with PID: $BACKUP_PID"

# Enter working directory
cd /SAN/vision/jo71_vla_wd/lerobot_memory

# Run training
python -m lerobot.scripts.lerobot_sequential_train \
  --policy.path="$MODEL_SCRATCH/checkpoints/last/pretrained_model" \
  --dataset.repo_id="$DATASET_SCRATCH" \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir="$OUTPUT_SCRATCH" \
  --steps=200000 \
  --batch_size=64 \
  --num_workers=8 \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --log_freq=200 \
  --wandb.enable=true \
  --wandb.project=vla-memory \
  --job_name=sequential_libero_10_smolvla_6_12_wide_film_dropout_lora_4 \
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
EOF