cat > train_pi05_libero_10_100k.sh << 'EOF'
#!/bin/bash
#$ -S /bin/bash
#$ -l tmem=64G
#$ -l h_rt=72:00:00
#$ -l gpu=true,gpu_type=(a100_80|h100)
#$ -pe gpu 2
#$ -R y
#$ -l tscratch=200G
#$ -N pi05_train
#$ -wd /SAN/vision/jo71_vla_wd/lerobot
#$ -j y
#$ -o /SAN/vision/jo71_vla_wd/lerobot/outputs/train/job_output_$JOB_ID.log

set -eo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $JOB_ID"

# Setup cleanup trap
function cleanup {
    echo "Cleaning up scratch space..."
    rm -rf /scratch0/johara/$JOB_ID
    echo "Cleanup completed at $(date)"
}
trap cleanup EXIT ERR INT TERM

# Create scratch directory
SCRATCH_DIR="/scratch0/johara/$JOB_ID"
mkdir -p "$SCRATCH_DIR"/{cache,data,outputs}

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
conda activate lerobot-full
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Copy dataset to scratch
echo "Copying dataset to scratch space..."
DATASET_SOURCE="/SAN/vision/jo71_vla_wd/lerobot/outputs/libero_10_quant"
DATASET_SCRATCH="$SCRATCH_DIR/data/libero_10_quant"
cp -r "$DATASET_SOURCE" "$DATASET_SCRATCH"
echo "Dataset copied to $DATASET_SCRATCH"

# Copy pretrained model to scratch
echo "Copying pretrained model to scratch space..."
MODEL_SOURCE="/SAN/vision/jo71_vla_wd/lerobot/outputs/pi05_base"
MODEL_SCRATCH="$SCRATCH_DIR/pi05_base"
cp -r "$MODEL_SOURCE" "$MODEL_SCRATCH"
echo "Model copied to $MODEL_SCRATCH"

# Configure PyTorch NCCL for long-running operations (eval with torch.compile)
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false


# Output directory in scratch
OUTPUT_SCRATCH="$SCRATCH_DIR/outputs/train/libero_10_pi05_100k"

# Enter working directory
cd /SAN/vision/jo71_vla_wd/lerobot

# Run training
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --policy.type=pi05 \
  --policy.dtype=bfloat16 \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.pretrained_path="$MODEL_SCRATCH" \
  --policy.repo_id=outputs/train/libero_10_pi05_100k \
  --dataset.repo_id="$DATASET_SCRATCH" \
  --output_dir="$OUTPUT_SCRATCH" \
  --steps=100000 \
  --batch_size=16 \
  --num_workers=12 \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=3 \
  --eval_freq=5000 \
  --save_freq=20000 \
  --policy.push_to_hub=false \
  --job_name=libero_10_pi05_100k \
  --wandb.enable=true

# Copy outputs back to permanent storage
echo "Copying outputs back to permanent storage..."
FINAL_OUTPUT_DIR="/SAN/vision/jo71_vla_wd/lerobot/outputs/train/libero_10_pi05_100k"
mkdir -p "$FINAL_OUTPUT_DIR"
cp -r "$OUTPUT_SCRATCH"/* "$FINAL_OUTPUT_DIR/"
echo "Outputs copied to $FINAL_OUTPUT_DIR"

# Copy wandb logs back
if [ -d "$WANDB_DIR" ]; then
    echo "Copying wandb logs..."
    mkdir -p /SAN/vision/jo71_vla_wd/lerobot/wandb
    cp -r "$WANDB_DIR"/* /SAN/vision/jo71_vla_wd/lerobot/wandb/ || true
fi

echo "Job completed at $(date)"
EOF