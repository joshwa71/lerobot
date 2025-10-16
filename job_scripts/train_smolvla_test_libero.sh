cat > train_smolvla_libero_10_test_libero.sh << 'EOF'
#!/bin/bash
#$ -S /bin/bash
#$ -l tmem=64G
#$ -l h_rt=72:00:00
#$ -l gpu=true,gpu_type=(a100_80|a40|h100|l40s|rtx6000ada)
#$ -pe gpu 1
#$ -R y
#$ -l tscratch=100G
#$ -N smolvla_train_test_libero
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

# Set Mujoco env vars
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

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
MODEL_SOURCE="/SAN/vision/jo71_vla_wd/lerobot/outputs/smolvla_base"
MODEL_SCRATCH="$SCRATCH_DIR/smolvla_base"
cp -r "$MODEL_SOURCE" "$MODEL_SCRATCH"
echo "Model copied to $MODEL_SCRATCH"

# Output directory in scratch
OUTPUT_SCRATCH="$SCRATCH_DIR/outputs/train/libero_10_smolvla_200k_test_libero"

# Enter working directory
cd /SAN/vision/jo71_vla_wd/lerobot

# Run training
lerobot-train \
  --policy.path="$MODEL_SCRATCH" \
  --policy.repo_id=outputs/train/libero_10_smolvla_200k_test_libero \
  --dataset.repo_id="$DATASET_SCRATCH" \
  --output_dir="$OUTPUT_SCRATCH" \
  --steps=20000 \
  --batch_size=32 \
  --num_workers=12 \
  --eval_freq=1000 \
  --env.type=libero \
  --eval.batch_size=1 \
  --eval.n_episodes=3 \
  --env.task=libero_10 \
  --save_freq=10000 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=15000 \
  --job_name=libero_10_smolvla_200k_test_libero \
  --wandb.enable=true

# Copy outputs back to permanent storage
echo "Copying outputs back to permanent storage..."
FINAL_OUTPUT_DIR="/SAN/vision/jo71_vla_wd/lerobot/outputs/train/libero_10_smolvla_200k"
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