cat > train_smolvla_meta_libero_10.sh << 'EOF'
#!/bin/bash
#$ -S /bin/bash
#$ -l tmem=64G
#$ -l h_rt=72:00:00
#$ -l gpu=true,gpu_type=(a100_80|a40|h100|l40s|rtx6000ada)
#$ -pe gpu 2
#$ -R y
#$ -l tscratch=200G
#$ -N smolvla_meta_libero_10_train
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
DATASET_SOURCE="/SAN/vision/jo71_vla_wd/lerobot/outputs/libero"
DATASET_SCRATCH="$SCRATCH_DIR/data/libero"
cp -r "$DATASET_SOURCE" "$DATASET_SCRATCH"
echo "Dataset copied to $DATASET_SCRATCH"

# Copy pretrained model to scratch
echo "Copying pretrained model to scratch space..."
MODEL_SOURCE="/SAN/vision/jo71_vla_wd/lerobot/outputs/smolvla_base"
MODEL_SCRATCH="$SCRATCH_DIR/smolvla_base"
cp -r "$MODEL_SOURCE" "$MODEL_SCRATCH"
echo "Model copied to $MODEL_SCRATCH"

# Configure PyTorch NCCL for long-running operations (eval with torch.compile)
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false


# Output directory in scratch
OUTPUT_SCRATCH="$SCRATCH_DIR/outputs/train/reptile_smolvla_libero"

# Enter working directory
cd /SAN/vision/jo71_vla_wd/lerobot

# Run training
lerobot-meta-train \
  --steps=100000 \
  --batch_size=16 \
  --log_freq=100 \
  --dataset.repo_id=$DATASET_SCRATCH \
  --policy.path=$MODEL_SCRATCH \
  --policy.repo_id=outputs/train/reptile_smolvla_libero \
  --lora.enable=true \
  --lora.r=8 \
  --lora.alpha=16 \
  --lora.dropout=0.05 \
  --algo.type=reptile \
  --algo.meta_step_size=0.1 \
  --inner_steps=3 \
  --inner_opt.lr=3e-4 \
  --inner_opt.grad_clip_norm=10 \
  --tasks_per_outer_step=4 \
  --support_frames_per_task=50000 \
  --query_frames_per_task=512 \
  --train_tasks=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39] \
  --eval_tasks=[0,1,2,3,4] \
  --eval_freq=2000 \
  --eval.batch_size=1 \
  --eval.n_episodes=3 \
  --env.type=libero \
  --output_dir=$OUTPUT_SCRATCH \
  --job_name=reptile_smolvla_libero \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --save_freq=10000 \
  --parallel.enable=on

# Copy outputs back to permanent storage
echo "Copying outputs back to permanent storage..."
FINAL_OUTPUT_DIR="/SAN/vision/jo71_vla_wd/lerobot/outputs/train/reptile_smolvla_libero"
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