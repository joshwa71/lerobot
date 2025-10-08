cat > train_smolvla_libero_10_300k.sh << 'EOF'
#!/bin/bash
#$ -S /bin/bash
#$ -l tmem=64G
#$ -l h_rt=72:00:00
#$ -l gpu=true,gpu_type=a100
#$ -pe gpu 1
#$ -R y
#$ -N smolvla_train
#$ -wd /SAN/vision/jo71_vla_wd/lerobot
#$ -j y
#$ -o /SAN/vision/jo71_vla_wd/lerobot/outputs/train/job_output_$JOB_ID.log

set -eo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $JOB_ID"

# Set cache directories to project space to avoid home quota issues
export TMPDIR="/SAN/vision/jo71_vla_wd/tmp"
export HF_DATASETS_CACHE="/SAN/vision/jo71_vla_wd/cache/hf_datasets"
export HUGGINGFACE_HUB_CACHE="/SAN/vision/jo71_vla_wd/cache/hf_hub"
export TRANSFORMERS_CACHE="/SAN/vision/jo71_vla_wd/cache/transformers"
export TORCH_HOME="/SAN/vision/jo71_vla_wd/cache/torch_home"
export WANDB_DIR="/SAN/vision/jo71_vla_wd/wandb"
export WANDB_CACHE_DIR="/SAN/vision/jo71_vla_wd/wandb/cache"

mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR"

# Setup conda
export PATH=/share/apps/miniconda3/bin:$PATH
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# Ensure conda libs are used first
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Enter directory
cd /SAN/vision/jo71_vla_wd/lerobot

# Run training
lerobot-train \
  --policy.path=outputs/smolvla_base \
  --policy.repo_id=outputs/train/libero_10_smolvla_300k \
  --dataset.repo_id=outputs/libero_10 \
  --output_dir=./outputs/train/libero_10_smolvla_300k \
  --steps=300000 \
  --batch_size=16 \
  --eval_freq=0 \
  --save_freq=20000 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true \
  --policy.scheduler_warmup_steps=20000 \
  --policy.scheduler_decay_steps=250000 \
  --job_name=libero_10_smolvla_300k \
  --wandb.enable=true

echo "Job completed at $(date)"
EOF