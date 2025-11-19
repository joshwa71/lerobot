cat > train_smolvla_memory_expert_vlm_memory_only_256.sh << 'EOF'
#!/bin/bash
#$ -S /bin/bash
#$ -l tmem=64G
#$ -l h_rt=96:00:00
#$ -l gpu=true,gpu_type=(h100)
#$ -pe gpu 1
#$ -R y
#$ -l tscratch=200G
#$ -N smolvla_memory_train_expert_vlm_memory_only_256
#$ -wd /SAN/vision/jo71_vla_wd/lerobot_memory
#$ -j y
#$ -o /SAN/vision/jo71_vla_wd/lerobot_memory/outputs/train/job_output_$JOB_ID.log

set -eo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $JOB_ID"

# Setup finish trap: sync outputs then cleanup scratch
function finish {
    set +e
    echo "Syncing outputs from scratch before cleanup..."
    if [ -n "$OUTPUT_SCRATCH" ] && [ -d "$OUTPUT_SCRATCH" ]; then
        mkdir -p "$FINAL_OUTPUT_DIR"
        cp -r "$OUTPUT_SCRATCH"/* "$FINAL_OUTPUT_DIR/" || true
    fi
    if [ -n "$WANDB_DIR" ] && [ -d "$WANDB_DIR" ]; then
        mkdir -p /SAN/vision/jo71_vla_wd/lerobot_memory/wandb
        cp -r "$WANDB_DIR"/* /SAN/vision/jo71_vla_wd/lerobot_memory/wandb/ || true
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

# ## Test ##
# # Force correct backend and GPU mapping for EGL
# export MUJOCO_GL=egl
# unset DISPLAY
# FIRST_VISIBLE="$(echo "${CUDA_VISIBLE_DEVICES}" | cut -d',' -f1)"
# export EGL_DEVICE_ID="${FIRST_VISIBLE}"
# export MUJOCO_EGL_DEVICE_ID="${FIRST_VISIBLE}"

# # Prefer system NVIDIA EGL over any conda-provided Mesa EGL
# if [ -e /usr/lib/x86_64-linux-gnu/libEGL.so.1 ]; then
#   export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
# fi
# # Stronger vendor pin (optional)
# if [ -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
#   export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
# fi

# echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# echo "EGL_DEVICE_ID=${EGL_DEVICE_ID}  MUJOCO_GL=${MUJOCO_GL}"

# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Copy dataset to scratch
echo "Copying dataset to scratch space..."
DATASET_SOURCE="/SAN/vision/jo71_vla_wd/lerobot/outputs/libero_90"
DATASET_SCRATCH="$SCRATCH_DIR/data/libero_90"
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
OUTPUT_SCRATCH="$SCRATCH_DIR/outputs/train/smolvla_libero_90_memory_expert_vlm_memory_only_256"
# Final output target (used by trap for sync-back)
FINAL_OUTPUT_DIR="/SAN/vision/jo71_vla_wd/lerobot_memory/outputs/train/smolvla_libero_90_memory_expert_vlm_memory_only_256"

# Enter working directory
cd /SAN/vision/jo71_vla_wd/lerobot_memory

# Run training
lerobot-train \
  --policy.path="$MODEL_SCRATCH" \
  --policy.repo_id=outputs/train/smolvla_libero_90_memory_expert_vlm_memory_only_256 \
  --dataset.repo_id="$DATASET_SCRATCH" \
  --env.type=libero \
  --env.task=libero_spatial \
  --output_dir="$OUTPUT_SCRATCH" \
  --save_freq=10000 \
  --steps=200000 \
  --batch_size=64 \
  --num_workers=12 \
  --eval.batch_size=1 \
  --eval.n_episodes=3 \
  --eval_freq=20000 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true \
  --policy.scheduler_warmup_steps=10000 \
  --policy.scheduler_decay_steps=150000 \
  --job_name=smolvla_libero_90_memory_expert_vlm_memory_only_256 \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.project=vla-memory \
  --wandb.disable_artifact=true \
  --policy.memory_layers=true \
  --policy.memory_layer.memory_only=true \
  --policy.memory_layer.layers="[10]" \
  --policy.memory_layer.vlm_layers="[10]" \
  --policy.memory_layer.log_usage=true \
  --policy.memory_layer.enabled=true \
  --policy.memory_layer.aggregate_usage=true \
  --policy.memory_layer.mem_n_keys=256 \
  --policy.memory_layer.mem_heads=4 \
  --policy.memory_layer.mem_knn=16 \
  --policy.memory_layer.mem_k_dim=256 \
  --policy.memory_layer.value_fixed_lr=0.001 \
  --policy.memory_layer.memory_lr=0.001


echo "Job completed at $(date)"
EOF