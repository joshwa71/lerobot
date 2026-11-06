#!/bin/bash
# End-to-end check of the cluster setup on one GPU: container + venv + CUDA, a short SmolVLA
# training run that checkpoints to /SAN, a resume from that checkpoint, and a LIBERO rollout with
# EGL rendering. Needs cluster/download_assets.sh to have run on a login node first.
#   qsub cluster/jobs/smoke_smolvla.sh
#$ -S /bin/bash
#$ -N smoke_smolvla
#$ -l tmem=32G,h_rt=2:00:00
#$ -l gpu=true,gpu_type=(a10|a100|a100_80|a40|a6000|h100|l4|l40s|rtx3090|rtx4070ti|rtx4090|rtx6000ada)
#$ -j y
#$ -o /SAN/vision/jo71_vla_wd/logs/
set -euo pipefail
source /SAN/vision/jo71_vla_wd/lerobot/cluster/env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
# A step that hangs is killed with SIGABRT, which PYTHONFAULTHANDLER turns into a traceback of every
# thread, so the log shows where it was stuck instead of the job silently burning its h_rt.
export PYTHONFAULTHANDLER=1
STEP_LIMIT=20m

RUN=$CLUSTER_ROOT/runs/smoke/smolvla_$JOB_ID
echo "== $(hostname) $(date) job=$JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "== cpus: nproc=$(nproc) NSLOTS=${NSLOTS:-} $(grep Cpus_allowed_list /proc/self/status)"
cd "$REPO"
git log -1 --format='== repo %h %s'

echo "== 1/4 CUDA inside the container"
ctr python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", float((x @ x).float().abs().mean()))
EOF

# LIBERO calls its cameras image/image2; smolvla_base expects camera1..3 (camera3 is padded)
RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
TRAIN_ARGS=(
    --policy.path=lerobot/smolvla_base
    --dataset.repo_id=HuggingFaceVLA/libero
    # explicit root: without it, lerobot treats an `hf download --local-dir` copy as a legacy
    # mirror and tries to re-fetch it from the hub, which fails offline
    --dataset.root="$HF_LEROBOT_HOME/HuggingFaceVLA/libero"
    --dataset.episodes="[0,1,2,3,4,5,6,7]"
    --rename_map="$RENAME"
    --batch_size=8 --num_workers=2
    --log_freq=20 --save_freq=100 --eval_freq=0
    --policy.push_to_hub=false --wandb.enable=false
)
echo "== 2/4 train 200 steps, checkpoint every 100 -> $RUN"
timeout -s ABRT $STEP_LIMIT ctr lerobot-train "${TRAIN_ARGS[@]}" --steps=200 --output_dir="$RUN" --job_name="smoke_$JOB_ID"
ls "$RUN/checkpoints"

echo "== 3/4 resume to 300 steps"
timeout -s ABRT $STEP_LIMIT ctr lerobot-train --config_path="$RUN/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true --steps=300
ls "$RUN/checkpoints"

echo "== 4/4 LIBERO rollout (1 episode per libero_10 task, EGL)"
timeout -s ABRT $STEP_LIMIT ctr lerobot-eval --policy.path="$RUN/checkpoints/last/pretrained_model" \
    --env.type=libero --env.task=libero_10 \
    --eval.n_episodes=1 --eval.batch_size=1 --rename_map="$RENAME" --output_dir="$RUN/eval"
echo "== done $(date)"
