# SmolVLA libero_90 base: the shared stage-1 checkpoint that the memory chains and the baselines start from.
# HF's own smolvla_base fine-tune recipe (docs/source/smolvla.mdx), unchanged. smolvla_base's config decides
# everything not set here: load_vlm_weights=true (HF's loading path and dtypes), expert only with the vision
# encoder frozen, no autocast, AdamW lr 1e-4, cosine warmup 1k / decay 30k to 2.5e-6.
# 30k steps = that schedule's own decay horizon, at HF's batch 64 (~3.3 epochs of libero_90's 574k frames).
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/l90_base.sh stage1
# Other configs use it as STAGE1_CKPT=$CLUSTER_ROOT/runs/smolvla_l90_base/stage1/checkpoints/030000/pretrained_model
NAME=smolvla_l90_base
WANDB=(--wandb.enable=false)
NUM_WORKERS=8

# bs64 peaks at ~14 GiB (measured), so any bf16 GPU with >= 24 GB; L4 is left out as too slow.
GPUS="gpu=true,gpu_type=(a10|a100|a100_80|a40|a6000|h100|l40s|rtx3090|rtx4090|rtx6000ada)"
RES_STAGE1="tmem=64G,h_rt=24:00:00,tscratch=120G"
RUN_MARGIN_MIN=30

PRETRAIN_DATASET=libero_90
STAGE1_CKPT=""
STAGE1_FROM=lerobot/smolvla_base
STAGE1_STEPS=30000
STAGE1_ARGS=(--batch_size=64 --save_freq=2500)
