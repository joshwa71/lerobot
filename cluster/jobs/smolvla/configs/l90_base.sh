# SmolVLA libero_90 base: the shared stage-1 checkpoint that the memory chains and the baselines start from.
# All weights, end to end, as the pi05 stage 1 (job_scripts/nebius/libero_90/staged/stage1_base50k_stage2_probe10k.sh
# -> libero_90_pi05_base_nomem_50k): raw base -> libero_90, no memory, 50k steps, bs32, train_expert_only=false +
# freeze_vision_encoder=false, warmup 4k / cosine decay over 50k, the policy's own default optimizer (SmolVLA's:
# AdamW 1e-4 -> 2.5e-6; pi05's run used pi05's own, 2.5e-5). HF's dtypes (load_vlm_weights=true from smolvla_base's
# config: VLM and expert in bf16, heads fp32), no autocast. SmolVLA's code keeps the last VLM layer, lm_head and the
# final norm frozen even here, because nothing reads their outputs.
# Not carried over from the pi05 script: gradient checkpointing (pi05 needed it to fit bs32), empty_cameras=1 (pi05's
# third camera slot; SmolVLA runs LIBERO on its two cameras, as HF's own LIBERO SmolVLA does), the in-run libero_10
# eval at 50k (run as a separate eval job), save_freq=50000 (resuming needs periodic checkpoints).
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/l90_base.sh stage1
# Other configs use it as STAGE1_CKPT=$CLUSTER_ROOT/runs/smolvla_l90_base/stage1/checkpoints/050000/pretrained_model
NAME=smolvla_l90_base
WANDB=(--wandb.enable=false)
NUM_WORKERS=8

# bs32 all weights without checkpointing: ~33 GiB estimated peak, so GPUs with >= 48 GB.
GPUS="gpu=true,gpu_type=(a100_80|a40|a6000|h100|l40s|rtx6000ada)"
RES_STAGE1="tmem=64G,h_rt=36:00:00,tscratch=120G"
RUN_MARGIN_MIN=30

PRETRAIN_DATASET=libero_90
STAGE1_CKPT=""
STAGE1_FROM=lerobot/smolvla_base
STAGE1_STEPS=50000
STAGE1_ARGS=(
  --batch_size=32 --save_freq=5000
  --policy.train_expert_only=false --policy.freeze_vision_encoder=false
  --policy.scheduler_warmup_steps=4000 --policy.scheduler_decay_steps=50000
)
