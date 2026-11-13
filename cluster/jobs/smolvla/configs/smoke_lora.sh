# Cluster smoke of the LoRA baselines on SmolVLA (never run before on this policy): every LoRA path for a few steps
# on the real l90 base, container and scratch, at the largest rank of the sweep (256) and batch 32, on 48 GB cards:
# a specialist with its own-env eval, naive sequential on two tasks with both triangle rows, multitask LoRA + eval.
#   rm -rf /SAN/vision/jo71_vla_wd/runs/smoke_lora && cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/smoke_lora.sh spec seq tri joint eval
RANK=256 LR=1e-4
source "$REPO/cluster/jobs/smolvla/configs/lora_common.sh"
NAME=smoke_lora
GPUS="gpu=true,gpu_type=(a40|a6000|l40s|rtx6000ada)"   # the 48 GB cards only
for st in SPEC SEQ TRI JOINT EVAL; do declare "RES_$st=tmem=48G,h_rt=3:00:00,tscratch=40G"; done
RUN_MARGIN_MIN=15
SEQ_INIT=$STAGE1_CKPT   # naive sequential
SEQ_TASKS='[0,1]'
SEQ_STEPS_PER_TASK=20
JOINT_STEPS=20
JOINT_ARGS+=(--save_freq=20 --log_freq=5)
EVAL_FROM=joint
EVAL_SEEDS=(1000)
EVAL_ARGS+=(--eval.n_episodes=2 --eval.batch_size=2)
