# Joint fine-tune baseline (research_log2 Entry 4): all weights from the SmolVLA libero_90 base on all ten libero_10
# tasks jointly, the ceiling row. Recipe as pi05's (E60 add-9), which is the stage-1 recipe: 50k steps, batch 32,
# warmup 4k / decay 50k, the policy's default optimizer (SmolVLA: AdamW 1e-4 -> 2.5e-6), no gradient checkpointing.
# Evaluated on all ten envs, 25 episodes x 4 paired seeds, 10 actions executed per chunk.
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/joint_ft.sh joint eval
NAME=smolvla_l10_joint_ft
WANDB=(--wandb.enable=false)
NUM_WORKERS=8
GPUS="gpu=true,gpu_type=(a100_80|a40|a6000|h100|l40s|rtx6000ada)"   # >= 48 GB: all-weights training at batch 32
RES_JOINT="tmem=64G,h_rt=48:00:00,tscratch=80G"   # libero_10 24 GB with cache + ten 2.4 GB checkpoints
RES_EVAL="tmem=48G,h_rt=24:00:00,tscratch=40G"
RUN_MARGIN_MIN=30

SEQ_DATASET=libero_10
SEQ_TASKS='[0,1,2,3,4,5,6,7,8,9]'   # the eval runs on these tasks' envs: 4,6,9,2,7,0,8,1,3,5
SEQ_STEPS_PER_TASK=0
DS_TO_ENV='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'
STAGE1_CKPT=$CLUSTER_ROOT/runs/smolvla_l90_base/stage1/checkpoints/050000/pretrained_model

JOINT_STEPS=50000
JOINT_ARGS=(
  --batch_size=32 --save_freq=5000
  --policy.train_expert_only=false --policy.freeze_vision_encoder=false
  --policy.scheduler_warmup_steps=4000 --policy.scheduler_decay_steps=50000
  --policy.n_action_steps=10
)

EVAL_FROM=joint
EVAL_SEEDS=(1000 2000 3000 4000)
EVAL_ARGS=(--env.type=libero --env.task=libero_10 --eval.n_episodes=25 --eval.batch_size=5 --policy.n_action_steps=10)
