# Zero-shot floor (research_log2 Entry 1 baselines): the SmolVLA libero_90 base, no memory, no adaptation,
# evaluated on all ten libero_10 envs with the headline instrument (25 episodes x 4 paired seeds, 10 actions
# executed per chunk). pi05's floor was 10.6 (single seed, 50-action chunks).
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/l90_zeroshot.sh eval
NAME=smolvla_l90_zeroshot
EVAL_MODEL=$CLUSTER_ROOT/runs/smolvla_l90_base/stage1/checkpoints/050000/pretrained_model

GPUS="gpu=true,gpu_type=(a100|a100_80|a40|a6000|h100|l40s|rtx6000ada)"
RES_EVAL="tmem=48G,h_rt=24:00:00,tscratch=40G"
RUN_MARGIN_MIN=30
NUM_WORKERS=4

SEQ_DATASET=libero_10
SEQ_TASKS='[0,1,2,3,4,5,6,7,8,9]'   # the eval runs on these tasks' envs: 4,6,9,2,7,0,8,1,3,5
SEQ_STEPS_PER_TASK=0
DS_TO_ENV='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'

EVAL_SEEDS=(1000 2000 3000 4000)
EVAL_ARGS=(--env.type=libero --env.task=libero_10 --eval.n_episodes=25 --eval.batch_size=5 --policy.n_action_steps=10)
