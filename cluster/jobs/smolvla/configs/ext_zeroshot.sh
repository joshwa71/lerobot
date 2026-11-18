# Zero-shot floor on libero_ext (projects/dataset_gen.md §3c): the SmolVLA libero_90 base, no adaptation, on all
# 60 libero_ext envs, 5 episodes each, one seed (a rough floor: most episodes run to the 520-step limit).
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/ext_zeroshot.sh eval
NAME=smolvla_l90_zeroshot_ext
EVAL_MODEL=$CLUSTER_ROOT/runs/smolvla_l90_base/stage1/checkpoints/050000/pretrained_model
GPUS="gpu=true,gpu_type=(a100|a100_80|a40|a6000|h100|l40s|rtx6000ada)"
RES_EVAL="tmem=48G,h_rt=24:00:00,tscratch=40G"
RUN_MARGIN_MIN=30
NUM_WORKERS=4
SEQ_DATASET=libero_ext
SEQ_TASKS="[$(seq -s, 0 59)]"   # libero_ext task_index = env id
SEQ_STEPS_PER_TASK=0
DS_TO_ENV="{$(for i in $(seq 0 59); do printf '"%d":%d,' "$i" "$i"; done | sed 's/,$//')}"
EVAL_SEEDS=(1000)
EVAL_ARGS=(--env.type=libero --env.task=libero_ext --eval.n_episodes=5 --eval.batch_size=5 --policy.n_action_steps=10)
