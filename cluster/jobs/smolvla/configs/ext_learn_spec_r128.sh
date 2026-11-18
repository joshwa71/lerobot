# Learnability check of the libero_ext dataset (projects/dataset_gen.md §3c): LoRA specialists with the libero_10
# recipe (rank 128, LR 1e-3, 5k steps from the SmolVLA libero_90 base; research_log2 Entry 4) on four libero_ext
# tasks, each evaluated on its own env. Trains on libero_ext_learn4 (those four tasks only, made by
# cluster/jobs/build_libero_ext.sh): its task_index 0-3 are libero_ext suite tasks 0, 12, 34, 45
#   0  put the cookies in the bowl (kitchen, one step)
#   12 open the top drawer and put the popcorn in it (kitchen, drawer opening)
#   34 put the ketchup on the shelf and put the cream cheese in the right of the caddy (study, two steps)
#   45 put the chocolate pudding in the bowl and put the milk in the tray (coffee table, two steps)
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/ext_learn_spec_r128.sh spec
RANK=128 LR=1e-3
source "$REPO/cluster/jobs/smolvla/configs/lora_common.sh"
NAME=ext_learn_spec_r128
SEQ_DATASET=libero_ext_learn4
SEQ_TASKS='[0,1,2,3]'
DS_TO_ENV='{"0":0,"1":12,"2":34,"3":45}'
SPEC_GROUP=2   # two jobs of two specialists each (~1.2 h training + eval per specialist)
RES_SPEC="tmem=48G,h_rt=24:00:00,tscratch=60G"
EVAL_SEEDS=(1000 2000)
EVAL_ARGS=(--env.type=libero --env.task=libero_ext --eval.n_episodes=25 --eval.batch_size=5 --policy.n_action_steps=10)
