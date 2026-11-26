# Task 59 (open the top drawer and put the popcorn in it) at 15k steps with LR 3e-4, beside the v6 panel
# (projects/dataset_gen.md §3e). The LR 1e-3 arm (configs/ext_panel_spec_r128_v6_t59long.sh, job 7852337) diverged at
# ~1.2k steps: loss 0.22 -> 1.33, never back below 1.0, gradient norm up to 235 by 5k. On the 15k schedule the LR
# was still 9e-4 there; the v4 5k-step specialist already spiked (0.18 -> 0.29 at 3k, LR ~5e-4) before recovering.
# Same data, evaluation and LoRA as the panel; LR 3e-4 -> 3e-5 linear over the 15k steps.
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/ext_panel_spec_r128_v6_t59long_lr3e4.sh spec
RANK=128 LR=3e-4
source "$REPO/cluster/jobs/smolvla/configs/lora_common.sh"
NAME=ext_panel_spec_r128_v6_t59long_lr3e4
SEQ_DATASET=libero_ext_panel_v6
SEQ_TASKS='[11]'   # libero_ext_panel_v6 task_index 11 = suite task 59
DS_TO_ENV='{"0":6,"1":11,"2":12,"3":19,"4":27,"5":32,"6":38,"7":42,"8":54,"9":56,"10":58,"11":59}'
SEQ_STEPS_PER_TASK=15000
SPEC_GROUP=1
RES_SPEC="tmem=48G,h_rt=24:00:00,tscratch=60G"
EVAL_SEEDS=(1000 2000)
EVAL_ARGS=(--env.type=libero --env.task=libero_ext --eval.n_episodes=25 --eval.batch_size=5 --policy.n_action_steps=10)
