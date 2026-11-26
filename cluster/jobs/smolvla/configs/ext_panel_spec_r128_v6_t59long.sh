# Task 59 (open the top drawer and put the popcorn in it) with 3x the training steps, beside the v6 panel
# (configs/ext_panel_spec_r128_v6.sh; projects/dataset_gen.md §3e). On v4 its specialist's loss was still falling
# steeply at 5k steps (0.13, after a spike to ~0.29 at 3k; 0.01-0.09 for the other specialists): is it a hard but
# learnable task that needs more steps? Same data, recipe and evaluation; 15k steps, the LR decay stretched over them.
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/ext_panel_spec_r128_v6_t59long.sh spec
source "$REPO/cluster/jobs/smolvla/configs/ext_panel_spec_r128_v6.sh"
NAME=ext_panel_spec_r128_v6_t59long
SEQ_TASKS='[11]'   # libero_ext_panel_v6 task_index 11 = suite task 59
SEQ_STEPS_PER_TASK=15000
SPEC_GROUP=1
