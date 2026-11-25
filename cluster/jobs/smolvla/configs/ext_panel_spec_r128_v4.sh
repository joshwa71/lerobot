# Learnability panel on v4 data (projects/dataset_gen.md §3d): LoRA r128 specialists with the libero_10 recipe on
# twelve libero_ext tasks covering every task type (faucet, drawer placement, microwave put-and-close, stove,
# lift with a rim grasp, lower shelf, bookcase, crowded coffee table, beside + caddy, floor scene, fridge, drawer
# opening), each evaluated on its own env. v4 = deterministic discrete choices, drawer-handle recovery only,
# narrower grasp slide, a safe approach path, 1 cm grasp clearance, LIBERO-like start states (no widening).
# Data: libero_ext_panel_v4 (cluster/jobs/build_libero_ext.sh with V=_v4 SUBSET_ONLY=1 LEARN_NAME=libero_ext_panel
# LEARN_TASKS=6,11,12,19,27,32,38,42,54,56,58,59); its task_index 0-11 are those suite tasks in that order.
#   HOLD_JID=<render job> cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/ext_panel_spec_r128_v4.sh spec
source "$REPO/cluster/jobs/smolvla/configs/ext_learn_spec_r128.sh"
NAME=ext_panel_spec_r128_v4
SEQ_DATASET=libero_ext_panel_v4
SEQ_TASKS='[0,1,2,3,4,5,6,7,8,9,10,11]'
DS_TO_ENV='{"0":6,"1":11,"2":12,"3":19,"4":27,"5":32,"6":38,"7":42,"8":54,"9":56,"10":58,"11":59}'
SPEC_GROUP=2
