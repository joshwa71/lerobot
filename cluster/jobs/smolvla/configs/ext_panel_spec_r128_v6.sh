# Learnability panel on v6 data (projects/dataset_gen.md §3d-3e): the v4 panel (configs/ext_panel_spec_r128_v4.sh)
# repeated on demos from expert v5 (one hand-orientation mode per task: a fixed grasp orientation per task,
# deterministic turns, only the most common mode's episodes kept) that start exactly as an evaluation episode does
# (LiberoEnv's reset with its 10 no-op steps). Same twelve tasks, recipe and evaluation.
# Data: libero_ext_panel_v6, the subset written by the full v6 render (cluster/jobs/build_libero_ext.sh with V=_v6
# LEARN_NAME=libero_ext_panel LEARN_TASKS=6:11:12:19:27:32:38:42:54:56:58:59); its task_index 0-11 are those suite
# tasks in that order.
#   HOLD_JID=<render job> cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/ext_panel_spec_r128_v6.sh spec
source "$REPO/cluster/jobs/smolvla/configs/ext_learn_spec_r128.sh"
NAME=ext_panel_spec_r128_v6
SEQ_DATASET=libero_ext_panel_v6
SEQ_TASKS='[0,1,2,3,4,5,6,7,8,9,10,11]'
DS_TO_ENV='{"0":6,"1":11,"2":12,"3":19,"4":27,"5":32,"6":38,"7":42,"8":54,"9":56,"10":58,"11":59}'
SPEC_GROUP=2
