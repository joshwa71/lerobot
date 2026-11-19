# Learnability retest (projects/dataset_gen.md §3c) after making the expert's discrete choices deterministic (step
# order = the instruction's, the grasp needing least wrist rotation, one faucet direction). Same recipe and tasks as
# ext_learn_spec_r128.sh, on libero_ext_learn4_v2 (the four tasks' new demos; cluster/jobs/build_libero_ext.sh with
# V=_v2 SUBSET_ONLY=1).
#   HOLD_JID=<render job> cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/ext_learn_spec_r128_v2.sh spec
source "$REPO/cluster/jobs/smolvla/configs/ext_learn_spec_r128.sh"
NAME=ext_learn_spec_r128_v2
SEQ_DATASET=libero_ext_learn4_v2
