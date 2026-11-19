# Learnability retest (projects/dataset_gen.md §3c), v3: v2's deterministic discrete choices plus recovery data (a
# quarter of first grasp attempts and drawer pulls missed on purpose, labels = the expert's correction) and a narrower
# grasp slide. Same recipe and tasks as
# ext_learn_spec_r128.sh, on libero_ext_learn4_v3 (the four tasks' new demos; cluster/jobs/build_libero_ext.sh with
# V=_v3 SUBSET_ONLY=1).
#   HOLD_JID=<render job> cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/ext_learn_spec_r128_v3.sh spec
source "$REPO/cluster/jobs/smolvla/configs/ext_learn_spec_r128.sh"
NAME=ext_learn_spec_r128_v3
SEQ_DATASET=libero_ext_learn4_v3
