# The 60-task sequential memory run (research_log2 Entry 5) restarted on libero_ext_v6 (projects/dataset_gen.md
# §3d-3f): cell_s2_ext60_rand1's settings unchanged (cell_s2_seq5's A-phase as the start, the same random task order
# ext60_order_rand1.json, 5,000 steps per task, the in-run loss matrix, every boundary checkpoint kept). Only the data
# differ: v6 demos (one hand-orientation mode per task, starts identical to an evaluation episode's) replace v1,
# whose run was stopped after 7 tasks when its learnability check failed.
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/cell_s2_ext60v6_rand1.sh seq
# The seq stage needs ~114 h (training + loss evals), more than one job's h_rt: re-run chain.sh to resume it.
source "$REPO/cluster/jobs/smolvla/configs/cell_s2_ext60_rand1.sh"
NAME=cell_s2_ext60v6_rand1
SEQ_DATASET=libero_ext_v6
