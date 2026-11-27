# Per-task LoRA specialists for the 20 tasks of cell_s2_ext20v6_rand1 (research_log2 Entry 5, updates of 27 Nov), on the
# full libero_ext_v6: the v6 panel's recipe (rank 128, alpha = r, LR 1e-3, 5k steps from the SmolVLA libero_90 base),
# each evaluated on its own env (25 episodes x seeds 1000, 2000). The per-task ceiling against which the memory run's
# diagonal is read. 4 jobs of 5 specialists (~8 h each, the dataset staged once per job).
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/ext20_spec_r128_v6.sh spec
source "$REPO/cluster/jobs/smolvla/configs/ext_learn_spec_r128.sh"
NAME=ext20_spec_r128_v6
SEQ_DATASET=libero_ext_v6
SEQ_TASKS='[26,59,23,9,2,5,18,43,47,22,57,25,19,10,3,11,29,37,15,35]'
DS_TO_ENV="{$(for i in $(seq 0 59); do printf '"%d":%d,' "$i" "$i"; done | sed 's/,$//')}"
SPEC_GROUP=5
RES_SPEC="tmem=64G,h_rt=24:00:00,tscratch=200G"   # the full libero_ext_v6: 70 GB + its HF datasets cache
