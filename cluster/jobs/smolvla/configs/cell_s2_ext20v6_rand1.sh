# The first 20 tasks of the 60-task memory run (cell_s2_ext60v6_rand1, research_log2 Entry 5) before committing to
# all 60: a K = 20 point (double the ten-task runs) at a third of the cost (~35 h), so the setup can be iterated on.
# Identical settings: cell_s2_seq5's A-phase as the start, the full libero_ext_v6 (so normalisation stats match the
# 60-task run), 5,000 steps per task, the in-run loss matrix. The tasks are the first 20 of ext60_order_rand1.json,
# so this run is exactly the first third of the 60-task run: 9 one-step and 11 two-step tasks from 17 of the 27
# scenes, including the drawer opening (2nd) and the stove (13th).
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/cell_s2_ext20v6_rand1.sh seq
source "$REPO/cluster/jobs/smolvla/configs/cell_s2_ext60v6_rand1.sh"
NAME=cell_s2_ext20v6_rand1
SEQ_TASKS='[26,59,23,9,2,5,18,43,47,22,57,25,19,10,3,11,29,37,15,35]'
