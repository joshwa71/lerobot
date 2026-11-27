# Smoke test of cell_s2_ext60v6_rand1 (as smoke_ext60 was of cell_s2_ext60_rand1): the run's exact settings on three
# tasks of libero_ext_panel_v6 (15 GB, quick to stage) for 30 steps each, with the in-run loss eval at each boundary.
# Panel task_index 3, 11, 9 = suite tasks 19 (stove + pan), 59 (drawer opening) and 56 (floor scene): the longest
# episodes and the two unusual arenas.
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/smoke_ext60_v6.sh seq
source "$REPO/cluster/jobs/smolvla/configs/cell_s2_ext60v6_rand1.sh"
NAME=smoke_ext60_v6
SEQ_DATASET=libero_ext_panel_v6
SEQ_TASKS='[3,11,9]'
DS_TO_ENV='{"0":6,"1":11,"2":12,"3":19,"4":27,"5":32,"6":38,"7":42,"8":54,"9":56,"10":58,"11":59}'
SEQ_STEPS_PER_TASK=30
RES_SEQ="tmem=48G,h_rt=02:00:00,tscratch=60G"
RUN_MARGIN_MIN=10
SEQ_ARGS+=(--log_freq=10)
