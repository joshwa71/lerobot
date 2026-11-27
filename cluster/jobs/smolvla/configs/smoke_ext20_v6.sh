# Smoke test of cell_s2_ext20v6_rand1's two new parts (as smoke_ext60_v6 was of the 60-task run): the seq stage with
# in-training current-task rollouts, and the triseed stage (one seed) polling the smoke run's boundary checkpoints.
# Two tasks of libero_ext_panel_v6 (15 GB, quick to stage) for 30 steps each, 2-episode evaluations:
# panel task_index 11 = suite task 59 (drawer opening), 9 = suite task 56 (floor scene).
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/smoke_ext20_v6.sh seq
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/smoke_ext20_v6.sh triseed
source "$REPO/cluster/jobs/smolvla/configs/cell_s2_ext20v6_rand1.sh"
NAME=smoke_ext20_v6
SEQ_DATASET=libero_ext_panel_v6
SEQ_TASKS='[11,9]'
DS_TO_ENV='{"0":6,"1":11,"2":12,"3":19,"4":27,"5":32,"6":38,"7":42,"8":54,"9":56,"10":58,"11":59}'
SEQ_STEPS_PER_TASK=30
SEQ_ARGS+=(--log_freq=10 --current_task_rollout_episodes=2 --env.task_ids='[59,56]')
RES_SEQ="tmem=48G,h_rt=02:00:00,tscratch=60G"
EVAL_SEEDS=(1000)
EVAL_ARGS+=(--eval.n_episodes=2)
RES_TRISEED="tmem=48G,h_rt=03:00:00,tscratch=40G"
TRI_WAIT_H=2
RUN_MARGIN_MIN=10
