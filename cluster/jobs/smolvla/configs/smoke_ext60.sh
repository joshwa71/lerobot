# Smoke test of cell_s2_ext60_rand1's seq stage (same SEQ_INIT and SEQ_ARGS): three tasks of libero_ext_learn4
# (5 GB, task_index 0-3 = libero_ext envs 0, 31, 42, 59), 30 steps each. Checks the trainer arguments (the in-run
# loss matrix), loading the A-phase with libero_ext's normalisation stats, and the boundary outputs
# (checkpoints, sequential_state.pt, memory_by_task, eval/loss_results.jsonl).
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/smoke_ext60.sh seq
source "$REPO/cluster/jobs/smolvla/configs/cell_s2_ext60_rand1.sh"
NAME=smoke_ext60
SEQ_DATASET=libero_ext_learn4
SEQ_TASKS='[2,0,1]'
DS_TO_ENV='{"0":0,"1":31,"2":42,"3":59}'
SEQ_STEPS_PER_TASK=30
RES_SEQ="tmem=48G,h_rt=02:00:00,tscratch=60G"
RUN_MARGIN_MIN=10
SEQ_ARGS+=(--log_freq=10)
