# The first 20 tasks of the 60-task memory run (cell_s2_ext60v6_rand1, research_log2 Entry 5) before committing to
# all 60: a K = 20 point (double the ten-task runs) at a third of the cost (~35 h), so the setup can be iterated on.
# Identical settings: cell_s2_seq5's A-phase as the start, the full libero_ext_v6 (so normalisation stats match the
# 60-task run), 5,000 steps per task, the in-run loss matrix. The tasks are the first 20 of ext60_order_rand1.json,
# so this run is exactly the first third of the 60-task run: 9 one-step and 11 two-step tasks from 17 of the 27
# scenes, including the drawer opening (2nd) and the stove (13th).
# Evaluation:
#   - in training, after each task: 20 rollouts of the just-trained task (seed = the run's seed), besides the loss
#     matrix: a quick check that each task is being acquired;
#   - the full retention triangle (every row, every seen task, 25 episodes x 4 seeds) from the boundary checkpoints,
#     as 4 jobs (one per seed, stage triseed) that run alongside training.
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/cell_s2_ext20v6_rand1.sh seq
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/cell_s2_ext20v6_rand1.sh triseed
source "$REPO/cluster/jobs/smolvla/configs/cell_s2_ext60v6_rand1.sh"
NAME=cell_s2_ext20v6_rand1
SEQ_TASKS='[26,59,23,9,2,5,18,43,47,22,57,25,19,10,3,11,29,37,15,35]'
SEQ_ARGS+=(
  --current_task_rollout_episodes=20
  --env.type=libero --env.task=libero_ext --env.task_ids="$SEQ_TASKS" --eval.batch_size=5
)
# one seed's whole triangle: 210 task evaluations of 25 episodes (~23 h), plus waiting on training
RES_TRISEED="tmem=48G,h_rt=72:00:00,tscratch=40G"
TRI_WAIT_H=8
