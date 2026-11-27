# Naive sequential LoRA through the 20 tasks of cell_s2_ext20v6_rand1, from the SmolVLA libero_90 base, on the full
# libero_ext_v6: the forgetting baseline. Rank 128, alpha = r, LR 3e-4 (libero_10's naive arm: at 1e-3 each later task
# trained worse; research_log2 Entry 4 addendum), 5k steps per task, one adapter throughout.
# Same in-run instruments as the memory run: the loss matrix (8 batches of 32 = the memory run's 16 of 16 samples) and
# 20 rollouts of the just-trained task after each. Then the final row at 4 seeds (stage eval, one job per seed). No full
# triangle: on libero_10 every earlier task was at 0 from the second block, and all-failure episodes, which run to the
# step limit, are the slowest to evaluate.
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/ext20_naive_r128_lr3e4_v6.sh seq eval
RANK=128 LR=3e-4
source "$REPO/cluster/jobs/smolvla/configs/lora_common.sh"
NAME=ext20_naive_r128_lr3e4_v6
SEQ_INIT=$STAGE1_CKPT
SEQ_DATASET=libero_ext_v6
SEQ_TASKS='[26,59,23,9,2,5,18,43,47,22,57,25,19,10,3,11,29,37,15,35]'
DS_TO_ENV="{$(for i in $(seq 0 59); do printf '"%d":%d,' "$i" "$i"; done | sed 's/,$//')}"
SEQ_ARGS+=(
  --eval.type=loss --eval_loss_n_batches=8 --log_freq=100
  --current_task_rollout_episodes=20
  --env.type=libero --env.task=libero_ext --env.task_ids="$SEQ_TASKS" --eval.batch_size=5
)
RES_SEQ="tmem=64G,h_rt=72:00:00,tscratch=200G"   # the full libero_ext_v6: 70 GB + its HF datasets cache
EVAL_ARGS=(--env.type=libero --env.task=libero_ext --eval.n_episodes=25 --eval.batch_size=5 --policy.n_action_steps=10)
