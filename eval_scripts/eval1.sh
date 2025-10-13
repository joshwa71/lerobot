lerobot-eval \
  --policy.path=/home/josh/phddev/lerobot-upstream/outputs/cluster_train/libero_10_smolvla_200k/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=2 \
  --eval.n_episodes=50 \
  --output_dir=/home/josh/phddev/lerobot-upstream/outputs/cluster_eval/libero_10_smolvla_200k

lerobot-eval \
  --policy.path=/home/josh/phddev/lerobot-upstream/outputs/cluster_train/mixed_libero_10_smolvla_200k/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=2 \
  --eval.n_episodes=50 \
  --output_dir=/home/josh/phddev/lerobot-upstream/outputs/cluster_eval/mixed_libero_10_smolvla_200k

lerobot-eval \
  --policy.path=/home/josh/phddev/lerobot-upstream/outputs/cluster_train/mixed_libero_10_smolvla_200k_curric_5050/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=2 \
  --eval.n_episodes=50 \
  --output_dir=/home/josh/phddev/lerobot-upstream/outputs/cluster_eval/mixed_libero_10_smolvla_200k_curric_5050