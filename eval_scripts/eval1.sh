lerobot-eval \
  --policy.path=/home/josh/phddev/lerobot/outputs/cluster_train/libero_10_smolvla_100k/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=2 \
  --eval.n_episodes=50 \
  --output_dir=/home/josh/phddev/lerobot/outputs/cluster_eval/libero_10_smolvla_100k

lerobot-eval \
  --policy.path=/home/josh/phddev/lerobot/outputs/cluster_train/mixed_libero_10_smolvla_100k/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=2 \
  --eval.n_episodes=50 \
  --output_dir=/home/josh/phddev/lerobot/outputs/cluster_eval/mixed_libero_10_smolvla_100k

lerobot-eval \
  --policy.path=/home/josh/phddev/lerobot/outputs/cluster_train/mixed_libero_10_smolvla_100k_curric_5050/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=2 \
  --eval.n_episodes=50 \
  --output_dir=/home/josh/phddev/lerobot/outputs/cluster_eval/mixed_libero_10_smolvla_100k_curric_5050

lerobot-eval \
  --policy.path=/home/josh/phddev/lerobot/outputs/cluster_train/mixed_libero_10_smolvla_100k_curric_6040/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=2 \
  --eval.n_episodes=50 \
  --output_dir=/home/josh/phddev/lerobot/outputs/cluster_eval/mixed_libero_10_smolvla_100k_curric_6040

lerobot-eval \
  --policy.path=/home/josh/phddev/lerobot/outputs/cluster_train/mixed_libero_10_smolvla_100k_curric_7030/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=2 \
  --eval.n_episodes=50 \
  --output_dir=/home/josh/phddev/lerobot/outputs/cluster_eval/mixed_libero_10_smolvla_100k_curric_7030

lerobot-eval \
  --policy.path=/home/josh/phddev/lerobot/outputs/cluster_train/mixed_libero_10_smolvla_100k_curric_8020/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=2 \
  --eval.n_episodes=50 \
  --output_dir=/home/josh/phddev/lerobot/outputs/cluster_eval/mixed_libero_10_smolvla_100k_curric_8020

lerobot-eval \
  --policy.path=/home/josh/phddev/lerobot/outputs/cluster_train/mixed_libero_10_smolvla_100k_curric_9010/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=2 \
  --eval.n_episodes=50 \
  --output_dir=/home/josh/phddev/lerobot/outputs/cluster_eval/mixed_libero_10_smolvla_100k_curric_9010