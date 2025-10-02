lerobot-train   --policy.type=smolvla   --policy.repo_id=outputs/train/mixed_libero_10_600k^C --dataset.repo_id=outputs/mixed_libero_10   --env.type=libero   --env.task=libero_10   --output_dir=./outputs/train/mixed_libero_10_6
00k   --steps=600000   --batch_size=8   --eval.batch_size=1   --eval.n_episodes=3   --eval_freq=20000   --policy.freeze_vision_encoder=false   --poli
cy.train_expert_only=false   --policy.train_state_proj=true   --policy.scheduler_warmup_steps=20000   --policy.scheduler_decay_steps=500000   --job_n
ame=mixed_libero_10_600k