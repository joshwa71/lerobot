#!/bin/bash
# E43 arm: GENERALIST-SLOT FREEZE (top-5k A-phase slots/layer) on the lr4xsched+topt3072 config.
#
# Single conceptual delta vs the E42 lr4xsched+topt3072 arm (VM2, 50-ep final 42.4): the
# prior-usefulness store is SEEDED with u=1.0 at the top-5000 A-phase read-mass slots per
# layer (18-19% of A-phase read mass; scripts/vla_analysis/data/a_phase_top5k_slots.json,
# baked from audit_libero90_usage_rwarmupB_A) and protect_hard_u=0.9 structurally vetoes
# them from top-t candidacy — never in mask => no gradient, no momentum => frozen for the
# whole run. The budget redistributes to the next-ranked slots automatically (S13/S14).
# Rank-mode beta4 protection for sequential-task cores runs unchanged on top (= VM2).
#
# Hypothesis (Josh, E42 addendum + E43 discussion): the hottest A-phase slots implement the
# generalist transforms every task's retrieval mixes in; sequential writes erode them a
# little from every direction (bleed-core measurements); on-demo MSE never registers it
# (each task's own adaptation compensates on-trail) but off-trail the eroded generalist
# transforms are the fallback => rollout cost with no MSE cost. Freezing removes the erosion.
# NB fit != perf is the POINT here: loss may be unchanged or slightly worse while rollouts
# improve. Read rollouts + off-trail/jitter probes, not MSE.
#
# Pre-registered reads (vs VM2, its exact twin):
#   - block-min MSEs within ~3% of VM2 (the veto cost is sub-linear: corrections route
#     through the other ~92-97% of each 144-slot mixture)
#   - 50-ep finals: family average up => the off-trail fallback story has legs; e6/e4 (the
#     marginal absorbers) are the expected movers
#   - jitter/off-trail probe: degradation slope shallower than VM2 at matched on-demo error
#   - FAIL: block-min MSE up >10% on any writer => freeze is starving fit (E19 shape)
set -eo pipefail
echo "E43 lr4xsched+topt3072+freeze5k started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
A_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_8_10_12_14_frozenroute_rwarmupB_values10k_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model"
SEED_JSON="$ROOT_DIR/scripts/vla_analysis/data/a_phase_top5k_slots.json"
SEQ_RUN=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_3072_protect_beta4_freeze5k_lr4xsched_steps5k_tasks5
SEQ_OUT="$ROOT_DIR/outputs/train/$SEQ_RUN"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$A_CKPT" ] || { echo "ERROR: stageB A checkpoint missing"; exit 1; }
[ -f "$SEED_JSON" ] || { echo "ERROR: seed JSON missing (git pull? scripts/vla_analysis/data/)"; exit 1; }

# 5 tasks x 5000 steps -> final checkpoint 025000
if [ -d "$SEQ_OUT/checkpoints/025000" ]; then
  echo "[seq] final checkpoint exists - skipping."
else
  lerobot-sequential-train \
    --policy.path="$A_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_10 \
    --dataset.root="$SEQ_DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_10 \
    --output_dir="$SEQ_OUT" \
    --steps=200000 \
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=20 \
    --eval_final_episodes=50 \
    --log_freq=200 \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --job_name="$SEQ_RUN" \
    --online_task_ids='[0,1,2,3,4]' \
    --online_steps_per_task=5000 \
    --policy.memory_layer.aggregate_usage=false \
    --policy.memory_layer.use_frozen_base_input_features=true \
    --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
    --save_after_each_task=true \
    --reinit_optimizer_each_task=true \
    --tfidf_enable=true \
    --tfidf_top_t=3072 \
    --use_online_idf_stats=true \
    --idf_exponent=1 \
    --protect_prior_slots=true \
    --protect_beta=4 \
    --protect_hard_u=0.9 \
    --protect_seed_path="$SEED_JSON" \
    --memory_value_lr=0.004 \
    --memory_value_lr_end=0.0002 \
    --memory_value_scheduler_type=linear
fi
echo "E43 lr4xsched+topt3072+freeze5k completed at $(date)"
