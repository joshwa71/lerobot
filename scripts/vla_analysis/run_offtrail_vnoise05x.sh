#!/bin/bash
# E58 full off-trail analysis for the dose05x arm (the E56 instrument on ITS OWN rollouts).
# Stages: harvest dose05x (50 eps, traces) -> score dose05x AND the e7 specialist on the
# SUPERSET of harvests (own + B's + spec's states; paired seeds) -> report with dose05x's
# own written sets. Gated on the e57-vnoise unit finishing.
# NB composition-read semantics: selfmass/churn rows join each harvest's TRACES against
# REPORT_MBT (dose05x's written sets) — coherent ONLY for harv_vnoise05x rows (its own
# retrievals); harv_B rows carry B's traces and are read for D/populations only.
set -eo pipefail
ROOT=/home/josh/lerobot
SP=$ROOT/outputs/analysis/e56_offtrail
LOG=$ROOT/outputs/e58_offtrail05x.log
exec >> "$LOG" 2>&1
echo "=== E58 dose05x full off-trail started (waiting for e57-vnoise) $(date -u) ==="
while systemctl is-active --quiet e57-vnoise; do sleep 120; done
echo "=== e57-vnoise finished -> starting $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

RUN05=libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_vnoise05x_steps5k
CKPT05=$ROOT/outputs/train/$RUN05/checkpoints/025000/pretrained_model
MBT05=$ROOT/outputs/train/$RUN05/memory_by_task/memory_usage_task_4.json
SPECKPT=$ROOT/outputs/train/loraft_baseline/task4_e7/checkpoints/005000/pretrained_model
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
EVAL_ARGS=(
  --policy.dtype=bfloat16
  --env.type=libero --env.task=libero_10 --env.task_ids="[7]"
  --rename_map="$RENAME"
  --eval.batch_size=1 --eval.n_episodes=50 --seed=1000
)
COMMON_ARGS=(
  --policy.empty_cameras=1 --policy.dtype=bfloat16
  --policy.gradient_checkpointing=false
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
  --dataset.repo_id=libero_10 --dataset.root="$ROOT/outputs/libero_10"
  --rename_map="$RENAME"
  --env.type=libero --env.task=libero_10
  --steps=200000 --batch_size=32 --num_workers=2
  --online_task_ids='[0,1,2,3,4]' --online_steps_per_task=5000
  --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'
  --wandb.enable=false
)
HARVS=$SP/harv_vnoise05x,$SP/harv_B,$SP/harv_spec

if [ ! -f "$SP/harv_vnoise05x/ep049.npz" ]; then
  echo "=== stage 1: harvest dose05x (50 eps, traces) $(date -u) ==="
  rm -rf "$SP/tmp_eval_vn05x"
  HARVEST_OUT=$SP/harv_vnoise05x HARVEST_EPISODES=50 HARVEST_TRACE=1 \
  python scripts/vla_analysis/probe_rollout_harvest.py \
    --policy.path="$CKPT05" "${EVAL_ARGS[@]}" --output_dir="$SP/tmp_eval_vn05x"
else echo "=== stage 1 skipped ==="; fi

if [ ! -f "$SP/chunks_vn05x_full.npz" ]; then
  echo "=== stage 2: score dose05x on the superset $(date -u) ==="
  rm -rf "$SP/score_out_vn05x_full"
  SCORE_HARVESTS=$HARVS SCORE_OUT_DIR=$SP SCORE_TAG=vn05x_full \
  SCORE_SEEDS=4 SCORE_TASK=4 SCORE_DEMO_N=120 SCORE_FEAT_LAYER=9 \
  python scripts/vla_analysis/probe_offtrail_score.py \
    --policy.path="$CKPT05" "${COMMON_ARGS[@]}" \
    --output_dir="$SP/score_out_vn05x_full" --job_name=offtrail_score_vn05x_full
else echo "=== stage 2 skipped ==="; fi

if [ ! -f "$SP/chunks_spec_full.npz" ]; then
  echo "=== stage 3: score specialist on the superset $(date -u) ==="
  rm -rf "$SP/score_out_spec_full"
  SCORE_HARVESTS=$HARVS SCORE_OUT_DIR=$SP SCORE_TAG=spec_full \
  SCORE_SEEDS=4 SCORE_TASK=4 SCORE_DEMO_N=120 SCORE_FEAT_LAYER= \
  python scripts/vla_analysis/probe_offtrail_score.py \
    --policy.path="$SPECKPT" "${COMMON_ARGS[@]}" \
    --output_dir="$SP/score_out_spec_full" --job_name=offtrail_score_spec_full
else echo "=== stage 3 skipped ==="; fi

echo "=== stage 4: report $(date -u) ==="
REPORT_DIR=$SP REPORT_TAG_A=vn05x_full REPORT_TAG_B=spec_full \
REPORT_HARVESTS=$HARVS \
REPORT_MBT=$MBT05 REPORT_TASKKEY=task_4 \
REPORT_OUT=$SP/offtrail_e7_vn05x_full.jsonl \
python scripts/vla_analysis/probe_offtrail_report.py
echo "=== E58 dose05x full off-trail COMPLETE $(date -u) ==="
