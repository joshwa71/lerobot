#!/bin/bash
# E56 off-trail probe campaign (B vs e7 specialist) — the conversion-gap instrument.
# Chain: harvest B rollouts (traces on) -> harvest specialist rollouts -> score B on all
# states (+ frozen-trunk features from LM layer 9, memory-free below B's first VLM bank
# at 10) -> score specialist on the same states (paired seeds/batching) -> report.
# Scripts: probe_rollout_harvest.py / probe_offtrail_score.py / probe_offtrail_report.py.
# Pre-registered reads + decision rules: research_log.md E56 note.
# SMOKE=1 runs a 2-episode / 2-seed / 24-demo end-to-end pass into a *_smoke dir.
set -eo pipefail
SMOKE=${SMOKE:-0}
ROOT=/home/josh/lerobot
SUF=""; [ "$SMOKE" = "1" ] && SUF="_smoke"
SP=$ROOT/outputs/analysis/e56_offtrail$SUF
LOG=$ROOT/outputs/e56_offtrail$SUF.log
mkdir -p "$SP"
exec >> "$LOG" 2>&1
echo "=== E56 off-trail probe started $(date -u) SMOKE=$SMOKE ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1   # E53: hub 429 masquerades as "vocabulary corrupted"

EPS=50; SEEDS=4; DEMO=120
if [ "$SMOKE" = "1" ]; then EPS=2; SEEDS=2; DEMO=24; fi

BASE=$ROOT/outputs/train
BRUN=$BASE/libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_steps5k
BCKPT=$BRUN/checkpoints/025000/pretrained_model
SPECKPT=$BASE/loraft_baseline/task4_e7/checkpoints/005000/pretrained_model
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
LAST=$(printf "ep%03d.npz" $((EPS-1)))

EVAL_ARGS=(
  --policy.dtype=bfloat16
  --env.type=libero --env.task=libero_10 --env.task_ids="[7]"
  --rename_map="$RENAME"
  --eval.batch_size=1 --eval.n_episodes=$EPS --seed=1000
)
# COMMON_ARGS = the probe convention (run_e55_gpu_gradB.sh), SequentialOnlineConfig
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

if [ ! -f "$SP/harv_B/$LAST" ]; then
  echo "=== stage 1: harvest B ($EPS eps, traces) $(date -u) ==="
  rm -rf "$SP/tmp_eval_B"
  HARVEST_OUT=$SP/harv_B HARVEST_EPISODES=$EPS HARVEST_TRACE=1 \
  python scripts/vla_analysis/probe_rollout_harvest.py \
    --policy.path="$BCKPT" "${EVAL_ARGS[@]}" --output_dir="$SP/tmp_eval_B"
else echo "=== stage 1 skipped (exists) ==="; fi

if [ ! -f "$SP/harv_spec/$LAST" ]; then
  echo "=== stage 2: harvest spec_e7 ($EPS eps) $(date -u) ==="
  rm -rf "$SP/tmp_eval_spec"
  HARVEST_OUT=$SP/harv_spec HARVEST_EPISODES=$EPS HARVEST_TRACE=1 \
  python scripts/vla_analysis/probe_rollout_harvest.py \
    --policy.path="$SPECKPT" "${EVAL_ARGS[@]}" --output_dir="$SP/tmp_eval_spec"
else echo "=== stage 2 skipped (exists) ==="; fi

if [ ! -f "$SP/chunks_B.npz" ]; then
  echo "=== stage 3: score B (+features L9) $(date -u) ==="
  rm -rf "$SP/score_out_B"
  SCORE_HARVESTS=$SP/harv_B,$SP/harv_spec SCORE_OUT_DIR=$SP SCORE_TAG=B \
  SCORE_SEEDS=$SEEDS SCORE_TASK=4 SCORE_DEMO_N=$DEMO SCORE_FEAT_LAYER=9 \
  python scripts/vla_analysis/probe_offtrail_score.py \
    --policy.path="$BCKPT" "${COMMON_ARGS[@]}" \
    --output_dir="$SP/score_out_B" --job_name=offtrail_score_B
else echo "=== stage 3 skipped (exists) ==="; fi

if [ ! -f "$SP/chunks_spec_e7.npz" ]; then
  echo "=== stage 4: score spec_e7 $(date -u) ==="
  rm -rf "$SP/score_out_spec"
  SCORE_HARVESTS=$SP/harv_B,$SP/harv_spec SCORE_OUT_DIR=$SP SCORE_TAG=spec_e7 \
  SCORE_SEEDS=$SEEDS SCORE_TASK=4 SCORE_DEMO_N=$DEMO SCORE_FEAT_LAYER= \
  python scripts/vla_analysis/probe_offtrail_score.py \
    --policy.path="$SPECKPT" "${COMMON_ARGS[@]}" \
    --output_dir="$SP/score_out_spec" --job_name=offtrail_score_spec
else echo "=== stage 4 skipped (exists) ==="; fi

echo "=== stage 5: report $(date -u) ==="
REPORT_DIR=$SP REPORT_TAG_A=B REPORT_TAG_B=spec_e7 \
REPORT_HARVESTS=$SP/harv_B,$SP/harv_spec \
REPORT_MBT=$BRUN/memory_by_task/memory_usage_task_4.json REPORT_TASKKEY=task_4 \
REPORT_OUT=$SP/offtrail_e7.jsonl \
python scripts/vla_analysis/probe_offtrail_report.py
echo "=== E56 off-trail probe COMPLETE $(date -u) ==="
