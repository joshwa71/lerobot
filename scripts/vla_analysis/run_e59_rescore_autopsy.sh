#!/bin/bash
# E59 second probe wave (Josh away, 5 Aug 26): (1) harvest-bank RESCORE of the
# interleaved final checkpoint on the E56/E57 e7 off-trail state bank vs the
# specialist — measures whether deep expert placement widened the value
# competence radius (the mechanism behind e7 20->38 and where the remaining -22
# lives); exact vnoise-rescore pattern (paired seeds/batching -> chunks
# comparable to chunks_B/chunks_spec_e7). SCORE_FEAT_LAYER=6 (below this
# model's first VLM bank at 7 -> genuinely memory-free stage-1 features; B used
# 9 for the same reason — the graded READ 1b axis is proprio distance, which is
# model-independent, so cross-model comparability is unaffected). REPORT_MBT =
# B's task-4 written sets, per the vnoise precedent: READ 2/3 join B's rollout
# traces and are context rows, not arm measurements.
# (2) slot autopsy interleave-vs-B (e59_slots.py) — completes the pre-registered
# prior-core-events=0 check and shows which banks carry each task's read mass
# (the depth story: does e7 live in E10/E12?).
set -o pipefail
ROOT=/home/josh/lerobot
SP=$ROOT/outputs/analysis/e56_offtrail
LOG=$ROOT/outputs/e59_rescore_autopsy.log
exec >> "$LOG" 2>&1
echo "=== E59 rescore+autopsy started $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

RD=$ROOT/outputs/train/libero_10_seq5_jw_interleave_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k
BRUN_MBT=$ROOT/outputs/train/libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_steps5k/memory_by_task/memory_usage_task_4.json
TAG=interleave
COMMON_ARGS=(
  --policy.empty_cameras=1 --policy.dtype=bfloat16
  --policy.gradient_checkpointing=false
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
  --dataset.repo_id=libero_10 --dataset.root="$ROOT/outputs/libero_10"
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
  --env.type=libero --env.task=libero_10
  --steps=200000 --batch_size=32 --num_workers=2
  --online_task_ids='[0,1,2,3,4]' --online_steps_per_task=5000
  --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'
  --wandb.enable=false
)

echo "=== stage 1: autopsy (CPU) $(date -u) ==="
python scripts/vla_analysis/e59_slots.py || echo "[autopsy] FAILED (continuing to rescore)"

echo "=== stage 2: rescore $TAG on the e7 harvest bank $(date -u) ==="
if [ ! -f "$SP/chunks_$TAG.npz" ]; then
  rm -rf "$SP/score_out_$TAG"
  SCORE_HARVESTS=$SP/harv_B,$SP/harv_spec SCORE_OUT_DIR=$SP SCORE_TAG=$TAG \
  SCORE_SEEDS=4 SCORE_TASK=4 SCORE_DEMO_N=120 SCORE_FEAT_LAYER=6 \
  python scripts/vla_analysis/probe_offtrail_score.py \
    --policy.path="$RD/checkpoints/025000/pretrained_model" "${COMMON_ARGS[@]}" \
    --output_dir="$SP/score_out_$TAG" --job_name=offtrail_score_$TAG \
    || { echo "[rescore:$TAG] scoring FAILED"; exit 1; }
fi
REPORT_DIR=$SP REPORT_TAG_A=$TAG REPORT_TAG_B=spec_e7 \
REPORT_HARVESTS=$SP/harv_B,$SP/harv_spec \
REPORT_MBT=$BRUN_MBT REPORT_TASKKEY=task_4 \
REPORT_OUT=$SP/offtrail_e7_$TAG.jsonl \
python scripts/vla_analysis/probe_offtrail_report.py \
  || echo "[rescore:$TAG] report FAILED"
echo "=== E59 rescore+autopsy COMPLETE $(date -u) ==="
