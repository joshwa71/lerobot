#!/bin/bash
# E57 value-input-noise dose queue: arm 1x -> harvest-bank rescore -> arm 0.5x -> rescore.
# Rescore = probe_offtrail_score.py on the arm's FINAL checkpoint against the SAME e56
# harvest states (paired seeds/batching -> chunks comparable to chunks_B/chunks_spec_e7),
# then the report with the arm as TAG_A vs the specialist. READ 1 (D vs distance) is the
# pre-screen; the trace-based rows (selfmass/churn) reuse B's ROLLOUT traces + MBT and
# are context, not arm measurements. Non-fatal stages: one arm's failure doesn't block
# the other.
set -o pipefail
ROOT=/home/josh/lerobot
SP=$ROOT/outputs/analysis/e56_offtrail
STAGED=$ROOT/job_scripts/nebius/libero_90/staged
LOG=$ROOT/outputs/e57_vnoise_queue.log
exec >> "$LOG" 2>&1
echo "=== E57 vnoise queue started $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

BRUN_MBT=$ROOT/outputs/train/libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_steps5k/memory_by_task/memory_usage_task_4.json
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

rescore () {  # $1 = run dir name, $2 = tag
  local RD=$ROOT/outputs/train/$1 TAG=$2
  if [ ! -d "$RD/checkpoints/025000/pretrained_model" ]; then
    echo "[rescore:$TAG] final checkpoint missing - skipping"; return 1
  fi
  if [ ! -f "$SP/chunks_$TAG.npz" ]; then
    rm -rf "$SP/score_out_$TAG"
    SCORE_HARVESTS=$SP/harv_B,$SP/harv_spec SCORE_OUT_DIR=$SP SCORE_TAG=$TAG \
    SCORE_SEEDS=4 SCORE_TASK=4 SCORE_DEMO_N=120 SCORE_FEAT_LAYER=9 \
    python scripts/vla_analysis/probe_offtrail_score.py \
      --policy.path="$RD/checkpoints/025000/pretrained_model" "${COMMON_ARGS[@]}" \
      --output_dir="$SP/score_out_$TAG" --job_name=offtrail_score_$TAG \
      || { echo "[rescore:$TAG] scoring FAILED"; return 1; }
  fi
  REPORT_DIR=$SP REPORT_TAG_A=$TAG REPORT_TAG_B=spec_e7 \
  REPORT_HARVESTS=$SP/harv_B,$SP/harv_spec \
  REPORT_MBT=$BRUN_MBT REPORT_TASKKEY=task_4 \
  REPORT_OUT=$SP/offtrail_e7_$TAG.jsonl \
  python scripts/vla_analysis/probe_offtrail_report.py \
    || echo "[rescore:$TAG] report FAILED"
}

echo "=== arm 1: dose1x $(date -u) ==="
bash $STAGED/seq5_gradB_vnoise_dose1x.sh || echo "[queue] dose1x arm FAILED (continuing)"
echo "=== rescore dose1x $(date -u) ==="
rescore libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_vnoise1x_steps5k vnoise1x

echo "=== arm 2: dose05x $(date -u) ==="
bash $STAGED/seq5_gradB_vnoise_dose05x.sh || echo "[queue] dose05x arm FAILED (continuing)"
echo "=== rescore dose05x $(date -u) ==="
rescore libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_vnoise05x_steps5k vnoise05x

echo "=== E57 vnoise queue COMPLETE $(date -u) ==="
