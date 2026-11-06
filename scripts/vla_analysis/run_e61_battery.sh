#!/bin/bash
# E61 landing battery (probes only — no sim envs; safe concurrent with the
# baseline-seeds evals): msemat + jitter -> autopsy w/ site-bleed -> harvest rescore.
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e61_battery.log
exec >> "$LOG" 2>&1
echo "=== E61 battery started $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

bash scripts/vla_analysis/run_e61_msemat_jitter.sh || echo "[battery] msemat/jitter FAILED"
python scripts/vla_analysis/e61_slots.py || echo "[battery] autopsy FAILED"

SP=$ROOT/outputs/analysis/e56_offtrail
RD=$ROOT/outputs/train/libero_10_seq5_jw_sharepairs_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k
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
TAG=sharepairs
if [ ! -f "$SP/chunks_$TAG.npz" ]; then
  rm -rf "$SP/score_out_$TAG"
  SCORE_HARVESTS=$SP/harv_B,$SP/harv_spec SCORE_OUT_DIR=$SP SCORE_TAG=$TAG \
  SCORE_SEEDS=4 SCORE_TASK=4 SCORE_DEMO_N=120 SCORE_FEAT_LAYER=6 \
  python scripts/vla_analysis/probe_offtrail_score.py \
    --policy.path="$RD/checkpoints/025000/pretrained_model" "${COMMON_ARGS[@]}" \
    --output_dir="$SP/score_out_$TAG" --job_name=offtrail_score_$TAG \
    || echo "[battery] rescore FAILED"
fi
REPORT_DIR=$SP REPORT_TAG_A=$TAG REPORT_TAG_B=spec_e7 \
REPORT_HARVESTS=$SP/harv_B,$SP/harv_spec \
REPORT_MBT=$BRUN_MBT REPORT_TASKKEY=task_4 \
REPORT_OUT=$SP/offtrail_e7_$TAG.jsonl \
python scripts/vla_analysis/probe_offtrail_report.py || echo "[battery] report FAILED"
echo "=== E61 BATTERY COMPLETE $(date -u) ==="
