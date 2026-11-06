#!/bin/bash
# E62 noise-arm calibration (E61-add-5 sequencing step 3 prerequisite):
# probe_value_input_calib on the merged-6x2 FINAL checkpoint — per-layer
# displacement ratios for its 12 sites (expert [4,6,8,10,14,16] + VLM
# [5,7,9,11,13,15]). The E58 sigmas were measured on B's layers and do not
# transfer. Uses the standing e56 harvest bank (harv_B). ~2h.
set -eo pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e62_vnoise_calib.log
exec >> "$LOG" 2>&1
echo "=== E62 vnoise calibration started $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

RD=$ROOT/outputs/train/libero_10_seq5_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
SP=$ROOT/outputs/analysis/e56_offtrail
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
CALIB_HARVEST=$SP/harv_B CALIB_OUT=$ROOT/outputs/analysis/e62/value_input_calib_merged6x2.json \
python scripts/vla_analysis/probe_value_input_calib.py \
  --policy.path="$RD/checkpoints/025000/pretrained_model" \
  "${COMMON_ARGS[@]}" \
  --output_dir=$ROOT/outputs/analysis/e62/calib_out --job_name=vnoise_calib_merged6x2
echo "=== E62 VNOISE CALIB COMPLETE $(date -u) ==="
