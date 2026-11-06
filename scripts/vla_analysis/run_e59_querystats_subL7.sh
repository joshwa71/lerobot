#!/bin/bash
# E59 follow-up (5 Aug 26): extend the E49 querystats-image measurement BELOW the
# measured region — layers [3,4,5,6,7] on stage-1 features (base no-mem checkpoint
# = the router input under frozen-route/prepass). L7 kept as the overlap point to
# calibrate against the E49 curve (known: inter 0.722 at L7 rising to 0.898 at L16;
# lower = better separation). Decides whether VLM banks at 3/5 earn slots in the
# go-big placement-search run, or whether the anchor's within-task conditionality
# collapses sub-L7 (palette-constancy risk: attention hasn't mixed image/proprio
# context into the instruction tokens yet at those depths).
set -eo pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e59_querystats_subL7.log
exec >> "$LOG" 2>&1
echo "=== E59 sub-L7 querystats started $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

BASE=$ROOT/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model

QS_LAYERS='[3,4,5,6,7]' \
OUT=$ROOT/outputs/analysis/e59/querystats_image_subL7.json \
python scripts/vla_analysis/probe_querystats_image.py \
  --policy.path="$BASE" \
  --policy.empty_cameras=1 --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=false \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --dataset.repo_id=libero_10 --dataset.root="$ROOT/outputs/libero_10" \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
  --env.type=libero --env.task=libero_10 \
  --steps=200000 --batch_size=8 --num_workers=2 \
  --online_task_ids='[0,1,2,3,4]' --online_steps_per_task=5000 \
  --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
  --wandb.enable=false --output_dir=/tmp/qs_subL7_out --job_name=qs_subL7
echo "=== E59 sub-L7 querystats COMPLETE $(date -u) ==="
