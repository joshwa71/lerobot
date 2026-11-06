#!/bin/bash
# E61 addendum 5 (9 Aug 26): share-criterion probe — cross-layer router-input
# similarity on the stage-1 checkpoint (= the router input under frozen-prepass).
# Decides the share/solo assignment of the 6x2 merged config from measurement
# instead of hand-picking.
#
# PRE-REGISTERED VALIDATION: the adopted metric must separate expert pair (6,8)
# [shareable in E61] from (10,12) [e7 38->22 in E61], and call VLM pairs
# shareable. E61's own overlap stats failed this test — passing it is a finding.
#
# Designed to run ALONGSIDE FT#1 (~92GB resident): forward-only, bs8, ~15-20GB.
set -eo pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e61_share_criterion.log
exec >> "$LOG" 2>&1
echo "=== E61 share-criterion probe started $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

BASE=$ROOT/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model

SC_EXPERT_LAYERS='[2,4,6,8,10,12,14,16]' \
SC_VLM_LAYERS='[3,5,7,9,11,13,15]' \
SC_NB=8 \
OUT=$ROOT/outputs/analysis/e61/share_criterion_stage1.json \
python scripts/vla_analysis/probe_share_criterion.py \
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
  --wandb.enable=false --output_dir=/tmp/share_criterion_out --job_name=share_criterion
echo "=== E61 share-criterion probe COMPLETE $(date -u) ==="
