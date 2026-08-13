#!/bin/bash
# E62 4-seed campaign row (the headline instrument: 25 eps x 4 paired seeds
# 1000/2000/3000/4000, front-5 envs) for the merged-6x2 final checkpoint.
# Gated on the e62-battery UNIT exiting (GPU consumer). NB is-active exits
# nonzero for inactive — never let that kill the loop (E61-add-6 lesson).
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e62_seeds.log
exec >> "$LOG" 2>&1
echo "=== e62-seeds: waiting on e62-battery $(date -u) ==="
while true; do
  st=$(systemctl is-active e62-battery 2>/dev/null) || true
  [ "$st" = "active" ] || [ "$st" = "activating" ] || break
  sleep 300
done
echo "=== e62-seeds: battery exited (state=$st) — starting campaign $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

RD=$ROOT/outputs/train/libero_10_seq5_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
export CAMP_SEEDS="1000,2000,3000,4000"
export CAMP_TAG=merged6x2
export CAMP_OUT=$ROOT/outputs/analysis/e60/seeds_merged6x2.json
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
if [ ! -f "$CAMP_OUT" ]; then
  python scripts/vla_analysis/eval_seeds_campaign.py \
    --policy.path="$RD/checkpoints/025000/pretrained_model" \
    --policy.dtype=bfloat16 \
    --env.type=libero --env.task=libero_10 --env.task_ids="[4,6,9,2,7]" \
    --rename_map="$RENAME" \
    --eval.batch_size=13 --eval.n_episodes=25 \
    --seed=1000 \
    --output_dir=/tmp/camp_merged6x2 \
    || echo "[FAIL] merged6x2 campaign"
fi
echo "=== E62-SEEDS COMPLETE $(date -u) ==="
