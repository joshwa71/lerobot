#!/bin/bash
# E61: sharepairs 4-seed campaign — gated behind the full-FT chain (needs sims +
# GPU free). Completes the one-instrument table's sharepairs row.
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e60_seeds_campaign.log
exec >> "$LOG" 2>&1
while true; do
  st=$(systemctl is-active fullft-l90l10 2>/dev/null)
  [ "$st" = "active" ] || [ "$st" = "activating" ] || break
  sleep 300
done
echo "=== sharepairs seeds: fullft chain exited (state=$st) — starting $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export CAMP_SEEDS="1000,2000,3000,4000"
export CAMP_TAG=sharepairs
export CAMP_OUT=$ROOT/outputs/analysis/e60/seeds_sharepairs.json
if [ ! -f "$CAMP_OUT" ]; then
  python scripts/vla_analysis/eval_seeds_campaign.py \
    --policy.path="$ROOT/outputs/train/libero_10_seq5_jw_sharepairs_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k/checkpoints/025000/pretrained_model" \
    --policy.dtype=bfloat16 \
    --env.type=libero --env.task=libero_10 --env.task_ids="[4,6,9,2,7]" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --eval.batch_size=13 --eval.n_episodes=25 \
    --seed=1000 \
    --output_dir=/tmp/camp_sharepairs \
    || echo "[FAIL] sharepairs campaign"
fi
echo "=== SHAREPAIRS SEEDS COMPLETE $(date -u) ==="
