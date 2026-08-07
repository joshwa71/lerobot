#!/bin/bash
# E60 multi-seed campaign (Josh, 7 Aug): 25 eps x 4 seeds — bigsearch final (5 envs),
# interleave final (5 envs), each specialist on its own env. Vec-batched bs=13
# (2 rounds per cell, 65 sim instances peak on the memory configs). ~3h total.
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e60_seeds_campaign.log
exec >> "$LOG" 2>&1
echo "=== E60 seeds campaign started $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

BASE=$ROOT/outputs/train
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
export CAMP_SEEDS="1000,2000,3000,4000"

run_camp () {  # $1 tag, $2 ckpt, $3 task_ids, $4 extra policy args
  export CAMP_TAG=$1
  export CAMP_OUT=$ROOT/outputs/analysis/e60/seeds_$1.json
  if [ -f "$CAMP_OUT" ]; then echo "[skip] $1 (exists)"; return 0; fi
  python scripts/vla_analysis/eval_seeds_campaign.py \
    --policy.path="$2" $4 \
    --policy.dtype=bfloat16 \
    --env.type=libero --env.task=libero_10 --env.task_ids="$3" \
    --rename_map="$RENAME" \
    --eval.batch_size=13 --eval.n_episodes=25 \
    --seed=1000 \
    --output_dir=/tmp/camp_$1 \
    || echo "[FAIL] $1"
}

run_camp bigsearch  "$BASE/libero_10_seq5_jw_bigsearch_e4to16_v5to13_prepass_beta4corefrac_topt3072_lr2x_steps5k/checkpoints/025000/pretrained_model" "[4,6,9,2,7]" ""
run_camp interleave "$BASE/libero_10_seq5_jw_interleave_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k/checkpoints/025000/pretrained_model" "[4,6,9,2,7]" ""
run_camp spec_e4 "$BASE/loraft_baseline/task0_e4/checkpoints/005000/pretrained_model" "[4]" "--policy.use_peft=true"
run_camp spec_e6 "$BASE/loraft_baseline/task1_e6/checkpoints/005000/pretrained_model" "[6]" "--policy.use_peft=true"
run_camp spec_e9 "$BASE/loraft_baseline/task2_e9/checkpoints/005000/pretrained_model" "[9]" "--policy.use_peft=true"
run_camp spec_e2 "$BASE/loraft_baseline/task3_e2/checkpoints/005000/pretrained_model" "[2]" "--policy.use_peft=true"
run_camp spec_e7 "$BASE/loraft_baseline/task4_e7/checkpoints/005000/pretrained_model" "[7]" "--policy.use_peft=true"

echo "=== E60 SEEDS CAMPAIGN COMPLETE $(date -u) ==="
