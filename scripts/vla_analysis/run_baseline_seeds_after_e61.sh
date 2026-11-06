#!/bin/bash
# Queued 4-seed baseline evals (Josh, 7 Aug): multitask-LoRA (standing 49.2,
# single-seed) + naive seq-LoRA r256 final (standing 17.6, the forgetting foil) —
# same instrument as the E60 campaign (25 eps x 4 paired seeds, finals only).
# Gated on the e61-sharepairs UNIT exiting (chain landing frees the GPU).
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e60_seeds_campaign.log   # append to the campaign log
exec >> "$LOG" 2>&1

while true; do
  st=$(systemctl is-active e61-sharepairs 2>/dev/null)
  [ "$st" = "active" ] || [ "$st" = "activating" ] || break
  sleep 300
done
echo "=== baseline seeds queue: e61 chain exited (state=$st) — starting $(date -u) ==="

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

run_camp multitask5 "$BASE/loraft_multitask5/checkpoints/last/pretrained_model" "[4,6,9,2,7]" "--policy.use_peft=true"
run_camp naive_final "$BASE/libero_10_seq5_naive_lora_r256_a64_steps5k/checkpoints/025000/pretrained_model" "[4,6,9,2,7]" "--policy.use_peft=true"

echo "=== BASELINE SEEDS QUEUE COMPLETE $(date -u) ==="
