#!/bin/bash
# E64c (Josh, 25 Aug): the r=64 SPECIALIST LADDER POINT, gated on `e64b-r128` exiting.
# r64/alpha16 (alpha/r=0.25 held), 5,000 steps each, recipe otherwise byte-identical
# to the r32 / r128 / r512 points, then a 4-seed row per specialist.
# WHY: with r128 measured, the bracket around our 65.1 is r32 (63.7) < ours < r128,
# and r64 is the only remaining rung inside it.
# ~2h42m each (~27h) + ~4.5h of rows.
# Outputs: seeds_spec_r64_e{env}.json in outputs/analysis/e60/.
# Skip-guarded per task and per row; relaunching after a preemption is safe
# (partial training dirs are moved aside and rerun -- no PEFT resume in lerobot-train).
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e64c_r64.log
exec >> "$LOG" 2>&1
echo "=== E64c r64 ladder: waiting on e64b-r128 $(date -u) ==="
while true; do
  st=$(systemctl is-active e64b-r128 2>/dev/null) || true
  [ "$st" = "active" ] || [ "$st" = "activating" ] || break
  sleep 300
done
echo "=== E64c r64 ladder: gate passed (e64b-r128=$st) $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'

echo "[e64c] training ten r64 specialists $(date -u)"
bash job_scripts/nebius/baselines/loraft_specialists10_r64.sh || echo "[FAIL] r64 specialist training"

declare -A ENV_ID=( [0]=4 [1]=6 [2]=9 [3]=2 [4]=7 [5]=0 [6]=8 [7]=1 [8]=3 [9]=5 )
for T in 0 1 2 3 4 5 6 7 8 9; do
  ENV=${ENV_ID[$T]}
  out=$ROOT/outputs/analysis/e60/seeds_spec_r64_e${ENV}.json
  [ -f "$out" ] && { echo "[camp] spec_r64_e$ENV exists - skipping."; continue; }
  pol=$ROOT/outputs/train/loraft_baseline_r64/task${T}_e${ENV}/checkpoints/005000/pretrained_model
  [ -d "$pol" ] || { echo "[camp] spec_r64_e$ENV: checkpoint missing - skipping."; continue; }
  echo "[e64c] 4-seed row spec_r64_e$ENV $(date -u)"
  CAMP_SEEDS="1000,2000,3000,4000" CAMP_TAG=spec_r64_e${ENV} CAMP_OUT=$out \
  python scripts/vla_analysis/eval_seeds_campaign.py \
    --policy.path="$pol" \
    --policy.dtype=bfloat16 --policy.use_peft=true \
    --env.type=libero --env.task=libero_10 --env.task_ids="[$ENV]" \
    --rename_map="$RENAME" \
    --eval.batch_size=13 --eval.n_episodes=25 \
    --seed=1000 \
    --output_dir=/tmp/camp_spec_r64_e${ENV} \
    || echo "[FAIL] campaign spec_r64_e$ENV"
done
echo "=== E64c r64 ladder COMPLETE $(date -u) ==="
