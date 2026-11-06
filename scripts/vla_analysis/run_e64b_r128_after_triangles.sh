#!/bin/bash
# E64b (Josh, 21 Aug): the r128 SPECIALIST LADDER POINT, gated on `e64-triangles`
# exiting. Priority: this runs BEFORE the protection-off ablation and the
# training-seed replicates (Josh's ordering: triangles -> cold-ship + r128 in
# parallel -> the rest). The cold-storage ship runs concurrently on the DESK PC
# (scripts/ops/ship_e64_batch_to_cold.sh) — different machine, no GPU contention.
#
# Ten per-task specialists at r=128 / alpha=32 (alpha/r=0.25 held), 5,000 steps
# each, recipe otherwise byte-identical to the r32 and r512 ladder points, then a
# 4-seed row per specialist. ~2.5-3 h each (~28 h) + ~3 h of rows.
# Outputs: seeds_spec_r128_e{env}.json in outputs/analysis/e60/.
# Skip-guarded per task and per row; relaunching after a preemption is safe
# (partial training dirs are moved aside and rerun — no PEFT resume in lerobot-train).
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e64b_r128.log
exec >> "$LOG" 2>&1
echo "=== E64b r128 ladder: waiting on e64-triangles $(date -u) ==="
while true; do
  st=$(systemctl is-active e64-triangles 2>/dev/null) || true
  [ "$st" = "active" ] || [ "$st" = "activating" ] || break
  sleep 300
done
echo "=== E64b r128 ladder: gate passed (e64-triangles=$st) $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'

echo "[e64b] training ten r128 specialists $(date -u)"
bash job_scripts/nebius/baselines/loraft_specialists10_r128.sh || echo "[FAIL] r128 specialist training"

declare -A ENV_ID=( [0]=4 [1]=6 [2]=9 [3]=2 [4]=7 [5]=0 [6]=8 [7]=1 [8]=3 [9]=5 )
for T in 0 1 2 3 4 5 6 7 8 9; do
  ENV=${ENV_ID[$T]}
  out=$ROOT/outputs/analysis/e60/seeds_spec_r128_e${ENV}.json
  [ -f "$out" ] && { echo "[camp] spec_r128_e$ENV exists - skipping."; continue; }
  pol=$ROOT/outputs/train/loraft_baseline_r128/task${T}_e${ENV}/checkpoints/005000/pretrained_model
  [ -d "$pol" ] || { echo "[camp] spec_r128_e$ENV: checkpoint missing - skipping."; continue; }
  echo "[e64b] 4-seed row spec_r128_e$ENV $(date -u)"
  CAMP_SEEDS="1000,2000,3000,4000" CAMP_TAG=spec_r128_e${ENV} CAMP_OUT=$out \
  python scripts/vla_analysis/eval_seeds_campaign.py \
    --policy.path="$pol" --policy.dtype=bfloat16 --policy.use_peft=true \
    --env.type=libero --env.task=libero_10 --env.task_ids="[$ENV]" \
    --rename_map="$RENAME" --eval.batch_size=13 --eval.n_episodes=25 \
    --seed=1000 --output_dir=/tmp/camp_spec_r128_e${ENV} \
    || echo "[FAIL] campaign spec_r128_e$ENV"
done
echo "=== E64b r128 LADDER COMPLETE $(date -u) ==="
