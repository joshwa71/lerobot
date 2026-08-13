#!/bin/bash
# WEEKEND BASELINES ORCHESTRATOR (Josh, 13 Aug: "pull all the baselines ... and
# get their baseline results"). Gated on the e62-vnoise-battery unit exiting
# (the last GPU consumer of the noise-arm chain). Stages, all skip-guarded:
#   1. Full-FT #1 (fresh base) — back-5 4-seed campaign (front-5 row exists;
#      envs [0,8,1,3,5] = task_index 5-9). Requires the checkpoint rsynced
#      back from cold storage (run_weekend_baselines waits for it).
#   2. Full-FT #2 (from-l90) — same.
#   3. Back-5 LoRA specialists: train t5-t9 (loraft_back5_specialists.sh),
#      then a 4-seed campaign per specialist on its own env.
#   4. 10-task multitask LoRA: train (loraft_multitask10.sh), then a 4-seed
#      campaign on all 10 envs.
# All campaigns: 25 eps x paired seeds 1000/2000/3000/4000 — the standing
# headline instrument; JSONs land in outputs/analysis/e60/ next to the
# existing rows.
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/weekend_baselines.log
exec >> "$LOG" 2>&1
echo "=== weekend baselines: waiting on e62-vnoise-battery $(date -u) ==="
while true; do
  st=$(systemctl is-active e62-vnoise-battery 2>/dev/null) || true
  [ "$st" = "active" ] || [ "$st" = "activating" ] || break
  sleep 300
done
echo "=== weekend baselines: gate passed (state=$st) — starting $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
BACK5='[0,8,1,3,5]'
ALL10='[4,6,9,2,7,0,8,1,3,5]'

camp () {  # $1 tag  $2 ckpt  $3 task_ids  $4 extra policy args
  local out=$ROOT/outputs/analysis/e60/seeds_$1.json
  [ -f "$out" ] && { echo "[camp] $1 exists - skipping."; return 0; }
  CAMP_SEEDS="1000,2000,3000,4000" CAMP_TAG=$1 CAMP_OUT=$out \
  python scripts/vla_analysis/eval_seeds_campaign.py \
    --policy.path="$2" \
    --policy.dtype=bfloat16 $4 \
    --env.type=libero --env.task=libero_10 --env.task_ids="$3" \
    --rename_map="$RENAME" \
    --eval.batch_size=13 --eval.n_episodes=25 \
    --seed=1000 \
    --output_dir=/tmp/camp_$1 \
    || echo "[FAIL] campaign $1"
}

# ---- stage 1+2: full-FT back-5 rows (checkpoints restored from cold) ----
FT1=$ROOT/outputs/train/libero_10_pi05_fullft_frombase_nomem_50k/checkpoints/050000/pretrained_model
FT2=$ROOT/outputs/train/libero_10_pi05_fullft_froml90_nomem_50k/checkpoints/050000/pretrained_model
for i in 1 2 3 4 5 6; do
  [ -d "$FT1" ] && [ -d "$FT2" ] && break
  echo "[ft] waiting for cold-storage restore ($(date -u))"; sleep 600
done
if [ -d "$FT1" ]; then camp fullft_l10_back5 "$FT1" "$BACK5" ""; else echo "[FAIL] FT1 checkpoint missing"; fi
if [ -d "$FT2" ]; then camp fullft_l90_l10_back5 "$FT2" "$BACK5" ""; else echo "[FAIL] FT2 checkpoint missing"; fi

# ---- stage 3: back-5 specialists ----
bash job_scripts/nebius/baselines/loraft_back5_specialists.sh || echo "[FAIL] back-5 specialist training"
camp spec_e0 "$ROOT/outputs/train/loraft_baseline/task5_e0/checkpoints/005000/pretrained_model" "[0]" "--policy.use_peft=true"
camp spec_e8 "$ROOT/outputs/train/loraft_baseline/task6_e8/checkpoints/005000/pretrained_model" "[8]" "--policy.use_peft=true"
camp spec_e1 "$ROOT/outputs/train/loraft_baseline/task7_e1/checkpoints/005000/pretrained_model" "[1]" "--policy.use_peft=true"
camp spec_e3 "$ROOT/outputs/train/loraft_baseline/task8_e3/checkpoints/005000/pretrained_model" "[3]" "--policy.use_peft=true"
camp spec_e5 "$ROOT/outputs/train/loraft_baseline/task9_e5/checkpoints/005000/pretrained_model" "[5]" "--policy.use_peft=true"

# ---- stage 4: 10-task multitask LoRA ----
bash job_scripts/nebius/baselines/loraft_multitask10.sh || echo "[FAIL] multitask10 training"
camp multitask10 "$ROOT/outputs/train/loraft_multitask10/checkpoints/010000/pretrained_model" "$ALL10" "--policy.use_peft=true"

echo "=== WEEKEND BASELINES COMPLETE $(date -u) ==="
