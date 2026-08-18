#!/bin/bash
# E64 (Josh, 18 Aug): re-provision the LoRA baseline rows at a uniform r=256 /
# alpha=64 so they sit ABOVE our method on every compute/optimization rung of the
# active-parameter ladder (per-token ~120x, per-step ~1.9x; below only on total
# storage, which the full-FT rows match). Gated on the E63 queue (naive-10 foil +
# its seed row) exiting.
# Stage 1: multitask-LoRA-10 at r256 / 50k steps (= 5k/task, our budget) — the
#          TRUE multitask baseline (the E43 r32/1k-per-task probe leaked into the
#          table) -> 4-seed row on all 10 envs -> seeds_multitask10_r256.json
# Stage 2: all ten per-task specialists at r256 / 5k -> per-specialist 4-seed rows
#          -> seeds_spec_r256_e{env}.json (the r32 rows stay as appendix sensitivity)
# All campaigns: 25 eps x paired seeds 1000/2000/3000/4000, vec bs=13 — the
# standing headline instrument; JSONs land in outputs/analysis/e60/ next to the
# existing rows. Stage-level skip-guards; failures logged, queue continues.
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e64_lora_r256.log
exec >> "$LOG" 2>&1
echo "=== E64 LoRA-r256 queue: waiting on e63-queue $(date -u) ==="
while true; do
  st=$(systemctl is-active e63-queue 2>/dev/null) || true
  [ "$st" = "active" ] || [ "$st" = "activating" ] || break
  sleep 300
done
echo "=== E64 LoRA-r256 queue: gate passed (e63-queue=$st) — starting $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
ALL10='[4,6,9,2,7,0,8,1,3,5]'

camp () {  # $1 tag  $2 ckpt  $3 task_ids  $4 extra policy args
  local out=$ROOT/outputs/analysis/e60/seeds_$1.json
  [ -f "$out" ] && { echo "[camp] $1 exists - skipping."; return 0; }
  [ -d "$2" ] || { echo "[camp] $1: checkpoint missing ($2) - skipping."; return 1; }
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

# ---- stage 1: multitask-LoRA-10 @ r256 / 50k ----
echo "[e64] stage 1: multitask10 r256/50k train $(date -u)"
bash job_scripts/nebius/baselines/loraft_multitask10_r256_50k.sh || echo "[FAIL] multitask10_r256 training"
echo "[e64] stage 1: multitask10 r256 4-seed row $(date -u)"
camp multitask10_r256 "$ROOT/outputs/train/loraft_multitask10_r256_50k/checkpoints/050000/pretrained_model" "$ALL10" "--policy.use_peft=true"

# ---- stage 2: ten specialists @ r256 / 5k + per-specialist rows ----
echo "[e64] stage 2: r256 specialists train (all 10) $(date -u)"
bash job_scripts/nebius/baselines/loraft_specialists10_r256.sh || echo "[FAIL] r256 specialists training (see per-task lines above)"
declare -A ENV_ID=( [0]=4 [1]=6 [2]=9 [3]=2 [4]=7 [5]=0 [6]=8 [7]=1 [8]=3 [9]=5 )
for T in 0 1 2 3 4 5 6 7 8 9; do
  ENV=${ENV_ID[$T]}
  camp spec_r256_e${ENV} "$ROOT/outputs/train/loraft_baseline_r256/task${T}_e${ENV}/checkpoints/005000/pretrained_model" "[$ENV]" "--policy.use_peft=true"
done
echo "=== E64 LoRA-r256 QUEUE COMPLETE $(date -u) ==="
