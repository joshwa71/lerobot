#!/bin/bash
# E64 (Josh, 18 Aug, revised): EVERY LoRA baseline row re-provisioned FROM SCRATCH
# at a uniform r=512 / alpha=128 — above our method on every rung of the
# active-parameter ladder that a reviewer could name (per-site bottleneck 288/128,
# per-token active ~240x, per-step trained ~3.7x), below only total storage (the
# full-FT rows match that). "Then we can defend every point against the reviewer."
# The r256 drafts and the running r256 naive-10 foil were killed before/at block 5;
# nothing carries over (adapter shapes differ).
# Stage 1: multitask-LoRA-10 r512 / 50k (= 5k/task) -> all-10 4-seed row
#          -> seeds_multitask10_r512.json
# Stage 2: ten per-task specialists r512 / 5k -> per-specialist rows
#          -> seeds_spec_r512_e{env}.json
# Stage 3: naive sequential LoRA r512 / 10 tasks (self-resuming wrapper)
#          -> all-10 4-seed row -> seeds_naive10_r512_final.json
# All campaigns: 25 eps x paired seeds 1000/2000/3000/4000, vec bs=13 — the
# standing headline instrument; JSONs in outputs/analysis/e60/. Stage-level
# skip-guards; failures logged, queue continues. Relaunching this unit after a
# preemption is safe: stages 1-2 skip finished runs (partial multitask/specialist
# dirs restart from scratch — no PEFT resume in lerobot-train), stage 3 resumes.
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e64_lora_r512.log
exec >> "$LOG" 2>&1
echo "=== E64 LoRA-r512 queue: gate (e63-queue / e64-lora-r256 must be inactive) $(date -u) ==="
while true; do
  a=$(systemctl is-active e63-queue 2>/dev/null) || true
  b=$(systemctl is-active e64-lora-r256 2>/dev/null) || true
  { [ "$a" = "active" ] || [ "$a" = "activating" ] || [ "$b" = "active" ] || [ "$b" = "activating" ]; } || break
  sleep 300
done
echo "=== E64 LoRA-r512 queue: gate passed (e63-queue=$a e64-lora-r256=$b) — starting $(date -u) ==="
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

# ---- stage 1: multitask-LoRA-10 @ r512 / 50k ----
echo "[e64] stage 1: multitask10 r512/50k train $(date -u)"
bash job_scripts/nebius/baselines/loraft_multitask10_r512_50k.sh || echo "[FAIL] multitask10_r512 training"
echo "[e64] stage 1: multitask10 r512 4-seed row $(date -u)"
camp multitask10_r512 "$ROOT/outputs/train/loraft_multitask10_r512_50k/checkpoints/050000/pretrained_model" "$ALL10" "--policy.use_peft=true"

# ---- stage 2: ten specialists @ r512 / 5k + per-specialist rows ----
echo "[e64] stage 2: r512 specialists train (all 10) $(date -u)"
bash job_scripts/nebius/baselines/loraft_specialists10_r512.sh || echo "[FAIL] r512 specialists training (see per-task lines above)"
declare -A ENV_ID=( [0]=4 [1]=6 [2]=9 [3]=2 [4]=7 [5]=0 [6]=8 [7]=1 [8]=3 [9]=5 )
for T in 0 1 2 3 4 5 6 7 8 9; do
  ENV=${ENV_ID[$T]}
  camp spec_r512_e${ENV} "$ROOT/outputs/train/loraft_baseline_r512/task${T}_e${ENV}/checkpoints/005000/pretrained_model" "[$ENV]" "--policy.use_peft=true"
done

# ---- stage 3: naive sequential LoRA r512, 10 tasks (self-resuming) + all-10 row ----
echo "[e64] stage 3: naive seq LoRA r512, 10 tasks $(date -u)"
bash job_scripts/nebius/baselines/naive_seq_lora_r512_10task.sh || echo "[FAIL] naive r512 10-task train"
NAIVE=$ROOT/outputs/train/libero_10_seq10_naive_lora_r512_a128_steps5k
camp naive10_r512_final "$NAIVE/checkpoints/050000/pretrained_model" "$ALL10" "--policy.use_peft=true"

echo "=== E64 LoRA-r512 QUEUE COMPLETE $(date -u) ==="
