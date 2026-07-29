#!/bin/bash
# E55 overnight queue (29 Jul 26): wait for loraft2 (e2 specialist) -> specialist
# chunk/jitter cells (the e7 threshold arbiter) -> P3 graduation chain.
# Stage-idempotent: safe to relaunch after a preemption (completed stages skip; the
# P3 chain has its own skip guards + sequential auto-resume).
set -o pipefail
exec >> /home/josh/lerobot/outputs/e55_queue.log 2>&1
echo "=== e55 overnight queue started $(date -u) ==="

# ---- stage 0: wait for the e2 specialist to finish (unit gone == not active == done) ----
while systemctl is-active --quiet loraft2; do sleep 60; done
echo "=== stage 0: loraft2 no longer active $(date -u) ==="
sleep 30

# ---- stage 1: specialist chunk/jitter cells (non-fatal; skip-if-done) ----
ROOT=/home/josh/lerobot
SP=$ROOT/outputs/analysis/e55
mkdir -p $SP
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

COMMON_ARGS=(
  --policy.empty_cameras=1 --policy.dtype=bfloat16
  --policy.gradient_checkpointing=false
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
  --dataset.repo_id=libero_10 --dataset.root="$ROOT/outputs/libero_10"
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
  --env.type=libero --env.task=libero_10
  --steps=200000 --batch_size=32 --num_workers=2
  --online_task_ids='[0,1,2,3,4]' --online_steps_per_task=5000
  --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'
  --wandb.enable=false
)

for SPEC in "4:e7" "3:e2"; do
  T=${SPEC%%:*}; ENV=${SPEC##*:}
  RD=$ROOT/outputs/train/loraft_baseline/task${T}_e${ENV#e}
  OUT=$SP/probe_jitter_specialist_${ENV}.jsonl
  if [ ! -d "$RD/checkpoints/005000/pretrained_model" ]; then
    echo "[stage1] WARNING: $RD checkpoint missing (specialist incomplete?) - SKIPPING ${ENV}; rerun its cell later"
    continue
  fi
  if [ -s "$OUT" ]; then
    echo "[stage1] ${ENV} probe output exists - skipping"
    continue
  fi
  echo "[stage1] chunk/jitter probe on ${ENV} specialist $(date -u)"
  export PROBE_RUN_DIR=$RD PROBE_CKPTS="t${T}:005000" PROBE_OUT=$OUT PROBE_SWAP_SLOTS=0
  python $ROOT/scripts/vla_analysis/probe_jitter.py \
    --policy.path="$RD/checkpoints/005000/pretrained_model" \
    "${COMMON_ARGS[@]}" \
    --output_dir=$SP/jitter_out_spec_${ENV} --job_name=jitter_spec_${ENV} \
    || echo "[stage1] ${ENV} probe FAILED (non-fatal - P3 proceeds; rerun probe later)"
done
echo "=== stage 1: specialist probes done $(date -u) ==="

# ---- stage 2: P3 graduation chain (A-phase + 5-task sequential w/ corefrac) ----
echo "=== stage 2: launching P3 graduation chain $(date -u) ==="
if bash $ROOT/job_scripts/nebius/libero_90/staged/grad_layermax_P3_sep8_corefrac.sh; then
  echo "=== e55 queue COMPLETE $(date -u) ==="
else
  echo "=== e55 queue: P3 CHAIN FAILED $(date -u) ==="
  exit 1
fi
