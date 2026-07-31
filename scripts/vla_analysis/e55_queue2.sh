#!/bin/bash
# E55 queue 2 (29 Jul): wait for the P3 graduation chain (e55queue) -> wipe the e2 stub
# dir (OOM attempt left wandb-only debris; the FileExistsError guard, IN the script this
# time) -> e2 specialist at bs16 x accum2 no-ckpt (Josh's config) -> e2 chunk/jitter probe.
# Stage-idempotent; safe to relaunch.
set -o pipefail
exec >> /home/josh/lerobot/outputs/e55_queue2.log 2>&1
echo "=== e55 queue2 started $(date -u) ==="

# stage 0: wait for the P3 chain's unit to leave the active state
while systemctl is-active --quiet e55queue; do sleep 120; done
echo "=== queue2 stage 0: e55queue no longer active $(date -u) ==="
sleep 30

ROOT=/home/josh/lerobot
SP=$ROOT/outputs/analysis/e55

# stage 1: stub guard - wipe a task3_e2 dir that has no completed checkpoint
RD=$ROOT/outputs/train/loraft_baseline/task3_e2
if [ -d "$RD" ] && [ ! -d "$RD/checkpoints/005000" ]; then
  echo "[queue2 guard] partial task3_e2 dir (no checkpoint) - wiping"
  rm -rf "$RD"
fi

# stage 2: e2 specialist (train + 50-ep eval; skip guards inside the wrapper)
if bash $ROOT/job_scripts/nebius/baselines/loraft_e2_bs16acc2.sh; then
  echo "=== queue2: e2 specialist done $(date -u) ==="
else
  echo "=== queue2: e2 specialist FAILED $(date -u) ==="
fi

# stage 3: e2 chunk/jitter probe (non-fatal)
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
OUT=$SP/probe_jitter_specialist_e2.jsonl
if [ -d "$RD/checkpoints/005000/pretrained_model" ] && [ ! -s "$OUT" ]; then
  rm -rf $SP/jitter_out_spec_e2
  export PROBE_RUN_DIR=$RD PROBE_CKPTS="t3:005000" PROBE_OUT=$OUT PROBE_SWAP_SLOTS=0
  python $ROOT/scripts/vla_analysis/probe_jitter.py \
    --policy.path="$RD/checkpoints/005000/pretrained_model" \
    --policy.empty_cameras=1 --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_10 --dataset.root="$ROOT/outputs/libero_10" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero --env.task=libero_10 \
    --steps=200000 --batch_size=32 --num_workers=2 \
    --online_task_ids='[0,1,2,3,4]' --online_steps_per_task=5000 \
    --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
    --wandb.enable=false \
    --output_dir=$SP/jitter_out_spec_e2 --job_name=jitter_spec_e2 \
    || echo "[queue2] e2 probe FAILED (non-fatal)"
else
  echo "[queue2] e2 probe skipped (missing checkpoint or output exists)"
fi
echo "=== e55 queue2 COMPLETE $(date -u) ==="
