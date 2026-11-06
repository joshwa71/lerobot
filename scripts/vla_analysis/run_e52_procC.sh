#!/bin/bash
# E52 proc C (chained behind proc B): fold-in INTERMEDIATE chunk cells to align the
# e4 rollout trajectory (55->40->25->35->18) with function (own 0.0333 -> final 0.0402)
# checkpoint-by-checkpoint, + e9 at the crash boundary (020000), + jitter t2 (e9
# brittleness read; proc A's grid covers t0/t3/t4 only).
set -eo pipefail
SP=/home/josh/lerobot/outputs/analysis/e52
SCRIPTS=/home/josh/lerobot/scripts/vla_analysis
ROOT=/home/josh/lerobot
BASE=$ROOT/outputs/train
FOLD=libero_10_seq5_jw_layermax_compact_e9to12_v13to16_beta4_topt3072_lr2x_steps5k
PROCB_LOG=/home/josh/lerobot/outputs/analysis/e52_procB.log

# wait for proc B to release its VRAM
while ! grep -q "E52 proc B COMPLETE" "$PROCB_LOG" 2>/dev/null; do sleep 60; done

source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false

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

RD=$BASE/$FOLD
export PROBE_RUN_DIR=$RD PROBE_OUT=$SP/probe_conversion_foldin.jsonl
export PROBE_LAYERS=""
export PROBE_CKPTS="t0:010000,t0:015000,t0:020000,t2:020000"
python $SCRIPTS/probe_conversion.py \
  --policy.path="$RD/checkpoints/005000/pretrained_model" \
  "${COMMON_ARGS[@]}" \
  --output_dir=$SP/probe_out_foldin_mid --job_name=probe_foldin_mid
echo "=== intermediate chunk cells done ==="

export PROBE_RUN_DIR=$RD PROBE_CKPTS="t2:025000" PROBE_OUT=$SP/probe_jitter_foldin.jsonl PROBE_SWAP_SLOTS=1
python $SCRIPTS/probe_jitter.py \
  --policy.path="$RD/checkpoints/025000/pretrained_model" \
  "${COMMON_ARGS[@]}" \
  --output_dir=$SP/jitter_out_foldin_t2 --job_name=jitter_foldin_t2
echo "=== E52 proc C COMPLETE ==="
