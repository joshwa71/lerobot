#!/bin/bash
# E52 battery, GPU proc B: chunk probes.
# 1) fold-in: own-block + final-grid, t0/t2 (the degrading tasks) front-loaded.
#    Probe A (gain) at expert:12 + vlm:14 via the new PROBE_LAYERS parameterization.
# 2) layermax-plain backfill: own-block t0-t3 + final t0-t3 (its own->final ladder was
#    never measured; final-grid previously came from jitter clean rows, t2 cell missing).
set -eo pipefail
SP=/home/josh/lerobot/outputs/analysis/e52
mkdir -p $SP
SCRIPTS=/home/josh/lerobot/scripts/vla_analysis
ROOT=/home/josh/lerobot
BASE=$ROOT/outputs/train
FOLD=libero_10_seq5_jw_layermax_compact_e9to12_v13to16_beta4_topt3072_lr2x_steps5k
PLAIN=libero_10_seq5_jw_layermax_compact_e9to12_v13to16_beta4_topt1536_steps5k
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

# --- fold-in: t0/t2 own+final first, then the rest ---
RD=$BASE/$FOLD
export PROBE_RUN_DIR=$RD PROBE_OUT=$SP/probe_conversion_foldin.jsonl
export PROBE_LAYERS="expert:12,vlm:14"
export PROBE_CKPTS="t0:005000,t2:015000,t0:025000,t2:025000,t1:010000,t3:020000,t4:025000,t1:025000,t3:025000"
python $SCRIPTS/probe_conversion.py \
  --policy.path="$RD/checkpoints/005000/pretrained_model" \
  "${COMMON_ARGS[@]}" \
  --output_dir=$SP/probe_out_foldin --job_name=probe_foldin
echo "=== fold-in chunk cells done ==="

# --- layermax-plain backfill (chunk only; skip gain probe) ---
RD=$BASE/$PLAIN
export PROBE_RUN_DIR=$RD PROBE_OUT=$SP/probe_conversion_plain.jsonl
export PROBE_LAYERS=""
export PROBE_CKPTS="t0:005000,t2:015000,t0:025000,t2:025000,t1:010000,t3:020000,t1:025000,t3:025000"
python $SCRIPTS/probe_conversion.py \
  --policy.path="$RD/checkpoints/005000/pretrained_model" \
  "${COMMON_ARGS[@]}" \
  --output_dir=$SP/probe_out_plain --job_name=probe_plain
echo "=== E52 proc B COMPLETE ==="
