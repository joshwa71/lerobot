#!/bin/bash
# E53 proc B resume: spread-A chunk + jitter ONLY (its MSE matrix completed before the
# OOM). Chained behind proc A's completion marker so the two chunk probes never run
# concurrently (the OOM cause: both gain probes' einsums peaked together, 70+66GB).
set -eo pipefail
SP=/home/josh/lerobot/outputs/analysis/e53
SCRIPTS=/home/josh/lerobot/scripts/vla_analysis
ROOT=/home/josh/lerobot
BASE=$ROOT/outputs/train
RUN=libero_10_seq5_jw_layermax_A_e2468_v10121416_beta4_topt3072_lr2x_steps5k
PROCA_LOG=/home/josh/lerobot/outputs/analysis/e53_procA.log

while ! grep -q "E53 proc A (corefrac) COMPLETE" "$PROCA_LOG" 2>/dev/null; do sleep 30; done

source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

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

RD=$BASE/$RUN
# clear the partial jsonl from the OOMed attempt (1 gain row)
: > $SP/probe_conversion_spreadA.jsonl
export PROBE_RUN_DIR=$RD PROBE_OUT=$SP/probe_conversion_spreadA.jsonl
export PROBE_LAYERS="expert:8,vlm:14"
export PROBE_CKPTS="t0:005000,t1:010000,t2:015000,t3:020000,t4:025000,t0:025000,t1:025000,t2:025000,t3:025000"
python $SCRIPTS/probe_conversion.py \
  --policy.path="$RD/checkpoints/005000/pretrained_model" \
  "${COMMON_ARGS[@]}" \
  --output_dir=$SP/probe_out_spreadA --job_name=probe_spreadA
echo "=== spreadA chunk cells done ==="

export PROBE_RUN_DIR=$RD PROBE_CKPTS="t0:025000,t2:025000,t3:025000,t4:025000" PROBE_OUT=$SP/probe_jitter_spreadA.jsonl PROBE_SWAP_SLOTS=1
python $SCRIPTS/probe_jitter.py \
  --policy.path="$RD/checkpoints/025000/pretrained_model" \
  "${COMMON_ARGS[@]}" \
  --output_dir=$SP/jitter_out_spreadA --job_name=jitter_spreadA
echo "=== E53 proc B (spreadA) COMPLETE ==="
