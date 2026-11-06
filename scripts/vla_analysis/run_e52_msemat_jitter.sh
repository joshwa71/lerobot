#!/bin/bash
# E52 battery, GPU proc A: fold-in MSE forgetting matrix (5 ckpts x 5 tasks) then
# jitter grid on the fold-in final checkpoint (t0/t3/t4, clean+state+image).
set -eo pipefail
SP=/home/josh/lerobot/outputs/analysis/e52
mkdir -p $SP
SCRIPTS=/home/josh/lerobot/scripts/vla_analysis
ROOT=/home/josh/lerobot
BASE=$ROOT/outputs/train
FOLD=libero_10_seq5_jw_layermax_compact_e9to12_v13to16_beta4_topt3072_lr2x_steps5k
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
  --steps=200000 --batch_size=32 --num_workers=4
  --online_task_ids='[0,1,2,3,4]' --online_steps_per_task=5000
  --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'
  --wandb.enable=false
)

RD=$BASE/$FOLD
export MSEMAT_RUN_DIR=$RD MSEMAT_STEPS=005000,010000,015000,020000,025000 MSEMAT_OUT=$SP/mse_matrix_foldin.jsonl
python $SCRIPTS/mse_matrix2.py \
  --policy.path="$RD/checkpoints/005000/pretrained_model" \
  "${COMMON_ARGS[@]}" \
  --output_dir=$SP/msemat_out_foldin --job_name=msemat_foldin
echo "=== MSE matrix done ==="

export PROBE_RUN_DIR=$RD PROBE_CKPTS="t0:025000,t3:025000,t4:025000" PROBE_OUT=$SP/probe_jitter_foldin.jsonl PROBE_SWAP_SLOTS=1
python $SCRIPTS/probe_jitter.py \
  --policy.path="$RD/checkpoints/025000/pretrained_model" \
  "${COMMON_ARGS[@]}" \
  --output_dir=$SP/jitter_out_foldin --job_name=jitter_foldin
echo "=== E52 proc A COMPLETE ==="
