#!/bin/bash
# E69: paired-noise flow-matching MSE matrix for the REVERSED-order paper cell (Josh, 15 Sep:
# "run an MSE probe for the final ckpt ... we need to know the delta for the paper").
#
# Instrument = mse_matrix2.py, the E39 matrix with the E65 add-16 shared-table loader fix (matches
# both `.mlp.mem.slot_*` AND `.mlp.mem._storage_shared_from.slot_*`, i.e. 14/14 tensors). Same
# instrument that produced the FORWARD 10-task own-task drift +28.5% (E65 add-25), so the two are
# directly comparable.
#
# WHY THE WHOLE MATRIX AND NOT JUST THE FINAL CHECKPOINT: the delta the paper quotes is "loss at a
# task's OWN boundary -> loss under the final model", both measured with THIS instrument. The
# training loss is a different measurement (different noise draw, different batches, running mean
# over the last 200 optimiser steps), so pairing it with a probe value would mix instruments. The
# diagonal is what supplies the baseline, and it costs one extra pass per checkpoint.
#
# Reversed training order is [9,8,7,6,5,4,3,2,1,0]: checkpoint k*5000 is taken AFTER dataset task
# order[k-1], so the diagonal pairs 005000<->task 9, 010000<->task 8, ..., 050000<->task 0.
# 10 checkpoints x 10 tasks x 16 batches of 32, seed 0.
set -eo pipefail
ROOT=/home/josh/lerobot
OUTDIR=$ROOT/outputs/analysis/e69; mkdir -p $OUTDIR
RUN=libero_10_seq10rev_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
RD=$ROOT/outputs/train/$RUN
OUT=$OUTDIR/mse_matrix_rev10.jsonl
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
[ -d "$RD/checkpoints/050000/pretrained_model" ] || { echo "ERROR: final checkpoint missing"; exit 1; }
export MSEMAT_RUN_DIR=$RD
export MSEMAT_STEPS=005000,010000,015000,020000,025000,030000,035000,040000,045000,050000
export MSEMAT_TASKS=0,1,2,3,4,5,6,7,8,9
export MSEMAT_OUT=$OUT
echo "=== E69 rev10 MSE matrix started $(date -u) -> $OUT ==="
python scripts/vla_analysis/mse_matrix2.py \
  --policy.path="$RD/checkpoints/005000/pretrained_model" \
  --policy.empty_cameras=1 --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=false \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --dataset.repo_id=libero_10 --dataset.root="$ROOT/outputs/libero_10" \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
  --env.type=libero --env.task=libero_10 \
  --output_dir=$OUTDIR/msemat_out_rev10 \
  --steps=200000 --batch_size=32 --num_workers=4 \
  --online_task_ids='[9,8,7,6,5,4,3,2,1,0]' --online_steps_per_task=5000 \
  --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
  --wandb.enable=false --job_name=msemat_rev10
echo "=== E69-REV10-MSEMAT-DONE $(date -u) ==="
