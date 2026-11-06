#!/bin/bash
set -eo pipefail
SP=/home/josh/lerobot/outputs/analysis/e41
SCRIPTS=/home/josh/lerobot/scripts/vla_analysis
ROOT=/home/josh/lerobot
BASE=$ROOT/outputs/train
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
run_one () {  # $1=run_dir_name $2=steps_csv $3=tag
  local RD=$BASE/$1
  local FIRST=$(echo $2 | cut -d, -f1)
  export MSEMAT_RUN_DIR=$RD MSEMAT_STEPS=$2 MSEMAT_OUT=$SP/mse_matrix_arms.jsonl
  python $SCRIPTS/mse_matrix2.py \
    --policy.path="$RD/checkpoints/$FIRST/pretrained_model" \
    --policy.empty_cameras=1 --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_10 --dataset.root="$ROOT/outputs/libero_10" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero --env.task=libero_10 \
    --output_dir=$SP/msemat_out_$3 \
    --steps=200000 --batch_size=32 --num_workers=4 \
    --online_task_ids='[0,1,2,3,4]' --online_steps_per_task=5000 \
    --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
    --wandb.enable=false --job_name=msemat_$3
  echo "=== $3 complete ==="
}
run_one libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_lr2x_steps5k_tasks5 "005000,010000,015000,020000,025000" lr2x
run_one libero_10_sequential_pi05_8_10_12_14_frozenroute_affine_nogate_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps5k_tasks5 "005000,010000,015000,020000,025000" affine
run_one libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps7k_tasks5 "007000,014000,021000,028000,035000" steps7k
echo "ALL MATRICES DONE"
