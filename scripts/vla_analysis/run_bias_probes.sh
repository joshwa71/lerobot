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
run_one () {  # $1=run_dir $2=ckpts $3=a_ckpt $4=tag
  local RD=$BASE/$1
  local FIRST=$(echo $2 | cut -d, -f1 | cut -d: -f2)
  export PROBE_RUN_DIR=$RD PROBE_CKPTS=$2 PROBE_A_CKPT=$3 PROBE_OUT=$SP/probe_bias.jsonl
  python $SCRIPTS/probe_bias.py \
    --policy.path="$RD/checkpoints/$FIRST/pretrained_model" \
    --policy.empty_cameras=1 --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_10 --dataset.root="$ROOT/outputs/libero_10" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero --env.task=libero_10 \
    --output_dir=$SP/biasprobe_out_$4 \
    --steps=200000 --batch_size=32 --num_workers=2 \
    --online_task_ids='[0,1,2,3,4]' --online_steps_per_task=5000 \
    --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
    --wandb.enable=false --job_name=biasprobe_$4
  echo "=== bias probe $4 complete ==="
}
A_B=$BASE/libero_90_pi05_8_10_12_14_frozenroute_rwarmupB_values10k_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model
R2244_PRE=$BASE/libero_90_pi05_8_10_12_14_film_lora_2244_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k/checkpoints/last/pretrained_model
LR2X=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_lr2x_steps5k_tasks5
S7K=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps7k_tasks5
STGB=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps5k_tasks5
R2244=libero_10_sequential_pi05_8_10_12_14_film_lora_2244_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536_protect_beta4_steps5k
run_one $STGB "t0:025000,t2:025000,t3:025000" $A_B stageB
run_one $LR2X "t0:005000,t2:015000,t3:020000" $A_B lr2x
run_one $S7K  "t0:007000,t2:021000,t3:028000" $A_B steps7k
run_one $R2244 "t0:050000,t2:050000,t3:050000" $R2244_PRE r2244
echo "ALL BIAS PROBES DONE"
