#!/bin/bash
# E42: probe battery over the 3 landed arms (bs64 / softprotect / topt3072) + grid fill-ins
# for stageB/lr2x/steps7k/affine so every arm has: own-block chunk error (fit) and final
# chunk error (retention) on t0(e4)/t1(e6)/t2(e9)/t3(e2). t1 rows are NEW (the e6 bleed cell,
# never probed in E41). Then the softprotect MSE forgetting matrix (stationarity check for
# the post-step blend machinery).
set -eo pipefail
SP=/home/josh/lerobot/outputs/analysis/e42
mkdir -p $SP
SCRIPTS=/home/josh/lerobot/scripts/vla_analysis
ROOT=/home/josh/lerobot
BASE=$ROOT/outputs/train
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false

probe () {  # $1=run_dir_name $2=ckpts $3=tag
  local RD=$BASE/$1
  local FIRST=$(echo $2 | cut -d, -f1 | cut -d: -f2)
  export PROBE_RUN_DIR=$RD PROBE_CKPTS=$2 PROBE_OUT=$SP/probe_conversion.jsonl
  python $SCRIPTS/probe_conversion.py \
    --policy.path="$RD/checkpoints/$FIRST/pretrained_model" \
    --policy.empty_cameras=1 --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_10 --dataset.root="$ROOT/outputs/libero_10" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero --env.task=libero_10 \
    --output_dir=$SP/probe_out_$3 \
    --steps=200000 --batch_size=32 --num_workers=2 \
    --online_task_ids='[0,1,2,3,4]' --online_steps_per_task=5000 \
    --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
    --wandb.enable=false --job_name=probe_$3
  echo "=== probe $3 complete ==="
}

BS64=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_bs64accum2_steps5k_tasks5
SOFTP=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_softprotect_cf_beta4_lr2x_steps5k_tasks5
TOP3K=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_3072_protect_beta4_steps5k_tasks5
STGB=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps5k_tasks5
LR2X=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_lr2x_steps5k_tasks5
S7K=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps7k_tasks5
AFF=libero_10_sequential_pi05_8_10_12_14_frozenroute_affine_nogate_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps5k_tasks5

GRID8="t0:005000,t1:010000,t2:015000,t3:020000,t0:025000,t1:025000,t2:025000,t3:025000"
# bs64 done in run_e42.sh pass
probe $STGB  "t1:025000" stageB
probe $SOFTP "$GRID8" softp
probe $LR2X  "t1:010000,t0:025000,t1:025000,t2:025000,t3:025000" lr2x
probe $TOP3K "$GRID8" top3k
probe $S7K   "t1:014000,t1:035000" steps7k
probe $AFF   "t1:010000,t1:025000" affine

# softprotect MSE forgetting matrix (post-step blend stationarity check)
export MSEMAT_RUN_DIR=$BASE/$SOFTP MSEMAT_STEPS="005000,010000,015000,020000,025000" MSEMAT_OUT=$SP/mse_matrix_arms.jsonl
python $SCRIPTS/mse_matrix2.py \
  --policy.path="$BASE/$SOFTP/checkpoints/005000/pretrained_model" \
  --policy.empty_cameras=1 --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=false \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --dataset.repo_id=libero_10 --dataset.root="$ROOT/outputs/libero_10" \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
  --env.type=libero --env.task=libero_10 \
  --output_dir=$SP/msemat_out_softp \
  --steps=200000 --batch_size=32 --num_workers=4 \
  --online_task_ids='[0,1,2,3,4]' --online_steps_per_task=5000 \
  --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
  --wandb.enable=false --job_name=msemat_softp
echo "ALL E42 PROBES DONE"
