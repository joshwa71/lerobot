#!/bin/bash
# E63 post-landing queue (Josh, 17 Aug: "kill the noise run and do the 4 seed.
# Then do the battery and then the naive 10 task sequential").
# Stage 1: 4-seed campaign on the 10-task merged6x2 final (ALL TEN envs) — the
#          headline instrument; makes the +4.1-over-oracle and the FT-fresh tie
#          quotable.
# Stage 2: battery — 10x10 MSE forgetting matrix (paired-noise, the noise-free
#          retention read that turns "front-5 ~= back-5" into a claim), jitter
#          grid, slot autopsy at 10-task exposure (5-pair site-bleed + prior-core
#          events incl. the solo E14/E16 depth cells).
# Stage 3: naive sequential LoRA r256 at 10 tasks (the catastrophic-forgetting
#          foil beside the 67.8 row) + its own 4-seed row.
# All stages skip-guarded; failures logged and the queue continues.
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e63_queue.log
exec >> "$LOG" 2>&1
echo "=== E63 queue started $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

SEQ10=$ROOT/outputs/train/libero_10_seq10_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
SP=$ROOT/outputs/analysis/e63
mkdir -p $SP
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
ALL10='[4,6,9,2,7,0,8,1,3,5]'
STEPS10=005000,010000,015000,020000,025000,030000,035000,040000,045000,050000

COMMON_ARGS=(
  --policy.empty_cameras=1 --policy.dtype=bfloat16
  --policy.gradient_checkpointing=false
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
  --dataset.repo_id=libero_10 --dataset.root="$ROOT/outputs/libero_10"
  --rename_map="$RENAME"
  --env.type=libero --env.task=libero_10
  --steps=200000 --batch_size=32 --num_workers=4
  --online_task_ids='[0,1,2,3,4,5,6,7,8,9]' --online_steps_per_task=5000
  --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'
  --wandb.enable=false
)

# ---------------- stage 1: 4-seed campaign ----------------
CAMP_OUT_10=$ROOT/outputs/analysis/e60/seeds_seq10_merged6x2.json
if [ ! -f "$CAMP_OUT_10" ]; then
  echo "[e63] stage 1: 4-seed campaign on the 10-task final $(date -u)"
  CAMP_SEEDS="1000,2000,3000,4000" CAMP_TAG=seq10_merged6x2 CAMP_OUT=$CAMP_OUT_10 \
  python scripts/vla_analysis/eval_seeds_campaign.py \
    --policy.path="$SEQ10/checkpoints/050000/pretrained_model" \
    --policy.dtype=bfloat16 \
    --env.type=libero --env.task=libero_10 --env.task_ids="$ALL10" \
    --rename_map="$RENAME" \
    --eval.batch_size=13 --eval.n_episodes=25 \
    --seed=1000 --output_dir=/tmp/camp_seq10 \
    || echo "[FAIL] stage 1 seeds"
else
  echo "[e63] stage 1: seeds row exists - skipping."
fi

# ---------------- stage 2: battery ----------------
echo "[e63] stage 2: 10x10 MSE forgetting matrix $(date -u)"
if [ ! -f "$SP/mse_matrix_seq10.jsonl" ]; then
  MSEMAT_RUN_DIR=$SEQ10 MSEMAT_STEPS=$STEPS10 MSEMAT_OUT=$SP/mse_matrix_seq10.jsonl \
  MSEMAT_TASKS=0,1,2,3,4,5,6,7,8,9 \
  python scripts/vla_analysis/mse_matrix2.py \
    --policy.path="$SEQ10/checkpoints/005000/pretrained_model" \
    "${COMMON_ARGS[@]}" \
    --output_dir=$SP/msemat_out_seq10 --job_name=msemat_seq10 \
    || echo "[FAIL] stage 2 msemat"
fi
echo "[e63] stage 2: jitter grid $(date -u)"
if [ ! -f "$SP/probe_jitter_seq10.jsonl" ]; then
  PROBE_RUN_DIR=$SEQ10 PROBE_CKPTS="t0:050000,t3:050000,t4:050000,t7:050000" \
  PROBE_OUT=$SP/probe_jitter_seq10.jsonl PROBE_SWAP_SLOTS=1 \
  python scripts/vla_analysis/probe_jitter.py \
    --policy.path="$SEQ10/checkpoints/050000/pretrained_model" \
    "${COMMON_ARGS[@]}" \
    --output_dir=$SP/jitter_out_seq10 --job_name=jitter_seq10 \
    || echo "[FAIL] stage 2 jitter"
fi
echo "[e63] stage 2: slot autopsy $(date -u)"
SLOTS_NTASKS=10 SLOTS_OUT_DIR=$SP SLOTS_TAG=e63 \
  python scripts/vla_analysis/e62_slots.py || echo "[FAIL] stage 2 autopsy"

# ---------------- stage 3: naive 10-task sequential LoRA ----------------
echo "[e63] stage 3: naive seq LoRA r256, 10 tasks $(date -u)"
NAIVE=$ROOT/outputs/train/libero_10_seq10_naive_lora_r256_a64_steps5k
if [ -d "$NAIVE" ] && [ ! -d "$NAIVE/checkpoints" ]; then
  echo "[e63] wiping stub naive dir (no checkpoints)"; rm -rf "$NAIVE"
fi
if [ ! -d "$NAIVE/checkpoints/050000" ]; then
  bash job_scripts/nebius/baselines/naive_seq_lora_r256_10task.sh || echo "[FAIL] stage 3 naive train"
fi
NAIVE_OUT=$ROOT/outputs/analysis/e60/seeds_naive10_final.json
if [ -d "$NAIVE/checkpoints/050000" ] && [ ! -f "$NAIVE_OUT" ]; then
  CAMP_SEEDS="1000,2000,3000,4000" CAMP_TAG=naive10_final CAMP_OUT=$NAIVE_OUT \
  python scripts/vla_analysis/eval_seeds_campaign.py \
    --policy.path="$NAIVE/checkpoints/050000/pretrained_model" \
    --policy.dtype=bfloat16 --policy.use_peft=true \
    --env.type=libero --env.task=libero_10 --env.task_ids="$ALL10" \
    --rename_map="$RENAME" \
    --eval.batch_size=13 --eval.n_episodes=25 \
    --seed=1000 --output_dir=/tmp/camp_naive10 \
    || echo "[FAIL] stage 3 naive seeds"
fi
echo "=== E63 QUEUE COMPLETE $(date -u) ==="
