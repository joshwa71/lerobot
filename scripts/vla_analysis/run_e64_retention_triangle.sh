#!/bin/bash
# E64 (Josh, 18 Aug): POST-HOC 4-SEED ROLLOUT RETENTION TRIANGLE.
#
# Replaces the sequential trainer's in-run boundary evals (20 eps, ONE seed, serial
# bs=1, ~0.8 min/episode — the instrument retired from decisions in E41) with the
# standing headline instrument applied at EVERY boundary: 25 eps x 4 paired seeds
# (1000/2000/3000/4000), vec-batched at bs=13, one policy load per checkpoint.
#
# For a 10-task sequential run this is the lower triangle: after block k, evaluate
# the k tasks seen so far -> 1+2+...+10 = 55 cells x 100 episodes = 5,500 episodes.
# Measured batched throughput ~200 eps/h (the E63 campaign: 1,000 eps in 4.65h), so
# ~28h per model; the serial in-run evals it replaces cost ~19h, i.e. the upgrade
# from 1-seed/20-ep to 4-seed/25-ep cells costs ~+3.5h net on the naive run.
#
# Usage:  run_e64_retention_triangle.sh naive|merged6x2  [BLOCKS]
#   BLOCKS defaults to "1 2 3 4 5 6 7 8 9 10" (env override for subsets/reruns).
# Outputs: outputs/analysis/e60/seeds_tri_<tag>_b<k>.json   (same schema as every
#   other campaign row: {tag, seeds, results:{env:{seed:{pc, successes[]}}}}).
#   Row k covers envs = the first k of the train order.
# Skip-guarded per cell-row; safe to relaunch after a preemption.
set -o pipefail
ROOT=/home/josh/lerobot
MODEL=${1:?usage: run_e64_retention_triangle.sh naive|merged6x2 [BLOCKS]}
BLOCKS=${2:-"1 2 3 4 5 6 7 8 9 10"}

source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
# train order (dataset task_index 0..9 -> env id), the order the blocks were trained in
ENVS=(4 6 9 2 7 0 8 1 3 5)

case "$MODEL" in
  naive)
    RUN=$ROOT/outputs/train/libero_10_seq10_naive_lora_r512_a128_steps5k
    TAG=naive10_r512; EXTRA="--policy.use_peft=true" ;;
  merged6x2)
    RUN=$ROOT/outputs/train/libero_10_seq10_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
    TAG=merged6x2_10task; EXTRA="" ;;
  *) echo "unknown model '$MODEL' (expected naive|merged6x2)"; exit 2 ;;
esac
[ -d "$RUN" ] || { echo "[tri:$TAG] FATAL: run dir missing: $RUN"; exit 1; }

# Row 10 (final checkpoint, all ten envs) is measurement-identical to a plain all-10
# final campaign, so if one already exists on disk, adopt it instead of spending ~5 h
# re-measuring the same cell:
#   merged6x2 -> seeds_seq10_merged6x2.json (the E63 row, 65.1)
#   naive     -> seeds_naive10_r512_final.json (written by the queue's stage-3 camp
#                call if the running bash is executing the pre-triangle revision of
#                the queue script — see E64 add-3)
case "$MODEL" in
  merged6x2) FINAL_EXISTING=$ROOT/outputs/analysis/e60/seeds_seq10_merged6x2.json ;;
  naive)     FINAL_EXISTING=$ROOT/outputs/analysis/e60/seeds_naive10_r512_final.json ;;
esac
B10=$ROOT/outputs/analysis/e60/seeds_tri_${TAG}_b10.json
if [ -f "$FINAL_EXISTING" ] && [ ! -f "$B10" ]; then
  cp "$FINAL_EXISTING" "$B10"
  echo "[tri:$TAG] b10 <- $(basename $FINAL_EXISTING) (identical checkpoint/seeds/episodes; not re-run)"
fi

echo "=== E64 retention triangle: $TAG ($(date -u)) ==="
for K in $BLOCKS; do
  CKPT=$(printf "%06d" $((K*5000)))
  POL="$RUN/checkpoints/$CKPT/pretrained_model"
  OUT=$ROOT/outputs/analysis/e60/seeds_tri_${TAG}_b${K}.json
  [ -f "$OUT" ] && { echo "[tri:$TAG] b$K exists - skipping."; continue; }
  [ -d "$POL" ] || { echo "[tri:$TAG] b$K: checkpoint $CKPT missing - skipping."; continue; }
  IDS=$(IFS=,; echo "[${ENVS[*]:0:$K}]")
  echo "[tri:$TAG] b$K ckpt=$CKPT envs=$IDS ($K x 4 seeds x 25 eps) $(date -u)"
  CAMP_SEEDS="1000,2000,3000,4000" CAMP_TAG=tri_${TAG}_b${K} CAMP_OUT=$OUT \
  python scripts/vla_analysis/eval_seeds_campaign.py \
    --policy.path="$POL" \
    --policy.dtype=bfloat16 $EXTRA \
    --env.type=libero --env.task=libero_10 --env.task_ids="$IDS" \
    --rename_map="$RENAME" \
    --eval.batch_size=13 --eval.n_episodes=25 \
    --seed=1000 \
    --output_dir=/tmp/tri_${TAG}_b${K} \
    || echo "[FAIL] triangle $TAG b$K"
done
echo "=== E64 retention triangle $TAG COMPLETE $(date -u) ==="
