#!/bin/bash
# E68: 5-task 4-seed ROLLOUT RETENTION TRIANGLE for the ablation arms and the paper-cell control —
# the E64 instrument (run_e64_retention_triangle.sh) on the five development tasks: after block k,
# evaluate the k tasks seen so far, 25 eps x 4 paired seeds (1000/2000/3000/4000), vec bs=13 (ladder
# 13 -> 8 -> 4 on OOM before any output). 1+2+3+4+5 = 15 cells x 100 episodes per arm.
# Usage: run_e68_retention_triangle.sh control|a1|a2|a3 [BLOCKS]     (BLOCKS default "1 2 3 4 5")
# Env:   GATE_FREE_MIB (default 0): wait for that much free VRAM before each row (co-tenancy).
# Outputs: outputs/analysis/e68/seeds_tri_<tag>_b<k>.json (campaign schema). Skip-guarded per row.
# Adoptions (identical checkpoint/seeds/episodes, never re-measured):
#   control b5 <- outputs/analysis/e60/seeds_merged6x2.json (E62 add-3, the 65.2 row)
#   a2 b1      <- control b1 (A2 forks the control at its task-1 boundary: same checkpoint)
set -o pipefail
ROOT=/home/josh/lerobot
MODEL=${1:?usage: run_e68_retention_triangle.sh control|a1|a2|a3 [BLOCKS]}
BLOCKS=${2:-"1 2 3 4 5"}
GATE_FREE_MIB=${GATE_FREE_MIB:-0}
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
ENVS=(4 6 9 2 7)   # dataset task 0..4 -> env id, the training order
case "$MODEL" in
  control) RUN=$ROOT/outputs/train/libero_10_seq5_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k; TAG=e68_control5 ;;
  a1) RUN=$ROOT/outputs/train/libero_10_seq5_jw_e68a1_live_merged6x2_e468101416_v579111315_beta4corefrac_topt3072_lr2x_steps5k; TAG=e68_a1_live ;;
  a2) RUN=$ROOT/outputs/train/libero_10_seq5_jw_e68a2_tfidfonly_merged6x2_e468101416_v579111315_prepass_noprotect_topt3072_lr2x_steps5k; TAG=e68_a2_tfidfonly ;;
  a3) RUN=$ROOT/outputs/train/libero_10_seq5_jw_e68a3_jointprep_lr1e-4_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k; TAG=e68_a3_jointprep ;;
  *) echo "unknown model '$MODEL'"; exit 2 ;;
esac
OUTDIR=$ROOT/outputs/analysis/e68; mkdir -p $OUTDIR
[ -d "$RUN/checkpoints" ] || { echo "[tri:$TAG] no checkpoints yet at $RUN"; exit 0; }
if [ "$MODEL" = control ] && [ -f $ROOT/outputs/analysis/e60/seeds_merged6x2.json ] && [ ! -f $OUTDIR/seeds_tri_${TAG}_b5.json ]; then
  cp $ROOT/outputs/analysis/e60/seeds_merged6x2.json $OUTDIR/seeds_tri_${TAG}_b5.json
  echo "[tri:$TAG] b5 <- e60/seeds_merged6x2.json (identical checkpoint/seeds/episodes; not re-run)"
fi
if [ "$MODEL" = a2 ] && [ -f $OUTDIR/seeds_tri_e68_control5_b1.json ] && [ ! -f $OUTDIR/seeds_tri_${TAG}_b1.json ]; then
  cp $OUTDIR/seeds_tri_e68_control5_b1.json $OUTDIR/seeds_tri_${TAG}_b1.json
  echo "[tri:$TAG] b1 <- control b1 (same checkpoint by construction)"
fi
wait_vram(){ [ "$GATE_FREE_MIB" -gt 0 ] || return 0
  while true; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${free:-0}" -ge "$GATE_FREE_MIB" ] && return 0; sleep 120; done; }
echo "=== E68 retention triangle: $TAG ($(date -u)) blocks: $BLOCKS ==="
for K in $BLOCKS; do
  CKPT=$(printf "%06d" $((K*5000)))
  POL="$RUN/checkpoints/$CKPT/pretrained_model"
  OUT=$OUTDIR/seeds_tri_${TAG}_b${K}.json
  [ -f "$OUT" ] && { echo "[tri:$TAG] b$K exists - skipping."; continue; }
  [ -d "$POL" ] || { echo "[tri:$TAG] b$K: checkpoint $CKPT missing - skipping."; continue; }
  IDS=$(IFS=,; echo "[${ENVS[*]:0:$K}]")
  wait_vram
  echo "[tri:$TAG] b$K ckpt=$CKPT envs=$IDS ($K x 4 seeds x 25 eps) $(date -u)"
  ok=0
  for bs in 13 8 4; do
    CAMP_SEEDS="1000,2000,3000,4000" CAMP_TAG=tri_${TAG}_b${K} CAMP_OUT=$OUT \
    python scripts/vla_analysis/eval_seeds_campaign.py \
      --policy.path="$POL" --policy.dtype=bfloat16 \
      --env.type=libero --env.task=libero_10 --env.task_ids="$IDS" \
      --rename_map="$RENAME" --eval.batch_size=$bs --eval.n_episodes=25 --seed=1000 \
      --output_dir=/tmp/tri_${TAG}_b${K} && { ok=1; break; }
    [ -f "$OUT" ] && { echo "[tri:$TAG] b$K failed AFTER writing output - not VRAM; aborting row."; break; }
    echo "[tri:$TAG] b$K bs=$bs failed before any output (treating as VRAM) - next rung"
  done
  [ "$ok" = 1 ] || echo "[FAIL] triangle $TAG b$K"
done
echo "=== E68 retention triangle $TAG COMPLETE ($(date -u)) ==="
