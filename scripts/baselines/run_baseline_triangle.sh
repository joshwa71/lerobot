#!/bin/bash
# E67: the post-hoc 4-seed ROLLOUT RETENTION TRIANGLE for the external baselines — the E64
# instrument (scripts/vla_analysis/run_e64_retention_triangle.sh) applied to the RETAIN and O-LoRA
# chains: after block k evaluate the k tasks seen so far, 25 eps x 4 paired seeds
# (1000/2000/3000/4000), vec-batched bs=13 (ladder 13 -> 8 -> 4 on OOM before any output).
# Usage: run_baseline_triangle.sh retain|olora [BLOCKS]        (BLOCKS default "1 .. 10")
# Env:   GATE_FREE_MIB (default 0): wait until nvidia-smi reports at least this much free VRAM
#        before each row, so rows can run alongside a training unit without starving it.
# Outputs: outputs/analysis/e67/seeds_tri_<tag>_b<k>.json (campaign schema). Skip-guarded per row.
set -o pipefail
ROOT=/home/josh/lerobot
MODEL=${1:?usage: run_baseline_triangle.sh retain|olora [BLOCKS]}
BLOCKS=${2:-"1 2 3 4 5 6 7 8 9 10"}
GATE_FREE_MIB=${GATE_FREE_MIB:-0}
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
ENVS=(4 6 9 2 7 0 8 1 3 5)   # dataset task 0..9 -> env id, the training order
case "$MODEL" in
  retain) RUN=$ROOT/outputs/train/libero_10_seq10_retain_a05_fullft_steps5k; TAG=retain10_a05; EXTRA="" ;;
  olora)  RUN=$ROOT/outputs/train/libero_10_seq10_olora_r64_a16_lam05_steps5k; TAG=olora10_r64; EXTRA="--policy.use_peft=true" ;;
  *) echo "unknown model '$MODEL'"; exit 2 ;;
esac
OUTDIR=$ROOT/outputs/analysis/e67; mkdir -p $OUTDIR
[ -d "$RUN/checkpoints" ] || exit 0   # chain not started yet: nothing to evaluate
wait_vram(){ # block until enough VRAM is free (0 = no gate)
  [ "$GATE_FREE_MIB" -gt 0 ] || return 0
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${free:-0}" -ge "$GATE_FREE_MIB" ] && return 0
    sleep 120
  done
}
echo "=== E67 retention triangle: $TAG ($(date -u)) blocks: $BLOCKS ==="
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
      --policy.path="$POL" --policy.dtype=bfloat16 $EXTRA \
      --env.type=libero --env.task=libero_10 --env.task_ids="$IDS" \
      --rename_map="$RENAME" --eval.batch_size=$bs --eval.n_episodes=25 --seed=1000 \
      --output_dir=/tmp/tri_${TAG}_b${K} && { ok=1; break; }
    [ -f "$OUT" ] && { echo "[tri:$TAG] b$K failed AFTER writing output - not VRAM; aborting row."; break; }
    echo "[tri:$TAG] b$K bs=$bs failed before any output (treating as VRAM) - next rung"
  done
  [ "$ok" = 1 ] || echo "[FAIL] triangle $TAG b$K"
done
echo "=== E67 retention triangle $TAG COMPLETE ($(date -u)) ==="
