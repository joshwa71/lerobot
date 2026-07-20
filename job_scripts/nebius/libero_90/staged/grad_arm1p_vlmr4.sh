#!/bin/bash
# E48 VLM-rank-4 arm (VM2): arm 1's certified router at vlm_lora_rank=4.
#
# Rank only changes the VLM slot tensor shapes; the warm-up trains keys/proj ONLY and
# under frozen-route the routing certificate is a function of backbone+keys+proj — so
# the r2 warm-up transfers without re-warming. torch raises on shape-mismatched tensors
# even at strict=False, so the graft step rewrites the checkpoint: drops the 4 r2-shaped
# VLM slot tensors (fresh r4 init fills them; slot_up zero => memory output 0 at start)
# and sets vlm_lora_rank=4 in its config (self-describing artifact, E37 rule).
# Smoked 20 Jul: 4 tensors fresh, 34 router tensors bit-identical, 48/895 trainable.
#
# VRAM: +16GiB value quadruple over r2 (~135GiB at bs32 vs ~137-139 usable) => A-phase
# keeps the common body's bs32->bs16xacc2 fallback; the SEQUENTIAL is forced to
# bs16xacc2 up front (evals spike mid-run and the try-fallback would lose hours;
# measured accum overhead +12% wall-clock).
# Pre-registered read: the palette is the always-read block — r4 doubles its per-slot
# expressivity (a 64-slot palette at r4 ~ doubling the task's always-on adapter rank).
# Fit read at t0 chunk vs arm 1's 0.1119; joint-era rank precedent (+2pp, E33) predates
# the compass — this is the first rank test where fit provably lives on the boosted tower.
set -eo pipefail
ROOT_DIR=/home/josh/lerobot
SRC_WARM=libero_90_pi05_jointwarm10k_arm1p_n256r2_vlmknn16_bcast
export WARM_RUN=libero_90_pi05_jointwarm10k_arm1p_vlmknn16_r4graft
export GRAD_TAG=arm1p_vlmr4
export SEQ_RUN=libero_10_seq5_jw_arm1p_vlmr4_beta4_topt1536_steps5k
export SEQ_BS=16
export SEQ_ACCUM=2

GRAFT_DST="$ROOT_DIR/outputs/train/$WARM_RUN/checkpoints/last/pretrained_model"
if [ ! -d "$GRAFT_DST" ]; then
  SRC="$ROOT_DIR/outputs/train/$SRC_WARM/checkpoints/last/pretrained_model"
  [ -d "$SRC" ] || { echo "ERROR: source warm-up missing: $SRC"; exit 1; }
  source /home/josh/miniforge3/etc/profile.d/conda.sh
  conda activate lerobot-memory-updated
  python "$ROOT_DIR/scripts/vla_analysis/graft_vlm_rank.py" --src "$SRC" --dst "$GRAFT_DST" --vlm-rank 4
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
