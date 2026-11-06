#!/bin/bash
# E50 layer-max FULL CHAIN (base box), per Josh 21 Jul: warm-up A -> automated gate ->
# (on fail: warm-up B -> gate -> on fail: STOP, drawing board) -> A-phase -> 5-task
# sequential at bs16 x accum2 (8-module VRAM: ~3.2B values, bs32 is ~139GiB marginal).
# Sequential = plain C-config (top_t 1536, 1x LR): the layer axis is measured as a
# SINGLE delta vs arm 1' (40.0 / block-min 0.0940); composition levers are being
# measured on the arm 1' substrate on the VMs in parallel (top-p / lr4x arms).
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE=/home/josh/lerobot/scripts/vla_analysis/gate_layermax.py
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated

WINNER_TAG=""
WINNER_EXP=""
WINNER_VLM=""

echo "=== [chain] attempt A: expert [2,4,6,8] / VLM [10,12,14,16] ==="
bash "$SCRIPT_DIR/joint_rwarmup_layermax_A.sh" || { echo "[chain] warm-up A crashed"; exit 1; }
if python "$GATE" audit_heldout_jointwarm_layermax_e2468_v10121416_10k "2,4,6,8" "10,12,14,16"; then
  WINNER_TAG=layermax_e2468_v10121416; WINNER_EXP="2,4,6,8"; WINNER_VLM="10,12,14,16"
else
  echo "=== [chain] gate A FAILED -> attempt B: expert [9,10,11,12] / VLM [13,14,15,16] ==="
  bash "$SCRIPT_DIR/joint_rwarmup_layermax_B.sh" || { echo "[chain] warm-up B crashed"; exit 1; }
  if python "$GATE" audit_heldout_jointwarm_layermax_compact_e9to12_v13to16_10k "9,10,11,12" "13,14,15,16"; then
    WINNER_TAG=layermax_compact_e9to12_v13to16; WINNER_EXP="9,10,11,12"; WINNER_VLM="13,14,15,16"
  else
    echo "=== [chain] gate B FAILED too - STOPPING (back to the drawing board) ==="
    exit 1
  fi
fi

echo "=== [chain] gate PASSED for $WINNER_TAG - graduating to A-phase + sequential ==="
export WARM_RUN=libero_90_pi05_jointwarm10k_${WINNER_TAG}
export GRAD_TAG=${WINNER_TAG}
export SEQ_BS=16 SEQ_ACCUM=2
export SEQ_RUN=libero_10_seq5_jw_${WINNER_TAG}_beta4_topt1536_steps5k
set -eo pipefail
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
