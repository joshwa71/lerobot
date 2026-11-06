#!/bin/bash
# E50 lr-max arm (VM): arm 1' substrate + top_t 3072, value LR 4e-3 -> 2e-4 (E43's
# exact lr4xsched schedule for comparability). SINGLE delta vs the composition
# frontier (46.0 @ 50ep: same substrate/top_t at 2e-3 -> 2e-4). The E43 ladder says
# give-back grows with amplitude (+7.7% at 2x -> +14.5% at 4x on the old substrate);
# this substrate absorbed doubled exposure at 2x with zero cost - the 4x cell decides.
# Reuses the existing arm 1' A-checkpoint. Health: lr:3.92e-03 in-log at peak.
set -eo pipefail
export WARM_RUN=libero_90_pi05_jointwarm10k_arm1p_vlmknn16_bcast
export GRAD_TAG=arm1p_vlmknn16
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.004
export SEQ_VALUE_LR_END=0.0002
export SEQ_RUN=libero_10_seq5_jw_arm1p_vlmknn16_beta4_topt3072_lr4xsched_steps5k
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
