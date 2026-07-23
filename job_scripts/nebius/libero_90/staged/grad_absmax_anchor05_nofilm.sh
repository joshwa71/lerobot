#!/bin/bash
# E52 absmax-anchor GRADUATION (stages 4+5 of joint_anchor05_nofilm_full_chain.sh,
# relaunched manually after the automated gate tripped on E9 famIoU 0.174 vs the
# 0.165 line — a gate-calibration miss, not an arm defect: the certificate is a
# decisive rescue (L4-L7: 0.242/0.195/0.201/0.219 -> 0.136/0.119/0.135/0.148, all
# BELOW compact's 0.140-0.154 band; expert bg 0.09-0.13 -> 0.024-0.038; VLM
# untouched; E8/E9 at un-anchored parity 0.166/0.174, inside every historical
# certificate standard). Josh: "gotcha. do it."
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WARM_RUN=libero_90_pi05_jointwarm10k_absmax_anchor05_nofilm_e4to9_v10to16
export GRAD_TAG=absmax_anchor05_nofilm_e4to9_v10to16
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=8
export SEQ_ACCUM=4
export SEQ_RUN=libero_10_seq5_jw_absmax_anchor05nofilm_beta4_topt3072_lr2x_steps5k
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
