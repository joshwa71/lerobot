#!/bin/bash
# E50 top-p arm (VM): arm 1' substrate + lr2x, write budget = top-p 0.9 with floor
# 3072 (k = min(n_read, max(3072, ceil(0.9 * n_read))), cap 16384) — "write ~everything
# you read; diffuse tasks (e9/e6, the two still mask-rotating at 3072) get their 90%".
# SINGLE delta vs the composition frontier (46.0 @ 50ep: lr2x + fixed top_t 3072).
# Reuses the existing arm 1' A-checkpoint (libero_90_pi05_jointA10k_arm1p_vlmknn16).
# Requires the E50 trainer code (tfidf_top_p) - git pull first. Health: config dump
# shows tfidf_top_p: 0.9; "[top_p] mask k/n_read per layer" lines appear in-log.
set -eo pipefail
export WARM_RUN=libero_90_pi05_jointwarm10k_arm1p_vlmknn16_bcast
export GRAD_TAG=arm1p_vlmknn16
export SEQ_TOP_T=3072
export SEQ_TOP_P=0.9
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_RUN=libero_10_seq5_jw_arm1p_vlmknn16_beta4_topp09f3072_lr2x_steps5k
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
