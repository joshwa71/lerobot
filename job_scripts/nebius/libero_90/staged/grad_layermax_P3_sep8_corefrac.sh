#!/bin/bash
# E55 GRADUATION — CELL P3: spread substrate + plain-FiLM sep8 router + corefrac
# ==============================================================================
# P3 = the second router certified in the E54 six-probe batch (B was the first; B ran
# 28 Jul -> 53.2). Certificate (audit_heldout_jointwarm_layermax_sep8_e2468_v10121416_10k):
#   expert famIoU 0.178/0.176/0.177/0.156 (L2/4/6/8; no grace needed)
#   expert bg     0.083-0.097  (~3x B's 0.025-0.037 — the axis under test)
#   expert core50 1383-1717    (~2.4x B's 585-732 — the capacity end)
#   VLM famIoU 0.132-0.154 PASS
# Router: w=0 (no anchor), sep 5->8, FiLM ON (lang_to_query=true, mpnet) — the sep8
# upgrade of the plain spread router that scored 47.6 with corefrac.
#
# WHAT THIS RUN IS FOR: the second point on the core-breadth <-> shoulder dose-response
# at matched substrate/protection/levers (E55 discussion, "sweet spot" frame). B holds
# the small-core/clean-shoulder end at 53.2 with give-back -0.8 and project-best chunks.
# P3 is the big-core/moderate-shoulder end. Pre-registered (E55):
#   - PREDICTION: double regression — fit toward spread-A's level (B beat spread-A's own
#     chunks 16-32% at 1/3 core50; t0 cell protection-free => pure router effect), and
#     give-back toward compact+corefrac's -5.6 (shoulder channel ~3x B's). Lands ~48-51.
#   - P3 > 53.2 => capacity end wins after all; router choice reopens (B still wins on
#     simplicity: no FiLM, no mpnet).
#   - P3 ~48-51 => B's profile confirmed as the active ingredient; dose-response figure
#     for the paper (small-core end 53.2, big-core end P3, plain-sep5 anchor 47.6).
#   - Watch: shoulder bleeds at E2 (leakiest module of the spread family), MSE matrix
#     diag drift (corefrac should hold ~flat regardless), e7 (expected ~20 — substrate).
# Config = B's graduation recipe VERBATIM; the ONLY delta is the warm-up checkpoint.
# ==============================================================================
set -eo pipefail
export HF_HUB_OFFLINE=1  # E53: hub 429s masquerade as tokenizer corruption; assets local

export WARM_RUN=libero_90_pi05_jointwarm10k_layermax_sep8_e2468_v10121416
export GRAD_TAG=layermax_sep8_e2468_v10121416
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_layermax_sep8_beta4corefrac_topt3072_lr2x_steps5k
export SEQ_LADDER="16:2:false,8:4:false,16:2:true"

# Partial-A guard (see B's wrapper): wipe a stub A-phase dir so a relaunch cannot
# silently demote the batch rung.
A_OUT_GUARD=/home/josh/lerobot/outputs/train/libero_90_pi05_jointA10k_${GRAD_TAG}
if [ -d "$A_OUT_GUARD" ] && [ ! -d "$A_OUT_GUARD/checkpoints/last/pretrained_model" ]; then
  echo "[guard] partial A-phase dir with no completed checkpoint - wiping $A_OUT_GUARD"
  rm -rf "$A_OUT_GUARD"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
