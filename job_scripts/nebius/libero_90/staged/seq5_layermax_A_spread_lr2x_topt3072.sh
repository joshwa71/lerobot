#!/bin/bash
# E51 Part-8 arm (nebius3): ATTEMPT-A SPREAD substrate + the composition levers.
#
# The coverage/spacing isolation (Josh's hypothesis: the core variable may be layer
# COVERAGE, not count): attempt A = expert [2,4,6,8] + VLM [10,12,14,16] — 8 modules
# SPREAD over depth 2-16 — vs compact's contiguous [9-12]+[13-16]. Count-matched,
# bank-matched, rank-matched; spacing is the only substrate delta. Levers (lr 2e-3 ->
# 2e-4 + top_t 3072) folded in to match the fold-in running on base, so the head-to-
# head is A+levers vs compact+levers at the frontier config.
#
# Certificate (E51 Part 7 re-audit, AUDIT_BS=16): GATE PASS — expert famIoU 0.163
# (L8) degrading to 0.212 (L2, individually over the 0.20 band; passes on 3-of-4);
# VLM 0.132-0.152 at PARITY with compact while sitting in better anchor geometry
# (E49: separation improves downward; E43 probe-A: lower layers transmit injected
# corrections better). Pre-registered: the certificate streak (5/5) predicts A lands
# BELOW compact's fold-in; the spacing hypothesis predicts at-or-above. Either branch
# settles the variable for the absolute-layer-max design (expert [4-9] + VLM [10-16]).
#
# Chain: A-phase (10k values-only both towers, routers frozen; auto bs32->bs16xacc2
# fallback — A's deep frozen-route forks OOMed the bs32 audit, expect the fallback)
# -> 5-task sequential (beta4 rank protection, top_t 3072, lr2x, bs16xacc2, 50-ep
# final). Comparators: fold-in (compact+levers, lands 23 Jul eve), layermax-plain
# 44.8, comp 46.0.
# Reads: beat the fold-in = spacing wins at the frontier; >= 49.2 crosses the
# multitask-LoRA line; watch e9 (fold-in's t2 block-min 0.0725 vs A's deeper VLM
# span) and e4 (A's VLM L10/12 sit nearest the E49 open-geometry band).
# REQUIRES on this box: outputs/train/libero_90_pi05_jointwarm10k_layermax_e2468_v10121416
# (the retained warm-up checkpoint — rsync from base if absent; the chain errors
# cleanly if missing).
set -eo pipefail
export WARM_RUN=libero_90_pi05_jointwarm10k_layermax_e2468_v10121416
export GRAD_TAG=layermax_e2468_v10121416
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_RUN=libero_10_seq5_jw_layermax_A_e2468_v10121416_beta4_topt3072_lr2x_steps5k
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
