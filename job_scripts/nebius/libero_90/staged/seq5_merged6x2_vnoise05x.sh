#!/bin/bash
# E62 NOISE ARM — value-input noise at the E58-winning HALF dose, RE-CALIBRATED on the
# merged-6x2 layout (E61-add-5 sequencing step 3; Josh 9 Aug: "noise arm on the winner,
# sigmas re-calibrated on the winner's layers").
# =====================================================================================
# DOSE (probe_value_input_calib on the merged6x2 FINAL ckpt, e56 harvest bank, mid
# band; outputs/analysis/e62/value_input_calib_merged6x2.json). E58 convention:
# dose1x sigma = measured mid dim_ratio x2 variance-matched at p=0.25; HALF dose
# (the E58 winner) = measured ratio x1.0:
#   expert [4,6,8,10,14,16]  measured 0.148/0.345/0.504/0.688/0.946/0.982
#     -> sigma [0.15,0.35,0.5,0.7,0.95,1.0]
#   vlm    [5,7,9,11,13,15]  measured 0.406/0.474/0.556/0.638/0.742/0.829
#     -> sigma [0.4,0.47,0.56,0.64,0.74,0.83]
#   p=0.25, per-row amp ~ U[0.5,1.5]. Depth-lawful ladder (expert 0.15->1.0) —
#   same monotone shape as B's calibration, now over 12 sites.
#
# Config otherwise = the E62 chain VERBATIM (merged 6x2: expert share (4,6)+(8,10),
# SOLO 14/16; VLM share (5,7)+(9,11)+(13,15); B router, prepass, corefrac,
# lr 2e-3->2e-4, top_t 3072, 5x5000, 50-ep final), reusing the E62 A-checkpoint
# (stage-A skip guard). Single delta = the noise flags.
#
# PRE-REGISTERED READS (the E61-add-5 redundancy question):
#   - Harvest-bank rescore: spec/succ Q4 D vs merged6x2's own 0.332 — noise must
#     ADD on top of sharing's cross-writing to earn its place; Q4 ~unchanged =
#     redundancy CONFIRMED (paper-worthy either way), drop noise from the recipe.
#   - Block-min mse_loss <= ~1.10x the E62 chain's per task (fit-cost guardrail).
#   - MSE matrix stays in the merged band (<= ~+5%; E62 was +0.0-4.2%).
#   - 50-ep final vs 66.8 (seed-1000); e7 vs 54 = the target cell. Only a
#     4-seed row (vs 65.2) decides any recipe change.
# =====================================================================================
set -eo pipefail
export HF_HUB_OFFLINE=1

export WARM_RUN=libero_90_pi05_jointwarm10k_merged6x2_e468101416_v579111315_anchor040_sep8_prepass
export GRAD_TAG=merged6x2_e468101416_v579111315_anchor040_sep8_prepass
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_vnoise05x_steps5k
export SEQ_LADDER="16:2:false,8:4:false,16:2:true"
# E58 noise flags at the recalibrated half dose (no spaces — expanded unquoted in the
# common body's seq stage)
export SEQ_EXTRA_ARGS="--policy.memory_layer.value_input_noise_p=0.25 --policy.memory_layer.value_input_noise_sigma=[0.15,0.35,0.5,0.7,0.95,1.0] --policy.memory_layer.vlm_value_input_noise_sigma=[0.4,0.47,0.56,0.64,0.74,0.83] --policy.memory_layer.value_input_noise_amp=[0.5,1.5]"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
