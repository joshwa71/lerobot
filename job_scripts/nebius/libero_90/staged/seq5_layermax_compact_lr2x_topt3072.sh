#!/bin/bash
# E51 Part-7 FOLD-IN (base box): compact layer-max substrate + the composition levers.
#
# The board's item-3 branch cell, gate CLEARED: compact layermax (expert [9-12] + VLM
# [13-16], 8 modules) landed 44.8 @ 50 eps at plain C-config vs arm 1's 40.0 — the
# largest matched-config substrate win on record (block-min 0.0528 vs 0.0940, -44%),
# and it CONVERTS (unlike vlmr4/imgspan). This cell folds in the exact levers that took
# arm 1' 40.0 -> 46.0 (comp): value_lr 2e-3 -> 2e-4 + top_t 3072. Sequential-only from
# the existing layermax A-checkpoint; bs16 x accum2 (8-module VRAM, same as the plain
# layermax sequential; top_t only masks gradients post-backward, no VRAM delta).
#
# Pre-registered reads (vs layermax-plain 44.8 = 34/36/44/78/32 and comp 46.0):
#   - block-min mean pushes below ~0.045 (levers on an already-lower floor);
#   - e4 >= 34 held, e9 >= ~40 held (the two real substrate wins must survive
#     amplitude; e9's bleed channels grow with displacement - the lr4x lesson);
#   - e6: watch only (misrank cell; its function is already best-ever 0.0188);
#   - BEAT 46.0 to take the frontier; >= 49.2 crosses the multitask-LoRA line
#     (the "must" target of the recalibrated ladder).
#   - Give-back tripwire: init-mean -> final <= ~-3 (comp was +1.8; lr4x -5.0).
set -eo pipefail
export WARM_RUN=libero_90_pi05_jointwarm10k_layermax_compact_e9to12_v13to16
export GRAD_TAG=layermax_compact_e9to12_v13to16
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_RUN=libero_10_seq5_jw_layermax_compact_e9to12_v13to16_beta4_topt3072_lr2x_steps5k
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
