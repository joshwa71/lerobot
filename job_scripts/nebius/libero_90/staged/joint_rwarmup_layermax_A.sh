#!/bin/bash
# E50 layer-max warm-up, attempt A (base box): expert memory moved LOW ([2,4,6,8]),
# VLM text-field memory x4 layers ([10,12,14,16]) — parameter-maxxing the tower the
# compass says carries fit, at the depths the E49 probe says have the best anchor
# geometry (composite b=0 inter: L7 0.722 -> L16 0.898, monotone in depth).
# Placement guard satisfied with existing code (all VLM layers > expert max 8), so
# expert prefix-KV stationarity + frozen-route need nothing new (policy smoke: 8/8
# modules attach, keys grads live at all of them). Known risk: expert routing at
# L2/L4 (feat-probe separability L4 89.5% vs the 98% plateau at L8+) — that is what
# the automated gate is for (gate_layermax.py; fallback = attempt B, compact).
set -eo pipefail
export ARM_TAG=layermax_e2468_v10121416
export EXP_LAYERS='[2,4,6,8]'
export VLM_LAYERS='[10,12,14,16]'
export EXP_N=256 EXP_R=2 EXP_KNN=36
export VLM_N=256 VLM_R=2 VLM_KNN=16
export ROUTER_FAST=true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_rwarmup_common.sh"
