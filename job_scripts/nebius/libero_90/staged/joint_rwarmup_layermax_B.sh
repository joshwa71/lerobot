#!/bin/bash
# E50 layer-max warm-up, attempt B (fallback if A fails the gate): compact split —
# expert [9,10,11,12], VLM [13,14,15,16]. Keeps expert mid-stack (closer to the
# action-proximal region E32 measured as load-bearing) and keeps the two certified
# VLM layers (15/16) while still adding two lower ones. Guard: 13 > 12 ok.
set -eo pipefail
export ARM_TAG=layermax_compact_e9to12_v13to16
export EXP_LAYERS='[9,10,11,12]'
export VLM_LAYERS='[13,14,15,16]'
export EXP_N=256 EXP_R=2 EXP_KNN=36
export VLM_N=256 VLM_R=2 VLM_KNN=16
export ROUTER_FAST=true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_rwarmup_common.sh"
