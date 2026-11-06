#!/bin/bash
# E49/1B (base box): graduation chain for the certified image-span warm-up.
#
# SINGLE-DELTA attribution cell vs arm 1' (40.0 final / block-min 0.0940 / chunk grid
# e4 0.112, e6 0.078, e9 0.223, e2 0.164, e7 0.096): identical C-config sequential
# (1x value LR, top_t 1536 — the composition levers are deliberately NOT stacked here;
# they are being measured on the text-only substrate on the VMs in parallel), the only
# delta = the VLM modules also serve the image block (8 pooled region keys/sample).
# Certificate (E49 addendum): img famIoU 0.091/0.128 at effnum 548/665 (state-
# conditional), instr floor BROKEN (0.103/0.134, bg 0.015), state palette L16 0.127
# watch-item, expert reproduced. Downstream flags handled by the common body:
# vlm_route_once=true + router_only_fast=false (E37 overrides — the warm-up ckpt
# carries false/true respectively).
# Reads on landing: t0 chunk vs 0.112 (does image adaptation move e4's function at
# matched optimization); VLM-tower RTO now includes image palettes (autopsy); the
# state-palette L16 watch-item via the post-run audit if warranted.
set -eo pipefail
export WARM_RUN=libero_90_pi05_jointwarm10k_imgspan_g2_n256_vlmknn16_bcast
export GRAD_TAG=imgspan_g2_vlmknn16
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
