#!/bin/bash
# E49/1A image-span warm-up (base box): arm 1' recipe + pooled image-region routing.
#
# Single conceptual delta vs the certified arm 1' warm-up (expert n256/r2/knn36,
# VLM n256/r2/knn16, anchored (1.0,0.5), broadcast losses): the VLM modules ALSO
# serve the image block — 2x2 spatial regions per ACTIVE camera (2 real cams -> 8
# region keys/sample, 64 positions each), key = rms_nrm(1.0*nrm(instr pool) +
# 0.5*nrm(region pool)) rescaled to the language-field token RMS. Empty-camera slots
# excluded via img_masks. Design frozen from the E49 querystats-image probe:
# g2 ~= g1 on separation with 4x the palette conditionality per camera; g4 buys
# nothing more; patch-level between/within 0.06-0.22 = the state-digit sprawl band,
# so per-token image routing stays rejected. NOTE step-1 discipline: layers stay
# [15,16] (the expert placement guard) although the probe shows anchor separation
# is best LOW in the stack (L7 inter 0.722 -> L16 0.898) — that upside is step 2's.
#
# Broadcast semantics at TRUE deployment mass (no per-region loss normalization —
# image keys carry ~90% of served positions and therefore of the loss mass; watched,
# not pre-engineered). router_only_fast=true makes the 571-row literal broadcast
# affordable (exact value-path skip at pinned-zero values; smoked bitwise).
# DOWNSTREAM (E37 rule): A-phase/sequential MUST override router_only_fast=false
# (already added to joint_aphase_seq5_common.sh) — values get no gradient under the
# skip. Chain: warm-up 10k -> held-out audit -> vlm/expert analyses -> region-split
# sub-span probe (route-once-aware row mapping) -> STOP for review.
#
# Review gates (vs arm 1' certificate 0.136/0.156 audit famIoU, palette 0.074/0.084
# effnum 685/851; instruction floor ~0.20): PASS = image-region famIoU <= ~0.25 with
# per-region effnum >= ~300 (no ~2-draw collapse) AND state palette/instr regions
# within ~20% of arm 1' (the new keys must not degrade the certified regions) AND
# expert certificate reproduced (~0.145-0.17). famIoU topline will be image-weighted
# — read the sub-span region table, not the topline alone.
set -eo pipefail
export ARM_TAG=imgspan_g2_n256_vlmknn16_bcast
export EXP_N=256 EXP_R=2 EXP_KNN=36
export VLM_N=256 VLM_R=2 VLM_KNN=16
export IMG_REGIONS=2
export IMG_POOL_W='[1.0,0.5]'
export ROUTER_FAST=true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_rwarmup_common.sh"
