#!/bin/bash
# E47 arm 1' (VM2): the incumbent shape — expert n256/r2/knn36, VLM n256/r2/knn16 —
# re-warmed with vlm_route_once=false (broadcast loss semantics: the palette counts
# once per served position, as in every pre-E46 warm-up). This is the E45 poolB
# recipe's exact replica plus the E46 protocol improvements (frozen-route inputs,
# per-tower topk alignment). Expected: VLM famIoU back at ~0.149/0.147 with palette
# famIoU ~0.08 and palette effnum ~600-800; expert certificate unchanged (~0.145-0.17).
ARM_TAG=arm1p_n256r2_vlmknn16_bcast
EXP_N=256; EXP_R=2; EXP_KNN=36
VLM_N=256; VLM_R=2; VLM_KNN=16
source "$(dirname "$0")/joint_rwarmup_common.sh"
