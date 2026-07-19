#!/bin/bash
# E46 arm 1 (VM2): the incumbent shape — expert n256/r2/knn36, VLM n256/r2/knn16.
# Re-warmed for protocol uniformity: frozen-route-consistent inputs (no bias-residual
# gap), per-tower topk alignment, deduplicated losses. The like-for-like baseline.
ARM_TAG=arm1_incumbent_n256r2_vlmknn16
EXP_N=256; EXP_R=2; EXP_KNN=36
VLM_N=256; VLM_R=2; VLM_KNN=16
source "$(dirname "$0")/joint_rwarmup_common.sh"
