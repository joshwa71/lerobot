#!/bin/bash
# E47 arm 3' (VM3): the knn axis — expert n256/r2/knn36, VLM n256/r2/knn36 — re-warmed
# with vlm_route_once=false (broadcast loss semantics; see arm1p header). With the
# palette's spreading force restored, this cleanly answers the original E46 question:
# does the 64->144-slot per-draw palette capacity pay, at matched loss semantics?
# Read against arm1p: same expert certificate expected; VLM palette famIoU/effnum are
# the comparison (E46's deduped arms had palette size pinned at ~2 draws, which
# confounded the knn comparison).
ARM_TAG=arm3p_n256r2_vlmknn36_bcast
EXP_N=256; EXP_R=2; EXP_KNN=36
VLM_N=256; VLM_R=2; VLM_KNN=36
source "$(dirname "$0")/joint_rwarmup_common.sh"
