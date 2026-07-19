#!/bin/bash
# E46 arm 3 (VM3): the knn axis — identical to arm 1 except VLM knn 16 -> 36
# (retrieval breadth = per-task palette capacity 64 -> 144 slots), with the keys
# TRAINED at topk 36 (the alignment arm 3 previously lacked). Route-once makes
# knn36 VRAM-cheap (measured 126.9GB A-phase).
ARM_TAG=arm3_n256r2_vlmknn36
EXP_N=256; EXP_R=2; EXP_KNN=36
VLM_N=256; VLM_R=2; VLM_KNN=36
source "$(dirname "$0")/joint_rwarmup_common.sh"
