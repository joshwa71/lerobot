#!/bin/bash
# E46 arm 2 (base box): the uniform shape — n128/r4/knn36 on ALL six memory layers,
# both towers. Iso-rank-unit with n256/r2 (16,384 slots x r4 = 65,536 rank-units)
# but 4x fewer, individually stronger slots: the read-write-product concentration
# play. Pre-registered read: expert core50 ~400-800 with famIoU ~0.145 = the bank-
# scaling law holds; famIoU up at scaled cores = the 144-slot per-query floor binds.
ARM_TAG=arm2_uniform_n128r4_knn36
EXP_N=128; EXP_R=4; EXP_KNN=36
VLM_N=128; VLM_R=4; VLM_KNN=36
source "$(dirname "$0")/joint_rwarmup_common.sh"
