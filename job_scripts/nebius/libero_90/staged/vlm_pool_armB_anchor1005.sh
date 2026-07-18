#!/bin/bash
# E45 arm B: anchored pool, (a,b)=(1.0,0.5) — keeps a direct (RMS-normalized) state
# channel in the key at measured init cost (inter 0.865/0.917, family 0.922/0.950 vs
# 0.800/0.882 at b=0). Tests whether trainable proj converts the state channel into
# within-task conditionality worth the separation it spends.
ARM_TAG=anchor1005_c0.05_sep5.0
POOL_MODE=anchored
POOL_W='[1.0,0.5]'
C_WEIGHT=0.05
SEP_WEIGHT=5.0
source "$(dirname "$0")/vlm_pool_chain_common.sh"
