#!/bin/bash
# E47 graduation: arm 1' (broadcast re-warm, VLM knn16 — the incumbent shape).
WARM_RUN=libero_90_pi05_jointwarm10k_arm1p_n256r2_vlmknn16_bcast
GRAD_TAG=arm1p_vlmknn16
source "$(dirname "$0")/joint_aphase_seq5_common.sh"
