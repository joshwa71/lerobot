#!/bin/bash
# E47 graduation: arm 2' (broadcast re-warm, uniform n192/r4/knn36 — the iso-budget
# concentration arm). Watch A-phase VRAM: r4 doubles the slot-gather (est ~130-135GB);
# the bs16 x accum2 fallback in the common body covers an OOM.
WARM_RUN=libero_90_pi05_jointwarm10k_arm2p_n192r4_knn36_bcast
GRAD_TAG=arm2p_n192r4
source "$(dirname "$0")/joint_aphase_seq5_common.sh"
