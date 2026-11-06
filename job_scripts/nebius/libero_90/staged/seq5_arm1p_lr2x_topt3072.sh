#!/bin/bash
# E48 composition arm (VM3): arm 1' substrate + the two proven amplitude/coverage levers.
#
# Sequential-ONLY: reuses arm 1's EXISTING A-phase checkpoint
# (libero_90_pi05_jointA10k_arm1p_vlmknn16 — rsync it to this box first; stage A is
# skipped by the common body's guard, and the warm-up ckpt is not required when A exists).
# Deltas vs the arm-1' graduation sequential (40.0 final / block-min 0.0940):
#   memory_value_lr 1e-3->1e-4  ==>  2e-3->2e-4   (lr2x)
#   tfidf_top_t     1536        ==>  3072          (coverage)
# Precedent on the stageB substrate: these two levers took 32.0 -> 40.4 (block-min
# 0.1274 -> 0.0969). Pre-registered read: block-min toward ~0.070-0.080; e6/e2/e7
# finals up (near-threshold tasks); e4/e9 expected pinned (thresholds ~0.03-0.07);
# watch own->final chunk on t0 for amplitude-induced give-back (arm 1' baseline was
# uniformly IMPROVING: -2.3..-5.0%). Steps stay 5000/task (E41: staged blocks converge
# by ~2.5-3k; 2x LR moves the knee earlier — 7k retired).
set -eo pipefail
export WARM_RUN=libero_90_pi05_jointwarm10k_arm1p_n256r2_vlmknn16_bcast
export GRAD_TAG=arm1p_vlmknn16
export SEQ_RUN=libero_10_seq5_jw_arm1p_vlmknn16_beta4_topt3072_lr2x_steps5k
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
