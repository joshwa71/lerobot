#!/bin/bash
# E47 arm 2' (VM1/base box): the concentration branch re-posed at n=192 — uniform
# n192/r4/knn36 BOTH towers, vlm_route_once=false (broadcast loss semantics, same as
# arm1p/arm3p so the three certificates are like-for-like).
#
# Why n192: the bank-scaling law held at n384->n256 (2.25x shrink, famIoU pinned at
# 0.145) and broke at n384->n128 (9x cumulative: expert famIoU +47%, bg +50%). n192
# = 4x cumulative — the unmeasured midpoint. Per-query draw = 144 slots = 0.39% of
# the 36,864-slot bank (n128: 0.88%, n256: 0.22%). And n192/r4 is the CLEAN
# iso-budget concentration arm: 36,864 x r4 = 147,456 rank-units/layer vs n256/r2's
# 131,072 (+12%) — n128/r4 was only half, so the original arm 2 never tested
# concentration at matched budget. (Rank is irrelevant to tonight's audit — values
# are pinned — but it is baked into the checkpoint for the eventual A-phase.)
#
# Pre-registered read: famIoU ~0.145-0.16 with expert core50 ~1,100-1,300 = law holds
# at 4x, concentration branch lives; famIoU ~0.18+ = gradual breakdown, branch closed.
#
# VRAM: broadcast + r4 doubles the slot-gather activations on the never-measured
# broadcast-knn36 term (estimate ~149GB at bs32 > 140.4 usable) -> launch with
# BATCH_SIZE=16 GRAD_ACCUM=2 (same 320k total samples; optimizer-step schedule
# unchanged; in-microbatch contrastive pool 16 vs 32 is covered by the 512 queues —
# footnote, not a confound).
ARM_TAG=arm2p_n192r4_knn36_bcast
EXP_N=192; EXP_R=4; EXP_KNN=36
VLM_N=192; VLM_R=4; VLM_KNN=36
source "$(dirname "$0")/joint_rwarmup_common.sh"
