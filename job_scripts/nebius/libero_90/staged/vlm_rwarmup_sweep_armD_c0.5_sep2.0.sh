#!/bin/bash
# E44 VLM router warm-up sweep — armD (REPLACES armC, which collapsed in <1k steps:
# top-1 share 0.7, support ~300 — in the aux-only warm-up regime sep's optimum with no
# counter-force is one-private-slot-per-task; standard SupCon's same-task-denominator
# repulsion is the ONLY anti-collapse force, so c=0 is structurally degenerate).
# armD probes the HIGH-c side: 10x anti-collapse, 2.5x less of the collapse driver.
# With armB (c0.0125/sep2) this gives a clean 40x c-contrast at fixed sep=2; armA holds
# (0.05, 5). Failure mode to watch = the OPPOSITE (sprawl: support >=8-10k, famIoU >=0.4,
# the P7 signature). See vlm_rwarmup_sweep_common.sh for design + gates.
export ARM_TAG=armD_c0.5_sep2.0 C_WEIGHT=0.5 SEP_WEIGHT=2.0
source "$(dirname "$0")/vlm_rwarmup_sweep_common.sh"
