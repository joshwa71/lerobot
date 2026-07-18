#!/bin/bash
# E44 VLM router warm-up sweep — armA_c0.05_sep5.0. See vlm_rwarmup_sweep_common.sh for design + gates.
export ARM_TAG=armA_c0.05_sep5.0 C_WEIGHT=0.05 SEP_WEIGHT=5.0
source "$(dirname "$0")/vlm_rwarmup_sweep_common.sh"
