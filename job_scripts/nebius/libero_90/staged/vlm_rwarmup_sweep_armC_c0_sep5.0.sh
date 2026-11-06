#!/bin/bash
# E44 VLM router warm-up sweep — armC_c0_sep5.0. See vlm_rwarmup_sweep_common.sh for design + gates.
export ARM_TAG=armC_c0_sep5.0 C_WEIGHT=0 SEP_WEIGHT=5.0
source "$(dirname "$0")/vlm_rwarmup_sweep_common.sh"
