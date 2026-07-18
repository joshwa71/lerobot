#!/bin/bash
# E44 VLM router warm-up sweep — armB_c0.0125_sep2.0. See vlm_rwarmup_sweep_common.sh for design + gates.
export ARM_TAG=armB_c0.0125_sep2.0 C_WEIGHT=0.0125 SEP_WEIGHT=2.0
source "$(dirname "$0")/vlm_rwarmup_sweep_common.sh"
