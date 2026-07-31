#!/bin/bash
# E57 ARM 2 — value-input noise, HALF dose (all sigmas x0.5 vs dose1x; p/amp unchanged).
# The dose-response companion to seq5_gradB_vnoise_dose1x.sh (full rationale + reads
# there). Two points on the dose axis discriminate "noise helps but 1x over-regularizes"
# from "noise is inert/harmful" in one overnight queue.
# =====================================================================================
set -eo pipefail
export HF_HUB_OFFLINE=1

export WARM_RUN=libero_90_pi05_jointwarm10k_layermax_A_anchor040_sep8_nofilm_e2468_v10121416
export GRAD_TAG=layermax_A_anchor040_sep8_nofilm_e2468_v10121416
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_vnoise05x_steps5k
export SEQ_LADDER="16:2:false,8:4:false,16:2:true"
export SEQ_EXTRA_ARGS="--policy.memory_layer.value_input_noise_p=0.25 --policy.memory_layer.value_input_noise_sigma=[0.05,0.15,0.375,0.525] --policy.memory_layer.vlm_value_input_noise_sigma=[0.575,0.7,0.825,0.875] --policy.memory_layer.value_input_noise_amp=[0.5,1.5]"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
