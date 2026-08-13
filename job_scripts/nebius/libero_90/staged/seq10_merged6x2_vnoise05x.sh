#!/bin/bash
# E62 10-TASK RUN — merged 6x2 WITH value-input noise at the recalibrated half
# dose (the noise-arm recipe, seq5_merged6x2_vnoise05x.sh, extended to 10
# tasks). Single delta vs seq10_merged6x2.sh = the noise flags. Runs SECOND
# (after the no-noise 10-task) per Josh's weekend ordering.
set -eo pipefail
export HF_HUB_OFFLINE=1

export WARM_RUN=libero_90_pi05_jointwarm10k_merged6x2_e468101416_v579111315_anchor040_sep8_prepass
export GRAD_TAG=merged6x2_e468101416_v579111315_anchor040_sep8_prepass
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_PROTECT_UNORM=corefrac
export SEQ_TASK_IDS='[0,1,2,3,4,5,6,7,8,9]'
export SEQ_FINAL_CKPT=050000
export SEQ_RUN=libero_10_seq10_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_vnoise05x_steps5k
export SEQ_LADDER="16:2:false,8:4:false,16:2:true"
export SEQ_EXTRA_ARGS="--policy.memory_layer.value_input_noise_p=0.25 --policy.memory_layer.value_input_noise_sigma=[0.15,0.35,0.5,0.7,0.95,1.0] --policy.memory_layer.vlm_value_input_noise_sigma=[0.4,0.47,0.56,0.64,0.74,0.83] --policy.memory_layer.value_input_noise_amp=[0.5,1.5]"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
