#!/bin/bash
# E60 RESUME (6 Aug 26, disk-full incident): the 015000 boundary save hit ENOSPC
# (SafetensorError) and killed the chain mid-e9-block. Freed ~300G (training_state
# of the two completed vnoise arms — pretrained_model/sequential_state kept), wiped
# the partial 015000. This wrapper re-enters stage B only (gate already passed;
# warm-up/audit/A-phase complete) via joint_aphase_seq5_common.sh's auto-resume
# (last -> 010000, --resume_sequential): the e9 block re-runs from its start.
# SEQ_LADDER starts at the KNOWN-GOOD rung (bs8 x acc4, no ckpt — the settled rung)
# because a bs16 rung-1 OOM after 005000 exists would abort loudly by design.
set -eo pipefail
export HF_HUB_OFFLINE=1

export WARM_RUN=libero_90_pi05_jointwarm10k_bigsearch_e4to16_v5to13_anchor040_sep8_prepass
export GRAD_TAG=bigsearch_e4to16_v5to13_anchor040_sep8_prepass
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=8
export SEQ_ACCUM=4
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_bigsearch_e4to16_v5to13_prepass_beta4corefrac_topt3072_lr2x_steps5k
export SEQ_LADDER="8:4:false,16:2:true"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
