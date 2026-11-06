#!/bin/bash
# E59 GRADUATION (GATE OVERRIDDEN — Josh's call, 3 Aug 26 eve): A-phase + 5-task
# sequential for the first interleaved substrate (expert [6,8,10,12] + VLM
# [7,9,11,13], n256/r2, anchored w0.40 + sep8 FiLM-free, frozen_prepass).
#
# The automated gate HARD-FAILED on expert L10/L12 famIoU (0.230/0.213 vs the
# <=0.20 ceiling; L8 0.180 took the one grace) — but the gate kept the E44-54
# famIoU-primary emphasis that E56 explicitly inverted ("gate on bg first"):
#   - bg 0.025-0.048 across ALL EIGHT modules = B's winning band (the causal axis
#     per E56; compact+corefrac won 51.6 at bg 0.080);
#   - capacity healthy everywhere (core50 600-1045, min-eff 523-850 — no collapse);
#   - VLM low-layer bet PAID: L7 famIoU 0.101 = best VLM cert in project history,
#     and the E49 depth gradient reproduces in the trained router (0.101 -> 0.161
#     monotone L7 -> L13);
#   - the expert famIoU rise with depth is the anchor-source gradient (per-layer
#     pairing inherits E49's geometry; consistent across B L8 0.192, absmax L9
#     0.174, here L10/L12 0.230/0.213) — lawful recipe behavior at new depths,
#     not a defect;
#   - precedent: absmax failed ITS gate at every expert layer and became the 53.6
#     frontier (E54: "a property of the GATE, not the router family").
# Sharpened e7 read from the override: if e7 lands low HERE (deep famIoU elevated,
# bg clean, corefrac protecting cores), the famIoU story revives with clean
# attribution; if e7 converts, famIoU is confirmed dead as a gate axis and
# bg-first becomes the standing certificate rule.
#
# Pre-registration otherwise per the chain wrapper: beat B 53.2 (pure placement,
# matched 3.2B budget); e7 >= 30; e9 >= ~56; e4 >= 40 AND e2 >= 80; give-back
# >= -3; prior-core events = 0; MSE matrix <= ~+5%; record updt_s (prepass cost).
set -eo pipefail
export HF_HUB_OFFLINE=1

export WARM_RUN=libero_90_pi05_jointwarm10k_interleave_e681012_v791113_anchor040_sep8_prepass
export GRAD_TAG=interleave_e681012_v791113_anchor040_sep8_prepass
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_interleave_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k
export SEQ_LADDER="16:2:false,8:4:false,16:2:true"

A_OUT_GUARD=/home/josh/lerobot/outputs/train/libero_90_pi05_jointA10k_${GRAD_TAG}
if [ -d "$A_OUT_GUARD" ] && [ ! -d "$A_OUT_GUARD/checkpoints/last/pretrained_model" ]; then
  echo "[guard] partial A-phase dir with no completed checkpoint - wiping $A_OUT_GUARD"
  rm -rf "$A_OUT_GUARD"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
