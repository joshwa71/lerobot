#!/bin/bash
# E62 10-TASK RUN — merged 6x2, NO noise (the REQUIRED shared-config validation,
# E61 add-5 point 4, + the ICRA catastrophe-elimination demonstration). Config =
# the E62 chain VERBATIM (share (4,6)+(8,10) expert / solo 14,16 / VLM pairs
# shared; corefrac, lr 2e-3->2e-4, top_t 3072), reusing the E62 A-checkpoint;
# deltas = online_task_ids [0..9] + final ckpt 050000. 10 blocks x 5000 steps,
# 20-ep boundaries + 50-ep final on all 10 tasks.
#
# PRE-REGISTERED (E61 add-5): the two shared-config degradation mechanisms —
# (a) cross-writing accumulates per block (early tasks sit under 9 later blocks
#     vs 4; the 5-task matrix band +0-4.2% could grow toward +8-15%);
# (b) protection crowding on shared tables (E61: 36% of a shared table at
#     u>0.5 after 5 tasks; late writers risk starvation — watch late-task
#     block-min losses vs their 5-task levels).
# Catastrophe read: seen-task trajectory vs the pre-stationarity 10-task runs
# (E19: 34.4 final w/ collapses; naive LoRA: 0s). Comparator for front-5 cells:
# the E62 5-task final (58/74/64/84/54 @ seed-1000).
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
export SEQ_RUN=libero_10_seq10_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
export SEQ_LADDER="16:2:false,8:4:false,16:2:true"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
