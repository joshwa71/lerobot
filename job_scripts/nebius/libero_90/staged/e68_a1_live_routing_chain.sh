#!/bin/bash
# E68 ablation A1 — LIVE ADDRESSING on the paper cell (merged 6x2: expert [4,6,8,10,14,16] with
# (4,6),(8,10) shared; VLM [5,7,9,11,13,15] all pairs shared; 12 sites / 7 tables).
# SINGLE DELTA from the E62 chain: routers, gates and expert anchors read the LIVE stream
#   use_frozen_base_input_features=false  frozen_prepass=false  allow_nonstationary_routing=true
# from value fill onward. The warm-up checkpoint is SHARED with the paper cell (values are zero at
# warm-up, so live == memory-free features there); stage A (value fill) and stage B (5-task
# sequential) both run live. Everything else C-config verbatim (top_t 3072, corefrac beta4, lr2x,
# 5k steps/task, bs eff 32). Replaces Table II B (IoU-only). Pre-registration: research_log E68.
# H100 80GB: ladders start at bs8xacc4 (E53: no-ckpt small micro-batch beats ckpt on a frozen
# backbone). Stage A saves every 5k steps and resumes via lerobot-train --resume (CLAUDE.md 9.4).
set -eo pipefail
export HF_HUB_OFFLINE=1
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "E68-A1 live-addressing chain started on $(hostname) at $(date -u)"

export WARM_RUN=libero_90_pi05_jointwarm10k_merged6x2_e468101416_v579111315_anchor040_sep8_prepass
export GRAD_TAG=e68a1_live_merged6x2_e468101416_v579111315_anchor040_sep8
LIVE_ARGS="--policy.memory_layer.frozen_prepass=false --policy.memory_layer.allow_nonstationary_routing=true"

# stage A (value fill, LIVE)
export A_FROZEN_ROUTE=false
export A_EXTRA_ARGS="$LIVE_ARGS"
export A_SAVE_FREQ=${A_SAVE_FREQ:-5000}
export A_LADDER="${A_LADDER:-8:4:false,4:8:false,8:4:true,4:8:true}"

# stage B (5-task sequential, LIVE) — C-config levers verbatim from the E62 chain
export SEQ_FROZEN_ROUTE=false
export SEQ_EXTRA_ARGS="$LIVE_ARGS"
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_e68a1_live_merged6x2_e468101416_v579111315_beta4corefrac_topt3072_lr2x_steps5k
export SEQ_LADDER="${SEQ_LADDER:-8:4:false,4:8:false,8:4:true,4:8:true}"

source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
echo "E68-A1 live-addressing chain COMPLETE at $(date -u)"
