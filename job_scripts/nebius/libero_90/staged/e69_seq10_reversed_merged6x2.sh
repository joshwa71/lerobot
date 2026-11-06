#!/bin/bash
# E69 (Josh, 12 Sep 26: "Reversed. DO it."): the PAPER CELL's 10-task sequential under the REVERSED
# task order — the order-robustness check for Table I's headline. Everything = seq10_merged6x2.sh
# (E62 10-task run: merged 6x2, corefrac, lr 2e-3 -> 2e-4, top_t 3072, same A-checkpoint), deltas:
#   * online_task_ids [9,8,...,0] (dataset order reversed; the ds->env map follows the ids);
#   * in-run rollout evals cut to 1 episode per seen task and no 50-ep final eval
#     (SEQ_EXTRA_ARGS overrides; draccus takes the last occurrence) — this run reports its FINAL ROW
#     from the 4-seed campaign only (queue_e69_spot.sh), so the ~15 h of in-run evals the control
#     spent (20 eps x seen tasks per boundary + 500 final eps) buy nothing here.
# Deadline maths: 10 x ~3.4 h train on the H200 at bs16x2 = ~34.5 h.
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
export SEQ_TASK_IDS='[9,8,7,6,5,4,3,2,1,0]'
export SEQ_FINAL_CKPT=050000
export SEQ_RUN=libero_10_seq10rev_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
export SEQ_LADDER="16:2:false,8:4:false,16:2:true"
export SEQ_EXTRA_ARGS="--eval.n_episodes=1 --eval_final_episodes=0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
