#!/bin/bash
# E53 ARM 2 (nebius3): spread substrate + corefrac — the single-delta retention patch
# on the best-fit substrate in the project.
#
# The spread run (expert [2,4,6,8] + VLM [10,12,14,16] + lr2x + top_t 3072, peak-norm
# beta4) landed 41.2 with give-back -14.8 — and the E53 battery showed its own-block
# functions are the best ever measured on ALL FIVE tasks (chunk 0.0272/0.0200/0.0683/
# 0.0309/0.0315; t0 BEATS the dense VLM-LoRA specialist 0.0298; e9 crossed its ~0.07
# threshold at own-block for the first time) with the family's lowest jitter absolutes
# (no brittleness) — then paid the full peak-norm drift tax: MSE diagonal +22.6/+15.9/
# +13.0/+4.3%, chunk give-backs +10-17%, e4 V16 core drift 56%/shoulder 89% (over the
# 44-55% cliff), 9.9M writer events into prior cores. Exactly the channel corefrac
# eliminated on compact (0 events, 0% core drift, zero writer tax, 43.6 -> 51.6).
#
# Single delta vs the 41.2 run: protect_u_norm peak -> corefrac. Reuses the existing
# spread A-checkpoint on this box (stage A auto-skips).
#
# Pre-registered reads (comparators: spread-peak twin 41.2, corefrac-compact 51.6):
#   - core events = 0 at all 8 modules; MSE matrix flat (<= ~+5%); function give-back
#     <= ~5%/task (twin: +10-17%);
#   - BEAT 51.6 = the frontier moves to the spread substrate (its fit lead is ~15-20%
#     across the board; the question is pure conversion);
#   - e7 is the DECISION CELL corefrac cannot help (last task, zero exposure; twin
#     rolled 20 at best-ever function vs compact-corefrac's 36 at worse function). If
#     e7 stays ~20, the spread substrate has a real conversion deficit that caps it
#     regardless of retention — that reads on the substrate, not the protection.
#   - E2 shoulder watch: the leakiest module of the family (bleeds to 34%, two ~548k
#     core-event cells in the twin) — corefrac zeroes its core channel but relocated
#     write pressure lands on the fattest shoulder in the project; expect the residual
#     give-back to sit slightly above compact-corefrac's -5.6 if E2 carries it.
set -eo pipefail
export HF_HUB_OFFLINE=1  # E53: hub 429s surface as bogus "vocabulary corrupted" tokenizer errors; all assets local
export WARM_RUN=libero_90_pi05_jointwarm10k_layermax_e2468_v10121416
export GRAD_TAG=layermax_e2468_v10121416
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_layermax_A_e2468_v10121416_beta4corefrac_topt3072_lr2x_steps5k
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
