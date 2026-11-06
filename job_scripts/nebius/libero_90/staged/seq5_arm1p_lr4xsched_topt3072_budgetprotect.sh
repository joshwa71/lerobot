#!/bin/bash
# E51 Part-4 arm (VM): lr4x + BUDGET protection v3 (Josh's conserved proportional allocation).
#
# Motivation: the amplitude dose-response on the arm 1' substrate is an inverted U
# (1x 40.0 / 2x 46.0 / 4x 40.4) - 4x wins every boundary (block-min 0.0651, project-low
# at the time) and gives it back via ~24% larger displacement on an IDENTICAL exposure
# topology. Rank-mode protection cannot touch that channel (candidacy-only; high-TF
# shared cores never rank out; survivors get full LR). grad_scale-mode protects the
# victims but taxes late writers cumulatively (E44: e7 block-min +49% - the union store
# attenuates without reallocating).
#
# HISTORY: v2 (deduct + water-fill refund) launched 22 Jul and NaN'd ~200 steps after
# the first task boundary - the E42 momentum-aware blend multiplied exp_avg by the
# BOOST scale (2.0) every in-mask step => ~1.8x/step compounding => overflow (E51
# Part 4). Its refund also had a u==0 eligibility cliff that stranded most of the
# deficit (prior tasks' binary read tails put u>0 on ~97% of later masks). v3 replaces
# the whole deduct/refund structure.
#
# protect_mode=budget v3: ONE score = tfidf * (1-u)^beta drives membership AND speed.
# Mask = top-t by that score (selection rule identical to rank mode => masks bitwise
# match the lr4x twin, whose u store evolves identically under the frozen router).
# Per-slot LR scale_i = min(2.0, lam*score_i) with lam solved exactly so
# sum(scale) == mask size (conservation exact by construction - no deficit, refund,
# eligibility set, or unspent remainder exists). u=1 slots score 0 -> never selected
# -> frozen, seats go to clean slots; below-average-score slots donate LR to
# above-average ones continuously. Momentum: exp_avg *= min(scale, 1) - attenuation
# damps the tail (E42 semantics unchanged), boost acts on the delta only (== per-row
# 2x-LR Adam, bounded; the v2 NaN is structurally impossible). Smoked S18a-f + S19a-g
# (lam-solver hand cases, exact conservation on heavy tails, twin-mask identity, u=1
# exclusion, 300-step boosted-row == 2x-LR Adam, v2 divergence reproduced).
#
# Delta vs the lr4x arm (40.4): protect_mode rank->budget + u_norm peak->corefrac.
# NOTE this is NOT a bitwise-t0 twin: with an empty store score == tfidf, so t0 runs
# TF-proportional allocation (hot slots ~2x, mask tail <1x) where lr4x ran flat.
# TRIPWIRES (kill the run, not the box):
#   [T0, ~2h in]  t0 block-min > ~0.075 (lr4x twin: 0.0651) => proportional
#                 allocation hurts a clean writer => kill, report.
#   [NaN watch]   loss/grdn must stay finite past the first boundary (~step 5.2k) -
#                 the v2 failure signature. Any nan => kill immediately, report.
# Pre-registered reads: e9 final back toward >=26 (comp's level; lr4x gave 18);
# e7 block-min <= ~0.085 (the E44 last-writer starvation tripwire; conserved
# allocation should hold ~0.075); beat 46.0 to displace the composition frontier.
# Health: config dump shows protect_mode: budget, protect_u_norm: corefrac, value_lr
# peak 3.92e-3; boundary logs "Updated prior-usefulness protection store after task N"
# each block; periodic "[budget] <layer>: capped m/k med-scale ... sum 3072" lines
# (sum must always read 3072 - conservation in production).
set -eo pipefail
export WARM_RUN=libero_90_pi05_jointwarm10k_arm1p_vlmknn16_bcast
export GRAD_TAG=arm1p_vlmknn16
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.004
export SEQ_VALUE_LR_END=0.0002
export SEQ_PROTECT_MODE=budget
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_arm1p_vlmknn16_beta4_topt3072_lr4xsched_budgetprop_steps5k
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
