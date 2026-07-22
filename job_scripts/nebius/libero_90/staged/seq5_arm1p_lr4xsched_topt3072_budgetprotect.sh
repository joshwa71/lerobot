#!/bin/bash
# E51 Part-1 arm (VM): lr4x + BUDGET-CONSERVING protection (Josh's reallocation design).
#
# Motivation: the amplitude dose-response on the arm 1' substrate is an inverted U
# (1x 40.0 / 2x 46.0 / 4x 40.4) - 4x wins every boundary (block-min 0.0651, project-low
# at the time) and gives it back via ~24% larger displacement on an IDENTICAL exposure
# topology. Rank-mode protection cannot touch that channel (candidacy-only; high-TF
# shared cores never rank out; survivors get full LR). grad_scale-mode protects the
# victims but taxes late writers cumulatively (E44: e7 block-min +49% - the union store
# attenuates without reallocating).
#
# protect_mode=budget: ranking stays pure TF-IDF; each slot consumes (1-u)^beta of a
# fixed budget B = min(top_t, n_read) full-LR slot-equivalents and receives that scaled
# LR (post-step momentum-aware blend); selection walks DOWN the ranking until B is
# spent. Total effective plasticity == B for every writer regardless of union size:
# protection is pure reallocation (deflected budget rolls into unprotected slots).
# u-norm corefrac (whole prior cores at u~1). Smoked S17a-g (identity/deep-reach/
# conservation-exact/edge/veto/regressions).
#
# SINGLE delta vs the lr4x arm (40.4): protect_mode rank->budget + u_norm
# peak->corefrac. Pre-registered: t0-t2 block-mins ~= 0.0651 (early writers untaxed by
# construction); e9 final back toward >=26 (comp's level; lr4x gave 18); e7 block-min
# <= ~0.085 (the E44 starvation tripwire - budget mode should hold it at ~0.075);
# beat 46.0 to displace the composition frontier.
# Health: config dump shows protect_mode: budget, protect_u_norm: corefrac,
# value_lr peak 3.92e-3; "[chain]"-era boundary logs "Updated prior-usefulness
# protection store after task N" each block.
set -eo pipefail
export WARM_RUN=libero_90_pi05_jointwarm10k_arm1p_vlmknn16_bcast
export GRAD_TAG=arm1p_vlmknn16
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.004
export SEQ_VALUE_LR_END=0.0002
export SEQ_PROTECT_MODE=budget
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_arm1p_vlmknn16_beta4_topt3072_lr4xsched_budgetcf_steps5k
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
