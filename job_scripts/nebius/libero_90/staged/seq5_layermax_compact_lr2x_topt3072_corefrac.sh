#!/bin/bash
# E52 protection patch (base box): the fold-in config + rank+corefrac protection.
#
# The fold-in (layermax compact + lr2x + top_t 3072) landed 43.6 with a -10.6 give-back
# concentrated on e4 (55->18) and e9 (55->30) — the E52 battery showed it produced the
# best per-task functions ever measured (all five own-block chunks best-ever by ~2x; e4
# 0.0333 / e9 0.0784 CROSSED their rollout-conversion thresholds for the first time),
# then later-task write drift at 2x LR pushed both back across: first materially rising
# MSE diagonal of the stationary era (e4 +22.6%), e4 V16 core drift 55% (dose ladder:
# plain 36% -> 34, comp ~44% -> 34, foldin 55% -> 18, top-p 59-69% -> 0 — cliff between
# 44 and 55%), e9's damage shoulder-borne (core 8-16%, shoulder 25-33%) replicating
# comp's e2-block crash. Rank+peak protection was inert (u ~0.035 at core boundary;
# 0.1-1.4M writer events into victim cores per module).
#
# THE PATCH (single delta vs the fold-in): protect_u_norm peak -> corefrac, mode stays
# rank. Budget-v3 measured corefrac's exclusion: whole prior cores at u=1 -> score
# tfidf*(1-u)^4 = 0 -> ZERO events into prior cores at every module; shoulder graded
# (u 0.2-1 -> 2.4-600x rank discount) attacks e9's channel too. Writer cost priced from
# this run's JSONs: only 1-5% of each writer's update events land on prior cores ->
# rank-mode relocation (candidacy-only, full LR) is near-free — avoids budget mode's
# proportional-speed writer tax (+13-17% block-mins) that cost budget-v3 the frontier.
#
# Pre-registered reads:
#   - later-task events into e4/e9 cores = 0 at all 8 modules (the mechanism check);
#   - e4 V16 core drift <= ~20% (was 55%); e4 final >= 34 with init ~50s; e9 >= 40;
#   - e6 >= ~50 and t1 block-min <= ~0.020 (writer-tax tripwire: e6 shares mug content
#     with e4's now-excluded core; fold-in t1 block-min was 0.0156);
#   - block-min mean <= ~0.045 (fit preserved; fold-in 0.0409);
#   - give-back init-mean -> final >= ~-3 (fold-in -10.6);
#   - BEAT 46.0 to take the frontier; >= 49.2 crosses multitask-LoRA (the "must" target).
# Reserve levers if the shoulder still leaks (e9 V-shoulder drift > ~15%): beta 4->8,
# protect_hard_u=0.9, budget mode (accepts the speed tax).
set -eo pipefail
export WARM_RUN=libero_90_pi05_jointwarm10k_layermax_compact_e9to12_v13to16
export GRAD_TAG=layermax_compact_e9to12_v13to16
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_layermax_compact_e9to12_v13to16_beta4corefrac_topt3072_lr2x_steps5k
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
