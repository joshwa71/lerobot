#!/bin/bash
# E51 Part-6 arm (nebius4): CLAMPED MASS-BASED TOP-P on the composition frontier.
#
# Single delta vs comp (46.0 = arm 1' + lr2x + static top_t 3072): the write budget
# becomes k = min(n_read, max(3072, k_p), 5120) where k_p covers 90% of the batch's
# ranking-score MASS (the colleague's nucleus rule; the E50 count-of-unique-slots rule
# measured the ~90%-binary-incidental tail and died at 19.2 — E51 Part 2). Ranking
# score unchanged (TF-IDF x (1-u)^beta rank discount); floor = the measured-best
# static; cap = 1.67x floor, bounding both E50 kill channels (core overwrite + tail
# churn scale with mask size) to ~1.7x comp's known-benign leak.
#
# Design frozen from measurement (E51 Part 6): comp block-aggregate k90 per module =
# 8.6-21.9k with the diagnostic ordering CONFIRMED (expert tower: e9 ~2x everyone;
# VLM tower FLIPS: e4 most diffuse - the adaptive rule hands e4 extra VLM budget,
# aligned with its perception-side deficit). Per-batch k90 ~0.2-0.3x block => working
# band ~2.5-6.5k => [3072, 5120] clip. NB if per-batch mass tails are fat, k pins at
# the cap everywhere and the arm degenerates to static top_t=5120 - still the band
# test both sides asked for; the in-run "[top_p] mask k/n_read per layer" lines tell
# us which regime we're in within the first hour (floor-pinned / in-band / cap-pinned
# per module per task).
#
# Pre-registered reads (vs comp 46.0 = 34/60/26/84/26):
#   - e9/e6 self-coverage UP in the block JSONs (the rotation the rule targets);
#   - e9 >= 26 and e6 >= 60 held or improved; e4 >= ~34 (its VLM-side k engages);
#   - beat 46.0 to displace the frontier; >= ~44 = band useful, adaptivity TBD by the
#     k-regime read; <= ~40 = the coverage axis is closed at every dose that's safe.
#   - KILL line: any top-p-shaped early cliff (e4 collapsing at the t1/t2 boundary
#     evals) - the autopsy channels are bounded by the cap but not disproven at 1.7x.
# t0 note: with an empty store the mass is over pure TF, so t0 masks may exceed
# comp's 3072 (bounded by 5120) - a small accepted t0 delta, same shape as budget-v3's.
set -eo pipefail
export WARM_RUN=libero_90_pi05_jointwarm10k_arm1p_vlmknn16_bcast
export GRAD_TAG=arm1p_vlmknn16
export SEQ_TOP_T=3072
export SEQ_TOP_P=0.9
export SEQ_TOP_P_CAP=5120
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_RUN=libero_10_seq5_jw_arm1p_vlmknn16_beta4_topp09mass_f3072c5120_lr2x_steps5k
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
