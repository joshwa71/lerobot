#!/bin/bash
# E53 ARM 1 (base box): absolute layer-max + corefrac, at bs8 x accum4 WITHOUT
# gradient checkpointing (Josh's batching call), sequential-only.
#
# Substrate: the certified anchored absmax — expert [4-9] + VLM [10-16] = 13 modules,
# 5.37B values, expert text-anchor (per-layer pooled LM instruction hidden, B=0.5) +
# FiLM off. Certificate (computed E53, was never summarized): expert famIoU
# 0.136/0.119/0.135/0.148/0.166/0.174 at L4-L9 (un-anchored: 0.242/0.195/0.201/0.219
# at L4-L7), expert bg 0.022-0.034 — the cleanest expert routing ever certified; VLM
# 0.132-0.150 all in band. The A-phase exists on this box (stage A auto-skips); the
# prior sequential (rank+peak, bs16xacc2+ckpt) was killed before stepping.
#
# TWO deltas vs that killed spec, both deliberate:
#   1. corefrac protection (E53: total core exclusion — 0 events into prior cores at
#      all modules, 0% core drift, zero writer tax — worth +8.0 on its compact twin).
#   2. bs8 x accum4, grad-ckpt OFF. Rationale (Josh): training is compute-bound
#      (E49: accum overhead only +12%/halving) and checkpointing re-runs the forward
#      (~+33% FLOPs) — trading recompute for smaller micro-batches should net
#      ~10-20% faster IF bs8-no-ckpt fits. CAVEAT carried from the E52 A-phase OOMs:
#      batch showed weak memory leverage at 5.37B values (bs16 demanded only ~5GB
#      less than bs32), which predicts bs8 may still OOM — hence the ladder below
#      (falls back to ckpt-on, then to the killed run's exact config). A rung failing
#      before the 005000 checkpoint is auto-retried on the next rung.
#
# Pre-registered reads (comparators: corefrac-compact 51.6 = frontier, spread-A 41.2,
# fold-in 43.6, comp 46.0):
#   - core events into prior cores = 0 at all 13 modules; MSE matrix flat (<= ~+5%);
#   - function give-back <= ~5% per task (corefrac-compact: +0-4.4%);
#   - BEAT 51.6 to take the frontier (49.2 multitask-LoRA line already crossed);
#   - block-min mean <= ~0.042 (13-module fit floor; compact-corefrac 0.0415) — the
#     capacity question IS this number plus e4/e9 own-block chunks vs 0.0333/0.0755;
#   - watch e9-family channels through L8/L9 (the two layers certified at 0.166/0.174,
#     above the compact band) — corefrac makes them shoulder-only, but they are the
#     widest shoulders on this substrate;
#   - log the winning ladder rung + s/step vs the killed config's ~(never stepped);
#     this is also the empirical answer to the bs8-no-ckpt viability question.
set -eo pipefail
export HF_HUB_OFFLINE=1  # E53: hub 429s surface as bogus "vocabulary corrupted" tokenizer errors; all assets local
export WARM_RUN=libero_90_pi05_jointwarm10k_absmax_anchor05_nofilm_e4to9_v10to16
export GRAD_TAG=absmax_anchor05_nofilm_e4to9_v10to16
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_PROTECT_UNORM=corefrac
export SEQ_LADDER="8:4:false,8:4:true,16:2:true"
export SEQ_RUN=libero_10_seq5_jw_absmax_anchor05nofilm_beta4corefrac_topt3072_lr2x_steps5k
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
