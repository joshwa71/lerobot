#!/bin/bash
# E53 ARM 3 (nebius4): SPREAD substrate + expert text-anchor + corefrac — the
# candidate-win chain (Josh). Inherits (A) the best fit in the project from the
# spread layout (E53: best own-block chunks on all five tasks), (B) the anchored
# expert routing that rescued absmax's low layers (E53 backfill audit: expert famIoU
# 0.242->0.136 at L4, bg 4x cleaner — the per-layer pooled-LM-instruction anchor,
# B=0.5, FiLM off), (C) corefrac's total core exclusion (0 events / 0% core drift /
# zero writer tax; compact twin 43.6 -> 51.6).
#
# Deltas vs arm 2 (nebius3): the anchored-nofilm router package, via a fresh warm-up
# (routers are trained objects — the anchor changes key construction, so the spread
# certificate must be re-earned). Everything else matched: substrate, levers, corefrac.
# Free structural bonus on this layout: expert layers [2,4,6,8] anchor from LM layers
# 2/4/6/8, ALL below the VLM memory at [10+] -> the anchors are memory-free and
# stationary by construction.
#
# Warm-up config note (the 9h-warm-up postmortem): router-only training on a frozen
# backbone needs neither grad-ckpt (router grads do not chain through the backbone)
# nor accum — the absmax warm-up's bs16/acc2/ckpt was a sequential-stage carry-over.
# This 8-module warm-up runs the common defaults (bs32, no accum, no ckpt) like every
# certified 8-module warm-up (~3-4h).
#
# GATE (automated; calibrated per the absmax lesson — its 0.165 hard line tripped on
# a 0.174 that was inside every historical certificate standard):
#   expert: famIoU <= 0.18 per layer (un-anchored spread audit: L2 0.212 -> the
#     anchor must pull L2 under band; L8 was 0.163) with ONE grace layer allowed up
#     to 0.20; bg <= 0.10; mean core50 >= 800; min-task effnum >= 300 (collapse guard
#     — B too high rebuilds the E21 constant-key pathology);
#   vlm: famIoU <= 0.165 all four + min-task effnum >= 150 (certified clean
#     un-anchored at 0.132-0.152 and never had FiLM — must come through untouched).
#   L2 is the honest unknown: the anchor-geometry ledger covered LM L4+ (improving
#   downward); L2 was never probed. The gate decides; a marginal L2 famIoU under
#   corefrac is shoulder-risk only, hence the grace layer.
# Fail -> STOP with warm-up + audit retained for review.
#
# Sequential pre-registered reads (comparators: arm 2's result when it lands,
# corefrac-compact 51.6, spread-peak 41.2):
#   - the arm-3-vs-arm-2 delta is the anchored router's conversion value at matched
#     substrate+protection — the cleanest attribution of the wave;
#   - BEAT 51.6 takes the frontier; core events = 0; give-back >= ~-5; e7 watch as in
#     arm 2 (anchoring is the one mechanism with a shot at e7: its language channel
#     routes the basket family apart where scene routing cannot — famIoU on the
#     audit's basket pairs is the leading indicator).
set -eo pipefail
export HF_HUB_OFFLINE=1  # E53: hub 429s surface as bogus "vocabulary corrupted" tokenizer errors; all assets local
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- stage 1+2: warm-up + audit + analyses (common body; STOPs after audit) ----
export ARM_TAG=layermax_A_anchor05_nofilm_e2468_v10121416
export EXP_LAYERS='[2,4,6,8]'
export VLM_LAYERS='[10,12,14,16]'
export EXP_N=256 EXP_R=2 EXP_KNN=36
export VLM_N=256 VLM_R=2 VLM_KNN=16
export ROUTER_FAST=true
# 8-module audit precedent (E50 attempt-A incident): full value path + both
# frozen-route forks OOM at bs32 -> bs16 x 200 steps (matched audited-sample coverage)
export AUDIT_BS=16 AUDIT_STEPS=200
export LANG_TO_QUERY=false
export EXPERT_ANCHOR=text
export EXPERT_ANCHOR_W=0.5
source "$SCRIPT_DIR/joint_rwarmup_common.sh"

# ---- stage 3: automated gate on the audit summaries ----
AUDIT_DIR="$ROOT_DIR/outputs/train/audit_heldout_jointwarm_${ARM_TAG}_10k"
if python - "$AUDIT_DIR" <<'EOF'
import json, sys
base = sys.argv[1] + "/"
exp = json.load(open(base + "expert_audit_summary.json"))
vlm = json.load(open(base + "vlm_audit_summary.json"))
fails, grace = [], []
def layers(d):
    return sorted({k.split("_")[0] for k in d if k.endswith("famIoU")}, key=lambda x: int(x[1:]))
for L in layers(exp):
    f = exp[f"{L}_famIoU"]
    bg = exp.get(f"{L}_bgIoU", 0.0)
    cores = [exp[f"{L}_t{t}"]["core50"] for t in range(10) if f"{L}_t{t}" in exp]
    effs = [exp[f"{L}_t{t}"]["effnum"] for t in range(10) if f"{L}_t{t}" in exp]
    print(f"[gate] expert {L}: famIoU {f:.3f} bg {bg:.3f} core50 mean {sum(cores)/len(cores):.0f} min-eff {min(effs):.0f}")
    if f > 0.20: fails.append(f"expert {L} famIoU {f:.3f} > 0.20")
    elif f > 0.18: grace.append(f"expert {L} famIoU {f:.3f} in (0.18, 0.20]")
    if bg > 0.10: fails.append(f"expert {L} bgIoU {bg:.3f} > 0.10")
    if sum(cores) / len(cores) < 800: fails.append(f"expert {L} mean core50 < 800")
    if min(effs) < 300: fails.append(f"expert {L} min-task effnum < 300")
for L in layers(vlm):
    f = vlm[f"{L}_famIoU"]
    effs = [vlm[f"{L}_t{t}"]["effnum"] for t in range(10) if f"{L}_t{t}" in vlm]
    print(f"[gate] vlm {L}: famIoU {f:.3f} min-eff {min(effs):.0f}")
    if f > 0.165: fails.append(f"vlm {L} famIoU {f:.3f} > 0.165")
    if min(effs) < 150: fails.append(f"vlm {L} min-task effnum < 150")
if len(grace) > 1: fails.append(f"more than one expert grace layer: {grace}")
if fails:
    print("GATE: HARD FAIL"); [print("  -", x) for x in fails]; sys.exit(1)
if grace:
    print("GATE: PASS (with grace):"); [print("  -", x) for x in grace]
else:
    print("GATE: PASS")
EOF
then
  echo "E53 arm-3 chain: gate PASSED — graduating to A-phase + sequential."
else
  echo "E53 arm-3 chain: GATE FAILED — stopping after certificate (warm-up + audit retained)."
  exit 1
fi

# ---- stage 4+5: A-phase + 5-task sequential (corefrac, fold-in levers) ----
export WARM_RUN=libero_90_pi05_jointwarm10k_${ARM_TAG}
export GRAD_TAG=${ARM_TAG}
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_layermax_A_anchor05nofilm_e2468_v10121416_beta4corefrac_topt3072_lr2x_steps5k
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
