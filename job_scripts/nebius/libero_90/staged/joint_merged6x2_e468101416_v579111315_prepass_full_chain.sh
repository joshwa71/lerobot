#!/bin/bash
# E62 MERGED 6x2 CELL — full gated chain. *** QUEUED: launch AFTER sharepairs-seeds. ***
# =====================================================================================
# The go-big + sharing merge (E61 addendum 5 discussion, 9 Aug): 6 layers per tower =
# 12 SITES (bigsearch's count) at 7 TABLES = 2.8B values (12.5% UNDER the 3.2B paper
# budget). Share/solo assignment set by the SHARE-CRITERION rule (E61 addendum 6):
# the similarity metric is a VETO — no expert pair in the high-similarity band
# ((10,12) 0.639, (14,16) 0.642), no deep expert sharing regardless.
#   Expert [4,6,8,10,14,16]: share (4,6) [sim 0.523] + (8,10) [0.566]; SOLO 14, 16
#     (the sites that took e7 20->38->58 keep dedicated content; L12 dropped —
#     its role subsumed by solo 14/16, which bigsearch's e7=58 validates).
#   VLM [5,7,9,11,13,15]: share (5,7) [0.653] + (9,11) [0.727] + (13,15) [0.605]
#     (all VLM pairs shareable per E61's direct evidence).
# B's router recipe verbatim (anchored w0.40, sep8, FiLM-free, broadcast losses),
# frozen_prepass on (V5 < E16 = interleaved).
#
# PRE-REGISTRATION (seed-1000/50-ep comparators: bigsearch-12 59.6 = 46/56/64/74/58
# at 4.8B; interleave-8 57.6; sharepairs 56.8 with e7 22):
#   1. e7 >= ~40 => solo deep tables preserve the depth lever under shallow sharing
#      (sharepairs' 22 = the failure mode this layout exists to avoid).
#   2. final >= ~57.6 => frontier band at 2.8B; >= ~59.6 => matches bigsearch at
#      58% of its params — the paper-cell branch.
#   3. Spread survival: e4 >= 40 AND e2 >= 80; e6 the known-noisy watch cell.
#   4. Risk cell (the one bet the calibration does not cover): e7/e9's interleave-8
#      read mass sat partly at E10, and 10 is inside a shared pair here — if e7/e9
#      hold, the bet clears; if e7 lands ~25-35 with e4/e2 held, suspect the (8,10)
#      share before the layout.
#   5. Matrix <= ~+8%/task (the E61 shared-write band; flat-matrix band <= +5%
#      expected on the SOLO tables); prior-core events = 0 at all 12 sites;
#      site-bleed on shared pairs expected in the E61 17-43% band (accepted — the
#      veto avoided the DANGEROUS pairs, it does not eliminate co-writing).
#   6. updt_s recorded (12-site prepass at 7 tables; halved optimizer work vs
#      bigsearch's 4.8B — expect the A-phase to hold an early ladder rung).
# =====================================================================================
set -eo pipefail
export HF_HUB_OFFLINE=1
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- stage 1+2: warm-up + audit + analyses (common body) ----
export ARM_TAG=merged6x2_e468101416_v579111315_anchor040_sep8_prepass
export EXP_LAYERS='[4,6,8,10,14,16]'
export VLM_LAYERS='[5,7,9,11,13,15]'
export SHARE_GROUPS='[[4,6],[8,10]]'
export VLM_SHARE_GROUPS='[[5,7],[9,11],[13,15]]'
export EXP_N=256 EXP_R=2 EXP_KNN=36
export VLM_N=256 VLM_R=2 VLM_KNN=16
export ROUTER_FAST=true
export AUDIT_BS=8 AUDIT_STEPS=400
export LANG_TO_QUERY=false
export EXPERT_ANCHOR=text
export EXPERT_ANCHOR_W=0.40
export SEP_W=8.0
export PREPASS=true
source "$SCRIPT_DIR/joint_rwarmup_common.sh"

# ---- stage 3: automated gate (BG-FIRST bands, per SITE — E59 standing rule) ----
AUDIT_DIR="$ROOT_DIR/outputs/train/audit_heldout_jointwarm_${ARM_TAG}_10k"
if python - "$AUDIT_DIR" <<'EOF'
import json, sys
base = sys.argv[1] + "/"
exp = json.load(open(base + "expert_audit_summary.json"))
vlm = json.load(open(base + "vlm_audit_summary.json"))
fails = []
def layers(d):
    return sorted({k.split("_")[0] for k in d if k.endswith("famIoU")}, key=lambda x: int(x[1:]))
for L in layers(exp):
    f = exp[f"{L}_famIoU"]
    bg = exp.get(f"{L}_bgIoU", 0.0)
    cores = [exp[f"{L}_t{t}"]["core50"] for t in range(10) if f"{L}_t{t}" in exp]
    effs = [exp[f"{L}_t{t}"]["effnum"] for t in range(10) if f"{L}_t{t}" in exp]
    print(f"[gate] expert {L}: bg {bg:.3f} core50 mean {sum(cores)/len(cores):.0f} min-eff {min(effs):.0f} (famIoU {f:.3f} informational)")
    if bg > 0.10: fails.append(f"expert {L} bgIoU {bg:.3f} > 0.10")
    if sum(cores) / len(cores) < 400: fails.append(f"expert {L} mean core50 < 400")
    if min(effs) < 300: fails.append(f"expert {L} min-task effnum < 300")
for L in layers(vlm):
    f = vlm[f"{L}_famIoU"]
    effs = [vlm[f"{L}_t{t}"]["effnum"] for t in range(10) if f"{L}_t{t}" in vlm]
    print(f"[gate] vlm {L}: min-eff {min(effs):.0f} (famIoU {f:.3f} informational)")
    if min(effs) < 150: fails.append(f"vlm {L} min-task effnum < 150 (palette-collapse tripwire)")
    if f >= 0.45: fails.append(f"vlm {L} famIoU {f:.3f} >= 0.45 backstop")
if fails:
    print("GATE: HARD FAIL"); [print("  -", x) for x in fails]; sys.exit(1)
print("GATE: PASS (bg-first bands)")
EOF
then
  echo "E62 merged6x2 chain: gate PASSED - graduating to A-phase + sequential."
else
  echo "E62 merged6x2 chain: GATE FAILED - stopping after certificate."
  exit 1
fi

# ---- stage 4+5: A-phase + 5-task sequential (C-config levers verbatim) ----
export WARM_RUN=libero_90_pi05_jointwarm10k_${ARM_TAG}
export GRAD_TAG=${ARM_TAG}
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
export SEQ_LADDER="32:1:false,16:2:false,8:4:false,16:2:true"

A_OUT_GUARD=$ROOT_DIR/outputs/train/libero_90_pi05_jointA10k_${GRAD_TAG}
if [ -d "$A_OUT_GUARD" ] && [ ! -d "$A_OUT_GUARD/checkpoints/last/pretrained_model" ]; then
  echo "[guard] partial A-phase dir with no completed checkpoint - wiping $A_OUT_GUARD"
  rm -rf "$A_OUT_GUARD"
fi
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
