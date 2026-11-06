#!/bin/bash
# E61 SHARED-PAIR CELL — full gated chain. *** QUEUED: launch AFTER E60 lands. ***
# =====================================================================================
# The E59 frontier layout with PAIRED STORAGE: expert [6,8,10,12] sharing (6,8)+(10,12),
# VLM [7,9,11,13] sharing (7,9)+(11,13). Same 8 SITES as interleave-8 (57.6), FOUR
# tables, 1.6B values = HALF the adaptation state. B's router recipe verbatim,
# frozen_prepass on. Single-delta vs interleave-8 = the storage topology.
#
# WHY (E59 verdict + Meta "Memory Layers at Scale" prior): capacity is not binding
# (worst core50 2.4K/65K) — sites are the axis; sharing decouples sites from params.
# Adjacent pairing keeps the two consumers' routing inputs maximally correlated
# (prepass features drift slowly); shared keys keep slot identity coherent across
# members (per-site keys would alias one memory under two unrelated addresses).
# MECHANISM BET: each shared slot's content trains under BOTH consumption contexts —
# the real-distribution analogue of E58's value-input noise, at NEGATIVE param cost.
#
# PRE-REGISTRATION (comparator interleave-8 57.6 = 42/68/56/84/38, matched sites):
#   1. 50-ep final >= ~55 => "sites, not slots" confirmed CAUSALLY; frontier
#      performance at half the adaptation state. >= 57.6 => the multi-consumer
#      regularizer is net-positive, not just free.
#   2. Harvest-bank rescore: spec/succ Q4 D <= 0.344 (interleave's radius) => the
#      two-consumer effect acts like vnoise; > B's 0.482 refutes the framing.
#   3. e7 >= ~30 with e2 >= 80 => content is NOT depth-specialized at pair distance 2.
#      e7 craters with shallow cells held => depth-specialized content is real =>
#      asymmetric capacity (solo deep tables, shared shallow) is the follow-up.
#   4. Autopsy: within-table site-bleed <= ~15% (per-task analog runs 2-8%);
#      prior-core events <= ~2K (interleave band); per-site bg <= 0.10 unchanged.
#   5. updt_s within ~5% of 0.93 (storage halves; retrieval count per site unchanged).
# Write semantics (built + smoked, E61): top_t=3072 PER SITE with UNION merge on the
# shared tables; protection stores synced (elementwise max) across group members at
# every boundary; corefrac protection binds through either member's reads.
# =====================================================================================
set -eo pipefail
export HF_HUB_OFFLINE=1
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- stage 1+2: warm-up + audit + analyses (common body) ----
export ARM_TAG=sharepairs_e681012_v791113_anchor040_sep8_prepass
export EXP_LAYERS='[6,8,10,12]'
export VLM_LAYERS='[7,9,11,13]'
export SHARE_GROUPS='[[6,8],[10,12]]'
export VLM_SHARE_GROUPS='[[7,9],[11,13]]'
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
  echo "E61 sharepairs chain: gate PASSED - graduating to A-phase + sequential."
else
  echo "E61 sharepairs chain: GATE FAILED - stopping after certificate."
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
export SEQ_RUN=libero_10_seq5_jw_sharepairs_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k
export SEQ_LADDER="16:2:false,8:4:false,16:2:true"

A_OUT_GUARD=$ROOT_DIR/outputs/train/libero_90_pi05_jointA10k_${GRAD_TAG}
if [ -d "$A_OUT_GUARD" ] && [ ! -d "$A_OUT_GUARD/checkpoints/last/pretrained_model" ]; then
  echo "[guard] partial A-phase dir with no completed checkpoint - wiping $A_OUT_GUARD"
  rm -rf "$A_OUT_GUARD"
fi
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
