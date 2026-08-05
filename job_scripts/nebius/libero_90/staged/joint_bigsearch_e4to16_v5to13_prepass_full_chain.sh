#!/bin/bash
# E60 GO-BIG PLACEMENT SEARCH — full gated chain (5 Aug 26, Josh: "write the
# scripts and fire them off").
# =====================================================================================
# Expert [4,6,8,10,12,14,16] + VLM [5,7,9,11,13] (12 modules, n256/r2 both = 4.8B
# values), B's router recipe verbatim (anchored w=0.40 + sep8, FiLM-free),
# frozen_prepass=true. This is a SEARCH run, not a paper cell: the capacity
# confound vs interleave-8 (3.2B) is accepted; the deliverable is the LESION MAP
# (zero-ablation per module at the final checkpoint) + a trimmed 3.2B-matched
# retrain as the confirmable cell.
#
# SITE SELECTION (evidence per site):
#   - Expert 14, 16: NEVER-RUN deep territory. The depth gradient (L8->20/20/12,
#     L9->28, L12->36, E59 e7=38 with read mass concentrated E10/E12) hasn't
#     turned over yet — these are the primary search targets.
#   - Expert 4 re-added: expected-dead control (E36 separability plateau >= L8;
#     E53 "L2 leakiest") — calibrates the lesion map's zero point.
#   - VLM 5: cleared by the E59 sub-L7 querystats extension (L5 inter 0.734 =
#     better than L9's 0.785 which carries a working bank; intra 0.895 = L13-band,
#     not pathological). The "lower is better" thesis is DEAD (L7 is the measured
#     optimum of the whole stack, E49 curve is a V) — V5 is a judgment-call site
#     the lesion map settles. V3/V4 EXCLUDED: L3 intra 0.926 = worst constancy in
#     the measured stack (the palette-constancy axis), no separation advantage.
#   - VLM [7,9,11,13] unchanged — sitting at the measured optimum.
#
# GATE — FIRST PRODUCTION USE OF THE BG-FIRST RULE (E59 addendum 3: "famIoU is
# dead as a gate axis; gate on bg (<=0.10, winning band 0.02-0.05) + capacity
# floors"). famIoU printed INFORMATIONALLY only (expect lawful elevation at expert
# 14/16 — the anchor-source depth gradient, 3 certs deep). Hard gates:
#   expert: bg <= 0.10 per layer; mean core50 >= 400; min-task effnum >= 300
#   vlm:    min-task effnum >= 150 (palette-collapse tripwire — THE V5 ARBITER,
#           first arbitration of a below-expert-band VLM bank); famIoU backstop
#           kill only at >= 0.45 (E48 permissive precedent)
#
# SEQUENTIAL PRE-REGISTRATION (comparator interleave-8 57.6 = 42/68/56/84/38,
# NOT budget-matched — search-run reads, scored at the 50-ep final):
#   - >= ~57.6-noise => the 12-module config breaks nothing; > 57.6 => more sites
#     help even before trimming; << 57.6 => interaction cost is real, the map
#     still tells us where.
#   - e7 vs 38: do E14/E16 push the depth lever further?
#   - e2 >= 80, e4 >= 40 (spread cells must continue to survive).
#   - give-back >= -3; MSE matrix <= ~+5%/task; prior-core autopsy vs the 1,684
#     (interleave) / 7,376 (B) ladder; updt_s recorded (12-module prepass cost).
#   - LESION BATTERY (implemented at landing, pre-registered here): zero-ablate
#     each module (additive memory => zero the value table is a clean lesion) at
#     the final ckpt -> delta-success per task at 20-ep (12x5x20 = 1,200 rollouts,
#     one overnight) + delta-chunk screen; greedy backward elimination to a
#     3.2B-budget 8-module layout; RETRAIN that layout as the paper cell. Trim
#     criterion = lesion delta, NOT read mass (usage != importance).
# =====================================================================================
set -eo pipefail
export HF_HUB_OFFLINE=1  # E53: hub 429s masquerade as tokenizer corruption; all assets local
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- stage 1+2: warm-up + audit + analyses (common body) ----
export ARM_TAG=bigsearch_e4to16_v5to13_anchor040_sep8_prepass
export EXP_LAYERS='[4,6,8,10,12,14,16]'
export VLM_LAYERS='[5,7,9,11,13]'
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

# ---- stage 3: automated gate on the audit summaries (BG-FIRST bands) ----
AUDIT_DIR="$ROOT_DIR/outputs/train/audit_heldout_jointwarm_${ARM_TAG}_10k"
SUBSPAN_JSON="$ROOT_DIR/outputs/analysis/e46/subspan_${ARM_TAG}.json"
[ -f "$SUBSPAN_JSON" ] && { echo "[gate] subspan (informational):"; python -c "import json,sys; d=json.load(open('$SUBSPAN_JSON')); print(json.dumps(d, indent=1)[:1200])" || true; }
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
  echo "E60 bigsearch chain: gate PASSED - graduating to A-phase + sequential."
else
  echo "E60 bigsearch chain: GATE FAILED - stopping after certificate (warm-up + audit retained)."
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
export SEQ_RUN=libero_10_seq5_jw_bigsearch_e4to16_v5to13_prepass_beta4corefrac_topt3072_lr2x_steps5k
export SEQ_LADDER="16:2:false,8:4:false,16:2:true"

A_OUT_GUARD=$ROOT_DIR/outputs/train/libero_90_pi05_jointA10k_${GRAD_TAG}
if [ -d "$A_OUT_GUARD" ] && [ ! -d "$A_OUT_GUARD/checkpoints/last/pretrained_model" ]; then
  echo "[guard] partial A-phase dir with no completed checkpoint - wiping $A_OUT_GUARD"
  rm -rf "$A_OUT_GUARD"
fi
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
