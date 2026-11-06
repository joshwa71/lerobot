#!/bin/bash
# E59 FIRST INTERLEAVED SUBSTRATE — full gated chain (base box, 3 Aug 26).
# =====================================================================================
# Expert [6,8,10,12] + VLM [7,9,11,13], n256/r2 both, B's router recipe verbatim
# (anchored w=0.40 + sep8, FiLM-free), memory_layer.frozen_prepass=true (E59 build:
# all routing inputs from one memory-free pre-pass; the placement guard is lifted —
# min(vlm)=7 <= max(expert)=12 is the first interleaved layout ever run).
#
# DESIGN (single-delta vs B = pure PLACEMENT at matched everything):
#   - Bank budget IDENTICAL to B (4 expert + 4 VLM, n256/r2 = 3.2B values).
#   - Expert top bank 12 = the e7-depth bet (pattern: L8->20/20/12, L9->28, L12->36);
#     expert bottom 6 exits the marginal L2/L4 routing band B needed the anchor to
#     rescue (E36 separability plateau >= L8; attempt-A L2 famIoU 0.212; E53 "L2
#     leakiest module").
#   - VLM [7,9,11,13] = the E49 geometry payload: instruction-anchor separation is
#     best LOW (L7 0.722 -> L16 0.898, measured on layers 7-16 only — hence no bank
#     below 7; sub-L7 risks the palette-constancy pathology since the anchor's
#     within-task conditionality needs attention-mixed context). Every VLM bank in
#     probe-measured territory; 13 bridges to compact's known-good [13-16].
#   - Even/odd offset: expert anchors pair with LM layers carrying no VLM wrapper.
#   - NOISE OFF (dose05x sigmas were calibrated for B's layers; recalibration is a
#     follow-up arm — comparator is B 53.2, the no-noise cell).
#
# CHAIN: warm-up 10k (router-only fast, broadcast losses, prepass on; ~4h with the
# pre-pass overhead) -> audit (AUDIT_BS=8x400 for headroom) -> AUTOMATED GATE ->
# A-phase (values both towers, 10k, bs ladder) -> 5-task sequential (C-config +
# corefrac + lr2x + top_t 3072, bs16xacc2 ladder, 50-ep final, per-task ckpts).
#
# GATE (E54-relaxed bands):
#   expert famIoU <= 0.18 with at most ONE grace layer <= 0.20; expert bg <= 0.10;
#   expert mean core50 >= 400 (constancy tripwire, not the old 800 floor);
#   expert min-task effnum >= 300; VLM famIoU <= 0.165 all layers; VLM min-task
#   effnum >= 150 (palette-collapse tripwire; subspan JSON printed informationally).
#   The low-VLM-layer bet (banks at 7/9 in never-run territory) is arbitrated HERE
#   for ~4h instead of after a chain.
#
# SEQUENTIAL PRE-REGISTRATION (comparator B 53.2 = 44/60/56/86/20; same budget,
# same recipe — any delta is placement):
#   - beat 53.2 => placement pays; >= 55 => new frontier band.
#   - e7 >= 30 (the depth bet; compact's top-12 e7 = 36); e9 >= ~56 (hold);
#   - e4 >= 40 AND e2 >= 80 (spread's cells must survive expert-bottom 2->6 and
#     the low VLM placement) — either cratering = the trade is architectural;
#   - give-back >= -3 (corefrac band); prior-core events = 0 at all 8 modules;
#   - MSE matrix flat (<= ~+5%/task); block-min mean <= ~0.045 band;
#   - RECORD updt_s: first production measure of the prepass training-step cost
#     (expect <= ~1.35x B's seq step time; smoke measured 1.31-1.39x fwd-only).
# =====================================================================================
set -eo pipefail
export HF_HUB_OFFLINE=1  # E53: hub 429s masquerade as tokenizer corruption; all assets local
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- stage 1+2: warm-up + audit + analyses (common body) ----
export ARM_TAG=interleave_e681012_v791113_anchor040_sep8_prepass
export EXP_LAYERS='[6,8,10,12]'
export VLM_LAYERS='[7,9,11,13]'
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

# ---- stage 3: automated gate on the audit summaries ----
AUDIT_DIR="$ROOT_DIR/outputs/train/audit_heldout_jointwarm_${ARM_TAG}_10k"
SUBSPAN_JSON="$ROOT_DIR/outputs/analysis/e46/subspan_${ARM_TAG}.json"
[ -f "$SUBSPAN_JSON" ] && { echo "[gate] subspan (informational):"; python -c "import json,sys; d=json.load(open('$SUBSPAN_JSON')); print(json.dumps(d, indent=1)[:1200])" || true; }
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
    elif f > 0.18: grace.append(f"expert {L} famIoU {f:.3f}")
    if bg > 0.10: fails.append(f"expert {L} bgIoU {bg:.3f} > 0.10")
    if sum(cores) / len(cores) < 400: fails.append(f"expert {L} mean core50 < 400")
    if min(effs) < 300: fails.append(f"expert {L} min-task effnum < 300")
if len(grace) > 1:
    fails.append(f"expert famIoU grace budget exceeded (1 allowed): {grace}")
elif grace:
    print(f"[gate] grace layer accepted: {grace[0]}")
for L in layers(vlm):
    f = vlm[f"{L}_famIoU"]
    effs = [vlm[f"{L}_t{t}"]["effnum"] for t in range(10) if f"{L}_t{t}" in vlm]
    print(f"[gate] vlm {L}: famIoU {f:.3f} min-eff {min(effs):.0f}")
    if f > 0.165: fails.append(f"vlm {L} famIoU {f:.3f} > 0.165")
    if min(effs) < 150: fails.append(f"vlm {L} min-task effnum < 150 (palette-collapse tripwire)")
if fails:
    print("GATE: HARD FAIL"); [print("  -", x) for x in fails]; sys.exit(1)
print("GATE: PASS")
EOF
then
  echo "E59 interleaved chain: gate PASSED - graduating to A-phase + sequential."
else
  echo "E59 interleaved chain: GATE FAILED - stopping after certificate (warm-up + audit retained)."
  exit 1
fi

# ---- stage 4+5: A-phase + 5-task sequential (B's levers verbatim) ----
export WARM_RUN=libero_90_pi05_jointwarm10k_${ARM_TAG}
export GRAD_TAG=${ARM_TAG}
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_interleave_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k
export SEQ_LADDER="16:2:false,8:4:false,16:2:true"

A_OUT_GUARD=$ROOT_DIR/outputs/train/libero_90_pi05_jointA10k_${GRAD_TAG}
if [ -d "$A_OUT_GUARD" ] && [ ! -d "$A_OUT_GUARD/checkpoints/last/pretrained_model" ]; then
  echo "[guard] partial A-phase dir with no completed checkpoint - wiping $A_OUT_GUARD"
  rm -rf "$A_OUT_GUARD"
fi
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
