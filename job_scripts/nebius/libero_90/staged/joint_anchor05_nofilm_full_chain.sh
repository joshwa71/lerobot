#!/bin/bash
# E52 ABSMAX-ANCHOR + FiLM-OFF full gated chain (base box, Josh's call — the rescue
# experiment the anchor mechanism was built for). TWO deltas vs the failed absmax
# warm-up (audit_heldout_jointwarm_absmax_e4to9_v10to16_10k: expert famIoU 0.171-0.242,
# bg 0.091-0.129 — L4/L6/L7 over band; verdict: representation-limited equilibrium,
# feature crowding through ~L7, no dose/steps rescue exists):
#   delta 1: expert_anchor_pool=text, B=0.5 — expert layer j routes on
#            0.5*nrm(W_a @ pooled LM instruction hidden at LM layer j) + 0.5*nrm(token).
#            Per-layer pairing puts the BEST anchors (ledger: LM L5/L7 inter-cos 0.67,
#            the most open task geometry measured) at exactly the depths that failed
#            on crowded expert features (inter-cos 0.966-0.976). Composite init
#            geometry at B=0.5: inter 0.81-0.82, conditionality proxy 0.83-0.87.
#   delta 2: lang_to_query=false — FiLM retired on the expert tower (E28: near-inert,
#            beta cos 0.945-0.99; the anchor is the strictly better language carrier).
#
# Board context: compact+corefrac (nebius4) owns the give-back axis; spread-A (nebius3)
# owns spacing. THIS arm owns the depth/capacity moonshot: 13 modules, 5.37B values —
# if the anchor pulls L4-L7 into band, the absolute-layer-max substrate comes back
# from the dead with the largest certified capacity in the project.
#
# Chain: warm-up (router-only fast, broadcast losses, ~3.5h) -> audit (AUDIT_BS=8x400,
# the 13-module VRAM precedent) -> AUTOMATED GATE -> on pass: A-phase (values both
# towers, 10k; common auto-falls-back bs32->bs16xacc2) -> 5-task sequential at the
# fold-in levers (top_t 3072, lr 2e-3->2e-4, rank+peak beta4) at SEQ_BS=8 x ACCUM=4
# (5.37B-value VRAM; ~20-24h), 50-ep final.
#
# GATE (automated; the RESCUE bar — pass = ALL of):
#   expert famIoU <= 0.165 on ALL SIX layers (un-anchored: 0.171-0.242; this demands
#     the anchor pull the whole stack to compact-parity band);
#   expert bgIoU  <= 0.110 on all six (un-anchored 0.091-0.129);
#   expert collapse guards: mean core50 >= 800, min-task effnum >= 300 (B too high
#     would rebuild the E21 constant-key pathology);
#   VLM famIoU <= 0.165 all seven + min-task effnum >= 150 (the VLM half certified
#     clean un-anchored at 0.125-0.164 and never had FiLM — must come through
#     untouched-to-noise).
# Fail -> STOP with the certificate on disk (warm-up checkpoint retained for review;
# per-layer table is the read either way — partial rescue shapes the next arm).
#
# Sequential pre-registered reads (comparators: fold-in 43.6, comp 46.0, plain 44.8;
# no 13-module baseline exists — scoreboard reads):
#   - beat 46.0 takes the frontier; >= 49.2 crosses multitask-LoRA;
#   - give-back watch: init-mean -> final (fold-in paid -10.6 at these levers; the
#     anchored-separated cores must shrink the collision cross-section — the
#     layers-vs-budget collision principle, now within-tower);
#   - e4/e9 (the cliff-edge tasks): own-block chunk + final vs E52's 0.0333/0.0784;
#   - block-min mean <= ~0.045 at 13 modules (fit floor with FiLM off).
set -eo pipefail
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- stage 1+2: warm-up + audit + analyses (common body; STOPs after audit) ----
export ARM_TAG=absmax_anchor05_nofilm_e4to9_v10to16
export EXP_LAYERS='[4,5,6,7,8,9]'
export VLM_LAYERS='[10,11,12,13,14,15,16]'
export EXP_N=256 EXP_R=2 EXP_KNN=36
export VLM_N=256 VLM_R=2 VLM_KNN=16
export ROUTER_FAST=true
export AUDIT_BS=8 AUDIT_STEPS=400
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
fails = []
def layers(d):
    return sorted({k.split("_")[0] for k in d if k.endswith("famIoU")}, key=lambda x: int(x[1:]))
for L in layers(exp):
    f = exp[f"{L}_famIoU"]
    bg = exp.get(f"{L}_bgIoU", 0.0)
    cores = [exp[f"{L}_t{t}"]["core50"] for t in range(10) if f"{L}_t{t}" in exp]
    effs = [exp[f"{L}_t{t}"]["effnum"] for t in range(10) if f"{L}_t{t}" in exp]
    print(f"[gate] expert {L}: famIoU {f:.3f} bg {bg:.3f} core50 mean {sum(cores)/len(cores):.0f} min-eff {min(effs):.0f}")
    if f > 0.165: fails.append(f"expert {L} famIoU {f:.3f} > 0.165")
    if bg > 0.110: fails.append(f"expert {L} bgIoU {bg:.3f} > 0.110")
    if sum(cores) / len(cores) < 800: fails.append(f"expert {L} mean core50 < 800")
    if min(effs) < 300: fails.append(f"expert {L} min-task effnum < 300")
for L in layers(vlm):
    f = vlm[f"{L}_famIoU"]
    effs = [vlm[f"{L}_t{t}"]["effnum"] for t in range(10) if f"{L}_t{t}" in vlm]
    print(f"[gate] vlm {L}: famIoU {f:.3f} min-eff {min(effs):.0f}")
    if f > 0.165: fails.append(f"vlm {L} famIoU {f:.3f} > 0.165")
    if min(effs) < 150: fails.append(f"vlm {L} min-task effnum < 150")
if fails:
    print("GATE: HARD FAIL"); [print("  -", x) for x in fails]; sys.exit(1)
print("GATE: PASS")
EOF
then
  echo "E52 absmax-anchor chain: gate PASSED — graduating to A-phase + sequential."
else
  echo "E52 absmax-anchor chain: GATE FAILED — stopping after certificate (warm-up + audit retained)."
  exit 1
fi

# ---- stage 4+5: A-phase + 5-task sequential at the fold-in levers ----
export WARM_RUN=libero_90_pi05_jointwarm10k_${ARM_TAG}
export GRAD_TAG=${ARM_TAG}
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=8
export SEQ_ACCUM=4
export SEQ_RUN=libero_10_seq5_jw_absmax_anchor05nofilm_beta4_topt3072_lr2x_steps5k
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
