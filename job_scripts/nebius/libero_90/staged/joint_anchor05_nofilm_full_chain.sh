#!/bin/bash
# E52 EXPERT-ANCHOR + FiLM-OFF full gated chain (base box, Josh's "feeling risky" call —
# deliberately a TWO-DELTA arm, logged as such):
#   delta 1: expert_anchor_pool=text, B=0.5 — every expert memory layer j routes on
#            0.5*nrm(W_a @ pooled LM instruction hidden at LM layer j) + 0.5*nrm(token)
#            (per-layer pairing; W_a trains in this warm-up, frozen downstream;
#            anchor memory-free/stationary by placement).
#   delta 2: lang_to_query=false — FiLM retired on the expert tower (E28 measured it
#            near-inert: film beta cos 0.945-0.99 across tasks, stripping it moved
#            routing ~0%; the anchor is the strictly better language carrier,
#            LM instruction pool inter-task cos 0.67-0.80).
#
# Substrate: layermax compact (expert [9-12] + VLM [13-16], n256/r2, expert knn36 /
# VLM knn16) — the frontier substrate; certificate reads against compact's expert
# 0.140-0.154 / VLM 0.145-0.154 (audit_heldout_jointwarm_layermax_compact_e9to12_v13to16_10k).
# Ledger prior (outputs/analysis/e52/anchor_ledger.json): composite init key geometry
# inter-task cos 0.95 -> 0.81 at B=0.5, conditionality proxy 0.87 (healthy band).
#
# Chain: warm-up (router-only, broadcast losses, ~3h) -> audit (AUDIT_BS=16 x 200,
# 8-module VRAM) -> AUTOMATED GATE -> on pass: A-phase (values both towers, 10k) ->
# 5-task sequential at the FOLD-IN levers (top_t 3072, lr 2e-3->2e-4, rank+peak
# beta4, bs16xacc2, 50-ep final) — the head-to-head vs the fold-in's 43.6 at matched
# config: does anchored separation cut the E52 give-back channel (e4 55->18, e9
# 55->30; V16 core drift 55%)?
#
# GATE (automated; pass = ALL of):
#   expert famIoU <= 0.155 on all 4 layers (parity-or-better vs compact's 0.140-0.154;
#     the pre-registered WIN read is <= ~0.12);
#   expert mean core50 >= 800 and min-task effnum >= 300 (anchor-collapse guard —
#     B too high would rebuild the E21 constant-key pathology);
#   VLM famIoU <= 0.165 all 4 + min-task effnum >= 150 (the VLM tower never had FiLM
#     and must come through untouched-to-noise).
# Fail -> STOP with the certificate on disk (warm-up checkpoint retained for review).
#
# Sequential pre-registered reads (vs fold-in 43.6 = 18/58/30/76/36, block-min 0.0409):
#   - beat 43.6; the give-back is the primary read: init-mean -> final >= -5 (fold-in
#     -10.6) and e4/e9 own->final chunk give-back < the fold-in's +21%/+9%;
#   - e4 V16-analog core drift < ~40% (fold-in 55%) — smaller/separated cores must
#     shrink the collision cross-section (the layers-vs-budget collision principle,
#     now within-tower);
#   - block-min mean <= ~0.045 (fit preserved under FiLM-off + anchored routing);
#   - >= 46.0 takes the frontier; >= 49.2 crosses multitask-LoRA.
set -eo pipefail
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- stage 1+2: warm-up + audit + analyses (common body; STOPs after audit) ----
export ARM_TAG=layermax_compact_anchor05_nofilm
export EXP_LAYERS='[9,10,11,12]'
export VLM_LAYERS='[13,14,15,16]'
export EXP_N=256 EXP_R=2 EXP_KNN=36
export VLM_N=256 VLM_R=2 VLM_KNN=16
export ROUTER_FAST=true
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
fails = []
def layers(d):
    return sorted({k.split("_")[0] for k in d if k.endswith("famIoU")}, key=lambda x: int(x[1:]))
for L in layers(exp):
    f = exp[f"{L}_famIoU"]
    cores = [exp[f"{L}_t{t}"]["core50"] for t in range(10) if f"{L}_t{t}" in exp]
    effs = [exp[f"{L}_t{t}"]["effnum"] for t in range(10) if f"{L}_t{t}" in exp]
    print(f"[gate] expert {L}: famIoU {f:.3f} core50 mean {sum(cores)/len(cores):.0f} min-eff {min(effs):.0f}")
    if f > 0.155: fails.append(f"expert {L} famIoU {f:.3f} > 0.155")
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
  echo "E52 anchor chain: gate PASSED — graduating to A-phase + sequential."
else
  echo "E52 anchor chain: GATE FAILED — stopping after certificate (warm-up + audit retained)."
  exit 1
fi

# ---- stage 4+5: A-phase + 5-task sequential at the fold-in levers ----
export WARM_RUN=libero_90_pi05_jointwarm10k_${ARM_TAG}
export GRAD_TAG=${ARM_TAG}
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_BS=16
export SEQ_ACCUM=2
export SEQ_RUN=libero_10_seq5_jw_layermax_compact_anchor05nofilm_beta4_topt3072_lr2x_steps5k
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
