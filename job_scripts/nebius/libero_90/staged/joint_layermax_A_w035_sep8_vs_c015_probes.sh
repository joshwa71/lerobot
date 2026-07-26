#!/bin/bash
# E54 probe pair — SPREAD substrate, GLOBAL loss-weight rejig at expert_anchor_w=0.35
# =====================================================================================
# Context: the E53 anchor-weight bisection (w 0.10->0.50, sep/c fixed) found no point
# passing famIoU AND the core50>=800 floor. The absmax frontier run (53.6, ~zero
# give-back, per-bank core50 425-648) falsified the 800 floor as a hard requirement,
# so the window is re-opened from the capacity-safe side: at w=0.35 the sweep measured
# expert famIoU 0.21-0.25 / core50 1058-1174 — capacity comfortable, famIoU needs only
# -0.03..-0.07. These two probes test which GLOBAL loss lever closes that gap (per-tower
# weight overrides deliberately avoided — Josh):
#   P1 sep 5->8   : translation route (E26: famIoU fell while capacity ROSE along the
#                   whole sep curve 0.5->5; >5 never swept). VLM risk = palette
#                   COMPACTION (sep pressure at fixed anti-collapse force -> the
#                   arm-3-old ~2-draw regime).
#   P2 c 0.05->0.15: contrastive route (with the anchor supplying the per-task axis,
#                   SupCon's cross-task push trains proj(x) to amplify the state-side
#                   task signal). VLM risk = palette SPRAWL (E45: c->breadth monotone
#                   on the pooled tower; famIoU headroom is thin, 0.132-0.152 vs 0.165).
# Both probes: fresh warm-up from base_nomem_50k (fresh ARM_TAGs — skip-guards key on
# tag; the E53 lesson), w=0.35, everything else the E53 arm-3 recipe verbatim.
#
# GATES (recalibrated; informational print at the end — no auto-graduation):
#   expert: famIoU <= 0.18 per layer (ONE grace layer <= 0.20), bg <= 0.10,
#           mean core50 >= 400 (constancy tripwire, NOT the old 800), min-task
#           effnum >= 300.
#   VLM   : famIoU <= 0.165 all layers; palette health read from the subspan JSON
#           (outputs/analysis/e46/subspan_<tag>.json) — palette effnum >= ~500
#           (anti-arm-3-old floor).
# Decision rule (manual, on the printed summary): one threads -> its A-phase + 5-task
# sequential w/ corefrac; both -> better joint margin; neither but one direction moved
# expert famIoU cleanly -> dose refinement (sep 12 / c 0.25 / w 0.375); both damage the
# VLM before fixing the expert -> per-LAYER anchor weight fallback (heavy L2/L4 only).
# =====================================================================================
set -o pipefail
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_OFFLINE=1  # E53: hub 429s surface as bogus tokenizer errors; all assets local

run_probe() {  # $1 = ARM_TAG   $2 = SEP_W   $3 = CONTRASTIVE_W
  local tag="$1" sep="$2" con="$3"
  echo "=== [chain] probe ${tag} (sep=${sep} contrastive=${con}) START $(date) ==="
  (
    set -eo pipefail
    export ARM_TAG="$tag"
    export EXP_LAYERS='[2,4,6,8]'
    export VLM_LAYERS='[10,12,14,16]'
    export EXP_N=256 EXP_R=2 EXP_KNN=36
    export VLM_N=256 VLM_R=2 VLM_KNN=16
    export ROUTER_FAST=true
    # 8-module audit precedent (E50 attempt-A OOM): bs16 x 200 steps = matched coverage
    export AUDIT_BS=16 AUDIT_STEPS=200
    export LANG_TO_QUERY=false
    export EXPERT_ANCHOR=text
    export EXPERT_ANCHOR_W=0.35
    export SEP_W="$sep"
    export CONTRASTIVE_W="$con"
    source "$SCRIPT_DIR/joint_rwarmup_common.sh"
  ) || echo "=== [chain] probe ${tag} FAILED (continuing to next probe) ==="
  echo "=== [chain] probe ${tag} END $(date) ==="
}

# P1: sep=8, c=0.05 (default)
run_probe layermax_A_anchor035_sep8_nofilm_e2468_v10121416 8.0 0.05
# P2: sep=5 (default), c=0.15
run_probe layermax_A_anchor035_c015_nofilm_e2468_v10121416 5.0 0.15

# ---- gate summary (informational; robust to missing files) ----
python - <<'EOF' || true
import json, os, statistics as st
base = "/home/josh/lerobot/outputs/train/"
ref = {"w035 sweep point (sep5/c0.05)": "famIoU 0.21-0.25 core50 1058-1174 (E53 sweep, gate-fail on famIoU only under relaxed gates)"}
print("\n================ E54 PROBE GATE SUMMARY ================")
for k, v in ref.items():
    print(f"  reference {k}: {v}")
for tag in ["layermax_A_anchor035_sep8_nofilm_e2468_v10121416",
            "layermax_A_anchor035_c015_nofilm_e2468_v10121416"]:
    audit = os.path.join(base, f"audit_heldout_jointwarm_{tag}_10k")
    print(f"\n--- {tag} ---")
    for tower, fam_lim in (("expert", 0.18), ("vlm", 0.165)):
        p = os.path.join(audit, f"{tower}_audit_summary.json")
        if not os.path.exists(p):
            print(f"  {tower}: MISSING {p}")
            continue
        d = json.load(open(p))
        layers = sorted({k.split("_")[0] for k in d if k.endswith("famIoU")}, key=lambda x: int(x[1:]))
        fails, grace = [], []
        for L in layers:
            f = d[f"{L}_famIoU"]; bg = d.get(f"{L}_bgIoU", 0.0)
            cores = [d[f"{L}_t{t}"]["core50"] for t in range(10) if f"{L}_t{t}" in d]
            effs  = [d[f"{L}_t{t}"]["effnum"] for t in range(10) if f"{L}_t{t}" in d]
            mc = st.mean(cores) if cores else float("nan")
            me = min(effs) if effs else float("nan")
            print(f"  {tower} {L}: famIoU {f:.3f}  bg {bg:.3f}  core50 mean {mc:.0f}  min-eff {me:.0f}")
            if tower == "expert":
                if f > 0.20: fails.append(f"{L} famIoU {f:.3f} > 0.20")
                elif f > fam_lim: grace.append(f"{L} famIoU {f:.3f} in (0.18,0.20]")
                if bg > 0.10: fails.append(f"{L} bg {bg:.3f} > 0.10")
                if cores and mc < 400: fails.append(f"{L} mean core50 {mc:.0f} < 400 (constancy tripwire)")
                if effs and me < 300: fails.append(f"{L} min-eff {me:.0f} < 300")
            else:
                if f > fam_lim: fails.append(f"{L} famIoU {f:.3f} > 0.165")
        if tower == "expert" and len(grace) > 1:
            fails.append(f"{len(grace)} grace layers (only 1 allowed): {grace}")
        verdict = "PASS" if not fails else "FAIL: " + "; ".join(fails)
        if not fails and grace: verdict += f" (grace used: {grace})"
        print(f"  {tower} VERDICT: {verdict}")
    sub = f"/home/josh/lerobot/outputs/analysis/e46/subspan_{tag}.json"
    print(f"  palette read (manual): {sub}" if os.path.exists(sub) else f"  subspan json MISSING: {sub}")
print("\n(decision rule in the script header; palette effnum >= ~500 is the VLM anti-collapse floor)")
EOF
echo "=== [chain] BOTH PROBES COMPLETE $(date) ==="
