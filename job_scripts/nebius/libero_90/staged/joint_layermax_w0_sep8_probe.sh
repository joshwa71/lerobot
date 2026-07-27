#!/bin/bash
# E54 probe 3 — SPREAD substrate, NO anchor, sep=8 (queued behind the P1/P2 chain)
# =====================================================================================
# Rationale (P1 postmortem): the famIoU-vs-anchor-w curve is U-shaped — plain spread
# (w=0) sits at 0.163-0.212 famIoU / core50 1842-2001, BETTER than every anchored point
# below w=0.5 (the pooled-instruction anchor injects a near-shared component for the
# lookalike basket family and pulls it together; it only cleans bg). Plain spread misses
# the gate by ~exactly sep8's measured effect (P1: -0.007..-0.031/layer at +3 sep).
# This probe = the plain-spread certificate + ONE delta: sep 5->8. FiLM stays ON
# (LANG_TO_QUERY unset -> default true) to match the plain-spread comparator; no anchor.
# PRE-REGISTERED (transfer P1's per-layer deltas onto plain): famIoU ~0.181/0.179/
# 0.182/0.151 (PASS w/ one-grace headroom), bg 0.08-0.10 (L2 borderline vs the <=0.10
# line), core50 ~1600-1800. FAIL-ROUTES: deltas don't transfer at w=0 -> sep 12 next;
# bg stuck >0.10 at L2 -> accept-with-note or mild anchor (w<=0.1) purely for bg;
# famIoU unmoved -> pivot to the anchored-w0.5 + corefrac sequential (absmax recipe on
# spread; warm-up already on disk).
# Gates unchanged: expert famIoU <=0.18 (one grace <=0.20), bg <=0.10, core50 >=400,
# min-eff >=300; VLM famIoU <=0.165.
# =====================================================================================
set -o pipefail
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_OFFLINE=1
CHAIN_LOG="$ROOT_DIR/outputs/e54_probes_w035.log"

# ---- wait for the P1/P2 chain (marker OR its tmux session gone) ----
echo "[probe3] waiting for the probes_w035 chain to finish... $(date)"
while true; do
  grep -q "BOTH PROBES COMPLETE" "$CHAIN_LOG" 2>/dev/null && { echo "[probe3] chain marker found."; break; }
  tmux has-session -t probes_w035 2>/dev/null || { echo "[probe3] probes_w035 session gone — proceeding."; break; }
  sleep 60
done

echo "=== [probe3] w=0 sep=8 START $(date) ==="
(
  set -eo pipefail
  export ARM_TAG=layermax_sep8_e2468_v10121416
  export EXP_LAYERS='[2,4,6,8]'
  export VLM_LAYERS='[10,12,14,16]'
  export EXP_N=256 EXP_R=2 EXP_KNN=36
  export VLM_N=256 VLM_R=2 VLM_KNN=16
  export ROUTER_FAST=true
  export AUDIT_BS=16 AUDIT_STEPS=200
  export EXPERT_ANCHOR=      # NO anchor (w=0 cell); LANG_TO_QUERY unset => FiLM on, matching the plain-spread comparator
  export SEP_W=8.0           # the single delta vs the plain-spread certificate
  source "$SCRIPT_DIR/joint_rwarmup_common.sh"
) || echo "=== [probe3] FAILED ==="

python - <<'EOF' || true
import json, os, statistics as st
base = "/home/josh/lerobot/outputs/train/"
tag = "layermax_sep8_e2468_v10121416"
print("\n======== PROBE 3 (w=0, sep8) GATE SUMMARY ========")
print("references: plain spread famIoU 0.212/0.195/0.189/0.163 bg 0.094-0.119 core50 1842-2001; P1 sep8@w035 deltas -0.031/-0.016/-0.007/-0.012")
for tower, fam_lim in (("expert", 0.18), ("vlm", 0.165)):
    p = os.path.join(base, f"audit_heldout_jointwarm_{tag}_10k", f"{tower}_audit_summary.json")
    if not os.path.exists(p):
        print(f"  {tower}: MISSING {p}"); continue
    d = json.load(open(p))
    layers = sorted({k.split("_")[0] for k in d if k.endswith("famIoU")}, key=lambda x: int(x[1:]))
    fails, grace = [], []
    for L in layers:
        f = d[f"{L}_famIoU"]; bg = d.get(f"{L}_bgIoU", 0.0)
        cores = [d[f"{L}_t{t}"]["core50"] for t in range(10) if f"{L}_t{t}" in d]
        effs  = [d[f"{L}_t{t}"]["effnum"] for t in range(10) if f"{L}_t{t}" in d]
        print(f"  {tower} {L}: famIoU {f:.3f}  bg {bg:.3f}  core50 mean {st.mean(cores):.0f}  min-eff {min(effs):.0f}")
        if tower == "expert":
            if f > 0.20: fails.append(f"{L} famIoU {f:.3f}")
            elif f > fam_lim: grace.append(L)
            if bg > 0.10: fails.append(f"{L} bg {bg:.3f}")
            if st.mean(cores) < 400: fails.append(f"{L} core50")
            if min(effs) < 300: fails.append(f"{L} min-eff")
        elif f > fam_lim: fails.append(f"{L} famIoU {f:.3f}")
    if tower == "expert" and len(grace) > 1: fails.append(f"{len(grace)} grace layers")
    print(f"  {tower} VERDICT: {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}{' (grace: '+','.join(grace)+')' if grace and not fails else ''}")
EOF
echo "=== [probe3] COMPLETE $(date) ==="
