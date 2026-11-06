#!/bin/bash
# E54 probe 4 — P3-nofilm: the ZERO-language-machinery router (queued behind cells B/C)
# =====================================================================================
# P3 (w=0, sep8, FiLM on) PASSED the full gate (famIoU 0.178/0.176/0.177/0.156, bg
# 0.083-0.097, core50 1383-1717; VLM 0.132-0.154). This cell = P3 with ONE delta:
# lang_to_query=false (no FiLM), and still no anchor -> the expert router is pure
# proj(x) + sep/contrastive. Josh wants FiLM gone (legacy machinery, mpnet dependency);
# E28 measured it near-inert at inference (gamma~0, beta ~5% nub, stripping it moved
# basket routing ~0.001 IoU) — but every certified non-anchored router trained WITH it,
# and warm-up aux losses do reach film_mlp, so this is the load-bearing test.
# PRE-REGISTERED: famIoU within ~0.01-0.02 of P3 (0.156-0.178) at similar capacity =>
# FiLM removable, THIS becomes the graduation candidate. Materially worse (any layer
# >0.20, or famIoU mean +>0.03) => FiLM was doing quiet separation work at warm-up;
# keep it with a one-line justification and graduate P3 instead.
# Gates unchanged (expert famIoU <=0.18 grace 0.20, bg <=0.10, core50 >=400,
# min-eff >=300; VLM famIoU <=0.165).
# =====================================================================================
set -o pipefail
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_OFFLINE=1
BC_LOG="$ROOT_DIR/outputs/e54_probesBC.log"

echo "[probe4] waiting for the B/C chain to finish... $(date)"
while true; do
  grep -q "BOTH CELLS COMPLETE" "$BC_LOG" 2>/dev/null && { echo "[probe4] BC marker found."; break; }
  tmux has-session -t probesBC 2>/dev/null || { echo "[probe4] probesBC session gone — proceeding."; break; }
  sleep 60
done

echo "=== [probe4] w=0 sep=8 NOFILM START $(date) ==="
(
  set -eo pipefail
  export ARM_TAG=layermax_sep8_nofilm_e2468_v10121416
  export EXP_LAYERS='[2,4,6,8]'
  export VLM_LAYERS='[10,12,14,16]'
  export EXP_N=256 EXP_R=2 EXP_KNN=36
  export VLM_N=256 VLM_R=2 VLM_KNN=16
  export ROUTER_FAST=true
  export AUDIT_BS=16 AUDIT_STEPS=200
  export EXPERT_ANCHOR=       # no anchor
  export LANG_TO_QUERY=false  # THE single delta vs P3: FiLM off -> pure proj(x) router
  export SEP_W=8.0
  source "$SCRIPT_DIR/joint_rwarmup_common.sh"
) || echo "=== [probe4] FAILED ==="

python - <<'EOF' || true
import json, os, statistics as st
base = "/home/josh/lerobot/outputs/train/"
tag = "layermax_sep8_nofilm_e2468_v10121416"
p3  = "layermax_sep8_e2468_v10121416"
print("\n======== PROBE 4 (w=0, sep8, NOFILM) GATE SUMMARY ========")
print("reference P3 (FiLM on): expert famIoU 0.178/0.176/0.177/0.156 bg 0.083-0.097 core50 1383-1717; VLM 0.132-0.154")
for tower, fam_lim in (("expert", 0.18), ("vlm", 0.165)):
    p = os.path.join(base, f"audit_heldout_jointwarm_{tag}_10k", f"{tower}_audit_summary.json")
    ref = os.path.join(base, f"audit_heldout_jointwarm_{p3}_10k", f"{tower}_audit_summary.json")
    if not os.path.exists(p):
        print(f"  {tower}: MISSING"); continue
    d = json.load(open(p))
    r = json.load(open(ref)) if os.path.exists(ref) else {}
    layers = sorted({k.split("_")[0] for k in d if k.endswith("famIoU")}, key=lambda x: int(x[1:]))
    fails, grace = [], []
    for L in layers:
        f = d[f"{L}_famIoU"]; bg = d.get(f"{L}_bgIoU", 0.0)
        rf = r.get(f"{L}_famIoU")
        cores = [d[f"{L}_t{t}"]["core50"] for t in range(10) if f"{L}_t{t}" in d]
        effs  = [d[f"{L}_t{t}"]["effnum"] for t in range(10) if f"{L}_t{t}" in d]
        delta = f" (P3 {rf:.3f}, d {f-rf:+.3f})" if rf is not None else ""
        print(f"  {tower} {L}: famIoU {f:.3f}{delta}  bg {bg:.3f}  core50 mean {st.mean(cores):.0f}  min-eff {min(effs):.0f}")
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
echo "=== [probe4] COMPLETE $(date) ==="
