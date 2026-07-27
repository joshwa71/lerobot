#!/bin/bash
# E54 cells B + C — the other two points on the anchor U-curve (queued behind probe 3)
# =====================================================================================
# With P1/P2 measured (sep8: famIoU -0.007..-0.031 & core50 -10-15%; c0.15: famIoU
# +/-0.00 & core50 +5-13%; BOTH VLM-null), the ladder spanning the curve is:
#   P3 (running): w=0    + sep8          — broad end (core50 ~1900)
#   CELL B:       w=0.40 + sep8          — absmax-band middle. Baseline anchor040
#                 fails the RELAXED gate by 0.001 (L8 famIoU 0.201 vs the 0.20 grace
#                 line; core50 680-824 passes >=400). sep8's measured L8 delta is
#                 -0.012. PRE-REGISTERED: PASS at famIoU ~0.155-0.19, core50 ~580-740.
#   CELL C:       w=0.5  + c=0.15        — family-clean end (Josh's cell). Baseline
#                 anchor05: famIoU 0.11-0.14 (0.05 of slack), core50 330-457. c-up =
#                 the measured broadener; response should be STRONGER at w=0.5 (nearer
#                 the pooled-key regime where E45 measured +100%/4x). PRE-REGISTERED:
#                 core50 -> ~400-650 at famIoU <= ~0.16; VLM watched (c015 safe at
#                 w=0.35). Doubles as the small-cores-vs-separation causal probe.
# Fail-routes: B famIoU misses -> the sep ladder on anchored points dead-ends, C
# primary; C cores don't inflate -> run w=0.5 AS-IS through corefrac sequential under
# the relaxed gate (the absmax-recipe-on-spread bet); both pass -> morning pick by
# joint margin. Gates unchanged (expert famIoU <=0.18 grace 0.20, bg <=0.10,
# core50 >=400, min-eff >=300; VLM famIoU <=0.165).
# =====================================================================================
set -o pipefail
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_OFFLINE=1
P3_LOG="$ROOT_DIR/outputs/e54_probe3_w0sep8.log"

echo "[BC] waiting for probe 3 to finish... $(date)"
while true; do
  grep -q "\[probe3\] COMPLETE" "$P3_LOG" 2>/dev/null && { echo "[BC] probe3 marker found."; break; }
  tmux has-session -t probe3_w0sep8 2>/dev/null || { echo "[BC] probe3 session gone — proceeding."; break; }
  sleep 60
done

run_probe() {  # $1=ARM_TAG  $2=EXPERT_ANCHOR_W  $3=SEP_W  $4=CONTRASTIVE_W
  local tag="$1" aw="$2" sep="$3" con="$4"
  echo "=== [BC] probe ${tag} (anchor_w=${aw} sep=${sep} c=${con}) START $(date) ==="
  (
    set -eo pipefail
    export ARM_TAG="$tag"
    export EXP_LAYERS='[2,4,6,8]'
    export VLM_LAYERS='[10,12,14,16]'
    export EXP_N=256 EXP_R=2 EXP_KNN=36
    export VLM_N=256 VLM_R=2 VLM_KNN=16
    export ROUTER_FAST=true
    export AUDIT_BS=16 AUDIT_STEPS=200
    export LANG_TO_QUERY=false
    export EXPERT_ANCHOR=text
    export EXPERT_ANCHOR_W="$aw"
    export SEP_W="$sep"
    export CONTRASTIVE_W="$con"
    source "$SCRIPT_DIR/joint_rwarmup_common.sh"
  ) || echo "=== [BC] probe ${tag} FAILED (continuing) ==="
  echo "=== [BC] probe ${tag} END $(date) ==="
}

run_probe layermax_A_anchor040_sep8_nofilm_e2468_v10121416 0.40 8.0 0.05
run_probe layermax_A_anchor05_c015_nofilm_e2468_v10121416  0.50 5.0 0.15

python - <<'EOF' || true
import json, os, statistics as st
base = "/home/josh/lerobot/outputs/train/"
cells = {
  "CELL B w040+sep8": ("layermax_A_anchor040_sep8_nofilm_e2468_v10121416",
                       "baseline anchor040: famIoU 0.167-0.201 core50 680-824"),
  "CELL C w05+c015":  ("layermax_A_anchor05_c015_nofilm_e2468_v10121416",
                       "baseline anchor05: famIoU 0.11-0.14 core50 330-457"),
}
print("\n======== E54 CELLS B/C GATE SUMMARY ========")
for name, (tag, ref) in cells.items():
    print(f"\n--- {name} ({ref}) ---")
    for tower, fam_lim in (("expert", 0.18), ("vlm", 0.165)):
        p = os.path.join(base, f"audit_heldout_jointwarm_{tag}_10k", f"{tower}_audit_summary.json")
        if not os.path.exists(p):
            print(f"  {tower}: MISSING"); continue
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
                if st.mean(cores) < 400: fails.append(f"{L} core50 {st.mean(cores):.0f}")
                if min(effs) < 300: fails.append(f"{L} min-eff")
            elif f > fam_lim: fails.append(f"{L} famIoU {f:.3f}")
        if tower == "expert" and len(grace) > 1: fails.append(f"{len(grace)} grace layers")
        print(f"  {tower} VERDICT: {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}{' (grace: '+','.join(grace)+')' if grace and not fails else ''}")
EOF
echo "=== [BC] BOTH CELLS COMPLETE $(date) ==="
