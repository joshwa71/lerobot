#!/bin/bash
# E68 certificates C1/C2 — COMMON BODY (sourced by e68_c1_cert_sep0.sh / e68_c2_cert_c0.sh, which set
# ARM_TAG + SEP_W + CONTRASTIVE_W). The paper cell's router warm-up recipe verbatim (E62 chain,
# joint_rwarmup_common.sh: anchored w0.40, FiLM-free, broadcast losses, prepass, router-only fast
# path, 10k) with ONE auxiliary-loss weight zeroed, then the held-out audit + analyses. No gate, no
# downstream: the certificate IS the result (replaces Table II D/E, which are joint-era probes).
# OOM fallback for the warm-up: bs32 -> bs16xacc2 (E11 caveat: accumulation shrinks the in-batch
# contrastive pool; queues cover it; note in the log).
set -eo pipefail
export HF_HUB_OFFLINE=1
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
export PREPASS=true
: "${ARM_TAG:?}" "${SEP_W:?}" "${CONTRASTIVE_W:?}"
export ARM_TAG SEP_W CONTRASTIVE_W
RUN=libero_90_pi05_jointwarm10k_${ARM_TAG}
OUT="$ROOT_DIR/outputs/train/$RUN"
CKPT="$OUT/checkpoints/last/pretrained_model"
echo "E68 certificate [$ARM_TAG] (sep $SEP_W, contrastive $CONTRASTIVE_W) started on $(hostname) at $(date -u)"
if [ -d "$CKPT" ]; then
  bash "$SCRIPT_DIR/joint_rwarmup_common.sh"      # warm-up skipped; audit/analyses skip-guarded
else
  if ! BATCH_SIZE=32 GRAD_ACCUM=1 bash "$SCRIPT_DIR/joint_rwarmup_common.sh"; then
    if [ -d "$CKPT" ]; then
      echo "[cert] warm-up completed but a later stage failed - rerunning audit/analyses only"
      bash "$SCRIPT_DIR/joint_rwarmup_common.sh"
    else
      echo "[cert] bs32 warm-up failed before its checkpoint (treating as VRAM) - retrying bs16 x accum2"
      rm -rf "$OUT"
      BATCH_SIZE=16 GRAD_ACCUM=2 bash "$SCRIPT_DIR/joint_rwarmup_common.sh"
    fi
  fi
fi
[ -d "$CKPT" ] || { echo "ERROR: certificate warm-up checkpoint missing: $CKPT"; exit 1; }
# Informational gate readout (the E62 bg-first bands); never exits non-zero.
AUDIT_DIR="$ROOT_DIR/outputs/train/audit_heldout_jointwarm_${ARM_TAG}_10k"
python - "$AUDIT_DIR" <<'PY' || echo "[cert] gate readout unavailable (audit incomplete?)"
import json, sys
base = sys.argv[1] + "/"
exp = json.load(open(base + "expert_audit_summary.json"))
vlm = json.load(open(base + "vlm_audit_summary.json"))
def layers(d):
    return sorted({k.split("_")[0] for k in d if k.endswith("famIoU")}, key=lambda x: int(x[1:]))
fails = []
for L in layers(exp):
    f = exp[f"{L}_famIoU"]; bg = exp.get(f"{L}_bgIoU", 0.0)
    cores = [exp[f"{L}_t{t}"]["core50"] for t in range(10) if f"{L}_t{t}" in exp]
    effs = [exp[f"{L}_t{t}"]["effnum"] for t in range(10) if f"{L}_t{t}" in exp]
    print(f"[cert] expert {L}: famIoU {f:.3f} bg {bg:.3f} core50 mean {sum(cores)/len(cores):.0f} min-eff {min(effs):.0f}")
    if bg > 0.10: fails.append(f"expert {L} bgIoU {bg:.3f} > 0.10")
    if sum(cores)/len(cores) < 400: fails.append(f"expert {L} mean core50 < 400")
    if min(effs) < 300: fails.append(f"expert {L} min-task effnum < 300")
for L in layers(vlm):
    f = vlm[f"{L}_famIoU"]
    effs = [vlm[f"{L}_t{t}"]["effnum"] for t in range(10) if f"{L}_t{t}" in vlm]
    print(f"[cert] vlm {L}: famIoU {f:.3f} min-eff {min(effs):.0f}")
    if min(effs) < 150: fails.append(f"vlm {L} min-task effnum < 150")
    if f >= 0.45: fails.append(f"vlm {L} famIoU {f:.3f} >= 0.45")
print("[cert] GATE (informational):", "PASS" if not fails else "FAIL -> " + "; ".join(fails))
PY
echo "E68 certificate [$ARM_TAG] COMPLETE at $(date -u)"
