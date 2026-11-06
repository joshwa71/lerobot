#!/bin/bash
# PROBE 6 (Entry 24 follow-up): aggressive separation isolation.
#   standard SupCon 0.05 (FIXED, no contrastive confound) + sep 0.5 -> 2.0 + rq512.
# Decisive test of whether held-out separation is scale-limited. Chains the 10k
# pretrain then the held-out routing audit. Single GPU -> sequential. NOT set -e.
# ~11h pretrain + ~35min audit.
#
# Read the audit against the rq512 anchors already on disk:
#   audit_heldout_standard_c0.05_sep0.5_rq512_10k  (sep0.5, the direct precursor)
#   audit_heldout_control_40k (broad 0.349) / audit_heldout_c005_40k (collapsed 0.133)
# Gate: GOOD = L14 famIoU <= ~0.28 AND core50 >= ~1500 (jointly).

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T=/home/josh/lerobot/outputs/train
LOG_DIR=/home/josh/lerobot/outputs/probe_logs
mkdir -p "$LOG_DIR"
log() { echo "[probe6-runner] $(date '+%F %T') $*" | tee -a "$LOG_DIR/probe6_runner.log"; }

log "=== probe 6 (sep 2.0 + rq512) sequence starting ==="

log "pretrain 1/1: standard SupCon c=0.05 + sep2.0 + rq512"
bash "$DIR/probe_10k_standard_c0.05_sep2.0_rq512.sh" > "$LOG_DIR/probe6_standard_c0.05_sep2.0_rq512.log" 2>&1
log "pretrain 1/1 exited with code $?"

log "audit 1/1: c0.05 sep2.0 rq512 @10k"
bash "$DIR/audit_heldout_routing.sh" \
  "$T/libero_90_pi05_8_10_12_14_probe10k_standard_c0.05_sep2.0_rq512/checkpoints/010000/pretrained_model" \
  "audit_heldout_standard_c0.05_sep2.0_rq512_10k" > "$LOG_DIR/probe6_audit_sep2.0_rq512.log" 2>&1
log "audit 1/1 exited with code $?"

log "=== probe 6 (sep 2.0 + rq512) sequence done ==="
