#!/bin/bash
# Runner for the Entry-22 follow-up: get separation INTO the capacity-safe regime.
#   probe 3: standard SupCon 0.1               (contrastive-weight axis)
#   probe 4: standard SupCon 0.05 + sep 0.5    (direct slot-space separation; favored)
# Then audits both 10k checkpoints.
#
# Single GPU -> sequential. NOT set -e: independent experiments.
# ~11h per pretrain + ~35min per audit -> ~23.5h total.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T=/home/josh/lerobot/outputs/train
LOG_DIR=/home/josh/lerobot/outputs/probe_logs
mkdir -p "$LOG_DIR"
log() { echo "[probes3-runner] $(date '+%F %T') $*" | tee -a "$LOG_DIR/probes3_runner.log"; }

log "=== probes 3/4 sequence starting ==="

log "pretrain 1/2: standard SupCon c=0.1"
bash "$DIR/probe_10k_standard_c0.1.sh" > "$LOG_DIR/probe3_standard_c0.1.log" 2>&1
log "pretrain 1/2 exited with code $?"

log "pretrain 2/2: standard SupCon c=0.05 + sep 0.5"
bash "$DIR/probe_10k_standard_c0.05_sep0.5.sh" > "$LOG_DIR/probe3_standard_c0.05_sep0.5.log" 2>&1
log "pretrain 2/2 exited with code $?"

log "audit 1/2: standard c=0.1 @10k"
bash "$DIR/audit_heldout_routing.sh" \
  "$T/libero_90_pi05_8_10_12_14_probe10k_standard_c0.1/checkpoints/010000/pretrained_model" \
  "audit_heldout_standard_c0.1_10k" > "$LOG_DIR/probe3_audit_c0.1.log" 2>&1
log "audit 1/2 exited with code $?"

log "audit 2/2: standard c=0.05 sep0.5 @10k"
bash "$DIR/audit_heldout_routing.sh" \
  "$T/libero_90_pi05_8_10_12_14_probe10k_standard_c0.05_sep0.5/checkpoints/010000/pretrained_model" \
  "audit_heldout_standard_c0.05_sep0.5_10k" > "$LOG_DIR/probe3_audit_sep0.5.log" 2>&1
log "audit 2/2 exited with code $?"

log "=== probes 3/4 sequence done ==="
