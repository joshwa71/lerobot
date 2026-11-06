#!/bin/bash
# Rerun of the Entry-22 "last pair" WITH the new routing-separation queue (rq=512),
# to isolate whether fixing the separation estimator (per-task coverage + multi-
# sample reference histograms via the cross-batch queue) unlocks held-out
# separation. Same configs as probes 3/4, only delta = routing_query_queue=512.
#   probe 3': standard SupCon 0.1  + rq512
#   probe 4': standard SupCon 0.05 + sep 0.5 + rq512  (favored)
# Compare audits directly against probes3 (no-queue): audit_heldout_standard_{c0.1,c0.05_sep0.5}_10k.
#
# Single GPU -> sequential. NOT set -e. ~11h/pretrain + ~35min/audit -> ~23.5h.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T=/home/josh/lerobot/outputs/train
LOG_DIR=/home/josh/lerobot/outputs/probe_logs
mkdir -p "$LOG_DIR"
log() { echo "[probes5-runner] $(date '+%F %T') $*" | tee -a "$LOG_DIR/probes5_runner.log"; }

log "=== probes 3'/4' (rq512) sequence starting ==="

log "pretrain 1/2: standard SupCon c=0.1 + rq512"
bash "$DIR/probe_10k_standard_c0.1_rq512.sh" > "$LOG_DIR/probe5_standard_c0.1_rq512.log" 2>&1
log "pretrain 1/2 exited with code $?"

log "pretrain 2/2: standard SupCon c=0.05 + sep0.5 + rq512"
bash "$DIR/probe_10k_standard_c0.05_sep0.5_rq512.sh" > "$LOG_DIR/probe5_standard_c0.05_sep0.5_rq512.log" 2>&1
log "pretrain 2/2 exited with code $?"

log "audit 1/2: c0.1 rq512 @10k"
bash "$DIR/audit_heldout_routing.sh" \
  "$T/libero_90_pi05_8_10_12_14_probe10k_standard_c0.1_rq512/checkpoints/010000/pretrained_model" \
  "audit_heldout_standard_c0.1_rq512_10k" > "$LOG_DIR/probe5_audit_c0.1_rq512.log" 2>&1
log "audit 1/2 exited with code $?"

log "audit 2/2: c0.05 sep0.5 rq512 @10k"
bash "$DIR/audit_heldout_routing.sh" \
  "$T/libero_90_pi05_8_10_12_14_probe10k_standard_c0.05_sep0.5_rq512/checkpoints/010000/pretrained_model" \
  "audit_heldout_standard_c0.05_sep0.5_rq512_10k" > "$LOG_DIR/probe5_audit_sep0.5_rq512.log" 2>&1
log "audit 2/2 exited with code $?"

log "=== probes 3'/4' (rq512) sequence done ==="
