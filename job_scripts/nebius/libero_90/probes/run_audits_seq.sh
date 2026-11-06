#!/bin/bash
# Sequential runner for the held-out routing audits (Entry 19 review).
# Order: control@40k (baseline) -> probe C (candidate) -> probe L (completeness).
# NOT set -e: audits are independent.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T=/home/josh/lerobot/outputs/train
LOG_DIR=/home/josh/lerobot/outputs/probe_logs
mkdir -p "$LOG_DIR"

log() { echo "[audit-runner] $(date '+%F %T') $*" | tee -a "$LOG_DIR/audit_runner.log"; }

log "=== audit sequence starting ==="

log "audit 1/3: control@40k"
bash "$DIR/audit_heldout_routing.sh" \
  "$T/libero_90_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.01_sep_0.25_loc_0.25_sup_128_2048_knn_36_40k/checkpoints/040000/pretrained_model" \
  "audit_heldout_control_40k" > "$LOG_DIR/audit_control.log" 2>&1
log "audit 1/3 exited with code $?"

log "audit 2/3: probeC@10k"
bash "$DIR/audit_heldout_routing.sh" \
  "$T/libero_90_pi05_8_10_12_14_probe10k_contrastive_0.05_negonly_q512/checkpoints/010000/pretrained_model" \
  "audit_heldout_probeC_10k" > "$LOG_DIR/audit_probeC.log" 2>&1
log "audit 2/3 exited with code $?"

log "audit 3/3: probeL@10k"
bash "$DIR/audit_heldout_routing.sh" \
  "$T/libero_90_pi05_8_10_12_14_probe10k_loc_1.0/checkpoints/010000/pretrained_model" \
  "audit_heldout_probeL_10k" > "$LOG_DIR/audit_probeL.log" 2>&1
log "audit 3/3 exited with code $?"

log "=== all audits done ==="
