#!/bin/bash
# Runner for the Entry-21 2-knob isolation: 2x 10k pretrain + held-out audit each.
#   probe 1: negonly=true,  weight=0.025  (dose knob)
#   probe 2: negonly=false, weight=0.05   (structure knob)
# Then audits both 10k checkpoints so capacity (core50/effnum) and separation
# (family IoU) can be read against control@40k and the failed negonly-0.05 anchors.
#
# Single GPU -> sequential. NOT set -e: stages are independent experiments.
# ~11h per pretrain + ~35min per audit  ->  ~23.5h total.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T=/home/josh/lerobot/outputs/train
LOG_DIR=/home/josh/lerobot/outputs/probe_logs
mkdir -p "$LOG_DIR"

log() { echo "[probes2-runner] $(date '+%F %T') $*" | tee -a "$LOG_DIR/probes2_runner.log"; }

log "=== 2-knob probe sequence starting ==="

log "pretrain 1/2: negonly c=0.025"
bash "$DIR/probe_10k_negonly_c0.025.sh" > "$LOG_DIR/probe2_negonly_c0.025.log" 2>&1
log "pretrain 1/2 exited with code $?"

log "pretrain 2/2: standard SupCon c=0.05"
bash "$DIR/probe_10k_standard_c0.05.sh" > "$LOG_DIR/probe2_standard_c0.05.log" 2>&1
log "pretrain 2/2 exited with code $?"

log "audit 1/2: negonly c=0.025 @10k"
bash "$DIR/audit_heldout_routing.sh" \
  "$T/libero_90_pi05_8_10_12_14_probe10k_negonly_c0.025/checkpoints/010000/pretrained_model" \
  "audit_heldout_negonly_c0.025_10k" > "$LOG_DIR/probe2_audit_negonly.log" 2>&1
log "audit 1/2 exited with code $?"

log "audit 2/2: standard c=0.05 @10k"
bash "$DIR/audit_heldout_routing.sh" \
  "$T/libero_90_pi05_8_10_12_14_probe10k_standard_c0.05/checkpoints/010000/pretrained_model" \
  "audit_heldout_standard_c0.05_10k" > "$LOG_DIR/probe2_audit_standard.log" 2>&1
log "audit 2/2 exited with code $?"

log "=== 2-knob probe sequence done ==="
