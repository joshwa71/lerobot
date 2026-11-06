#!/bin/bash
# PROBES 7-10 (Entry 25): isolate whether SEPARATION ALONE is the lever.
# All single-knob deltas from the P6 anchor (c0.05 / sep2.0 / loc0.25 / rq512;
# held-out famIoU 0.311 / core50 2368 — the first decoupled separation result).
#   probe 7  : contrastive 0.05 -> 0      (goal A: is contrastive needed at all?)
#   probe 8  : sep 2.0 -> 3.0             (goal B: sep curve)
#   probe 9  : sep 2.0 -> 5.0             (goal B: sep curve, far end / turnover)
#   probe 10 : locality 0.25 -> 0         (goal C: does locality do anything?)
# NOTE: contrastive read as 0.05 (project's capacity-safe value); "0.5" in the
#   request taken as a typo — 0.5 would over-compact and confound B/C. See Entry 25.
#
# Interleaved pretrain->audit per probe so a mid-batch failure still leaves earlier
# audits complete. Single GPU -> sequential. NOT set -e. ~11.6h each -> ~2 days total.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T=/home/josh/lerobot/outputs/train
LOG_DIR=/home/josh/lerobot/outputs/probe_logs
mkdir -p "$LOG_DIR"
log() { echo "[probes7-10-runner] $(date '+%F %T') $*" | tee -a "$LOG_DIR/probes7_10_runner.log"; }

# name : pretrain_script : run_name : audit_name
run_one() {
  local label="$1" script="$2" run="$3" audit="$4"
  log "pretrain: $label"
  bash "$DIR/$script" > "$LOG_DIR/${audit}_pretrain.log" 2>&1
  log "pretrain $label exited code $?"
  log "audit: $label"
  bash "$DIR/audit_heldout_routing.sh" \
    "$T/$run/checkpoints/010000/pretrained_model" \
    "$audit" > "$LOG_DIR/${audit}_audit.log" 2>&1
  log "audit $label exited code $?"
}

log "=== probes 7-10 sequence starting ==="

run_one "P7 contrastive=0 sep2.0" \
  "probe_10k_standard_c0_sep2.0_rq512.sh" \
  "libero_90_pi05_8_10_12_14_probe10k_standard_c0_sep2.0_rq512" \
  "audit_heldout_standard_c0_sep2.0_rq512_10k"

run_one "P8 c0.05 sep3.0" \
  "probe_10k_standard_c0.05_sep3.0_rq512.sh" \
  "libero_90_pi05_8_10_12_14_probe10k_standard_c0.05_sep3.0_rq512" \
  "audit_heldout_standard_c0.05_sep3.0_rq512_10k"

run_one "P9 c0.05 sep5.0" \
  "probe_10k_standard_c0.05_sep5.0_rq512.sh" \
  "libero_90_pi05_8_10_12_14_probe10k_standard_c0.05_sep5.0_rq512" \
  "audit_heldout_standard_c0.05_sep5.0_rq512_10k"

run_one "P10 c0.05 sep2.0 NO-locality" \
  "probe_10k_standard_c0.05_sep2.0_noloc_rq512.sh" \
  "libero_90_pi05_8_10_12_14_probe10k_standard_c0.05_sep2.0_noloc_rq512" \
  "audit_heldout_standard_c0.05_sep2.0_noloc_rq512_10k"

log "=== probes 7-10 sequence done ==="
