#!/bin/bash
# Sequential runner for the two 10k routing-prior probes (Entry 19).
# Deliberately NOT `set -e`: the probes are independent experiments, so probe C
# must run even if probe L crashes.
#
# Per-probe stdout/stderr -> outputs/probe_logs/<probe>.log
# Runner narration         -> outputs/probe_logs/runner.log (and the tmux pane)

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=/home/josh/lerobot/outputs/probe_logs
mkdir -p "$LOG_DIR"

log() { echo "[runner] $(date '+%F %T') $*" | tee -a "$LOG_DIR/runner.log"; }

log "=== probe sequence starting ==="

log "starting probe L (locality 1.0)"
bash "$DIR/probe_10k_pretrain_loc_1.0.sh" > "$LOG_DIR/probe_loc_1.0.log" 2>&1
log "probe L exited with code $?"

log "starting probe C (contrastive 0.05, negatives_only, queue 512)"
bash "$DIR/probe_10k_pretrain_c_0.05_negonly_q512.sh" > "$LOG_DIR/probe_c_0.05.log" 2>&1
log "probe C exited with code $?"

log "=== all probes done ==="
