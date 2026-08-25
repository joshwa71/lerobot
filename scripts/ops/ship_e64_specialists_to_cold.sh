#!/bin/bash
# Ship the r128 and r64 LoRA specialist batches to cold storage (Josh, 25 Aug:
# "ship the specialists to cold storage after their evals are done and logged so we
# don't run out of storage"). LOCAL script -- the destination drive is on the desk PC.
#
# Per-directory gate, evaluated independently so each ships as soon as IT is eligible:
#   loraft_baseline_r128 : unit e64b-r128 inactive AND 10 seeds_spec_r128_e*.json
#                          AND the oracle table is in the research log
#   loraft_baseline_r64  : unit e64c-r64  inactive AND 10 seeds_spec_r64_e*.json
#                          AND the oracle table is in the research log
# The log check is what makes "and logged" real: the checkpoints stay hot until the
# numbers are actually written down, so a shipping bug can never cost us a result.
#
# Protocol identical to the E64 r512 batch (manifest, 26 Jul): rsync -aH with 3
# retries -> verify with `rsync -aHc --dry-run` showing zero differing files AND
# `du -sb` byte-exact both sides -> VM-side rm -rf ONLY on PASS -> manifest row.
# NB the verify counts differences WITHOUT `... | grep -c) || delta=999`: grep -c
# exits 1 on zero matches, which is the PASS case, and that inverted every result
# in the 24 Aug run. Command failure and zero-differences are separated below.
#
# LAUNCH DETACHED, and from a copy, not from the repo path:
#   cp scripts/ops/ship_e64_specialists_to_cold.sh /tmp/.../ship_spec.sh
#   nohup setsid bash /tmp/.../ship_spec.sh >/dev/null 2>&1 &
# (bash reads a script lazily by byte offset; editing the file under a running
# instance re-executes shifted text -- that happened on 24 Aug.)
# Session-side processes do NOT survive a desk-PC reboot. Re-running is safe:
# PASS dirs are skipped and nothing re-transfers.
set -uo pipefail
SRC_HOST=nebius-spot
SRC_BASE=/home/josh/lerobot/outputs/train
DEST=/media/josh/Backup/memory-models
REPO=/home/josh/phddev/lerobot
LOG=$DEST/_archive_e64spec.log
STATUS=$DEST/_archive_e64spec_status.txt
MANIFEST=$DEST/_ARCHIVE_MANIFEST_complete.md
touch "$STATUS"

# dir : unit that must be inactive : row glob : marker that must appear in the log
SPECS=(
  "loraft_baseline_r128:e64b-r128:seeds_spec_r128_e:r128 SPECIALIST ORACLE"
  "loraft_baseline_r64:e64c-r64:seeds_spec_r64_e:r64 SPECIALIST ORACLE"
)

echo "=== ship_e64spec: armed $(date -u) ===" >> "$LOG"

eligible() {  # $1 unit  $2 row-glob-prefix  $3 log marker
  local unit=$1 glob=$2 marker=$3 st rows
  st=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$SRC_HOST" "systemctl is-active $unit 2>/dev/null; true") || return 1
  [ "$st" = "active" ] || [ "$st" = "activating" ] && return 1
  rows=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$SRC_HOST" \
        "ls /home/josh/lerobot/outputs/analysis/e60/${glob}*.json 2>/dev/null | wc -l") || return 1
  [ "${rows:-0}" -ge 10 ] || return 1
  grep -q "$marker" "$REPO/projects/research_log.md" || return 1
  return 0
}

ship() {  # $1 dir
  local c=$1 ok=0 attempt delta vout vrc vm_b co_b
  grep -q "^PASS $c$" "$STATUS" && { echo "[skip] $c already PASS" >> "$LOG"; return 0; }
  ssh -o BatchMode=yes "$SRC_HOST" "[ -d $SRC_BASE/$c ]" || { echo "[skip] $c not on VM" >> "$LOG"; return 0; }
  echo "=== $c: transferring $(date -u) ===" >> "$LOG"
  for attempt in 1 2 3; do
    rsync -aH --partial --timeout=120 "$SRC_HOST:$SRC_BASE/$c" "$DEST/" >> "$LOG" 2>&1 && { ok=1; break; }
    echo "[retry $attempt] $c" >> "$LOG"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "FAIL-TRANSFER $c" | tee -a "$STATUS" >> "$LOG"; return 1; }
  vout=$(rsync -aHc --dry-run --itemize-changes "$SRC_HOST:$SRC_BASE/$c" "$DEST/" 2>/dev/null); vrc=$?
  if [ "$vrc" -ne 0 ]; then
    delta=999
  else
    delta=$(printf '%s\n' "$vout" | grep -c '^[<>ch]'); true
    delta=${delta:-999}
  fi
  vm_b=$(ssh -o BatchMode=yes "$SRC_HOST" "du -sb $SRC_BASE/$c | cut -f1")
  co_b=$(du -sb "$DEST/$c" | cut -f1)
  if [ "$delta" = "0" ] && [ -n "$vm_b" ] && [ "$vm_b" = "$co_b" ]; then
    echo "PASS $c" >> "$STATUS"
    echo "[verified] $c ($co_b bytes)" >> "$LOG"
    ssh -o BatchMode=yes "$SRC_HOST" "rm -rf $SRC_BASE/$c" && echo "[vm-deleted] $c" >> "$LOG"
    printf '| %s | %s bytes | E64 specialist ladder batch | %s |\n' "$c" "$co_b" "$(date -u +%Y-%m-%d)" >> "$MANIFEST"
  else
    echo "FAIL-VERIFY $c delta=$delta vm=$vm_b cold=$co_b" | tee -a "$STATUS" >> "$LOG"; return 1
  fi
}

while true; do
  done_all=1
  for spec in "${SPECS[@]}"; do
    IFS=: read -r dir unit glob marker <<< "$spec"
    grep -q "^PASS $dir$" "$STATUS" && continue
    done_all=0
    if eligible "$unit" "$glob" "$marker"; then
      echo "[gate passed] $dir $(date -u)" >> "$LOG"
      ship "$dir"
    fi
  done
  [ "$done_all" = 1 ] && break
  sleep 600
done
echo "=== SHIP-E64SPEC-DONE $(date -u) ===" >> "$LOG"
