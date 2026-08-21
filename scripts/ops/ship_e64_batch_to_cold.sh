#!/bin/bash
# Ship the E64 r512 baseline batch to cold storage (Josh, 21 Aug: "ship this batch
# of baselines to cold storage when you're done the triangles"). LOCAL script —
# the destination drive is on the desk PC, so this cannot be a VM systemd unit.
# Runs CONCURRENTLY with the r128 ladder on the VM (different machine, no GPU use).
#
# Gate: the `e64-triangles` unit has exited AND all 20 triangle rows exist — i.e.
# every measurement that needs these checkpoints hot has been taken.
#   naive triangle : 10 rows, merged6x2 triangle : 10 rows
# Protocol (Batch-2/3 manifest, unchanged): rsync -aH with 3 retries -> verify with
# rsync -aHc --dry-run zero-transfer AND du -sb byte-exact both sides -> VM-side
# rm -rf ONLY on PASS -> append to the manifest.
#
# LAUNCH DETACHED so a closed session does not kill it (the 8 Aug lesson: a
# session-side shipper died in a desk-PC reboot having transferred nothing):
#   nohup setsid bash scripts/ops/ship_e64_batch_to_cold.sh >/dev/null 2>&1 &
# Re-running is safe: dirs marked PASS are skipped.
set -uo pipefail
SRC_HOST=nebius-spot
SRC_BASE=/home/josh/lerobot/outputs/train
DEST=/media/josh/Backup/memory-models
STATUS=$DEST/_archive_e64_status.txt
LOG=$DEST/_archive_e64.log
MANIFEST=$DEST/_ARCHIVE_MANIFEST_complete.md
touch "$STATUS"

# ~158 G total measured 21 Aug: multitask 47G, ten specialists 93G,
# naive-r512 (grows to ~19G as blocks land), plus the killed r256 naive partial.
DIRS=(
  loraft_multitask10_r512_50k
  loraft_baseline_r512
  libero_10_seq10_naive_lora_r512_a128_steps5k
  libero_10_seq10_naive_lora_r256_a64_steps5k
)

echo "=== ship_e64: waiting on e64-triangles + 20 triangle rows $(date -u) ===" >> "$LOG"
while true; do
  st=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$SRC_HOST" \
    'systemctl is-active e64-triangles 2>/dev/null; \
     ls /home/josh/lerobot/outputs/analysis/e60/seeds_tri_naive10_r512_b*.json \
        /home/josh/lerobot/outputs/analysis/e60/seeds_tri_merged6x2_10task_b*.json 2>/dev/null | wc -l' 2>/dev/null) || { sleep 600; continue; }
  unit=$(echo "$st" | sed -n 1p); nrows=$(echo "$st" | sed -n 2p)
  if [ "$unit" != "active" ] && [ "$unit" != "activating" ] && [ "${nrows:-0}" -ge 20 ]; then break; fi
  sleep 600
done
echo "=== ship_e64: gate passed (unit=$unit rows=$nrows) — transferring $(date -u) ===" >> "$LOG"

fail=0
for c in "${DIRS[@]}"; do
  grep -q "^PASS $c$" "$STATUS" && { echo "[skip] $c already PASS" >> "$LOG"; continue; }
  ssh -o BatchMode=yes "$SRC_HOST" "[ -d $SRC_BASE/$c ]" || { echo "[skip] $c not on VM" >> "$LOG"; continue; }
  ok=0
  for attempt in 1 2 3; do
    rsync -aH --partial --timeout=120 "$SRC_HOST:$SRC_BASE/$c" "$DEST/" >> "$LOG" 2>&1 && { ok=1; break; }
    echo "[retry $attempt] $c" >> "$LOG"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "FAIL-TRANSFER $c" | tee -a "$STATUS" >> "$LOG"; fail=1; continue; }
  # verify: checksum dry-run must show zero transfer, and byte counts must match
  delta=$(rsync -aHc --dry-run --itemize-changes "$SRC_HOST:$SRC_BASE/$c" "$DEST/" 2>/dev/null | grep -c '^[<>ch]') || delta=999
  vm_b=$(ssh -o BatchMode=yes "$SRC_HOST" "du -sb $SRC_BASE/$c | cut -f1")
  co_b=$(du -sb "$DEST/$c" | cut -f1)
  if [ "$delta" = "0" ] && [ "$vm_b" = "$co_b" ]; then
    echo "PASS $c" >> "$STATUS"
    echo "[verified] $c ($co_b bytes)" >> "$LOG"
    ssh -o BatchMode=yes "$SRC_HOST" "rm -rf $SRC_BASE/$c" && echo "[vm-deleted] $c" >> "$LOG"
    printf '| %s | %s bytes | E64 r512 baseline batch | %s |\n' "$c" "$co_b" "$(date -u +%Y-%m-%d)" >> "$MANIFEST"
  else
    echo "FAIL-VERIFY $c delta=$delta vm=$vm_b cold=$co_b" | tee -a "$STATUS" >> "$LOG"; fail=1
  fi
done
echo "=== SHIP-E64-DONE fail=$fail $(date -u) ===" >> "$LOG"
