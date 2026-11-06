#!/bin/bash
# Ship the two full-FT baseline run dirs straight to cold storage once both are
# done (Josh, 7 Aug: "when 3 and 4 are done, just ship them straight into cold").
# Local script. Gate: the fullft-l90l10 unit on the VM has exited AND both
# 4-seed JSONs exist (evals complete => checkpoints no longer needed hot).
# Protocol per the Batch-2 manifest: rsync -aH (3 retries) -> checksum dry-run
# zero-delta + du -sb byte-exact -> VM-side delete ONLY on PASS -> manifest.
set -uo pipefail
SRC_HOST=nebius-spot
SRC_BASE=/home/josh/lerobot/outputs/train
DEST=/media/josh/Backup/memory-models
STATUS=$DEST/_archive_fullft_status.txt
LOG=$DEST/_archive_fullft.log
touch "$STATUS"

DIRS=(
  libero_10_pi05_fullft_frombase_nomem_50k
  libero_10_pi05_fullft_froml90_nomem_50k
)

echo "=== ship_fullft: waiting on fullft-l90l10 + both seeds JSONs $(date -u) ===" >> "$LOG"
while true; do
  st=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$SRC_HOST" \
    'systemctl is-active fullft-l90l10 2>/dev/null; \
     ls /home/josh/lerobot/outputs/analysis/e60/seeds_fullft_l10.json \
        /home/josh/lerobot/outputs/analysis/e60/seeds_fullft_l90_l10.json 2>/dev/null | wc -l' 2>/dev/null) || { sleep 600; continue; }
  unit=$(echo "$st" | sed -n 1p); njson=$(echo "$st" | sed -n 2p)
  if [ "$unit" != "active" ] && [ "$unit" != "activating" ] && [ "$njson" = "2" ]; then break; fi
  sleep 600
done
echo "=== ship_fullft: gate passed (unit=$unit jsons=$njson) — transferring $(date -u) ===" >> "$LOG"

fail=0
for c in "${DIRS[@]}"; do
  grep -q "^PASS $c$" "$STATUS" && { echo "[skip] $c" >> "$LOG"; continue; }
  ok=0
  for attempt in 1 2 3; do
    rsync -aH --partial --timeout=120 "$SRC_HOST:$SRC_BASE/$c" "$DEST/" >> "$LOG" 2>&1 && { ok=1; break; }
    echo "[retry $attempt] $c" >> "$LOG"; sleep 60
  done
  [ "$ok" = "1" ] || { echo "FAIL-TRANSFER $c" >> "$STATUS"; fail=1; continue; }
  n_delta=$(rsync -aHc --dry-run --itemize-changes "$SRC_HOST:$SRC_BASE/$c" "$DEST/" 2>>"$LOG" | grep -cv '^\.d\|^cd' || true)
  src_b=$(ssh -o BatchMode=yes "$SRC_HOST" "du -sb $SRC_BASE/$c" | cut -f1)
  dst_b=$(du -sb "$DEST/$c" | cut -f1)
  if [ "$n_delta" = "0" ] && [ -n "$src_b" ] && [ "$src_b" = "$dst_b" ]; then
    echo "PASS $c" >> "$STATUS"
    ssh -n -o BatchMode=yes "$SRC_HOST" "rm -rf $SRC_BASE/$c" && echo "[VM-deleted] $c ($src_b bytes cold)" >> "$LOG"
  else
    echo "FAIL-VERIFY $c (delta=$n_delta src=$src_b dst=$dst_b)" >> "$STATUS"; fail=1
  fi
done
{
  echo ""
  echo "# ---- Batch 3: full-FT baselines, $(date -u +%Y-%m-%d) (auto-shipped on eval completion) ----"
  grep "^PASS" "$STATUS" | sed 's/^PASS /ARCHIVED  /'
} >> "$DEST/_ARCHIVE_MANIFEST_complete.md"
echo "SHIP-FULLFT-DONE fail=$fail" >> "$LOG"
echo "SHIP-FULLFT-DONE fail=$fail"
