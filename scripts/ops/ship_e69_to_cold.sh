#!/bin/bash
# Ship the E69 deliverable checkpoints to cold storage (Josh, 15 Sep 26: "Ship the naive
# sequential and reversed checkpoints to cold storage (Backup drive)"). LOCAL script — the
# destination drive is on the desk PC, so this cannot be a VM systemd unit.
#
# Payload (optimizer state already pruned from both, 15 Sep — 520 GiB freed across the two VMs):
#   nebius-spot : libero_10_seq10rev_jw_merged6x2_...  the REVERSED-order paper cell (E69 add-7,
#                 final row 62.4 vs forward 65.1)                                    ~190 G
#   nebius2     : libero_10_seq10_naive_fullft_steps5k the NAIVE sequential full FT (E69 add-6,
#                 final row 7.7 %)                                                    ~88 G
#
# `ft_pretrained_model` is EXCLUDED from the naive run: at alpha = 1.0 the boundary merge is the
# identity, so it is a byte-identical copy of `pretrained_model`. Verified md5-equal at b1 and b10,
# and `ratio_merged_over_ft` is exactly 1.0 at all ten boundaries (retain_boundary.json). Shipping
# one copy halves the payload and loses nothing.
#
# Protocol (Batch-2/3 manifest, unchanged from ship_e64_batch_to_cold.sh):
#   rsync -aH with 3 retries -> verify with rsync -aHc --dry-run zero-transfer AND du -sb byte-exact
#   both sides (VM side with the same --exclude) -> append to the manifest on PASS.
# DELIBERATE DEVIATION: no VM-side rm. The ICRA deadline is 07:59 UK 16 Sep; restoring 190 G from a
# desk drive to a VM over home broadband inside that window would be painful, and neither box is
# short of disk (spot 728 G free, nebius2 929 G free after the prune). Delete after submission.
#
# LAUNCH DETACHED so a closed session does not kill it (the 8 Aug lesson):
#   nohup setsid bash scripts/ops/ship_e69_to_cold.sh >/dev/null 2>&1 &
# Re-running is safe: dirs marked PASS are skipped; rsync --partial resumes.
set -uo pipefail
DEST=/media/josh/Backup/memory-models
STATUS=$DEST/_archive_e69_status.txt
LOG=$DEST/_archive_e69.log
MANIFEST=$DEST/_ARCHIVE_MANIFEST_complete.md
mountpoint -q /media/josh/Backup || { echo "Backup drive not mounted — abort"; exit 1; }
mkdir -p "$DEST"; touch "$STATUS" "$MANIFEST"

# host : run dir : rsync excludes (space separated, may be empty)
ENTRIES=(
  "nebius-spot:libero_10_seq10rev_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k:"
  "nebius2:libero_10_seq10_naive_fullft_steps5k:ft_pretrained_model"
)
SRC_BASE=/home/josh/lerobot/outputs/train

echo "=== ship_e69 START $(date -u) ===" >> "$LOG"
fail=0
for entry in "${ENTRIES[@]}"; do
  host="${entry%%:*}"; rest="${entry#*:}"; c="${rest%%:*}"; excl="${rest#*:}"
  grep -q "^PASS $c$" "$STATUS" && { echo "[skip] $c already PASS" >> "$LOG"; continue; }
  ssh -o BatchMode=yes "$host" "[ -d $SRC_BASE/$c ]" || { echo "[skip] $c not on $host" >> "$LOG"; continue; }
  EX=(); DUEX=()
  if [ -n "$excl" ]; then for e in $excl; do EX+=(--exclude="$e"); DUEX+=(--exclude="$e"); done; fi
  ok=0
  for attempt in 1 2 3; do
    echo "[xfer $attempt] $host:$c $(date -u +%H:%M:%SZ)" >> "$LOG"
    rsync -aH --partial --timeout=300 "${EX[@]}" "$host:$SRC_BASE/$c" "$DEST/" >> "$LOG" 2>&1 && { ok=1; break; }
    echo "[retry $attempt] $c" >> "$LOG"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "FAIL-TRANSFER $c" | tee -a "$STATUS" >> "$LOG"; fail=1; continue; }
  # verify: checksum dry-run must itemize zero differences, and byte counts must match.
  # NB ship_e64's lesson: capture rsync's output first and separate "no differences" (delta 0)
  # from "the verify command itself failed" (delta 999) — `grep -c` exits 1 on a zero count.
  vout=$(rsync -aHc --dry-run --itemize-changes "${EX[@]}" "$host:$SRC_BASE/$c" "$DEST/" 2>/dev/null); vrc=$?
  if [ "$vrc" -ne 0 ]; then delta=999; else delta=$(printf '%s\n' "$vout" | grep -c '^[<>ch]'); delta=${delta:-999}; fi
  vm_b=$(ssh -o BatchMode=yes "$host" "du -sb ${DUEX[*]} $SRC_BASE/$c | cut -f1")
  co_b=$(du -sb "$DEST/$c" | cut -f1)
  if [ "$delta" = "0" ] && [ "$vm_b" = "$co_b" ]; then
    echo "PASS $c" >> "$STATUS"
    echo "[verified] $c ($co_b bytes) $(date -u +%H:%M:%SZ)" >> "$LOG"
    printf '| %s | %s bytes | E69 (%s); optimizer state pruned%s | %s |\n' \
      "$c" "$co_b" "$host" "${excl:+; excluded $excl (byte-identical duplicate at alpha=1)}" \
      "$(date -u +%Y-%m-%d)" >> "$MANIFEST"
  else
    echo "FAIL-VERIFY $c delta=$delta vm=$vm_b cold=$co_b" | tee -a "$STATUS" >> "$LOG"; fail=1
  fi
done
echo "=== SHIP-E69-DONE fail=$fail $(date -u) ===" >> "$LOG"
