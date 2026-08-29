#!/bin/bash
# E65 (Josh, 29 Aug): ship the sim-era checkpoints still hot on the VM to cold storage,
# CONCURRENTLY with the running real-world chain. LOCAL script — the destination drive is
# on the desk PC, so this cannot be a VM systemd unit.
#
# Protocol (identical to _archive_e60 / _archive_e64, incl. their fixed verify logic):
#   rsync -aH (3 retries) -> verify = `rsync -aHc --dry-run` zero itemized changes AND
#   `du -sb` byte-exact both sides -> manifest append. NB `grep -c` exits 1 on a count of
#   ZERO, which is the PASS case — capture rsync's output first (the 24 Aug FAIL-VERIFY bug).
#
# DELETION IS OPT-IN AND NEVER WHOLE-DIR:
#   COPY_ONLY=1 (default) — transfer + verify only. Nothing on the VM is touched.
#   PRUNE=1                — after a PASS, remove ONLY `<dir>/checkpoints` on the VM, keeping
#                            memory_by_task/, eval/, logs (~1% of the dir) so every analysis
#                            artifact stays live. The full copy is in cold either way.
# Re-running is safe: PASS dirs skip their transfer; PRUNE can be run later over the same list.
#
# HARD EXCLUSIONS (belt-and-braces against a typo): anything matching the active real-world
# chain, its stage-1 base, or its A-phase is refused even if named in DIRS.
#
# LAUNCH DETACHED (the 8 Aug lesson — a session-side shipper died in a desk-PC reboot):
#   nohup setsid bash scripts/ops/ship_e65_batch_to_cold.sh >/dev/null 2>&1 &
set -uo pipefail
SRC_HOST=nebius-spot
SRC_BASE=/home/josh/lerobot/outputs/train
DEST=/media/josh/Backup/memory-models
STATUS=$DEST/_archive_e65_status.txt
LOG=$DEST/_archive_e65.log
MANIFEST=$DEST/_ARCHIVE_MANIFEST_complete.md
COPY_ONLY=${COPY_ONLY:-1}
PRUNE=${PRUNE:-0}
touch "$STATUS"

# Ordered most-superseded first; the E62 paper cell is LAST (prune it only on an explicit call).
DIRS=(
  libero_10_seq5_jw_bigsearch_e4to16_v5to13_prepass_beta4corefrac_topt3072_lr2x_steps5k
  libero_90_pi05_jointwarm10k_bigsearch_e4to16_v5to13_anchor040_sep8_prepass
  libero_90_pi05_jointA10k_bigsearch_e4to16_v5to13_anchor040_sep8_prepass
  libero_10_seq5_jw_sharepairs_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k
  libero_10_seq5_jw_interleave_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k
  libero_90_pi05_jointwarm10k_interleave_e681012_v791113_anchor040_sep8_prepass
  libero_90_pi05_jointA10k_interleave_e681012_v791113_anchor040_sep8_prepass
  libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_steps5k
  libero_10_seq5_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_vnoise05x_steps5k
  libero_10_seq10_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
  libero_10_seq5_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
)
FORBID='realworld_v5_seq5_jw_|realworld_v5_pi05_base_nomem_50k|realworld_v5_pi05_jointA10k|realworld_v5_pi05_jointwarm10k'

echo "=== ship_e65 started $(date -u) COPY_ONLY=$COPY_ONLY PRUNE=$PRUNE — ${#DIRS[@]} dirs ===" >> "$LOG"
fail=0
for c in "${DIRS[@]}"; do
  if printf '%s' "$c" | grep -qE "$FORBID"; then
    echo "[REFUSED] $c matches the active-run exclusion list" >> "$LOG"; continue
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$SRC_HOST" "[ -d $SRC_BASE/$c ]" || { echo "[skip] $c not on VM" >> "$LOG"; continue; }
  if grep -q "^PASS $c$" "$STATUS"; then
    echo "[skip-transfer] $c already PASS" >> "$LOG"
  else
    echo "=== $c : transfer $(date -u) ===" >> "$LOG"
    ok=0
    for attempt in 1 2 3; do
      rsync -aH --partial --timeout=180 --bwlimit=40000 "$SRC_HOST:$SRC_BASE/$c" "$DEST/" >> "$LOG" 2>&1 && { ok=1; break; }
      echo "[retry $attempt] $c" >> "$LOG"; sleep 60
    done
    [ "$ok" = 1 ] || { echo "FAIL-TRANSFER $c" | tee -a "$STATUS" >> "$LOG"; fail=1; continue; }
    vout=$(rsync -aHc --dry-run --itemize-changes "$SRC_HOST:$SRC_BASE/$c" "$DEST/" 2>/dev/null); vrc=$?
    if [ "$vrc" -ne 0 ]; then delta=999; else delta=$(printf '%s\n' "$vout" | grep -c '^[<>ch]'); true; delta=${delta:-999}; fi
    vm_b=$(ssh -o BatchMode=yes "$SRC_HOST" "du -sb $SRC_BASE/$c | cut -f1")
    co_b=$(du -sb "$DEST/$c" | cut -f1)
    if [ "$delta" = "0" ] && [ "$vm_b" = "$co_b" ]; then
      echo "PASS $c" >> "$STATUS"; echo "[verified] $c ($co_b bytes)" >> "$LOG"
      printf '| %s | %s bytes | E65 sim-era archival (copy) | %s |\n' "$c" "$co_b" "$(date -u +%Y-%m-%d)" >> "$MANIFEST"
    else
      echo "FAIL-VERIFY $c delta=$delta vm=$vm_b cold=$co_b" | tee -a "$STATUS" >> "$LOG"; fail=1; continue
    fi
  fi
  if [ "$PRUNE" = "1" ] && [ "$COPY_ONLY" != "1" ] && grep -q "^PASS $c$" "$STATUS"; then
    if ! grep -q "^PRUNED $c$" "$STATUS"; then
      ssh -o BatchMode=yes "$SRC_HOST" "rm -rf $SRC_BASE/$c/checkpoints" \
        && { echo "PRUNED $c" >> "$STATUS"; echo "[vm-pruned checkpoints/] $c" >> "$LOG"; } \
        || { echo "FAIL-PRUNE $c" | tee -a "$STATUS" >> "$LOG"; fail=1; }
    fi
  fi
done
free=$(ssh -o BatchMode=yes "$SRC_HOST" 'df -h /home/josh | tail -1' 2>/dev/null)
echo "=== ship_e65 done $(date -u) fail=$fail | VM: $free ===" >> "$LOG"
echo "SHIP-E65-DONE fail=$fail"
