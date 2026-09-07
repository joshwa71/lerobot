#!/bin/bash
# E60-era cold-storage archival (6 Aug 26): VM -> external drive, ~630G.
# Copy + verify only — NO deletion here. VM-side rm -rf happens separately,
# per-dir, only for dirs marked PASS in the status file.
# Protocol per _ARCHIVE_MANIFEST_complete.md (26 Jul): rsync -aH transfer,
# then rsync -aHc --dry-run zero-transfer check + du -sb byte-exact both sides.
# Resumable: re-running skips dirs already marked PASS.
set -uo pipefail
SRC_HOST=nebius-spot
SRC_BASE=/home/josh/lerobot/outputs/train
DEST=/media/josh/Backup/memory-models
STATUS=$DEST/_archive_e60_status.txt
LOG=$DEST/_archive_e60.log
touch "$STATUS"

CKPTS=(
  # E58 value-input-noise dose arms (superseded by placement; analyses+rescores extracted)
  libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_vnoise1x_steps5k
  libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_vnoise05x_steps5k
  # E54 absmax family (pre-B 53.6 frontier, now two frontiers stale)
  libero_90_pi05_jointwarm10k_absmax_e4to9_v10to16
  libero_90_pi05_jointwarm10k_absmax_anchor05_nofilm_e4to9_v10to16
  libero_90_pi05_jointA10k_absmax_anchor05_nofilm_e4to9_v10to16
  libero_10_seq5_jw_absmax_anchor05nofilm_beta4corefrac_topt3072_lr2x_steps5k
  # E53 anchor/sep certificate-grid warm-ups not in the 26 Jul archive
  libero_90_pi05_jointwarm10k_layermax_sep8_nofilm_e2468_v10121416
  libero_90_pi05_jointwarm10k_layermax_sep8_e2468_v10121416
  libero_90_pi05_jointwarm10k_layermax_e2468_v10121416
  libero_90_pi05_jointwarm10k_layermax_compact_e9to12_v13to16
  libero_90_pi05_jointwarm10k_layermax_A_anchor05_c015_nofilm_e2468_v10121416
  libero_90_pi05_jointwarm10k_layermax_A_anchor040_sep8_nofilm_e2468_v10121416
  libero_90_pi05_jointwarm10k_layermax_A_anchor035_sep8_nofilm_e2468_v10121416
  libero_90_pi05_jointwarm10k_layermax_A_anchor035_c015_nofilm_e2468_v10121416
  # Dead A-phases (incl. B's — flagged in chat: future B-init arms would restore from cold)
  libero_90_pi05_jointA10k_layermax_sep8_e2468_v10121416
  libero_90_pi05_jointA10k_layermax_e2468_v10121416
  libero_90_pi05_jointA10k_layermax_compact_e9to12_v13to16
  libero_90_pi05_jointA10k_layermax_A_anchor040_sep8_nofilm_e2468_v10121416
  # E53-era sequential arms not archived (corefrac variants)
  libero_10_seq5_jw_layermax_sep8_beta4corefrac_topt3072_lr2x_steps5k
  libero_10_seq5_jw_layermax_A_e2468_v10121416_beta4corefrac_topt3072_lr2x_steps5k
  # small: E49 imgspan sequential
  libero_10_seq5_jw_imgspan_g2_vlmknn16_beta4_topt1536_steps5k
)

echo "=== archive_e60 started $(date -u) — ${#CKPTS[@]} dirs ===" >> "$LOG"
fail=0
for c in "${CKPTS[@]}"; do
  if grep -q "^PASS $c$" "$STATUS"; then
    echo "[skip, already PASS] $c" >> "$LOG"
    continue
  fi
  echo "=== $c : transfer $(date -u) ===" >> "$LOG"
  ok=0
  for attempt in 1 2 3; do
    if rsync -aH --partial --timeout=120 "$SRC_HOST:$SRC_BASE/$c" "$DEST/" >> "$LOG" 2>&1; then
      ok=1; break
    fi
    echo "[retry $attempt failed] $c — backing off 60s" >> "$LOG"
    sleep 60
  done
  if [ "$ok" != "1" ]; then
    echo "FAIL-TRANSFER $c" >> "$STATUS"; fail=1; continue
  fi
  # verify: checksum dry-run must transfer nothing, byte sizes must match exactly
  n_delta=$(rsync -aHc --dry-run --itemize-changes "$SRC_HOST:$SRC_BASE/$c" "$DEST/" 2>>"$LOG" | grep -cv '^\.d\|^cd' || true)
  src_b=$(ssh -o BatchMode=yes "$SRC_HOST" "du -sb $SRC_BASE/$c" | cut -f1)
  dst_b=$(du -sb "$DEST/$c" | cut -f1)
  if [ "$n_delta" = "0" ] && [ -n "$src_b" ] && [ "$src_b" = "$dst_b" ]; then
    echo "PASS $c" >> "$STATUS"
    echo "[VERIFIED $src_b bytes] $c" >> "$LOG"
  else
    echo "FAIL-VERIFY $c (delta_items=$n_delta src=$src_b dst=$dst_b)" >> "$STATUS"; fail=1
  fi
done
echo "=== archive_e60 done $(date -u) fail=$fail ===" >> "$LOG"
grep -c "^PASS" "$STATUS" | xargs -I{} echo "ARCHIVE-E60-COMPLETE: {} PASS of ${#CKPTS[@]}"
[ "$fail" = "0" ] && exit 0 || exit 1
