#!/bin/bash
# E65 (Josh, 29 Aug): archive the sim-era runs still hot on the VM to cold storage, prune the
# optimizer states, and free the VM. LOCAL script — the destination drive is on the desk PC.
# Runs CONCURRENTLY with the real-world chain (different machine; network-bound, bwlimited).
#
# Josh's spec: "get all them sim runs in cold storage and prune any old training state
# safetensors. When each is done, verify manually then delete from the VM. Copy the analysis dir
# into cold too BUT keep it in the VM too in case we need to revisit."
#
# PER DIR:
#   1. rsync -aH --exclude=training_state (3 retries, bwlimit) -> cold.
#      training_state/ = optimizer_state.safetensors + rng + scheduler (21G per checkpoint on the
#      10-task run, ~200G total; every other dir was already trimmed). Useless for a COMPLETED run;
#      the project already auto-prunes it (E60-add-6 / E62-add-8, rw_stage1_base.sh). NB the
#      cross-task protection store sequential_state.pt lives OUTSIDE training_state and IS kept.
#   2. verify: `rsync -aHc --dry-run` zero itemized changes AND `du -sb` byte-exact, both under the
#      SAME exclusion. (24 Aug bug: `grep -c` exits 1 on a count of zero, which is the PASS case —
#      capture rsync's output first and separate "no differences" from "command failed".)
#   3. preserve the diagnostic surface ON THE VM before deleting: memory_by_task/ (the slot-usage
#      JSONs the E65 saturation analysis ran on), eval/, wandb/ and top-level files ->
#      outputs/analysis/_run_artifacts/<run>/  (~390M per run vs ~100G of checkpoints).
#   4. rm -rf the run dir on the VM, ONLY on a strict PASS.
# THEN: rsync outputs/analysis -> cold (copy only; the VM keeps its copy, per Josh).
#
# HARD EXCLUSIONS: anything matching the live real-world chain, its stage-1 base or its A-phase is
# refused even if named in DIRS. Re-running is safe (PASS/DELETED recorded and skipped).
# LAUNCH DETACHED (8 Aug lesson — a session-side shipper died in a desk-PC reboot):
#   nohup setsid bash scripts/ops/ship_e65_batch_to_cold.sh >/dev/null 2>&1 &
set -uo pipefail
SRC_HOST=nebius-spot
SRC_BASE=/home/josh/lerobot/outputs/train
ANALYSIS=/home/josh/lerobot/outputs/analysis
DEST=/media/josh/Backup/memory-models
STATUS=$DEST/_archive_e65_status.txt
LOG=$DEST/_archive_e65.log
MANIFEST=$DEST/_ARCHIVE_MANIFEST_complete.md
DELETE=${DELETE:-1}
touch "$STATUS"

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

ship_analysis () {   # copy-only; the VM keeps its copy
  echo "=== analysis dir -> cold ($1) $(date -u) ===" >> "$LOG"
  if rsync -aH --partial --timeout=180 --bwlimit=40000 "$SRC_HOST:$ANALYSIS" "$DEST/_analysis_vm/" >> "$LOG" 2>&1; then
    echo "[analysis-copied] $1 ($(du -sh "$DEST/_analysis_vm" 2>/dev/null | cut -f1))" >> "$LOG"
  else
    echo "[analysis-FAILED] $1" >> "$LOG"
  fi
}

mkdir -p "$DEST/_analysis_vm"
echo "=== ship_e65 v2 started $(date -u) DELETE=$DELETE — ${#DIRS[@]} dirs, training_state EXCLUDED ===" >> "$LOG"
ship_analysis first-pass

fail=0
for c in "${DIRS[@]}"; do
  if printf '%s' "$c" | grep -qE "$FORBID"; then
    echo "[REFUSED] $c matches the active-run exclusion list" >> "$LOG"; continue
  fi
  if grep -q "^DELETED $c\$" "$STATUS"; then echo "[skip] $c already DELETED" >> "$LOG"; continue; fi
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$SRC_HOST" "[ -d $SRC_BASE/$c ]" || { echo "[skip] $c not on VM" >> "$LOG"; continue; }

  if ! grep -q "^PASS $c\$" "$STATUS"; then
    echo "=== $c : transfer (training_state excluded) $(date -u) ===" >> "$LOG"
    ok=0
    for attempt in 1 2 3; do
      rsync -aH --exclude=training_state --partial --timeout=180 --bwlimit=40000 \
        "$SRC_HOST:$SRC_BASE/$c" "$DEST/" >> "$LOG" 2>&1 && { ok=1; break; }
      echo "[retry $attempt] $c" >> "$LOG"; sleep 60
    done
    [ "$ok" = 1 ] || { echo "FAIL-TRANSFER $c" | tee -a "$STATUS" >> "$LOG"; fail=1; continue; }
    # idempotence: drop any training_state a previous (v1, unexcluded) pass may have landed
    find "$DEST/$c" -type d -name training_state -prune -exec rm -rf {} + 2>/dev/null
    vout=$(rsync -aHc --exclude=training_state --dry-run --itemize-changes \
             "$SRC_HOST:$SRC_BASE/$c" "$DEST/" 2>/dev/null); vrc=$?
    if [ "$vrc" -ne 0 ]; then
      delta=999
    else
      delta=$(printf '%s\n' "$vout" | grep -c '^[<>ch]'); true
      delta=${delta:-999}
    fi
    vm_b=$(ssh -o BatchMode=yes "$SRC_HOST" "du -sb --exclude=training_state $SRC_BASE/$c | cut -f1")
    co_b=$(du -sb "$DEST/$c" | cut -f1)
    if [ "$delta" = "0" ] && [ -n "$vm_b" ] && [ "$vm_b" = "$co_b" ]; then
      echo "PASS $c" >> "$STATUS"
      echo "[verified] $c ($co_b bytes; training_state pruned)" >> "$LOG"
      printf '| %s | %s bytes | E65 sim-era archival (training_state PRUNED; weights + sequential_state kept) | %s |\n' \
        "$c" "$co_b" "$(date -u +%Y-%m-%d)" >> "$MANIFEST"
    else
      echo "FAIL-VERIFY $c delta=$delta vm=$vm_b cold=$co_b" | tee -a "$STATUS" >> "$LOG"; fail=1; continue
    fi
  fi

  if [ "$DELETE" = "1" ] && grep -q "^PASS $c\$" "$STATUS"; then
    if ssh -o BatchMode=yes "$SRC_HOST" "A=$ANALYSIS/_run_artifacts/$c; mkdir -p \$A; \
        for s in memory_by_task eval wandb; do [ -d $SRC_BASE/$c/\$s ] && cp -an $SRC_BASE/$c/\$s \$A/ 2>/dev/null; done; \
        find $SRC_BASE/$c -maxdepth 1 -type f -exec cp -an {} \$A/ \; 2>/dev/null; \
        du -sh \$A | cut -f1" >> "$LOG" 2>&1; then
      if ssh -o BatchMode=yes "$SRC_HOST" "rm -rf $SRC_BASE/$c"; then
        echo "DELETED $c" >> "$STATUS"
        echo "[vm-deleted] $c (artifacts preserved under outputs/analysis/_run_artifacts/)" >> "$LOG"
      else
        echo "FAIL-DELETE $c" | tee -a "$STATUS" >> "$LOG"; fail=1
      fi
    else
      echo "FAIL-PRESERVE $c (NOT deleted)" | tee -a "$STATUS" >> "$LOG"; fail=1
    fi
    echo "[vm-disk] $(ssh -o BatchMode=yes "$SRC_HOST" 'df -h /home/josh | tail -1' 2>/dev/null)" >> "$LOG"
  fi
done

ship_analysis final-pass
echo "=== ship_e65 done $(date -u) fail=$fail | VM: $(ssh -o BatchMode=yes "$SRC_HOST" 'df -h /home/josh | tail -1' 2>/dev/null) ===" >> "$LOG"
echo "SHIP-E65-DONE fail=$fail"
