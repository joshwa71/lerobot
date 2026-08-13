#!/bin/bash
# WEEKEND 10-TASK QUEUE (Josh, 13 Aug: "If that's done before I'm back, get the
# 10 task results for the 6x2 arms, noise and no noise"). Gated on the
# weekend-baselines unit exiting; runs the no-noise 10-task chain, then the
# noise one. Both chains are stage-level idempotent + sequential-resume-safe:
# relaunching this unit after a preemption continues from the last completed
# task boundary.
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/weekend_10task.log
exec >> "$LOG" 2>&1
echo "=== weekend 10-task queue: waiting on weekend-baselines $(date -u) ==="
while true; do
  st=$(systemctl is-active weekend-baselines 2>/dev/null) || true
  [ "$st" = "active" ] || [ "$st" = "activating" ] || break
  sleep 600
done
echo "=== weekend 10-task queue: gate passed (state=$st) — no-noise chain first $(date -u) ==="
bash $ROOT/job_scripts/nebius/libero_90/staged/seq10_merged6x2.sh \
  || echo "[FAIL] seq10 no-noise chain"
echo "=== no-noise 10-task chain exited $(date -u) — starting noise chain ==="
bash $ROOT/job_scripts/nebius/libero_90/staged/seq10_merged6x2_vnoise05x.sh \
  || echo "[FAIL] seq10 vnoise chain"
echo "=== WEEKEND 10-TASK QUEUE COMPLETE $(date -u) ==="
