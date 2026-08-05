#!/bin/bash
# E59: gate the landing battery (run_e59_msemat_jitter.sh) on the sequential
# actually finishing. Polls the e59-interleave unit ON the VM (no SSH churn);
# when it exits, requires the step-25000 row in eval/results.jsonl (final eval
# landed) before touching the GPU. A dead/preempted run must NOT trigger probes.
set -uo pipefail
RUN=/home/josh/lerobot/outputs/train/libero_10_seq5_jw_interleave_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k
RESULTS=$RUN/eval/results.jsonl

while true; do
  state=$(systemctl is-active e59-interleave 2>/dev/null || true)
  if [ "$state" != "active" ] && [ "$state" != "activating" ]; then
    break
  fi
  sleep 120
done
echo "[gate] e59-interleave unit exited (state=$state)"

if ! grep -q '"step": 25000' "$RESULTS" 2>/dev/null; then
  echo "[gate] FAIL: no step-25000 row in $RESULTS — run did not complete cleanly. Refusing to probe."
  exit 1
fi
echo "[gate] final eval row present — launching probe battery"
exec bash /home/josh/lerobot/scripts/vla_analysis/run_e59_msemat_jitter.sh
