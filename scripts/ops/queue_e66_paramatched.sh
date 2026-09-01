#!/bin/bash
# E66 queue: wait for the E63 rematrix to release the GPU, smoke the parameter-matched wrapper
# (fatal on failure), then run it. VM-side; launch under systemd-run.
# Gate: the E63 corrected matrix file has all 10 rows, OR a 3h timeout (so a rematrix failure
# cannot block this indefinitely), AND the GPU is free.
set -uo pipefail
ROOT=/home/josh/lerobot
W=$ROOT/job_scripts/nebius/baselines/naive_seq_lora_r1216_paramatched_10task.sh
M=$ROOT/outputs/analysis/e65_rematrix/mse_matrix_e63_seq10_FIXED.jsonl
say(){ echo "[e66] $* $(date -u +%H:%M:%SZ)"; }

say "waiting for the E63 rematrix to finish (or 3h timeout)"
end=$(( $(date +%s) + 10800 ))
while [ "$(date +%s)" -lt "$end" ]; do
  rows=$(grep -c '^{' "$M" 2>/dev/null || echo 0)
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${rows:-0}" -ge 10 ] && [ "${gpu:-0}" -lt 2000 ] && { say "E63 matrix complete ($rows rows), GPU free"; break; }
  sleep 120
done
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "${gpu:-0}" -ge 2000 ]; then say "GPU still busy (${gpu}MiB) after the wait - aborting rather than contending"; exit 1; fi

say "SMOKE (2 tasks x 20 steps) - checks the r1216 adapter builds, wraps and fits in VRAM"
if SMOKE=1 bash "$W"; then
  say "smoke OK"
  rm -rf $ROOT/outputs/train/smoke_naive_lora_r1216_paramatched
else
  say "SMOKE FAILED - not launching the full run"; echo "E66-SMOKE-FAIL"; exit 1
fi

say "full run: 10 tasks x 5,000 steps"
bash "$W" || { say "RUN FAILED (self-resuming on relaunch)"; echo "E66-RUN-FAIL"; exit 1; }
say "done"
echo "E66-DONE"
