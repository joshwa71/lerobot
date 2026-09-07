#!/bin/bash
# E67 TRAIN queue (VM-side; launch under systemd-run, unit e67-train). Two modes:
#   smoke : RETAIN smoke, then O-LoRA smoke (each 2 tasks x 20 steps + stop/resume + drift
#           instrument). Writes outputs/e67/smoke_retain_ok / smoke_olora_ok on success.
#   full  : requires both smoke markers; RETAIN chain (10 x 5k full FT + merge, ~31h), then the
#           O-LoRA chain (10 x 5k, ~27h). Both trainers resume from their own state, so relaunching
#           this unit after a preemption continues where it stopped (the heartbeat does that).
# Markers on stdout (the heartbeat greps the log): E67-RETAIN-SMOKE-OK/FAIL, E67-OLORA-SMOKE-OK/FAIL,
# RETAIN-BOUNDARY-k, RETAIN-CHAIN-DONE, OLORA-BOUNDARY-k, OLORA-CHAIN-DONE, E67-TRAIN-DONE.
set -uo pipefail
MODE=${1:-full}
ROOT=/home/josh/lerobot
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
mkdir -p $ROOT/outputs/e67
say(){ echo "[e67-train] $* $(date -u +%H:%M:%SZ)"; }
echo "=== E67 TRAIN QUEUE START mode=$MODE $(date -u) ==="
if [ "$MODE" = "smoke" ]; then
  say "RETAIN smoke"
  if SMOKE=1 bash job_scripts/nebius/baselines/retain_a05_10task.sh; then touch outputs/e67/smoke_retain_ok; say "RETAIN smoke OK"
  else echo "E67-RETAIN-SMOKE-FAIL"; say "RETAIN smoke FAILED"; fi
  say "O-LoRA smoke"
  if SMOKE=1 bash job_scripts/nebius/baselines/olora_r64_10task.sh; then touch outputs/e67/smoke_olora_ok; say "O-LoRA smoke OK"
  else echo "E67-OLORA-SMOKE-FAIL"; say "O-LoRA smoke FAILED"; fi
  echo "E67-SMOKE-DONE"; exit 0
fi
[ -f outputs/e67/smoke_retain_ok ] && [ -f outputs/e67/smoke_olora_ok ] || { say "smoke markers missing - run 'queue_e67_train.sh smoke' first"; echo "E67-TRAIN-FAIL"; exit 1; }
say "RETAIN chain (alpha 0.5)"
bash job_scripts/nebius/baselines/retain_a05_10task.sh || { say "RETAIN wrapper exited non-zero (resumable)"; echo "E67-RETAIN-FAIL"; exit 1; }
P=outputs/train/libero_10_seq10_retain_a05_fullft_steps5k/retain_state/progress.json
if [ "$(python3 -c "import json;print(json.load(open('$P'))['completed_tasks'])" 2>/dev/null)" != "10" ]; then
  say "RETAIN exited before task 10 (preemption save) - relaunch to resume"; echo "E67-TRAIN-PAUSED"; exit 0
fi
echo "E67-RETAIN-DONE"
say "O-LoRA chain (r64, lambda1 0.5)"
bash job_scripts/nebius/baselines/olora_r64_10task.sh || { say "O-LoRA wrapper exited non-zero (resumable)"; echo "E67-OLORA-FAIL"; exit 1; }
P=outputs/train/libero_10_seq10_olora_r64_a16_lam05_steps5k/olora_state/progress.json
if [ "$(python3 -c "import json;print(json.load(open('$P'))['completed_tasks'])" 2>/dev/null)" != "10" ]; then
  say "O-LoRA exited before task 10 (preemption save) - relaunch to resume"; echo "E67-TRAIN-PAUSED"; exit 0
fi
echo "E67-OLORA-DONE"
echo "E67-TRAIN-DONE"
