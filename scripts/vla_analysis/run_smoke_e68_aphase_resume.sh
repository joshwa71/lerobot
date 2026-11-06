#!/bin/bash
# E68: stage-A periodic-save + resume smoke on the REAL merged-6x2 warm-up checkpoint with the A1
# LIVE flags, driven through joint_aphase_seq5_common.sh itself (A_ONLY=1) so the exact production
# command is what gets exercised. Doubles as the H100 profiling gate: log_freq 10 => updt_s lines,
# and a background nvidia-smi poll records peak VRAM for the rung that fits.
#   1. A_STEPS=40, A_SAVE_FREQ=20 -> checkpoints/000020 and /000040 must exist
#   2. delete 000040 (+ point last -> 000020) -> relaunch -> the body must RESUME from 000020's
#      train_config.json (lerobot-train --resume) and recreate 000040
# Cleans its run dir on success. Exit 1 on any failure (run dir kept for inspection).
set -o pipefail
ROOT=/home/josh/lerobot
J=$ROOT/job_scripts/nebius/libero_90/staged
export HF_HUB_OFFLINE=1
export WARM_RUN=libero_90_pi05_jointwarm10k_merged6x2_e468101416_v579111315_anchor040_sep8_prepass
export GRAD_TAG=smoke_e68a1_live_resume
export A_FROZEN_ROUTE=false
export A_EXTRA_ARGS="--policy.memory_layer.frozen_prepass=false --policy.memory_layer.allow_nonstationary_routing=true"
export A_STEPS=40 A_SAVE_FREQ=20 A_LOG_FREQ=10 A_ONLY=1
export A_LADDER="${A_LADDER:-8:4:false,4:8:false,8:4:true,4:8:true}"
A_OUT=$ROOT/outputs/train/libero_90_pi05_jointA10k_${GRAD_TAG}
rm -rf "$A_OUT"
PEAK=/tmp/smoke_e68_resume_peak.txt; : > $PEAK
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 >> $PEAK; sleep 5; done ) & POLL=$!
trap 'kill $POLL 2>/dev/null' EXIT
echo "=== step 1: fresh stage A (40 steps, save every 20) ==="
bash $J/joint_aphase_seq5_common.sh > /tmp/smoke_e68_resume_1.log 2>&1; RC=$?
grep -E "\[A-phase\]|updt_s|Traceback|Error" /tmp/smoke_e68_resume_1.log | tail -12
[ $RC -eq 0 ] && [ -d "$A_OUT/checkpoints/000020/pretrained_model" ] && [ -d "$A_OUT/checkpoints/000040/pretrained_model" ] \
  || { echo "[FAIL] step 1: rc=$RC or checkpoints missing"; exit 1; }
echo "[PASS] step 1: 000020 + 000040 written"
RUNG=$(grep -oE "\[A-phase\] rung: bs=[0-9]+ accum=[0-9]+ grad_ckpt=[a-z]+" /tmp/smoke_e68_resume_1.log | tail -1)
echo "=== step 2: delete 000040, resume from 000020 ==="
rm -rf "$A_OUT/checkpoints/000040"; ln -sfn 000020 "$A_OUT/checkpoints/last"
bash $J/joint_aphase_seq5_common.sh > /tmp/smoke_e68_resume_2.log 2>&1; RC=$?
grep -E "\[A-phase\]|RESUM|resum|step:|updt_s|Traceback|Error" /tmp/smoke_e68_resume_2.log | tail -12
grep -q "RESUMING from" /tmp/smoke_e68_resume_2.log || { echo "[FAIL] step 2: body did not take the resume path"; exit 1; }
[ $RC -eq 0 ] && [ -d "$A_OUT/checkpoints/000040/pretrained_model" ] || { echo "[FAIL] step 2: rc=$RC or 000040 not recreated"; exit 1; }
# the resumed run must have started above step 20 (no log line at or below step 20)
if grep -oE "step:[0-9]+" /tmp/smoke_e68_resume_2.log | sed 's/step://' | awk '$1<20{f=1} END{exit f?1:0}'; then
  echo "[PASS] step 2: resumed from step 20 and finished at 40"
else
  echo "[FAIL] step 2: resumed run logged steps <= 20"; exit 1
fi
kill $POLL 2>/dev/null
PK=$(sort -n $PEAK | tail -1)
echo "[PROFILE] fit rung: ${RUNG:-unknown}; peak VRAM ${PK:-NA} MiB; updt_s lines:"
grep -oE "updt_s:[0-9.]+" /tmp/smoke_e68_resume_1.log | tail -3
rm -rf "$A_OUT"
echo "A-PHASE RESUME SMOKE PASS"
