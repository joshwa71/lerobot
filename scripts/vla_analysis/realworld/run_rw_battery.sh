#!/bin/bash
# E65 real-world landing battery — realworld duplicate of run_e62_battery.sh:
#   msemat + jitter (run_rw_msemat_jitter.sh) -> slot autopsy (rw_slots.py: site-bleed on the 5
#   shared pairs + prior-core write events incl. the solo E14/E16 cells) -> raw matrix reports
#   (in-run eval/loss_results.jsonl + the mse matrix).
# The harvest-bank rescore has NO real-world analogue (it needs simulator rollouts) — omitted.
# Launch under systemd-run (never tmux):
#   sudo systemd-run --unit=rw-battery --property=User=josh --property=KillSignal=SIGTERM \
#     --property=TimeoutStopSec=45 --property=WorkingDirectory=/home/josh/lerobot \
#     --setenv=RW_TAG=v5 /bin/bash scripts/vla_analysis/realworld/run_rw_battery.sh
set -o pipefail
source /home/josh/lerobot/job_scripts/nebius/realworld/rw_env.sh
BATTERY_TAG=${BATTERY_TAG:-merged6x2}
RW_SEQ_RUN=${RW_SEQ_RUN:-${RUN_PREFIX}realworld_${RW_TAG}_seq${RW_N_SEQ}_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k}
RD=$ROOT_DIR/outputs/train/$RW_SEQ_RUN
SP=${SP:-$ROOT_DIR/outputs/analysis/realworld/${RUN_PREFIX}e65}
export BATTERY_TAG RW_SEQ_RUN SP
LOG=$ROOT_DIR/outputs/${RUN_PREFIX}rw_battery_${RW_TAG}.log
mkdir -p "$SP"
exec >> "$LOG" 2>&1
echo "=== RW battery started $(date -u) tag=$RW_TAG smoke=$SMOKE run=$RW_SEQ_RUN HEAD=$(git rev-parse --short HEAD) ==="
bash scripts/vla_analysis/realworld/run_rw_msemat_jitter.sh || echo "[battery] msemat/jitter FAILED"
SLOTS_RUN_DIR=$RD SLOTS_NTASKS=$RW_N_SEQ SLOTS_OUT_DIR=$SP SLOTS_TAG=$BATTERY_TAG \
  python scripts/vla_analysis/realworld/rw_slots.py || echo "[battery] autopsy FAILED"
OUT=$SP/inrun_matrix_${BATTERY_TAG}.json python scripts/vla_analysis/realworld/rw_matrix_report.py inrun "$RD" "$SEQ_STEPS" \
  || echo "[battery] inrun report FAILED"
OUT=$SP/mse_matrix_${BATTERY_TAG}_report.json python scripts/vla_analysis/realworld/rw_matrix_report.py msemat "$SP/mse_matrix_${BATTERY_TAG}.jsonl" "$SEQ_STEPS" \
  || echo "[battery] msemat report FAILED"
echo "=== RW BATTERY COMPLETE $(date -u) ==="
echo "RW-BATTERY-DONE"
