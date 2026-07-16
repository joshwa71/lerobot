#!/bin/bash
# E42 VM3 chain: (1) A-phase libero-90 usage audit + generalist-slot overlap analysis (~2h),
# then (2) the per-task LoRA-FT baseline (~13h). The other two E42 arms (lr2x+topt3072,
# lr4xsched+topt3072) run on VM1/VM2.
set -eo pipefail
ROOT_DIR=/home/josh/lerobot
cd "$ROOT_DIR"
echo "=== VM3 chain started on $(hostname) at $(date) ==="
bash job_scripts/nebius/libero_90/probes/audit_a_phase_usage_and_overlap.sh
echo "=== stage 1 (audit+overlap) done at $(date); starting LoRA-FT baseline ==="
bash job_scripts/nebius/baselines/loraft_pertask_baseline.sh
echo "=== VM3 chain completed at $(date) ==="
