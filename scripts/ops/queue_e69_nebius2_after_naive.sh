#!/usr/bin/env bash
# E69: on nebius2, after the naive full-FT queue (unit e69-naive) has finished, run A3's 4-seed
# retention triangle here (moved off spot so the H200 can start the reversed-order paper cell the
# moment A3's sequential ends). Needs A3's five sequential checkpoints mirrored from spot first
# (outputs/train/libero_10_seq5_jw_e68a3_.../checkpoints/{005000..025000}/pretrained_model) — this
# queue waits for them too. Launched as an inline unit payload (this file is the committed record).
set -uo pipefail
ROOT=/home/josh/lerobot; cd "$ROOT" || exit 1
A3=$ROOT/outputs/train/libero_10_seq5_jw_e68a3_jointprep_lr1e-4_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
say(){ echo "[e69-n2] $* $(date -u +%H:%M:%SZ)"; }
echo "=== E69 NEBIUS2 AFTER-NAIVE QUEUE START $(date -u) ==="
while systemctl is-active e69-naive >/dev/null 2>&1; do sleep 300; done
say "e69-naive finished"
while [ "$(ls -d $A3/checkpoints/0[0-9]*/pretrained_model 2>/dev/null | wc -l)" -lt 5 ]; do sleep 300; done
say "A3 checkpoints present - triangle"
bash scripts/vla_analysis/run_e68_retention_triangle.sh a3 2>&1 | grep --line-buffered -v "exists - skipping\|checkpoint .* missing - skipping"
n=$(ls $ROOT/outputs/analysis/e68/seeds_tri_e68_a3_jointprep_b*.json 2>/dev/null | wc -l)
say "A3 triangle rows: $n/5"
[ "$n" -ge 5 ] && { echo "E68-A3-TRIANGLE-DONE"; exit 0; }
echo "E68-A3-TRIANGLE-INCOMPLETE"; exit 1
