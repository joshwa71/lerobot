#!/usr/bin/env bash
# E69 (Josh, 12 Sep 26): the reviewer-requested NAIVE SEQUENTIAL FULL FINE-TUNE of pi0.5 — every
# parameter free, no memory, no adapter, no merge, no replay — from the libero_90 stage-1 base,
# 10 LIBERO-10 tasks x 5,000 steps under the sequential protocol (= the RETAIN recipe with alpha=1.0:
# theta <- theta_ft, so retain_a05_10task.sh is reused byte-for-byte with ALPHA=1.0 and RUN_NAME).
# Then the FINAL-ROW 4-seed campaign only (b10: all ten envs x 25 eps x 4 seeds), like the r1216 row.
# nebius2 (H100 80 GB): the H200 rung bs8x4 peaked at 88.5 GiB, so the ladder starts at bs4x8.
# Resume-safe (the trainer resumes from retain_state/ on relaunch); skip-guarded at both stages.
set -uo pipefail
ROOT=/home/josh/lerobot
cd "$ROOT" || exit 1
export ALPHA=1.0
export RUN_NAME=libero_10_seq10_naive_fullft_steps5k
export LADDER=${LADDER:-"4:8,4:8,2:16"}
say(){ echo "[e69-naive] $* $(date -u +%H:%M:%SZ)"; }
echo "=== E69 NAIVE FULL-FT QUEUE START $(date -u) (alpha=$ALPHA run=$RUN_NAME ladder=$LADDER) ==="
FINAL=$ROOT/outputs/train/$RUN_NAME/checkpoints/050000/pretrained_model/model.safetensors
if [ -f "$FINAL" ]; then say "chain already complete"; else
  bash job_scripts/nebius/baselines/retain_a05_10task.sh
  [ -f "$FINAL" ] || { say "chain ended without the final boundary"; echo "E69-NAIVE-FAIL"; exit 1; }
fi
say "chain complete - final-row campaign (b10)"
bash scripts/baselines/run_baseline_triangle.sh naive_fullft 10
if [ -f "$ROOT/outputs/analysis/e67/seeds_tri_naive10_fullft_b10.json" ]; then echo "E69-NAIVE-DONE"; exit 0; fi
say "final row missing"; echo "E69-NAIVE-FAIL"; exit 1
