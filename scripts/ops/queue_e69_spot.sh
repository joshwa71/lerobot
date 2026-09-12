#!/usr/bin/env bash
# E69 spot queue (nebius-spot, H200) — runs hands-free once A3's 5-task sequential (unit e68-a3) is
# COMPLETE, in this order:
#   1. replay SMOKE: retain_a05_10task.sh SMOKE=1 with ALPHA=1.0 + the replay flags (2 tasks x 20 steps,
#      forced stop/resume, dense loss matrix) — must print E67-RETAIN-SMOKE-OK and E69-REPLAY-SMOKE-OK;
#   2. A3's 4-seed retention triangle (run_e68_retention_triangle.sh a3; skip-guarded per row);
#   3. the REPLAY chain: naive full FT (alpha=1.0) + 1 episode/task replay buffer at 25 %, 10 tasks;
#   4. its final-row 4-seed campaign (run_baseline_triangle.sh replay_fullft 10).
# A smoke failure still runs the A3 triangle (never blocked by a replay bug) and then stops with
# E69-REPLAY-SMOKE-FAIL so the code can be fixed by hand. Every stage is resumable / skip-guarded.
set -uo pipefail
ROOT=/home/josh/lerobot
cd "$ROOT" || exit 1
A3_SEQ=$ROOT/outputs/train/libero_10_seq5_jw_e68a3_jointprep_lr1e-4_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
A3_LOG=$ROOT/outputs/e68/a3_train.log
REPLAY_ARGS="--replay_episodes_per_task=1 --replay_fraction=0.25 --replay_seed=0"
REPLAY_RUN=libero_10_seq10_replay1ep_f025_fullft_steps5k
POLL=${POLL:-300}
say(){ echo "[e69-spot] $* $(date -u +%H:%M:%SZ)"; }
echo "=== E69 SPOT QUEUE START $(date -u) ==="
# ---- gate: A3 sequential complete (unit gone, final checkpoint, chain marker) ----
while true; do
  if ! systemctl is-active e68-a3 >/dev/null 2>&1 && [ -d "$A3_SEQ/checkpoints/025000/pretrained_model" ] \
     && grep -q "E68-A3 joint-preparation chain COMPLETE" "$A3_LOG" 2>/dev/null; then
    say "A3 sequential complete - GPU released"; break
  fi
  sleep "$POLL"
done
# ---- 1. replay smoke ----
if [ -f "$ROOT/outputs/e69/replay_smoke_ok" ]; then say "replay smoke already OK - skipping"; smoke_ok=1
else
  say "replay smoke"
  if SMOKE=1 ALPHA=1.0 EXTRA_ARGS="$REPLAY_ARGS" bash job_scripts/nebius/baselines/retain_a05_10task.sh > "$ROOT/outputs/e69/replay_smoke.log" 2>&1 \
     && grep -q "E67-RETAIN-SMOKE-OK" "$ROOT/outputs/e69/replay_smoke.log" && grep -q "E69-REPLAY-SMOKE-OK" "$ROOT/outputs/e69/replay_smoke.log"; then
    touch "$ROOT/outputs/e69/replay_smoke_ok"; say "replay smoke OK"; smoke_ok=1
  else
    say "replay smoke FAILED (outputs/e69/replay_smoke.log)"; echo "E69-REPLAY-SMOKE-FAIL"; smoke_ok=0
  fi
fi
# ---- 2. A3 triangle ----
say "A3 triangle"
bash scripts/vla_analysis/run_e68_retention_triangle.sh a3 2>&1 | grep --line-buffered -v "exists - skipping\|checkpoint .* missing - skipping"
na=$(ls $ROOT/outputs/analysis/e68/seeds_tri_e68_a3*_b*.json 2>/dev/null | wc -l)
say "A3 triangle rows: $na/5"; [ "$na" -ge 5 ] && echo "E68-A3-TRIANGLE-DONE" || echo "E68-A3-TRIANGLE-INCOMPLETE"
[ "$smoke_ok" = 1 ] || { say "stopping: replay smoke failed"; exit 1; }
# ---- 3. replay chain (H200 ladder = the E67 RETAIN rung first) ----
FINAL=$ROOT/outputs/train/$REPLAY_RUN/checkpoints/050000/pretrained_model/model.safetensors
if [ -f "$FINAL" ]; then say "replay chain already complete"; else
  say "replay chain: $REPLAY_RUN"
  ALPHA=1.0 RUN_NAME=$REPLAY_RUN EXTRA_ARGS="$REPLAY_ARGS" LADDER="8:4,8:4,4:8" bash job_scripts/nebius/baselines/retain_a05_10task.sh
  [ -f "$FINAL" ] || { say "replay chain ended without the final boundary"; echo "E69-REPLAY-FAIL"; exit 1; }
fi
# ---- 4. final row ----
say "replay final-row campaign (b10)"
bash scripts/baselines/run_baseline_triangle.sh replay_fullft 10
if [ -f "$ROOT/outputs/analysis/e67/seeds_tri_replay10_1ep_f025_b10.json" ]; then echo "E69-REPLAY-DONE"; exit 0; fi
say "final row missing"; echo "E69-REPLAY-FAIL"; exit 1
