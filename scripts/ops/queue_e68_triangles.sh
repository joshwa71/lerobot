#!/usr/bin/env bash
# E68 A1 4-seed rollout retention triangle on nebius-spot, QUEUED BEHIND the E67 eval queue.
#
# Waits until E67 is finished end to end (both triangles 10/10 rows AND both drift matrices 10/10
# rows — the same condition queue_e67_eval.sh uses to exit), then runs the A1 (live-addressing)
# 5-task triangle: blocks 1..5, 15 cells, 25 eps x 4 paired seeds per cell, ~6.5 h.
# A1's checkpoints ONLY — no control arm (Josh, 10 Sep; the control is already measured).
#
# Resume-safe: run_e68_retention_triangle.sh is skip-guarded per row, so a preemption relaunch
# picks up at the first missing row. Idempotent: exits immediately once both arms have 5 rows.
#
# Env: GATE_FREE_MIB (default 22000) — passed through so a row waits for VRAM if anything else is
#      on the card. POLL (default 600s) — how often to re-check the E67 gate.
set -uo pipefail
ROOT=/home/josh/lerobot
cd "$ROOT" || exit 1
E67=$ROOT/outputs/analysis/e67
A=$ROOT/outputs/analysis/e68
export GATE_FREE_MIB=${GATE_FREE_MIB:-22000}
POLL=${POLL:-600}
mkdir -p "$A"

say(){ echo "[e68-eval] $* $(date -u +%H:%M:%SZ)"; }
rows(){ ls "$A"/seeds_tri_$1_b*.json 2>/dev/null | wc -l; }
jrows(){ grep -c '^{' "$1" 2>/dev/null || echo 0; }

echo "=== E68 TRIANGLE QUEUE START $(date -u) (gate ${GATE_FREE_MIB} MiB, waits for E67) ==="

# ---- gate: E67 complete end to end -------------------------------------------------------------
while true; do
  nr=$(ls $E67/seeds_tri_retain10_a05_b*.json 2>/dev/null | wc -l)
  no=$(ls $E67/seeds_tri_olora10_r64_b*.json 2>/dev/null | wc -l)
  mr=$(jrows $E67/mse_matrix_retain10_a05.jsonl)
  mo=$(jrows $E67/mse_matrix_olora10_r64.jsonl)
  if [ "$nr" -ge 10 ] && [ "$no" -ge 10 ] && [ "$mr" -ge 10 ] && [ "$mo" -ge 10 ]; then
    say "E67 complete (retain $nr/10 + matrix $mr/10 | olora $no/10 + matrix $mo/10) - starting E68"
    break
  fi
  say "waiting on E67: retain rows $nr/10 matrix $mr/10 | olora rows $no/10 matrix $mo/10"
  sleep "$POLL"
done

# ---- A1 only (Josh, 10 Sep: new checkpoints only, no control arm) ------------------------------
tag=e68_a1_live
if [ "$(rows $tag)" -ge 5 ]; then
  say "a1 already 5/5 rows - nothing to do"
else
  say "a1 triangle starting (tag $tag, blocks 1-5, 15 cells)"
  bash scripts/vla_analysis/run_e68_retention_triangle.sh a1 2>&1 \
    | grep --line-buffered -v "exists - skipping\|checkpoint .* missing - skipping"
fi

na=$(rows $tag)
say "final: a1 $na/5 rows"
if [ "$na" -ge 5 ]; then echo "E68-TRIANGLES-DONE"; exit 0; fi
echo "E68-TRIANGLES-INCOMPLETE"; exit 1
