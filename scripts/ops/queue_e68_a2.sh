#!/usr/bin/env bash
# E68 A2 ("TF-IDF-only writes", protect_prior_slots=false) 4-seed retention triangle,
# QUEUED BEHIND the A1 triangle on nebius-spot.
#
# Waits until A1 has all 5 rows, then runs run_e68_retention_triangle.sh a2 — 15 cells
# (blocks 1..5), 25 eps x 4 paired seeds per cell, ~6.5 h.
#
# b1 is MEASURED, not adopted: the script's adoption rule copies seeds_tri_e68_control5_b1.json
# only if it exists, and no control arm is being run. A2's checkpoint 005000 is a byte copy of
# the paper cell's task-1 boundary (the fork is exact — with an empty protection store the write
# score reduces to plain TF-IDF), so the measured b1 doubles as the matched 5-task control's b1
# and should land near the comparator's 54.0. A b1 far from that would indicate a fork or
# instrument problem, not an ablation effect.
#
# Comparator (no control arm): outputs/analysis/e60/seeds_tri_merged6x2_10task_b{1..5}.json
# (E64 add-12) = 54.0 / 59.5 / 62.3 / 64.2 / 66.0.
#
# Resume-safe: rows are skip-guarded, so a preemption relaunch picks up at the first missing row.
set -uo pipefail
ROOT=/home/josh/lerobot
cd "$ROOT" || exit 1
A=$ROOT/outputs/analysis/e68
export GATE_FREE_MIB=${GATE_FREE_MIB:-22000}
POLL=${POLL:-600}
mkdir -p "$A"

say(){ echo "[e68-a2] $* $(date -u +%H:%M:%SZ)"; }
rows(){ ls "$A"/seeds_tri_$1_b*.json 2>/dev/null | wc -l; }

echo "=== E68 A2 TRIANGLE QUEUE START $(date -u) (gate ${GATE_FREE_MIB} MiB, waits for A1) ==="

while true; do
  na=$(rows e68_a1_live)
  [ "$na" -ge 5 ] && { say "A1 complete ($na/5) - starting A2"; break; }
  say "waiting on A1: $na/5 rows"
  sleep "$POLL"
done

if [ "$(rows e68_a2_tfidfonly)" -ge 5 ]; then
  say "a2 already 5/5 rows - nothing to do"
else
  say "a2 triangle starting (blocks 1-5, 15 cells)"
  bash scripts/vla_analysis/run_e68_retention_triangle.sh a2 2>&1 \
    | grep --line-buffered -v "exists - skipping\|checkpoint .* missing - skipping"
fi

n=$(rows e68_a2_tfidfonly)
say "final: a2 $n/5 rows"
[ "$n" -ge 5 ] && { echo "E68-A2-DONE"; exit 0; }
echo "E68-A2-INCOMPLETE"; exit 1
