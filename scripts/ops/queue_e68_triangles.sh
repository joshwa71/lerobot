#!/usr/bin/env bash
# E68 4-seed rollout retention triangles on nebius-spot, QUEUED BEHIND the E67 eval queue.
#
# Waits until E67 is finished end to end (both triangles 10/10 rows AND both drift matrices 10/10
# rows — the same condition queue_e67_eval.sh uses to exit), then runs the A1 (live-addressing)
# triangle followed by the paper-cell control. 15 cells each (blocks 1..5), ~6.5 h each.
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

# ---- the two arms, A1 first --------------------------------------------------------------------
for arm in a1 control; do
  case $arm in a1) tag=e68_a1_live ;; control) tag=e68_control5 ;; esac
  if [ "$(rows $tag)" -ge 5 ]; then say "$arm already 5/5 rows - skipping"; continue; fi
  say "$arm triangle starting (tag $tag)"
  bash scripts/vla_analysis/run_e68_retention_triangle.sh "$arm" 2>&1 \
    | grep --line-buffered -v "exists - skipping\|checkpoint .* missing - skipping"
  say "$arm triangle returned: $(rows $tag)/5 rows"
done

na=$(rows e68_a1_live); nc=$(rows e68_control5)
say "final: a1 $na/5 | control $nc/5"
if [ "$na" -ge 5 ] && [ "$nc" -ge 5 ]; then echo "E68-TRIANGLES-DONE"; exit 0; fi
echo "E68-TRIANGLES-INCOMPLETE"; exit 1
