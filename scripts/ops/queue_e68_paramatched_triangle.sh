#!/usr/bin/env bash
# E68 add-25: 4-seed rollout retention triangle for the PARAMETER-MATCHED naive sequential LoRA
# (E66: r=1216 / alpha=304, 2.681B added), rows b1..b9 — the paper's Table I row "(b) Naive seq.
# LoRA, matched parameters" has only the final (b10) row so far. b10 already exists as the E66
# campaign (outputs/analysis/e60/seeds_naive10_paramatched_r1216.json — same instrument: all ten
# envs, 25 eps x 4 paired seeds 1000/2000/3000/4000, bs13) and is NOT re-run.
#
# Runs on nebius2 (H100) once the A3 held-out audit unit (e68-a3-audit) has released the GPU; the
# nine adapter checkpoints (pretrained_model only, 10.7 GB each) and the LoRA base
# (libero_90_pi05_base_nomem_50k) are mirrored from nebius-spot beforehand.
# Resume-safe: run_baseline_triangle.sh is skip-guarded per row, so a preemption relaunch picks
# up at the first missing row. Idempotent: exits 0 once every row in BLOCKS exists.
# Env: GATE_FREE_MIB (default 22000), BLOCKS (default "1 .. 9"), POLL (default 120 s),
#      WAIT_UNIT (default e68-a3-audit): the systemd unit to wait for before starting — set it to
#      e68-pm-tri-a to chain the second half behind the first (add-27: two halves side by side were
#      SLOWER in aggregate, 3.4 vs 4.9 env-steps/s — 16 vCPUs saturate on osmesa rendering).
# Two units with DISJOINT BLOCKS may run side by side on the same box (Josh, 12 Sep: "half that if
# possible") - rollouts are env/CPU-bound, the H100 has VRAM for two eval processes, and the per-row
# skip guard makes the shared output dir safe. Cell-balanced split: "1 2 3 4 5 6" (21) | "9 8 7" (24).
set -uo pipefail
ROOT=/home/josh/lerobot
cd "$ROOT" || exit 1
export GATE_FREE_MIB=${GATE_FREE_MIB:-22000}
POLL=${POLL:-120}
BLOCKS=${BLOCKS:-"1 2 3 4 5 6 7 8 9"}
TAG=naive10_paramatched_r1216
say(){ echo "[e68-pm-tri] $* $(date -u +%H:%M:%SZ)"; }
rows(){ ls "$ROOT"/outputs/analysis/e67/seeds_tri_${TAG}_b*.json 2>/dev/null | wc -l; }
echo "=== E68 PARAM-MATCHED TRIANGLE QUEUE START $(date -u) (blocks: $BLOCKS; gate ${GATE_FREE_MIB} MiB) ==="
WAIT_UNIT=${WAIT_UNIT:-e68-a3-audit}
if systemctl is-active "$WAIT_UNIT" >/dev/null 2>&1; then
  say "waiting for unit $WAIT_UNIT to finish"
  while systemctl is-active "$WAIT_UNIT" >/dev/null 2>&1; do sleep "$POLL"; done
  say "$WAIT_UNIT finished - GPU released"
fi
bash scripts/baselines/run_baseline_triangle.sh paramatched "$BLOCKS" 2>&1 \
  | grep --line-buffered -v "exists - skipping"
n=0; want=0
for K in $BLOCKS; do want=$((want+1)); [ -f "$ROOT/outputs/analysis/e67/seeds_tri_${TAG}_b${K}.json" ] && n=$((n+1)); done
say "final: $n/$want rows for blocks [$BLOCKS]; $(rows)/9 rows on this box overall (b10 = the E66 campaign row)"
if [ "$n" -ge "$want" ]; then echo "E68-PM-TRIANGLE-DONE blocks=[$BLOCKS]"; exit 0; fi
echo "E68-PM-TRIANGLE-INCOMPLETE blocks=[$BLOCKS]"; exit 1
