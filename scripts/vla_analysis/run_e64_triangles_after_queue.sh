#!/bin/bash
# E64 (18 Aug): the two retention TRIANGLES, as a unit GATED on `e64-lora-r512`
# exiting — deliberately NOT relying on the queue script's own stage 3b/4.
# Reason (E64 add-3): the queue script was already executing under bash when the
# triangle stages were added to it, and bash reads a script lazily, so the running
# process may hold the pre-triangle revision. This unit makes the outcome identical
# either way — every stage is skip-guarded on its output JSON, so whichever copy of
# the queue ran, nothing is measured twice:
#   - running bash has the NEW queue  -> triangles already done -> this unit skips all.
#   - running bash has the OLD queue  -> its stage 3 wrote seeds_naive10_r512_final.json
#     (adopted as naive b10 by the triangle script) and this unit runs the rest.
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e64_triangles.log
exec >> "$LOG" 2>&1
echo "=== E64 triangles: waiting on e64-lora-r512 $(date -u) ==="
while true; do
  st=$(systemctl is-active e64-lora-r512 2>/dev/null) || true
  [ "$st" = "active" ] || [ "$st" = "activating" ] || break
  sleep 300
done
echo "=== E64 triangles: gate passed (e64-lora-r512=$st) $(date -u) ==="
bash $ROOT/scripts/vla_analysis/run_e64_retention_triangle.sh naive     || echo "[FAIL] naive triangle"
bash $ROOT/scripts/vla_analysis/run_e64_retention_triangle.sh merged6x2 || echo "[FAIL] merged6x2 triangle"
echo "=== E64 TRIANGLES COMPLETE $(date -u) ==="
