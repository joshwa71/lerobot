#!/bin/bash
# E62 queued launch (Josh, 11 Aug: "write the script and queue up the 6x2").
# Waits for the sharepairs-seeds unit (the E61 4-seed eval, last GPU consumer of
# the baseline chain) to release the GPU, then runs the E62 merged-6x2 chain
# in-place (this unit IS the chain — all output in one journal).
# Gate is on the UNIT, not a completion marker: if the eval dies early the GPU is
# free and E62 should launch anyway. NB is-active exits nonzero for inactive —
# never let that kill the loop (E61-add-6 lesson).
while true; do
  st=$(systemctl is-active sharepairs-seeds 2>/dev/null) || true
  [ "$st" = "active" ] || [ "$st" = "activating" ] || break
  sleep 300
done
echo "[queue] sharepairs-seeds exited (state=$st) — launching E62 merged6x2 chain $(date -u)"
exec /bin/bash /home/josh/lerobot/job_scripts/nebius/libero_90/staged/joint_merged6x2_e468101416_v579111315_prepass_full_chain.sh
