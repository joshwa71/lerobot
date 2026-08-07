#!/bin/bash
# E61 queued launch (Josh, 7 Aug: "make sure the layer share gets going").
# Waits for the e60-seeds campaign unit to release the GPU, then runs the E61
# shared-pairs chain in-place (this unit IS the chain — all output in one journal).
# The gate is on the UNIT, not the completion marker: if the campaign dies early,
# the GPU is free and E61 should launch anyway.
while true; do
  st=$(systemctl is-active e60-seeds 2>/dev/null)
  [ "$st" = "active" ] || [ "$st" = "activating" ] || break
  sleep 120
done
echo "[queue] e60-seeds exited (state=$st) — launching E61 sharepairs chain $(date -u)"
exec /bin/bash /home/josh/lerobot/job_scripts/nebius/libero_90/staged/joint_sharepairs_e681012_v791113_prepass_full_chain.sh
