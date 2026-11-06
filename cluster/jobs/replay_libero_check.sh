#!/bin/bash
# Replay-check the built LIBERO datasets against the simulator (see
# scripts/datasets/replay_libero_actions.py); one JSON report per dataset in $CLUSTER_ROOT/logs/.
# Reference numbers from the local validation (6 Nov 26): rot180 orientation on every episode,
# median start offset ~1.8-1.9 cm, tracking error ~1.9 cm over the first 30 steps, gripper
# agreement ~1.0; open-loop success 5% (libero_10) / 48% (libero_90).
#   qsub cluster/jobs/replay_libero_check.sh
#$ -S /bin/bash
#$ -N replay_libero
#$ -l tmem=32G,h_rt=4:00:00
#$ -l gpu=true,gpu_type=(a10|a100|a100_80|a40|a6000|h100|l4|l40s|rtx3090|rtx4070ti|rtx4090|rtx6000ada)
#$ -j y
#$ -o /SAN/vision/jo71_vla_wd/logs/
set -euo pipefail
source /SAN/vision/jo71_vla_wd/lerobot/cluster/env.sh
export HF_HUB_OFFLINE=1 PYTHONFAULTHANDLER=1
echo "== $(hostname) $(date) job=$JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
cd "$REPO/scripts/datasets"
for name in libero_10 libero_goal libero_object libero_spatial libero_90; do
    echo "== $name $(date)"
    timeout -s ABRT 90m "$REPO/cluster/bin/ctr" python replay_libero_actions.py --root "$HF_LEROBOT_HOME/$name" --suite "$name" \
        --episodes-per-task 3 --report "$CLUSTER_ROOT/logs/replay_$name.json" 2>&1 \
        | grep -E '^(task|skip)|"(replay|tasks_with|rot180|raw|median)' || true
done
echo "== done $(date)"
