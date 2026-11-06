#!/usr/bin/env bash
# Download hub models/datasets that jobs need. Run on a LOGIN node: compute nodes must not download,
# and jobs run with HF_HUB_OFFLINE=1 so a missing asset fails fast instead of downloading.
#   bash cluster/download_assets.sh [extra_model_or_dataset ...]
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"

MODELS=(lerobot/smolvla_base HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
# HuggingFaceVLA/libero and jesbu1/libero_90_lerobot are sources for scripts/datasets/build_libero_*.py
DATASETS=(HuggingFaceVLA/libero jesbu1/libero_90_lerobot)

for m in "${MODELS[@]}"; do
    echo "[$(date '+%F %T')] model $m"
    ctr hf download "$m" > /dev/null
done
for d in "${DATASETS[@]}"; do
    echo "[$(date '+%F %T')] dataset $d -> $HF_LEROBOT_HOME/$d"
    ctr hf download "$d" --repo-type dataset --local-dir "$HF_LEROBOT_HOME/$d" > /dev/null
done
echo "[$(date '+%F %T')] done"
