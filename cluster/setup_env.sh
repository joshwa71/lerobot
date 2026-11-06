#!/usr/bin/env bash
# (Re)build the Python venv at $VENV from uv.lock, inside the container.
#
# Run on a LOGIN node: it downloads packages, and compute nodes must not download anything.
# Login nodes are slow on /SAN, so this takes a long time; that is expected. It is safe to re-run
# (uv's cache keeps everything already downloaded), so run it detached:
#   nohup bash cluster/setup_env.sh > $CLUSTER_ROOT/logs/setup_env.log 2>&1 &
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"

EXTRAS=(smolvla pi libero training peft evaluation test)

[ -e "$SIF" ] || { echo "No container image at $SIF (see cluster/Dockerfile)"; exit 1; }
mkdir -p "$UV_CACHE_DIR" "$LIBERO_CONFIG_PATH" "$HF_HOME" "$HF_LEROBOT_HOME"
cd "$REPO"

echo "[$(date '+%F %T')] uv sync (extras: ${EXTRAS[*]})"
ctr uv sync --locked $(printf -- '--extra %s ' "${EXTRAS[@]}")

echo "[$(date '+%F %T')] import check"
ctr python - <<'EOF'
import sys, torch, transformers, lerobot
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| transformers", transformers.__version__, "| lerobot", lerobot.__version__)
EOF
echo "[$(date '+%F %T')] done"
