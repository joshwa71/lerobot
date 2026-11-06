#!/bin/bash
# Resume probe C (SupCon 0.05 + negatives_only + queue 512) from 10k -> 40k.
# Probe C passed both its in-run gates and the held-out routing audit
# (family IoU -46%, footprints ~7x smaller; see research_log Entry 20).
#
# The probe was a truncated full run (scheduler warmup 4000 / decay 40000), so
# resuming continues the original LR trajectory from step 10001. Saved config
# provides everything; only --steps is overridden (10000 -> 40000).
#   - save_freq=10000 in saved config -> checkpoints at 20k/30k/40k (audit points)
#   - eval_freq=20000 -> held-in evals at 20k and 40k (compare control: 76.4/81.1)
#
# NOTE: if the CLI --steps override is ignored on resume (older lerobot resume
# semantics use the checkpoint config verbatim), the run exits immediately at
# step 10000 with "End of training" — check the log a few minutes after launch;
# the fallback is editing steps in checkpoints/last/pretrained_model/train_config.json.
#
# Watch during the run (vs control 7wu3dyax):
#   - train/mse_loss tracking control's curve (0.196 @20k, 0.133 @40k)
#   - eval/pc_success @20k/@40k vs control 76.4 / 81.1
#   - routing_intra_task_support_* and query_inter/intra_sim stability
#     (does compaction hold as LR decays, or re-broaden?)

set -eo pipefail

echo "Resume probe C -> 40k started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot
OUT="$ROOT_DIR/outputs/train/libero_90_pi05_8_10_12_14_probe10k_contrastive_0.05_negonly_q512"
CFG="$OUT/checkpoints/last/pretrained_model/train_config.json"

if [ ! -f "$CFG" ]; then
  echo "ERROR: saved train config not found: $CFG"; exit 1
fi

export MUJOCO_GL=osmesa
unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated

cd "$ROOT_DIR"

lerobot-train \
  --config_path="$CFG" \
  --resume=true \
  --steps=40000

echo "Resume probe C -> 40k completed at $(date)"
