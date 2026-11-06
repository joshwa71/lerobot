#!/bin/bash
# 50-ep re-eval of the [2,2,4,4] sequential FINAL (Entry 33 frontier, 46.5 @ 20 eps).
# De-noises the headline number (20-ep cells are +-3-4pp). Eval-only, ~7.5h at bs1.
set -eo pipefail
echo "reeval started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
CKPT="$ROOT_DIR/outputs/train/libero_10_sequential_pi05_8_10_12_14_film_lora_2244_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536_protect_beta4_steps5k/checkpoints/last/pretrained_model"
OUT="$ROOT_DIR/outputs/eval/r2244_final_50ep"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$CKPT" ] || { echo "ERROR: checkpoint missing"; exit 1; }
N_EPS="${1:-50}"
lerobot-eval \
  --policy.path="$CKPT" \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.n_episodes="$N_EPS" \
  --eval.batch_size=1 \
  --output_dir="$OUT" \
  --seed=1000
echo "reeval completed at $(date)"
