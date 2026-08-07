#!/bin/bash
# E60: e2 seed re-eval (noise check for the 74-vs-84 give-back read).
# Same instrument as the historical finals except seed 2000 (vs 1000).
set -eo pipefail
ROOT=/home/josh/lerobot
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
CKPT=$ROOT/outputs/train/libero_10_seq5_jw_interleave_e681012_v791113_prepass_beta4corefrac_topt3072_lr2x_steps5k/checkpoints/025000/pretrained_model
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
D=$ROOT/outputs/analysis/e60/e2_reeval_interleave_seed2000
rm -rf "$D"
lerobot-eval \
  --policy.path="$CKPT" \
  --policy.dtype=bfloat16 \
  --env.type=libero --env.task=libero_10 --env.task_ids="[2]" \
  --rename_map="$RENAME" \
  --eval.batch_size=10 --eval.n_episodes=50 \
  --seed=2000 \
  --output_dir="$D"
python -c "
import json
d = json.load(open('$D/eval_info.json'))
print('E2-REEVAL-RESULT seed2000:', d.get('overall', {}).get('pc_success'))
"
