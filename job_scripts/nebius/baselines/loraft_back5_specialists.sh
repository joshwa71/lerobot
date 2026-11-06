#!/bin/bash
# WEEKEND BASELINES (Josh, 13 Aug): the five MISSING per-task LoRA specialists —
# dataset task_index 5-9 (envs 0/8/1/3/5) — for the 10-task oracle row. Recipe
# byte-identical to the front-5 anchors (E42/E55 convention): frozen stage-1
# backbone, LoRA r=32 attn+MLP both towers + action projections, 5000 steps,
# bs16 x accum2 NO grad-ckpt (the E55 standard LoRA-cell config), lr 1e-4
# cosine w/ warmup 200 / decay 5000. NO 50-ep serial eval here — the 4-seed
# campaign (run_weekend_baselines.sh) is the instrument.
# Episode ranges verified from meta/episodes parquet (13 Aug):
#   t5 192-224 (env0 soup+sauce) t6 225-253 (env8 both-mokas)
#   t7 254-302 (env1 cheese+butter) t8 303-337 (env3 bowl+drawer)
#   t9 338-378 (env5 book+caddy)
set -eo pipefail
echo "Weekend back-5 LoRA specialists started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
BASE_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model"
OUT_ROOT="$ROOT_DIR/outputs/train/loraft_baseline"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$BASE_CKPT" ] || { echo "ERROR: stage-1 base checkpoint missing"; exit 1; }
python -c "import peft" || { echo "ERROR: peft not installed"; exit 1; }

TARGETS='(.*\.gemma_expert\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|.*\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'

declare -A ENV_ID=( [5]=0   [6]=8   [7]=1   [8]=3   [9]=5 )
declare -A EP_LO=(  [5]=192 [6]=225 [7]=254 [8]=303 [9]=338 )
declare -A EP_HI=(  [5]=224 [6]=253 [7]=302 [8]=337 [9]=378 )

for T in 5 6 7 8 9; do
  ENV=${ENV_ID[$T]}
  RUN_DIR="$OUT_ROOT/task${T}_e${ENV}"
  # stub-dir guard (E55/E60 lesson)
  if [ -d "$RUN_DIR" ] && [ ! -d "$RUN_DIR/checkpoints" ]; then
    echo "[t$T/e$ENV] wiping stub output dir (no checkpoints): $RUN_DIR"
    rm -rf "$RUN_DIR"
  fi
  EPS="[$(seq -s, ${EP_LO[$T]} ${EP_HI[$T]})]"
  if [ -d "$RUN_DIR/checkpoints/005000" ]; then
    echo "[t$T/e$ENV] final checkpoint exists - skipping train."
    continue
  fi
  echo "[t$T/e$ENV] training LoRA adapter ($(date))"
  lerobot-train \
    --policy.path="$BASE_CKPT" \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --gradient_accumulation_steps=2 \
    --policy.optimizer_lr=1e-4 \
    --policy.scheduler_warmup_steps=200 \
    --policy.scheduler_decay_steps=5000 \
    --policy.scheduler_decay_lr=1e-5 \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --peft.method_type=LORA \
    --peft.r=32 \
    --peft.target_modules="$TARGETS" \
    --peft.full_training_modules='[]' \
    --dataset.repo_id=libero_10 \
    --dataset.root="$ROOT_DIR/outputs/libero_10" \
    --dataset.episodes="$EPS" \
    --rename_map="$RENAME" \
    --output_dir="$RUN_DIR" \
    --steps=5000 \
    --batch_size=16 \
    --num_workers=8 \
    --log_freq=200 \
    --save_freq=5000 \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --job_name="loraft_baseline_t${T}_e${ENV}"
done
echo "Weekend back-5 LoRA specialists completed at $(date)"
