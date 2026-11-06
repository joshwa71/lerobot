#!/bin/bash
# E43: MULTI-TASK LoRA — the within-architecture discriminator for the breadth/threshold law
# (heuristic 1, E43 addendum). ONE r32 adapter (full targets = the 53.2M config) trained on
# ALL 5 tasks' episodes (0-191) at the SAME 5k-step budget as each specialist => 1/5 the
# per-task exposure; only breadth changes vs the per-task baseline.
#
# Pre-registered reads (vs specialists e4 58 / e6 44 / e9 70 and staged 32-42):
#   - per-task train MSE / chunk HIGHER than the specialists (less per-task data+budget)
#   - average rollout: breadth law says the support gain partially offsets the fit loss;
#     specifically e6 UP vs 44 (specialist's n=1 penalty removed) and e4 DOWN-or-equal vs 58
#     (e4 is precision-bound; less per-task budget hurts where threshold is tight)
#   - if avg ~> specialists' avg at clearly worse per-task fit => breadth law measured
#     within one architecture, everything else controlled; if avg collapses => e6's staged
#     advantage was NOT breadth (revisit the substrate story)
set -eo pipefail
echo "E43 multi-task LoRA (5 tasks, one adapter) started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
BASE_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model"
RUN_DIR="$ROOT_DIR/outputs/train/loraft_multitask5"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$BASE_CKPT" ] || { echo "ERROR: stage-1 base checkpoint missing"; exit 1; }

TARGETS='(.*\.gemma_expert\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|.*\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
EPS="[$(seq -s, 0 191)]"   # t0-t4 episode ranges (0-37,38-73,74-107,108-148,149-191)

if [ -d "$RUN_DIR/checkpoints/005000" ]; then
  echo "[mt5] final checkpoint exists - skipping train."
else
  lerobot-train \
    --policy.path="$BASE_CKPT" \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
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
    --batch_size=32 \
    --num_workers=8 \
    --log_freq=200 \
    --save_freq=5000 \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --job_name="loraft_multitask5"
fi
if [ -f "$RUN_DIR/eval/eval_info.json" ]; then
  echo "[mt5] eval exists - skipping."
else
  echo "[mt5] evaluating on envs 4,6,9,2,7 @ 50 eps each ($(date))"
  lerobot-eval \
    --policy.path="$RUN_DIR/checkpoints/005000/pretrained_model" \
    --policy.dtype=bfloat16 \
    --env.type=libero --env.task=libero_10 --env.task_ids="[4,6,9,2,7]" \
    --rename_map="$RENAME" \
    --eval.batch_size=1 \
    --eval.n_episodes=50 \
    --output_dir="$RUN_DIR/eval"
fi
echo "E43 multi-task LoRA completed at $(date)"
