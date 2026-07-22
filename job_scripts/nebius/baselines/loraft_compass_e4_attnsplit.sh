#!/bin/bash
# E51 COMPASS EXTENSION: split the VLM tower's dense-LoRA gain (e4: 0.030 chunk / 40 roll)
# into ATTENTION vs MLP. Decides whether attention-side memory (the colleague's
# "architectural shift") is a live +15 candidate or dies in one night.
#
# Two single-task e4 arms, byte-identical recipe to loraft_compass_e4.sh except TARGETS:
#   ATTN: language_model self_attn q/k/v/o (all 18 blocks) + action/state projections
#         (~7.8M trainable at r32 - GQA makes k/v tiny: out-dim 256)
#   MLP:  language_model mlp gate/up/down (all 18 blocks) + same projections (~32M)
# Anchors (e4, 50-ep, chunk): full 0.020/58 | VLM-full(attn+MLP) 0.030/40 |
#   expert-only 0.229/14 | our memory frontier (comp) 0.0753/34.
#
# Pre-registered reads:
#   attn <= ~0.04 chunk AND roll >= ~35  -> attention alone reproduces the VLM gain =>
#       attention is a sufficient substrate; o-first sparse build justified.
#   attn >= ~0.10 AND train-loss PLATEAUED -> attention insufficient at any density =>
#       direction dead. (Plateau diagnostic per E44: expert-only ceilinged at 0.18.
#       If still DESCENDING at 5k, it's the 4.3x param asymmetry, not placement ->
#       rerun at r=64/128 before any verdict.)
#   mlp ~= 0.030-0.045 -> our MLP-site substrate is placed right; the 0.075->0.030
#       gap is sparse-capture/conversion, not location.
#   both mid -> complementary families; no single-site memory closes the gap.
# ARMS env var controls which arms run (default "attn mlp", attn FIRST - it is the
# must-complete cell for the bedtime review; mlp may be killed mid-train if the box
# is needed).
set -eo pipefail
echo "E51 LoRA compass attn-split (e4) started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
BASE_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model"
OUT_ROOT="$ROOT_DIR/outputs/train/loraft_compass"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$BASE_CKPT" ] || { echo "ERROR: stage-1 base checkpoint missing"; exit 1; }

COMMONS='model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out)'
VLM_ATTN_ONLY="(.*\\.language_model\\.layers\\.\\d+\\.self_attn\\.(q|k|v|o)_proj|$COMMONS)"
VLM_MLP_ONLY="(.*\\.language_model\\.layers\\.\\d+\\.mlp\\.(gate|up|down)_proj|$COMMONS)"
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
EPS="[$(seq -s, 0 37)]"   # t0/e4 episodes (E42-verified range)

run_arm () {  # $1=arm_name $2=targets
  local RUN_DIR="$OUT_ROOT/$1_t0_e4"
  if [ -d "$RUN_DIR/checkpoints/005000" ]; then
    echo "[$1] final checkpoint exists - skipping train."
  else
    echo "[$1] training ($(date))"
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
      --peft.target_modules="$2" \
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
      --job_name="loraft_compass_$1_t0_e4"
  fi
  if [ -f "$RUN_DIR/eval/eval_info.json" ]; then
    echo "[$1] eval exists - skipping."
  else
    echo "[$1] evaluating on env 4 @ 50 eps ($(date))"
    lerobot-eval \
      --policy.path="$RUN_DIR/checkpoints/005000/pretrained_model" \
      --policy.dtype=bfloat16 \
      --env.type=libero --env.task=libero_10 --env.task_ids="[4]" \
      --rename_map="$RENAME" \
      --eval.batch_size=1 \
      --eval.n_episodes=50 \
      --output_dir="$RUN_DIR/eval"
  fi
}

for ARM in ${ARMS:-attn mlp}; do
  case "$ARM" in
    attn) run_arm vlm_attn_only "$VLM_ATTN_ONLY" ;;
    mlp)  run_arm vlm_mlp_only "$VLM_MLP_ONLY" ;;
    *)    echo "unknown arm: $ARM"; exit 1 ;;
  esac
done
echo "E51 LoRA compass attn-split (e4) completed at $(date)"
