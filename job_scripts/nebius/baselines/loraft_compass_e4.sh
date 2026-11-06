#!/bin/bash
# E43 COMPASS: locate WHERE full-LoRA's e4 win (58% vs staged 20-35) lives.
# Two single-task e4 arms, same recipe as loraft_pertask_baseline.sh t0 except TARGETS:
#   A: EXPERT-ONLY  (gemma_expert attn+MLP + action/state projections; no VLM)
#   B: VLM-ONLY     (language_model attn+MLP + action/state projections; no expert)
# Full-LoRA anchor: e4=58 (attn+MLP both towers + proj, 53.2M). Reading grid (E43):
#   A~58, B low  -> expert side sufficient => n256/r4 staged build; VLM memory dead
#   A low, B~58  -> perception adaptation is the carrier => VLM-side memory is the build
#   both mid     -> both contribute => expert build first (cheaper), VLM second
#   both high    -> redundant capacity => cheapest build wins (n256/r4)
# Read each arm through 50-ep success AND the chunk/jitter probes (probe battery, E41 rule).
set -eo pipefail
echo "E43 LoRA compass (e4) started on $(hostname) at $(date)"
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

EXPERT_ONLY='(.*\.gemma_expert\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
VLM_ONLY='(.*\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
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

run_arm expert_only "$EXPERT_ONLY"
run_arm vlm_only "$VLM_ONLY"
echo "E43 LoRA compass (e4) completed at $(date)"
