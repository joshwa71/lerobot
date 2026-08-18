#!/bin/bash
# E64 (Josh, 18 Aug): the TRUE 10-task multitask-LoRA baseline — one adapter,
# ALL 379 libero_10 episodes, provisioned to match our method's budget instead
# of the E43 breadth-probe convention (r=32, 1k steps/task) that leaked into
# the paper table:
#   r=512 / lora_alpha=128 (alpha/r = 0.25 held, same as the r32@a8 specialists
#                           and the r512 naive foil -> same effective
#                           update scale; 852M trainable, ~240x our per-token
#                           active params, ~1.8x our per-site bottleneck (288), ~3.7x per-step budget)
#   steps=50000            (= 5k/task, matched to our sequential's 10 x 5k)
# Everything else byte-identical to loraft_multitask10.sh (frozen stage-1 base,
# same attn+MLP targets both towers, bs16 x accum2 no-ckpt, lr 1e-4 -> 1e-5,
# warmup 200; decay scaled to 50000 so the schedule is honored, E20 gotcha).
# Eval = the 4-seed campaign (run_e64_lora_r512_queue.sh), not in-run.
#
# PREEMPTION NOTE: lerobot-train has NO validated PEFT resume path (E58 add-5
# built one for the SEQUENTIAL trainer only; here --resume would re-wrap and
# double-adapter). save_freq=10000 keeps salvage checkpoints, but a preempted
# run RESTARTS FROM SCRATCH: partial dirs are moved aside, never resumed.
set -eo pipefail
echo "E64 multitask-LoRA-10 r512/50k started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
BASE_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model"
RUN_DIR="$ROOT_DIR/outputs/train/loraft_multitask10_r512_50k"
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

if [ -d "$RUN_DIR/checkpoints/050000" ]; then
  echo "[mt10-r512] final checkpoint exists - skipping train."
elif [ -d "$RUN_DIR" ]; then
  # stub dir (no checkpoints) OR a partial run (preempted mid-way): no PEFT resume
  # exists for lerobot-train, so move it aside and restart clean.
  ASIDE="${RUN_DIR}_partial_$(date -u +%Y%m%dT%H%M%S)"
  echo "[mt10-r512] partial/stub output dir found -> moving aside to $ASIDE and RESTARTING FROM SCRATCH"
  mv "$RUN_DIR" "$ASIDE"
fi
if [ ! -d "$RUN_DIR/checkpoints/050000" ]; then
  lerobot-train \
    --policy.path="$BASE_CKPT" \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --gradient_accumulation_steps=2 \
    --policy.optimizer_lr=1e-4 \
    --policy.scheduler_warmup_steps=200 \
    --policy.scheduler_decay_steps=50000 \
    --policy.scheduler_decay_lr=1e-5 \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --peft.method_type=LORA \
    --peft.r=512 \
    --peft.lora_alpha=128 \
    --peft.target_modules="$TARGETS" \
    --peft.full_training_modules='[]' \
    --dataset.repo_id=libero_10 \
    --dataset.root="$ROOT_DIR/outputs/libero_10" \
    --rename_map="$RENAME" \
    --output_dir="$RUN_DIR" \
    --steps=50000 \
    --batch_size=16 \
    --num_workers=8 \
    --log_freq=200 \
    --save_freq=10000 \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --job_name="loraft_multitask10_r512_50k"
fi
[ -d "$RUN_DIR/checkpoints/050000" ] || { echo "[mt10-r512] FATAL: 050000 missing"; exit 1; }
echo "E64 multitask-LoRA-10 r512/50k completed at $(date)"
