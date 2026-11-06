#!/bin/bash
# E64 (18 Aug): ALL TEN per-task LoRA specialists re-provisioned at r=512 /
# lora_alpha=128 — the uniform adapter rank across every LoRA row of the paper
# table (multitask-10 r512/50k, naive-seq r512, and now the specialist oracle),
# chosen to sit above our method on every compute/optimization rung of the
# active-parameter ladder (per-site bottleneck 288/128 -> 512 is above both; per-step ~3.7x) while below only its
# total storage (matched by the full-FT rows). The r=32 specialists stay as the
# appendix sensitivity row.
# Everything else byte-identical to the r32 anchors (loraft_pertask_baseline.sh /
# loraft_back5_specialists.sh): frozen stage-1 backbone, attn+MLP both towers +
# action projections, 5000 steps (= our per-task budget), bs16 x accum2 no-ckpt,
# lr 1e-4 cosine, warmup 200 / decay 5000. NO in-run eval — the 4-seed campaign
# (run_e64_lora_r512_queue.sh) is the instrument. Per task ~3h; ~30h all ten.
# Episode ranges verified from meta/episodes parquet (13 Aug):
#   t0 0-37 (e4)   t1 38-73 (e6)   t2 74-107 (e9)  t3 108-148 (e2)  t4 149-191 (e7)
#   t5 192-224 (e0) t6 225-253 (e8) t7 254-302 (e1) t8 303-337 (e3) t9 338-378 (e5)
# 5k-step runs are short enough that a preempted task simply reruns (partial dir
# moved aside; skip-guard on the final checkpoint).
set -eo pipefail
echo "E64 r512 LoRA specialists (all 10) started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
BASE_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model"
OUT_ROOT="$ROOT_DIR/outputs/train/loraft_baseline_r512"
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

declare -A ENV_ID=( [0]=4  [1]=6  [2]=9   [3]=2   [4]=7   [5]=0   [6]=8   [7]=1   [8]=3   [9]=5 )
declare -A EP_LO=(  [0]=0  [1]=38 [2]=74  [3]=108 [4]=149 [5]=192 [6]=225 [7]=254 [8]=303 [9]=338 )
declare -A EP_HI=(  [0]=37 [1]=73 [2]=107 [3]=148 [4]=191 [5]=224 [6]=253 [7]=302 [8]=337 [9]=378 )

# TASKS env override lets the queue (or a relaunch) run a subset; default all ten in train order.
for T in ${TASKS:-0 1 2 3 4 5 6 7 8 9}; do
  ENV=${ENV_ID[$T]}
  RUN_DIR="$OUT_ROOT/task${T}_e${ENV}"
  if [ -d "$RUN_DIR/checkpoints/005000" ]; then
    echo "[t$T/e$ENV] final checkpoint exists - skipping train."
    continue
  fi
  if [ -d "$RUN_DIR" ]; then
    ASIDE="${RUN_DIR}_partial_$(date -u +%Y%m%dT%H%M%S)"
    echo "[t$T/e$ENV] partial/stub dir found -> moving aside to $ASIDE, rerunning from scratch"
    mv "$RUN_DIR" "$ASIDE"
  fi
  EPS="[$(seq -s, ${EP_LO[$T]} ${EP_HI[$T]})]"
  echo "[t$T/e$ENV] training r512 LoRA specialist ($(date))"
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
    --peft.r=512 \
    --peft.lora_alpha=128 \
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
    --job_name="loraft_baseline_r512_t${T}_e${ENV}"
  [ -d "$RUN_DIR/checkpoints/005000" ] || echo "[t$T/e$ENV] WARNING: 005000 missing after training"
done
echo "E64 r512 LoRA specialists completed at $(date)"
