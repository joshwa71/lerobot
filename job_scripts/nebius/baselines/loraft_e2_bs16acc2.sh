#!/bin/bash
# grad-ckpt OFF (29 Jul 26): LoRA cells measured 30GB/141GB with ckpt ON; on a frozen backbone ckpt recompute ~triples the backward (E53 mechanism) -> 3.32 s/step. Numerically equivalent to the ckpt-on t0-t2/t4 anchors.
# bs16 x accum2, NO grad-ckpt (Josh, 29 Jul): bs32 no-ckpt OOMs (measured 138.4GiB demand); bs32+ckpt works but recompute ~3x backward. Effective batch 32 preserved.
# E42 arm 3 (VM3): per-task LoRA-FT baseline — the Layer-3 arbiter (queued since E31/E36/E40).
#
# Five INDEPENDENT LoRA finetunes of the frozen stage-1 backbone (libero_90_pi05_base_nomem_50k,
# read-only), one per libero_10 task (dataset task_index 0-4 = envs 4/6/9/2/7), matched budget
# (5000 steps, bs32, like our sequential blocks), then a 50-ep eval of each adapter on ITS env.
#
# What it decides (pre-registered, E40/E41): the memory arms sit at inits 35-47 vs base-joint
# 74.8 on these 5 tasks. LoRA-FT is the fit CEILING of standard adapter adaptation on this
# exact frozen backbone:
#   - e4/e9 at ~35-50  => backbone sufficient; the deficit is OUR sparse-mixture machinery's
#     conversion tax (closable in-protocol, and now priced).
#   - e4/e9 at ~<=15   => frozen-backbone adaptation itself is the ceiling; thesis claim
#     scopes to "adapter-level fit with zero forgetting".
#   - e4/e9 at ~>=60   => large machinery tax; rethink the value path before more knob arms.
# Read through the probe battery (chunk metric on its checkpoints), not just success cells.
#
# Config notes: LoRA r=32 (PEFT default lora_alpha=8 => scale 0.25; LR compensates), targets =
# attn(q,k,v,o)+MLP(gate,up,down) of BOTH gemma_expert and the paligemma language model +
# action projections (53.2M adapters; vision tower untouched). lr 1e-4 cosine, warmup 200,
# decay 5000 == steps (schedule honored, E20 gotcha). modules_to_save=[] (pure adapter).
# Requires: pip install peft (0.19.1 smoked); the wrap_with_peft get_optim_params fix (E42).
set -eo pipefail
echo "E55 LoRA-FT e2 specialist (bs16xacc2 no-ckpt) started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
BASE_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model"
OUT_ROOT="$ROOT_DIR/outputs/train/loraft_baseline"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$BASE_CKPT" ] || { echo "ERROR: stage-1 base checkpoint missing"; exit 1; }
python -c "import peft" || { echo "ERROR: peft not installed (pip install peft)"; exit 1; }

TARGETS='(.*\.gemma_expert\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|.*\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'

# dataset task_index -> env id (ds_to_env_map) and contiguous episode ranges (E42, verified)
declare -A ENV_ID=( [0]=4 [1]=6 [2]=9 [3]=2 [4]=7 )
declare -A EP_LO=( [0]=0  [1]=38 [2]=74  [3]=108 [4]=149 )
declare -A EP_HI=( [0]=37 [1]=73 [2]=107 [3]=148 [4]=191 )

for T in 3; do
  ENV=${ENV_ID[$T]}
  RUN_DIR="$OUT_ROOT/task${T}_e${ENV}"
  EPS="[$(seq -s, ${EP_LO[$T]} ${EP_HI[$T]})]"
  if [ -d "$RUN_DIR/checkpoints/005000" ]; then
    echo "[t$T/e$ENV] final checkpoint exists - skipping train."
  else
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
  fi
  if [ -f "$RUN_DIR/eval/eval_info.json" ] || [ -d "$RUN_DIR/eval" ]; then
    echo "[t$T/e$ENV] eval dir exists - skipping eval."
  else
    echo "[t$T/e$ENV] evaluating adapter on env $ENV @ 50 eps ($(date))"
    lerobot-eval \
      --policy.path="$RUN_DIR/checkpoints/005000/pretrained_model" \
      --policy.dtype=bfloat16 \
      --env.type=libero --env.task=libero_10 --env.task_ids="[$ENV]" \
      --rename_map="$RENAME" \
      --eval.batch_size=1 \
      --eval.n_episodes=50 \
      --output_dir="$RUN_DIR/eval"
  fi
done
echo "E55 LoRA-FT e2 specialist completed at $(date)"
