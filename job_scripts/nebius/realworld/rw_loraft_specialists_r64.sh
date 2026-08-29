#!/bin/bash
# E65 REAL-WORLD r64 LoRA SPECIALISTS — realworld duplicate of baselines/loraft_specialists10_r64.sh
# (the E64c "equivalent-rank" rung: r=64 / lora_alpha=16, alpha/r = 0.25). One dense adapter per
# held-out task, trained from the RW stage-1 finetune (realworld_${RW_TAG}_pi05_base_nomem_50k) on that
# task's episodes of the RW SEQ split. Recipe byte-identical to the sim rung: attn+MLP both towers +
# action/state projections, 5000 steps (= our per-task budget), bs16 x acc2 no-ckpt, lr 1e-4 cosine
# (warmup 200 / decay 5000 -> 1e-5), NO in-run eval (no simulator; own-task MSE via mse_matrix_peft.py
# in the weekend queue; robot evals are Josh's). Deltas vs sim: datasets/maps from rw_env.sh,
# --policy.empty_cameras=1 (explicit; the stage-1 config carries it), --eval_freq=0 (no --env.*),
# --policy.push_to_hub=false.
# Episode ranges of realworld_seq_v5 (verified from meta/episodes 29 Aug; RE-ASSERTED at runtime):
#   t0 0-50 (mustard-basket)  t1 51-99 (push white lego brick)  t2 100-149 (stack yellow bricks)
#   t3 150-199 (screwdriver-tub)  t4 200-250 (red bow-plate)
# 5k-step runs are short enough that a preempted task simply reruns (partial dir moved aside;
# skip-guard on the final checkpoint). TASKS env = subset override (default "0 1 2 3 4", train order).
# SMOKE=1: 20 steps, task 1 only, throwaway _smoke_ dir, wandb off.  DRYRUN=1: print commands only.
set -eo pipefail
source /home/josh/lerobot/job_scripts/nebius/realworld/rw_env.sh
DRYRUN=${DRYRUN:-0}
LORA_R=${LORA_R:-64}; LORA_ALPHA=${LORA_ALPHA:-16}
STEPS=${SPEC_STEPS:-5000}
TASK_LIST=${TASKS:-"0 1 2 3 4"}
WARMUP=200
if [ "$SMOKE" = "1" ]; then STEPS=20; WARMUP=2; TASK_LIST=${TASKS:-"1"}; fi
FINAL=$(printf '%06d' "$STEPS")
# the REAL stage-1 checkpoint even under SMOKE=1 (rw_env prefixes STAGE1_CKPT with _smoke_; no smoke stage-1 exists)
BASE_CKPT=${BASE_CKPT:-$ROOT_DIR/outputs/train/${STAGE1_RUN}/checkpoints/last/pretrained_model}
OUT_ROOT=$ROOT_DIR/outputs/train/${RUN_PREFIX}rw_${RW_TAG}_loraft_baseline_r${LORA_R}
TARGETS='(.*\.gemma_expert\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|.*\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
declare -A EP_LO=( [0]=0  [1]=51 [2]=100 [3]=150 [4]=200 )
declare -A EP_HI=( [0]=50 [1]=99 [2]=149 [3]=199 [4]=250 )
echo "RW r${LORA_R}/a${LORA_ALPHA} LoRA specialists (tasks: $TASK_LIST; $STEPS steps; smoke=$SMOKE dryrun=$DRYRUN) started on $(hostname) at $(date -u)"
[ -d "$BASE_CKPT" ] || { echo "ERROR: stage-1 checkpoint missing: $BASE_CKPT"; exit 1; }
python -c "import peft" || { echo "ERROR: peft not installed"; exit 1; }
# Runtime assertion: every episode in each hardcoded range belongs to that seq task_index.
python - "$RW_SEQ_ROOT" "0:0-50,1:51-99,2:100-149,3:150-199,4:200-250" <<'PY'
import glob, os, sys
import pandas as pd
root, spec = sys.argv[1], sys.argv[2]
t = pd.read_parquet(os.path.join(root, "meta", "tasks.parquet"))
name_to_idx = {str(n): int(r["task_index"]) for n, r in t.iterrows()}
e = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True))])
e = e.sort_values("episode_index")
idx = {int(r.episode_index): name_to_idx[str(list(r.tasks)[0])] for r in e.itertuples()}
bad = 0
for tok in spec.split(","):
    ti, rng = tok.split(":"); lo, hi = (int(x) for x in rng.split("-"))
    got = {idx[i] for i in range(lo, hi + 1)}
    if got != {int(ti)}:
        print(f"[spec-assert] FAIL task {ti} eps {lo}-{hi}: task_index set {sorted(got)}"); bad += 1
    else:
        print(f"[spec-assert] ok task {ti} eps {lo}-{hi} ({hi-lo+1} eps)")
sys.exit(1 if bad else 0)
PY
if [ "$SMOKE" = "1" ] && [ "$DRYRUN" != "1" ]; then rm -rf "$OUT_ROOT"; fi
for T in $TASK_LIST; do
  RUN_DIR="$OUT_ROOT/task${T}"
  if [ -d "$RUN_DIR/checkpoints/$FINAL" ]; then
    echo "[spec t$T] final checkpoint exists - skipping train."; continue
  fi
  if [ -d "$RUN_DIR" ] && [ "$DRYRUN" != "1" ]; then
    ASIDE="${RUN_DIR}_partial_$(date -u +%Y%m%dT%H%M%S)"
    echo "[spec t$T] partial/stub dir found -> moving aside to $ASIDE, rerunning from scratch"
    mv "$RUN_DIR" "$ASIDE"
  fi
  EPS="[$(seq -s, ${EP_LO[$T]} ${EP_HI[$T]})]"
  CMD=(lerobot-train
    --policy.path="$BASE_CKPT"
    --policy.empty_cameras=1
    --policy.dtype=bfloat16
    --policy.gradient_checkpointing=false
    --policy.push_to_hub=false
    --gradient_accumulation_steps=2
    --policy.optimizer_lr=1e-4
    --policy.scheduler_warmup_steps=$WARMUP
    --policy.scheduler_decay_steps=$STEPS
    --policy.scheduler_decay_lr=1e-5
    --policy.normalization_mapping="$RW_NORM_MAP"
    --peft.method_type=LORA
    --peft.r=$LORA_R
    --peft.lora_alpha=$LORA_ALPHA
    --peft.target_modules="$TARGETS"
    --peft.full_training_modules='[]'
    --dataset.repo_id="$RW_SEQ_ID"
    --dataset.root="$RW_SEQ_ROOT"
    --dataset.episodes="$EPS"
    --rename_map="$RW_RENAME_MAP"
    --output_dir="$RUN_DIR"
    --steps=$STEPS
    --batch_size=16
    --num_workers=8
    --eval_freq=0
    --log_freq=200
    --save_freq=$STEPS
    --wandb.enable=$WANDB
    --wandb.project=vla-memory
    --job_name="${RUN_PREFIX}rw_${RW_TAG}_loraft_r${LORA_R}_t${T}")
  echo "[spec t$T] r$LORA_R/a$LORA_ALPHA LoRA specialist, eps ${EP_LO[$T]}-${EP_HI[$T]}, $STEPS steps ($(date -u))"
  if [ "$DRYRUN" = "1" ]; then printf '  %q' "${CMD[@]}"; echo; continue; fi
  "${CMD[@]}"
  [ -d "$RUN_DIR/checkpoints/$FINAL" ] || { echo "[spec t$T] ERROR: $FINAL missing after training"; exit 1; }
done
echo "RW r${LORA_R} LoRA specialists ($TASK_LIST) COMPLETE at $(date -u)"
