#!/bin/bash
# E47 graduation chain — COMMON BODY (sourced by wrappers setting WARM_RUN / GRAD_TAG).
# For a certified joint router warm-up checkpoint (both towers' routers in ONE ckpt —
# no merge step, unlike the E45 pool chain):
#   stage A: joint A-phase, 10k values-only on libero_90, BOTH towers, routers frozen
#            (train_memory_only + freeze_memory_router; explicit E37 overrides for the
#            flags the warm-up ckpt carries: train_router_only=false, and
#            vlm_route_once=true — the E47 bcast warm-ups save False, but with the
#            router frozen the compact path is numerically identical and saves 6-10GB);
#   stage B: 5-task sequential, C-config (beta4 protection, top_t 1536, 5000 steps/task,
#            value lr 1e-3 -> 1e-4, 20-ep intermediates + 50-ep FINAL AT 50 EPS),
#            comparable to stageB (32.0 final / 35.0 init) and the E45 e4 anchors.
# GATE 2 (central, not automated here): the sequential's t0 block IS the e4 probe
# (same C-config 5k steps) — run the chunk probe on checkpoints/005000 while t1+ train;
# kill the run at chunk >= ~0.12 (anchors 0.153 staged-best / 0.0994 poolB / 0.020 LoRA).
# Per-task checkpoints are kept (save_after_each_task) for the probe battery.
set -eo pipefail
echo "E47 graduation chain [$GRAD_TAG] from $WARM_RUN started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
WARM_CKPT="$ROOT_DIR/outputs/train/$WARM_RUN/checkpoints/last/pretrained_model"
A_RUN=libero_90_pi05_jointA10k_${GRAD_TAG}
A_OUT="$ROOT_DIR/outputs/train/$A_RUN"
A_CKPT="$A_OUT/checkpoints/last/pretrained_model"
# E48 parametrization (defaults = the E47 graduation config, byte-identical commands):
# wrappers may override the sequential's write budget / value LR / micro-batching and
# must then set SEQ_RUN so the run name reflects the actual config.
SEQ_TOP_T=${SEQ_TOP_T:-1536}
SEQ_VALUE_LR=${SEQ_VALUE_LR:-0.001}
SEQ_VALUE_LR_END=${SEQ_VALUE_LR_END:-0.0001}
SEQ_BS=${SEQ_BS:-32}
SEQ_ACCUM=${SEQ_ACCUM:-1}
SEQ_TOP_P=${SEQ_TOP_P:-0}
SEQ_TOP_P_CAP=${SEQ_TOP_P_CAP:-16384}
SEQ_PROTECT_MODE=${SEQ_PROTECT_MODE:-rank}
SEQ_PROTECT_UNORM=${SEQ_PROTECT_UNORM:-peak}
SEQ_RUN=${SEQ_RUN:-libero_10_seq5_jw_${GRAD_TAG}_beta4_topt1536_steps5k}
SEQ_OUT="$ROOT_DIR/outputs/train/$SEQ_RUN"
# E68 hooks (BYTE-IDENTICAL defaults; set only by ablation wrappers):
#   A_STEPS / A_SAVE_FREQ  - A-phase length and periodic-save cadence (save_freq < steps enables
#                            preemption resume via lerobot-train --resume; CLAUDE.md 9.4)
#   A_LADDER               - "bs:accum:ckpt,..." rungs for stage A (default = the E47/E52 ladder)
#   A_FROZEN_ROUTE / SEQ_FROZEN_ROUTE - use_frozen_base_input_features for stage A / stage B
#   A_EXTRA_ARGS           - extra CLI args appended verbatim to stage A (cf. SEQ_EXTRA_ARGS)
#   SEQ_PROTECT            - protect_prior_slots for stage B (E68 A2: TF-IDF-only writes)
A_STEPS=${A_STEPS:-10000}
A_SAVE_FREQ=${A_SAVE_FREQ:-10000}
A_LADDER=${A_LADDER:-32:1:false,16:2:false,16:2:true}
A_FROZEN_ROUTE=${A_FROZEN_ROUTE:-true}
SEQ_FROZEN_ROUTE=${SEQ_FROZEN_ROUTE:-true}
SEQ_PROTECT=${SEQ_PROTECT:-true}
A_LOG_FREQ=${A_LOG_FREQ:-200}
A_ONLY=${A_ONLY:-0}          # 1 = stop after stage A (smokes / profiling)
A_FINAL="$A_OUT/checkpoints/$(printf '%06d' "$A_STEPS")/pretrained_model"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
# The warm-up checkpoint is only needed when stage A actually runs (a seq-only reuse of
# an existing A checkpoint on a box without the warm-up dir is legitimate — E48).
if [ ! -d "$A_FINAL" ] && ! ls -d "$A_OUT"/checkpoints/[0-9]*/pretrained_model/train_config.json >/dev/null 2>&1; then
  [ -d "$WARM_CKPT" ] || { echo "ERROR: warm-up checkpoint missing: $WARM_CKPT (rsync it to this box first)"; exit 1; }
fi

# ---------- stage A: joint A-phase (values both towers, routers frozen) ----------
a_phase () {
  lerobot-train \
    --policy.path="$WARM_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$A_RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_90 \
    --dataset.root="$DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_90 \
    --output_dir="$A_OUT" \
    --save_freq=$A_SAVE_FREQ \
    --steps=$A_STEPS \
    --batch_size=$1 \
    --gradient_accumulation_steps=$2 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=4 \
    --eval_freq=20000 \
    --log_freq=$A_LOG_FREQ \
    --policy.train_router_only=false \
    --policy.train_memory_only=true \
    --policy.freeze_memory_router=true \
    --policy.memory_layer.use_frozen_base_input_features=$A_FROZEN_ROUTE \
    --policy.memory_layer.vlm_route_once=true \
    --policy.memory_layer.router_only_fast=false \
    --policy.optimizer_lr=2.5e-5 \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=40000 \
    --job_name="$A_RUN" \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --wandb.disable_artifact=true \
    --policy.gradient_checkpointing=${3:-false} \
    $A_EXTRA_ARGS
}
# Preemption/crash resume for stage A (only reachable when A_SAVE_FREQ < A_STEPS): the standard
# lerobot-train resume — everything (dataset, flags, output_dir, rung) comes from the saved
# train_config.json; optimizer/scheduler/step from training_state/.
a_phase_resume () {
  lerobot-train --resume=true --config_path="$1"
}
if [ -d "$A_FINAL" ]; then
  echo "[A-phase] final checkpoint exists - skipping."
else
  A_PARTIAL=$(ls -d "$A_OUT"/checkpoints/[0-9]*/pretrained_model/train_config.json 2>/dev/null | sort | tail -1)
  if [ -n "$A_PARTIAL" ]; then
    echo "[A-phase] RESUMING from $A_PARTIAL (periodic save)"
    a_phase_resume "$A_PARTIAL" || { echo "ERROR: A-phase resume failed - NOT wiping $A_OUT; inspect and relaunch"; exit 1; }
  else
    echo "[A-phase] ladder: $A_LADDER (rungs bs:accum:grad_ckpt; a rung that fails before any checkpoint is treated as VRAM)"
    ok=0
    for rung in ${A_LADDER//,/ }; do
      IFS=: read -r rb ra rc <<< "$rung"
      echo "[A-phase] rung: bs=$rb accum=$ra grad_ckpt=$rc"
      if a_phase "$rb" "$ra" "$rc"; then ok=1; break; fi
      if ls -d "$A_OUT"/checkpoints/[0-9]*/pretrained_model >/dev/null 2>&1; then
        echo "[A-phase] rung failed AFTER a periodic checkpoint - not a VRAM failure; aborting (relaunch resumes)."
        exit 1
      fi
      echo "[A-phase] rung failed before any checkpoint (treating as VRAM) - wiping and trying next rung"
      rm -rf "$A_OUT"
    done
    [ "$ok" = 1 ] || { echo "ERROR: all A_LADDER rungs failed"; exit 1; }
  fi
fi
[ -d "$A_FINAL" ] || { echo "ERROR: A-phase finished but final checkpoint missing ($A_FINAL)"; exit 1; }
if [ "$A_ONLY" = 1 ]; then
  echo "[A-phase] A_ONLY=1 - stopping after stage A."
  return 0 2>/dev/null || exit 0
fi

# ---------- stage B: 5-task sequential (C-config; t0 block == the e4 probe) ----------
# train_memory_only + freeze_memory_router + frozen-route ride in the A-ckpt config
# (desired downstream); train_router_only + vlm_route_once are explicitly overridden.
# E53: optional VRAM ladder — set SEQ_LADDER="bs:accum:ckpt,bs:accum:ckpt,..." to try
# rungs in order. A rung that fails BEFORE the first per-task checkpoint (005000) is
# treated as a config/VRAM failure: the run dir is wiped and the next rung tried. A
# failure AFTER 005000 exists is NOT VRAM (peak memory is reached within task 0) and
# aborts the chain loudly. SEQ_LADDER unset => single attempt, byte-identical to E47.
seq_stage () {
  lerobot-sequential-train \
    --policy.path="$SEQ_POLICY_PATH" $SEQ_RESUME_FLAG \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=${3:-false} \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_10 \
    --dataset.root="$SEQ_DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_10 \
    --output_dir="$SEQ_OUT" \
    --steps=200000 \
    --batch_size=$1 \
    --gradient_accumulation_steps=$2 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=20 \
    --eval_final_episodes=50 \
    --log_freq=200 \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --job_name="$SEQ_RUN" \
    --online_task_ids="${SEQ_TASK_IDS:-[0,1,2,3,4]}" \
    --online_steps_per_task=5000 \
    --policy.train_router_only=false \
    --policy.memory_layer.vlm_route_once=true \
    --policy.memory_layer.router_only_fast=false \
    --policy.memory_layer.aggregate_usage=false \
    --policy.memory_layer.use_frozen_base_input_features=$SEQ_FROZEN_ROUTE \
    --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
    --save_after_each_task=true \
    --reinit_optimizer_each_task=true \
    --tfidf_enable=true \
    --tfidf_top_t=$SEQ_TOP_T \
    --tfidf_top_p=$SEQ_TOP_P \
    --tfidf_top_p_cap=$SEQ_TOP_P_CAP \
    --use_online_idf_stats=true \
    --idf_exponent=1 \
    --protect_prior_slots=$SEQ_PROTECT \
    --protect_beta=4 \
    --protect_mode=$SEQ_PROTECT_MODE \
    --protect_u_norm=$SEQ_PROTECT_UNORM \
    --memory_value_lr=$SEQ_VALUE_LR \
    --memory_value_lr_end=$SEQ_VALUE_LR_END \
    --memory_value_scheduler_type=linear \
    $SEQ_EXTRA_ARGS
}
# SEQ_EXTRA_ARGS (E57): optional space-separated extra CLI args appended verbatim to the
# sequential stage (e.g. the value-input-noise flags). Unset => byte-identical.
# Auto-resume (preemption / crash recovery, per the "resume by default" rule): if a previous
# attempt left a COMPLETED task boundary behind, continue from it rather than redoing the whole
# sequential stage. sequential_state.pt carries the cross-task state the model checkpoint does
# not (protection store, online-IDF accumulators, eval histories); without that file the trainer
# refuses to resume rather than silently running later tasks unprotected.
SEQ_POLICY_PATH="$A_CKPT"
SEQ_RESUME_FLAG=""
if [ -f "$SEQ_OUT/checkpoints/last/sequential_state.pt" ] && [ ! -d "$SEQ_OUT/checkpoints/${SEQ_FINAL_CKPT:-025000}" ]; then
  SEQ_POLICY_PATH="$SEQ_OUT/checkpoints/last/pretrained_model"
  SEQ_RESUME_FLAG="--resume_sequential=true"
  echo "[seq5] RESUMING from $(readlink -f "$SEQ_OUT/checkpoints/last" 2>/dev/null)"
fi

if [ -d "$SEQ_OUT/checkpoints/${SEQ_FINAL_CKPT:-025000}" ]; then
  echo "[seq5] final checkpoint exists - skipping."
elif [ -z "$SEQ_LADDER" ]; then
  seq_stage $SEQ_BS $SEQ_ACCUM ${SEQ_GRAD_CKPT:-false}
else
  ok=0
  for rung in ${SEQ_LADDER//,/ }; do
    IFS=: read -r rb ra rc <<< "$rung"
    echo "[seq5] ladder rung: bs=$rb accum=$ra grad_ckpt=$rc"
    if seq_stage "$rb" "$ra" "$rc"; then ok=1; break; fi
    if [ -d "$SEQ_OUT/checkpoints/005000" ]; then
      echo "[seq5] rung failed AFTER task-0 checkpoint - not a VRAM failure; aborting."
      exit 1
    fi
    echo "[seq5] rung failed before 005000 (treating as VRAM) - wiping and trying next rung"
    rm -rf "$SEQ_OUT"
  done
  [ "$ok" = 1 ] || { echo "ERROR: all SEQ_LADDER rungs failed"; exit 1; }
fi
echo "E47 graduation chain [$GRAD_TAG] COMPLETE at $(date)"
