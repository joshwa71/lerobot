#!/bin/bash
# Autonomous batch (research_log Entry 29): protection + plasticity levers on the sep5 prior.
# All sequential-only — reuse the EXISTING sep5 40k pretrain (no new pretrain). Single-knob
# deltas from the standing baseline (protect beta=4, the +6.5pp Entry-28 win, 40.5% @20ep).
# Runs in order C -> B -> D -> A, ~18-19h each, ~3 days total, 20 eval eps for comparability.
#
#   C  steps5k          : online_steps_per_task 3000 -> 5000        (plasticity, safe)
#   B  lr2x             : memory_value_lr 1e-3/1e-4 -> 2e-3/2e-4     (plasticity; watch 2e-3 peak instability)
#   D  lr2x_steps5k     : both                                       (plasticity ceiling / additivity)
#   A  beta8            : protect_beta 4 -> 8                         (protection curve)
#
# Baseline already in hand: ..._top_t_1536_protect_beta4 (40.5% @20ep).
# Each run is a clean single-knob delta; all 20-ep eval. Robust: one failure does NOT abort the batch.

set -uo pipefail
echo "BATCH started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
PRETRAIN_RUN=libero_90_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k
PRETRAIN_CHECKPOINT="$ROOT_DIR/outputs/train/$PRETRAIN_RUN/checkpoints/last/pretrained_model"
BASE=libero_10_sequential_pi05_8_10_12_14_film_lora_2_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536
LOGDIR="$ROOT_DIR/outputs/batch_logs"; mkdir -p "$LOGDIR"

export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"

if [ ! -d "$PRETRAIN_CHECKPOINT" ]; then echo "ERROR: sep5 prior missing at $PRETRAIN_CHECKPOINT"; exit 1; fi

# run_seq <suffix> <beta> <steps_per_task> <lr> <lr_end>
run_seq () {
  local SUFFIX="$1" BETA="$2" STEPS="$3" LR="$4" LR_END="$5"
  local RUN="${BASE}_${SUFFIX}"
  local OUT="$ROOT_DIR/outputs/train/$RUN"
  local FINAL=$(printf "%06d" $((STEPS*10)))
  local LOG="$LOGDIR/${RUN}.log"
  echo "=============================================================="
  echo "[$(date)] RUN $SUFFIX  beta=$BETA steps/task=$STEPS lr=$LR->$LR_END"
  echo "  out: $OUT"; echo "  log: $LOG"
  if [ -d "$OUT/checkpoints/$FINAL" ]; then
    echo "  -> final checkpoint $FINAL already exists; SKIP"; return 0
  fi
  lerobot-sequential-train \
    --policy.path="$PRETRAIN_CHECKPOINT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_10 \
    --dataset.root="$SEQ_DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_10 \
    --output_dir="$OUT" \
    --steps=200000 \
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=20 \
    --log_freq=200 \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --job_name="$RUN" \
    --online_task_ids='[0,1,2,3,4,5,6,7,8,9]' \
    --online_steps_per_task="$STEPS" \
    --policy.memory_layer.aggregate_usage=false \
    --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
    --save_after_each_task=true \
    --reinit_optimizer_each_task=true \
    --tfidf_enable=true \
    --tfidf_top_t=1536 \
    --use_online_idf_stats=true \
    --idf_exponent=1 \
    --protect_prior_slots=true \
    --protect_beta="$BETA" \
    --memory_value_lr="$LR" \
    --memory_value_lr_end="$LR_END" \
    --memory_value_scheduler_type=linear \
    > "$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "  -> DONE ok"; else echo "  -> FAILED rc=$rc (continuing batch)"; fi
}

run_seq "protect_beta4_steps5k"      4 5000 0.001 0.0001   # C
run_seq "protect_beta4_lr2x"         4 3000 0.002 0.0002   # B
run_seq "protect_beta4_lr2x_steps5k" 4 5000 0.002 0.0002   # D
run_seq "protect_beta8"              8 3000 0.001 0.0001   # A

echo "BATCH completed at $(date)"
