#!/bin/bash
# Layer-wise LoRA rank probes (research_log Entry 30 append) — test where to spend a
# per-slot capacity increase under the frozen-router continual setup, WITHOUT growing
# the footprint to rank-4-on-all-4 (4.8B, too large).
#
#   P1  ranks [2,2,2,4] on layers [8,10,12,14]  -> keep all 4, boost the action-proximal L14 (+25%, 3.0B values)
#   P2  ranks [4,4,4]   on layers [8,10,12]     -> drop L14, rank-4 on the rest (+50%, 3.6B values)
#
# Everything else = C's prior recipe (the sep5 40k pretrain: contrastive 0.05 / sep 5.0 /
# locality OFF / rq512 / knn 36), truncated to 10k (compressed cosine, warmup 4000->1000,
# decay 40000->10000; identical to every prior probe -> audits are comparable, incl. the
# existing rank-2 sep5/P9 audit as the [2,2,2,2] baseline).
# Only deltas vs that recipe: lora_rank scalar -> layer_ranks list (+ layers for P2).
#
# ONE H200 can't hold two full-backbone pretrains, so P1 then P2 sequentially (~11h each).
# Robust: one failure does NOT abort; skip-if-final-ckpt-exists. Poll first steps + the
# "Per-layer LoRA ranks for EXPERT" line to confirm the config parsed and applied.

set -uo pipefail
echo "LAYERRANK PROBES started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot
PI05_BASE="/home/josh/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30"
PRETRAIN_DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
LOGDIR="$ROOT_DIR/outputs/probe_logs"; mkdir -p "$LOGDIR"

export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
if [ ! -d "$PI05_BASE" ]; then echo "ERROR: pi05_base missing at $PI05_BASE"; exit 1; fi

# run_probe <run_name> <layers> <layer_ranks>
run_probe () {
  local RUN="$1" LAYERS="$2" RANKS="$3"
  local OUT="$ROOT_DIR/outputs/train/$RUN"
  local LOG="$LOGDIR/${RUN}.log"
  echo "=============================================================="
  echo "[$(date)] PROBE $RUN"
  echo "  layers=$LAYERS  layer_ranks=$RANKS"
  echo "  out: $OUT"; echo "  log: $LOG"
  if [ -d "$OUT/checkpoints/010000" ]; then
    echo "  -> final checkpoint 010000 already exists; SKIP"; return 0
  fi
  lerobot-train \
    --policy.path="$PI05_BASE" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_90 \
    --dataset.root="$PRETRAIN_DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_90 \
    --output_dir="$OUT" \
    --save_freq=10000 \
    --steps=10000 \
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=4 \
    --eval_freq=20000 \
    --log_freq=200 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=40000 \
    --job_name="$RUN" \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --wandb.disable_artifact=true \
    --policy.gradient_checkpointing=true \
    --policy.memory_layers=true \
    --policy.memory_layer.memory_only=false \
    --policy.memory_layer.layers="$LAYERS" \
    --policy.memory_layer.log_usage=true \
    --policy.memory_layer.enabled=true \
    --policy.memory_layer.aggregate_usage=true \
    --policy.memory_layer.mem_n_keys=384 \
    --policy.memory_layer.mem_heads=4 \
    --policy.memory_layer.mem_knn=36 \
    --policy.memory_layer.mem_k_dim=512 \
    --policy.memory_layer.value_fixed_lr=0.001 \
    --policy.memory_layer.memory_lr=0.001 \
    --policy.memory_layer.lang_to_query=true \
    --policy.memory_layer.fuse_method=film \
    --policy.memory_layer.embedding_model=all-mpnet-base-v2 \
    --policy.memory_layer.value_type=lora \
    --policy.memory_layer.layer_ranks="$RANKS" \
    --policy.memory_layer.contrastive_method=sample \
    --policy.memory_layer.contrastive_loss_weight=0.05 \
    --policy.memory_layer.contrastive_margin=0.0 \
    --policy.memory_layer.contrastive_query_queue=512 \
    --policy.memory_layer.routing_loss_topk=36 \
    --policy.memory_layer.routing_intra_task_locality_weight=0 \
    --policy.memory_layer.routing_inter_task_separation_weight=5.0 \
    --policy.memory_layer.routing_query_queue=512 \
    > "$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "  -> DONE ok"; else echo "  -> FAILED rc=$rc (continuing)"; fi
}

# P1 first (keep-all-4, boost L14; lower footprint), then P2 (drop L14, 3x rank4).
run_probe "libero_90_pi05_8_10_12_14_probe10k_c0.05_sep5.0_noloc_rq512_ranks_2224" "[8,10,12,14]" "[2,2,2,4]"
run_probe "libero_90_pi05_8_10_12_probe10k_c0.05_sep5.0_noloc_rq512_ranks_444"      "[8,10,12]"    "[4,4,4]"

echo "LAYERRANK PROBES completed at $(date)"
