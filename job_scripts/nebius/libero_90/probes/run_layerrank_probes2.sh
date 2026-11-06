#!/bin/bash
# [2,2,4,4] A/B probes + interleaved routing audits (research_log Entry 32).
# Rank mental model: capacity ~ footprint x per-slot rank; interference-damage ~ overlap x
# per-slot destructiveness — rank multiplies BOTH sides, so the rank-2-tuned contrastive
# ceiling (c=0.05) need not carry to rank 4. Two cells, single-knob delta, sep=5.0 held:
#
#   A  ranks [2,2,4,4], c=0.05  -> arch question: does r4-at-L12 perturb L14 routing from below?
#   B  ranks [2,2,4,4], c=0.1   -> contrastive re-look: rank-4 makes compaction affordable;
#                                  does it convert to lower famIoU via sep's translation?
#
# Sequence: run A -> audit A -> run B -> audit B  (~11h + 35min each; ~23.5h total).
# All else = C's prior recipe (sep5 noloc rq512 knn36), compressed 10k schedule (comparable
# to every prior probe audit). Gate (rank-adjusted): famIoU<=~0.28 AND query_intra<=~0.93
# AND r4-layer core50>=~1300 (rank-units >= baseline) AND L8/L10 core50 >= ~50% of P9's.
# Robust: one failure does not abort; skip-if-exists on both runs and audits.

set -uo pipefail
echo "LAYERRANK PROBES 2 started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot
PI05_BASE="/home/josh/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30"
PRETRAIN_DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
AUDIT_SH="$ROOT_DIR/job_scripts/nebius/libero_90/probes/audit_heldout_routing.sh"
LOGDIR="$ROOT_DIR/outputs/probe_logs"; mkdir -p "$LOGDIR"

export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
if [ ! -d "$PI05_BASE" ]; then echo "ERROR: pi05_base missing"; exit 1; fi

# run_probe <run_name> <contrastive_weight>
run_probe () {
  local RUN="$1" CONTRASTIVE="$2"
  local OUT="$ROOT_DIR/outputs/train/$RUN"
  local LOG="$LOGDIR/${RUN}.log"
  echo "=============================================================="
  echo "[$(date)] PROBE $RUN  (layers=[8,10,12,14] ranks=[2,2,4,4] c=$CONTRASTIVE sep=5.0)"
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
    --policy.memory_layer.layers="[8,10,12,14]" \
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
    --policy.memory_layer.layer_ranks="[2,2,4,4]" \
    --policy.memory_layer.contrastive_method=sample \
    --policy.memory_layer.contrastive_loss_weight="$CONTRASTIVE" \
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

# run_audit <run_name> <audit_name>
run_audit () {
  local RUN="$1" AUD="$2"
  local CKPT="$ROOT_DIR/outputs/train/$RUN/checkpoints/010000/pretrained_model"
  local LOG="$LOGDIR/${AUD}.log"
  echo "=============================================================="
  echo "[$(date)] AUDIT $AUD  (ckpt: $RUN)"
  if [ ! -d "$CKPT" ]; then echo "  -> checkpoint missing; SKIP audit"; return 1; fi
  if [ "$(ls $ROOT_DIR/outputs/train/$AUD/memory_by_task/*.json 2>/dev/null | wc -l)" -ge 10 ]; then
    echo "  -> audit already complete (10/10 JSONs); SKIP"; return 0
  fi
  bash "$AUDIT_SH" "$CKPT" "$AUD" > "$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "  -> AUDIT ok"; else echo "  -> AUDIT FAILED rc=$rc (continuing)"; fi
}

RUN_A=libero_90_pi05_8_10_12_14_probe10k_c0.05_sep5.0_noloc_rq512_ranks_2244
RUN_B=libero_90_pi05_8_10_12_14_probe10k_c0.1_sep5.0_noloc_rq512_ranks_2244

run_probe "$RUN_A" 0.05
run_audit "$RUN_A" audit_heldout_ranks2244_c005_10k
run_probe "$RUN_B" 0.1
run_audit "$RUN_B" audit_heldout_ranks2244_c01_10k

echo "LAYERRANK PROBES 2 completed at $(date)"
