#!/bin/bash
# STAGED PRETRAINING — stage-2 probe rerun with a NONLINEAR QUERY HEAD (research_log
# Entry 34). Single-knob delta from the failed frozen-base probe
# (libero_90_pi05_8_10_12_14_frozenbase_probe10k_c0.05_sep5.0_noloc_rq512):
#   --policy.memory_layer.query_proj_layers=2   (state-path query proj: Linear ->
#   Linear(1024,1024)+SiLU+Linear(1024,2048); +16 router tensors, ~2M params/layer)
#
# Why: on the frozen backbone the routing losses lost their actuator (backbone
# co-adaptation). The depth-1 probe FAILED the audit gate (famIoU 0.390 > 0.28,
# bg 0.188, inverted layer ladder L8 worst 0.538) with in-run sep flatlined at 0.18
# from step 2.5k while aux terms already dominate MSE 10:1 -> representation-limited,
# not weight-limited. A 2-layer head can carve nonlinear task boundaries in the
# frozen features. Gate (same as always): famIoU <= ~0.28 AND core50 >= ~1500 AND
# q_intra <= ~0.93 — read against BOTH the depth-1 frozenbase audit (0.390/6456/0.71)
# and the joint P9 anchor (0.264/2679/0.91).
set -eo pipefail
echo "qproj2 probe started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot
DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
AUDIT_SH="$ROOT_DIR/job_scripts/nebius/libero_90/probes/audit_heldout_routing.sh"
STAGE1_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model"
RUN=libero_90_pi05_8_10_12_14_frozenbase_probe10k_qproj2_c0.05_sep5.0_noloc_rq512
AUDIT_RUN=audit_heldout_frozenbase_qproj2_10k
OUT="$ROOT_DIR/outputs/train/$RUN"
CKPT="$OUT/checkpoints/last/pretrained_model"

export MUJOCO_GL=osmesa
unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
if [ ! -d "$STAGE1_CKPT" ]; then echo "ERROR: stage-1 checkpoint missing"; exit 1; fi

if [ -d "$CKPT" ]; then
  echo "[probe] checkpoint exists - skipping."
else
  lerobot-train \
    --policy.path="$STAGE1_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_90 \
    --dataset.root="$DATASET_ROOT" \
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
    --policy.train_memory_only=true \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=40000 \
    --job_name="$RUN" \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --wandb.disable_artifact=true \
    --policy.gradient_checkpointing=false \
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
    --policy.memory_layer.query_proj_layers=2 \
    --policy.memory_layer.contrastive_method=sample \
    --policy.memory_layer.contrastive_loss_weight=0.05 \
    --policy.memory_layer.contrastive_margin=0.0 \
    --policy.memory_layer.contrastive_query_queue=512 \
    --policy.memory_layer.routing_loss_topk=36 \
    --policy.memory_layer.routing_intra_task_locality_weight=0 \
    --policy.memory_layer.routing_inter_task_separation_weight=5.0 \
    --policy.memory_layer.routing_query_queue=512
fi
if [ ! -d "$CKPT" ]; then echo "ERROR: probe finished but $CKPT missing"; exit 1; fi

echo "[audit] $AUDIT_RUN"
if [ "$(ls $ROOT_DIR/outputs/train/$AUDIT_RUN/memory_by_task/*.json 2>/dev/null | wc -l)" -ge 10 ]; then
  echo "[audit] already complete - skipping."
else
  bash "$AUDIT_SH" "$CKPT" "$AUDIT_RUN" || echo "[audit] AUDIT FAILED (probe checkpoint retained; rerun manually)"
fi
echo "qproj2 probe completed at $(date)"
