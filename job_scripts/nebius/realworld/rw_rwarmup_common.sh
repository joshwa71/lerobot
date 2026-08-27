#!/bin/bash
# Joint both-tower router warm-up + held-out audit — REAL-WORLD duplicate of
# libero_90/staged/joint_rwarmup_common.sh (E46/E47 protocol, byte-parity on every memory /
# routing flag). Sourced by rw_merged6x2_full_chain.sh after rw_env.sh + rw_stage1_base.sh,
# with ARM_TAG / EXP_* / VLM_* / SHARE_GROUPS / ROUTER_FAST / PREPASS / anchor+loss knobs exported.
#
# Protocol: (1) retrain BOTH towers' routers jointly on the RW pretrain split — values pinned
# at zero, aux losses only, router lr 1e-4, 10k compressed schedule, frozen-route + prepass,
# broadcast (vlm_route_once=false) loss semantics (E47); (2) held-out audit = inert sweep of
# the RW SEQ split -> expert + VLM analyses (famIoU family from RW_FAMILY, informational);
# (3) return to the chain for the bg-first gate.
# Deltas vs the LIBERO body: datasets/rename map; no --env.* (the LIBERO body built 90 gym envs
# it never rolled out); --eval_freq=0 and no --eval.*; task-count guards from RW_N_SEQ; the
# sub-span probe is NOT run here (its token-position REGIONS were calibrated on LIBERO's 8-D
# state prompt; recalibrate before using it on the 7-D WidowX prompt).
set -eo pipefail
echo "RW joint router warm-up [$ARM_TAG] (exp n$EXP_N/r$EXP_R/knn$EXP_KNN | vlm n$VLM_N/r$VLM_R/knn$VLM_KNN) started on $(hostname) at $(date)"
SCRIPT_DIR_RW="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_SH="$SCRIPT_DIR_RW/rw_audit_heldout_routing.sh"
RUN=${RUN_PREFIX}realworld_${RW_TAG}_pi05_jointwarm10k_${ARM_TAG}
AUDIT_RUN=${RUN_PREFIX}audit_heldout_rw_${RW_TAG}_jointwarm_${ARM_TAG}_10k
OUT="$ROOT_DIR/outputs/train/$RUN"
CKPT="$OUT/checkpoints/last/pretrained_model"
[ -d "$STAGE1_CKPT" ] || { echo "ERROR: stage-1 checkpoint missing: $STAGE1_CKPT"; exit 1; }

if [ -d "$CKPT" ]; then
  echo "[warmup] checkpoint exists - skipping."
else
  lerobot-train \
    --policy.path="$STAGE1_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$RUN" \
    --policy.push_to_hub=false \
    --policy.normalization_mapping="$RW_NORM_MAP" \
    --dataset.repo_id="$RW_PRETRAIN_ID" \
    --dataset.root="$RW_PRETRAIN_ROOT" \
    --rename_map="$RW_RENAME_MAP" \
    --output_dir="$OUT" \
    --save_freq=10000 \
    --steps=$WARM_STEPS \
    --batch_size=${BATCH_SIZE:-32} \
    --gradient_accumulation_steps=${GRAD_ACCUM:-1} \
    --num_workers=8 \
    --eval_freq=0 \
    --log_freq=200 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.train_router_only=true \
    --policy.optimizer_lr=1e-4 \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=40000 \
    --job_name="$RUN" \
    --wandb.enable=$WANDB \
    --wandb.project=vla-memory \
    --wandb.disable_artifact=true \
    --policy.gradient_checkpointing=false \
    --policy.memory_layers=true \
    --policy.memory_layer.enabled=true \
    --policy.memory_layer.memory_only=false \
    --policy.memory_layer.layers="${EXP_LAYERS:-[8,10,12,14]}" \
    --policy.memory_layer.mem_n_keys=$EXP_N \
    --policy.memory_layer.lora_rank=$EXP_R \
    --policy.memory_layer.mem_knn=$EXP_KNN \
    --policy.memory_layer.routing_loss_topk=$EXP_KNN \
    --policy.memory_layer.vlm_layers="${VLM_LAYERS:-[15,16]}" \
    --policy.memory_layer.vlm_mem_n_keys=$VLM_N \
    --policy.memory_layer.vlm_lora_rank=$VLM_R \
    --policy.memory_layer.vlm_mem_knn=$VLM_KNN \
    --policy.memory_layer.vlm_text_span=200 \
    --policy.memory_layer.vlm_router_pool=anchored \
    --policy.memory_layer.vlm_router_pool_weights='[1.0,0.5]' \
    --policy.memory_layer.vlm_route_once=false \
    --policy.memory_layer.vlm_image_regions=${IMG_REGIONS:-0} \
    --policy.memory_layer.vlm_image_pool_weights="${IMG_POOL_W:-[1.0,0.5]}" \
    --policy.memory_layer.router_only_fast=${ROUTER_FAST:-false} \
    --policy.memory_layer.use_frozen_base_input_features=true \
    --policy.memory_layer.frozen_prepass=${PREPASS:-false} \
    --policy.memory_layer.share_groups="${SHARE_GROUPS:-[]}" \
    --policy.memory_layer.vlm_share_groups="${VLM_SHARE_GROUPS:-[]}" \
    --policy.memory_layer.log_usage=true \
    --policy.memory_layer.aggregate_usage=true \
    --policy.memory_layer.mem_heads=4 \
    --policy.memory_layer.mem_k_dim=512 \
    --policy.memory_layer.value_fixed_lr=0.001 \
    --policy.memory_layer.memory_lr=0.001 \
    --policy.memory_layer.lang_to_query=${LANG_TO_QUERY:-true} \
    --policy.memory_layer.expert_anchor_pool="${EXPERT_ANCHOR:-}" \
    --policy.memory_layer.expert_anchor_weight=${EXPERT_ANCHOR_W:-0.5} \
    --policy.memory_layer.fuse_method=film \
    --policy.memory_layer.embedding_model=all-mpnet-base-v2 \
    --policy.memory_layer.value_type=lora \
    --policy.memory_layer.contrastive_method=sample \
    --policy.memory_layer.contrastive_loss_weight=${CONTRASTIVE_W:-0.05} \
    --policy.memory_layer.contrastive_margin=0.0 \
    --policy.memory_layer.contrastive_query_queue=512 \
    --policy.memory_layer.routing_intra_task_locality_weight=0 \
    --policy.memory_layer.routing_inter_task_separation_weight=${SEP_W:-5.0} \
    --policy.memory_layer.routing_query_queue=512
fi
[ -d "$CKPT" ] || { echo "ERROR: warmup finished but checkpoint missing"; exit 1; }

if [ "$(ls $ROOT_DIR/outputs/train/$AUDIT_RUN/memory_by_task/*.json 2>/dev/null | wc -l)" -ge "$RW_N_SEQ" ]; then
  echo "[audit] already complete ($RW_N_SEQ task JSONs) - skipping."
else
  bash "$AUDIT_SH" "$CKPT" "$AUDIT_RUN" || echo "[audit] AUDIT FAILED (warmup checkpoint retained; rerun manually)"
fi
VBANK=$((VLM_N * VLM_N)); EBANK=$((EXP_N * EXP_N))
VLM_L_CSV=$(echo "${VLM_LAYERS:-[15,16]}" | tr -d "[] ")
EXP_L_CSV=$(echo "${EXP_LAYERS:-[8,10,12,14]}" | tr -d "[] ")
AUDIT_FAMILY="$RW_FAMILY" python scripts/vla_analysis/realworld/vlm_audit_analysis_rw.py "$AUDIT_RUN" $VLM_L_CSV $VBANK vlm || true
AUDIT_FAMILY="$RW_FAMILY" python scripts/vla_analysis/realworld/vlm_audit_analysis_rw.py "$AUDIT_RUN" $EXP_L_CSV $EBANK expert || true
echo "RW joint router warm-up [$ARM_TAG] + audit COMPLETE at $(date)"
