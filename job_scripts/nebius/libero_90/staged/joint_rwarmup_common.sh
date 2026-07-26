#!/bin/bash
# E46 joint both-tower router warm-up — COMMON BODY (sourced by arm wrappers setting
# ARM_TAG / EXP_N / EXP_R / EXP_KNN / VLM_N / VLM_R / VLM_KNN).
#
# All E46/E47 arms share ONE protocol so the certificates are like-for-like:
# (1) retrain BOTH towers' routers jointly (values pinned at zero, aux losses only,
#     router lr 1e-4, 10k compressed schedule) with FROZEN-ROUTE ON — the routers
#     train on exactly the memory-free features they deploy on (no value_proj-bias
#     residual gap), and the routing-loss candidate pool aligns to each tower's knn
#     (expert = routing_loss_topk below; VLM auto-derived = vlm_mem_knn in code);
# (2) held-out audit (inert sweep of libero_10) -> expert + VLM analyses + the
#     region-split sub-span probe;
# (3) STOP. A-phases run only on arms whose routers certify (no point filling
#     values on a useless router).
#
# E47: warm-ups run vlm_route_once=false — the legacy BROADCAST loss semantics, in
# which the shared state-region key enters the routing/contrastive losses and queues
# once per served position (~35x). The E46 arms ran the route-once DEDUPLICATED
# losses and the palette's spreading force collapsed (palette famIoU 0.08 -> 0.19-0.24,
# effnum -> ~2 query-draws): dedup under-weights the palette relative to its
# deployment read mass. Downstream stages (A-phase / sequential / inference) keep
# vlm_route_once=true — with the router frozen the two paths are numerically
# interchangeable and the compact path saves ~6-10GB VRAM.
# OOM fallback: BATCH_SIZE=16 GRAD_ACCUM=2 (effective 32; NB accumulation shrinks the
# in-batch contrastive pool per the E11 caveat — queues cover it, but note it in the log).
#
# Reads at the audit: expert side expect the bank-scaling law (n384->n256 held
# famIoU at exactly 0.145 with core50 scaling ~1.7x per 2.25x bank) — at n128
# core50 ~400-800 with famIoU ~0.145 = law holds; famIoU up at scaled cores = the
# fixed 144-slot per-query draw floor is binding. VLM side vs last night's arm B
# (0.149/0.147); pooled-audit famIoU is palette-weighted.
set -eo pipefail
echo "E46 joint router warm-up [$ARM_TAG] (exp n$EXP_N/r$EXP_R/knn$EXP_KNN | vlm n$VLM_N/r$VLM_R/knn$VLM_KNN) started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
AUDIT_SH="$ROOT_DIR/job_scripts/nebius/libero_90/probes/audit_heldout_routing.sh"
STAGE1_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model"
RUN=libero_90_pi05_jointwarm10k_${ARM_TAG}
AUDIT_RUN=audit_heldout_jointwarm_${ARM_TAG}_10k
OUT="$ROOT_DIR/outputs/train/$RUN"
CKPT="$OUT/checkpoints/last/pretrained_model"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$STAGE1_CKPT" ] || { echo "ERROR: stage-1 checkpoint missing"; exit 1; }

if [ -d "$CKPT" ]; then
  echo "[warmup] checkpoint exists - skipping."
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
    --batch_size=${BATCH_SIZE:-32} \
    --gradient_accumulation_steps=${GRAD_ACCUM:-1} \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=4 \
    --eval_freq=20000 \
    --log_freq=200 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.train_router_only=true \
    --policy.optimizer_lr=1e-4 \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=40000 \
    --job_name="$RUN" \
    --policy.push_to_hub=false \
    --wandb.enable=true \
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

if [ "$(ls $ROOT_DIR/outputs/train/$AUDIT_RUN/memory_by_task/*.json 2>/dev/null | wc -l)" -ge 10 ]; then
  echo "[audit] already complete - skipping."
else
  bash "$AUDIT_SH" "$CKPT" "$AUDIT_RUN" || echo "[audit] AUDIT FAILED (warmup checkpoint retained; rerun manually)"
fi
VBANK=$((VLM_N * VLM_N)); EBANK=$((EXP_N * EXP_N))
VLM_L_CSV=$(echo "${VLM_LAYERS:-[15,16]}" | tr -d "[] ")
python scripts/vla_analysis/vlm_audit_analysis.py "$AUDIT_RUN" $VLM_L_CSV $VBANK vlm || true
EXP_L_CSV=$(echo "${EXP_LAYERS:-[8,10,12,14]}" | tr -d "[] ")
python scripts/vla_analysis/vlm_audit_analysis.py "$AUDIT_RUN" $EXP_L_CSV $EBANK expert || true
mkdir -p outputs/analysis/e46
ARM="$ARM_TAG" OUT="$ROOT_DIR/outputs/analysis/e46/subspan_${ARM_TAG}.json" \
python scripts/vla_analysis/probe_subspan.py \
  --policy.path="$CKPT" \
  --policy.empty_cameras=1 --policy.dtype=bfloat16 \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --dataset.repo_id=libero_10 --dataset.root="$SEQ_DATASET_ROOT" \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
  --output_dir="$ROOT_DIR/outputs/train/_subspan_tmp_${ARM_TAG}" \
  --steps=1 --batch_size=8 --wandb.enable=false --job_name=subspan_${ARM_TAG} \
  --online_task_ids='[0,1,2,3,4,5,6,7,8,9]' --online_steps_per_task=1 --save_checkpoint=false \
  || echo "[subspan] probe failed (non-fatal)"
echo "E46 joint router warm-up [$ARM_TAG] COMPLETE (stopped after audit) at $(date)"
