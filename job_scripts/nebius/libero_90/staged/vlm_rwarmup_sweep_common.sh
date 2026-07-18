#!/bin/bash
# E44 VLM-router warm-up sweep — COMMON BODY (sourced by the 3 arm scripts, which set
# ARM_TAG / C_WEIGHT / SEP_WEIGHT before sourcing this file).
#
# VLM text-span memory (the E44 build): modules on paligemma LM layers [15,16], attached to
# the last-200 prefix positions (instruction + state-as-text), bank n_keys=256 (65,536
# slots), r=2 on the 2048-dim hidden, knn=16 (routing_loss_topk aligned =16, E24 rule).
# NO expert memory attached (the prefix never attends to the suffix, so VLM router training
# is fully independent of the expert side). train_router_only: values pinned at init =>
# pure contrastive+sep signal at router lr 1e-4 (the E37 protocol).
#
# Regime hazards this sweep brackets (E44 discussion): (1) the text-field geometry is
# already OPEN (inter-task cos 0.73-0.86 vs the 0.87-0.93 the expert recipe was tuned on)
# => sep may need less force; (2) the instruction component is CONSTANT within task =>
# contrastive intra-pull can snap into the E21 per-task-bias collapse; the state tokens
# carry the within-task variation the router must keep.
#
# GATES (audit on held-out libero_10, scripts/vla_analysis/vlm_audit_analysis.py;
# re-anchored for the 65,536 bank = 0.44x expert): PASS = famIoU <= ~0.25 AND per-task
# core50 >= ~650 AND effnum >= ~500. COLLAPSE tripwire: effnum <= ~150 (per-task-bias
# signature — kill the arm regardless of IoU). Stretch: famIoU <= 0.15.
set -eo pipefail
echo "E44 VLM router warm-up [$ARM_TAG] (c=$C_WEIGHT sep=$SEP_WEIGHT) started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
AUDIT_SH="$ROOT_DIR/job_scripts/nebius/libero_90/probes/audit_heldout_routing.sh"
STAGE1_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model"
RUN=libero_90_pi05_vlm1516_txtspan_rwarmup10k_padfix_${ARM_TAG}_n256_r2_knn16_rq512
AUDIT_RUN=audit_heldout_vlmrwarmup_padfix_${ARM_TAG}_10k
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
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
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
    --policy.memory_layer.layers='[]' \
    --policy.memory_layer.vlm_layers='[15,16]' \
    --policy.memory_layer.vlm_mem_n_keys=256 \
    --policy.memory_layer.vlm_lora_rank=2 \
    --policy.memory_layer.vlm_mem_knn=16 \
    --policy.memory_layer.vlm_text_span=200 \
    --policy.memory_layer.log_usage=true \
    --policy.memory_layer.aggregate_usage=true \
    --policy.memory_layer.mem_heads=4 \
    --policy.memory_layer.mem_k_dim=512 \
    --policy.memory_layer.value_fixed_lr=0.001 \
    --policy.memory_layer.memory_lr=0.001 \
    --policy.memory_layer.lang_to_query=false \
    --policy.memory_layer.value_type=lora \
    --policy.memory_layer.contrastive_method=sample \
    --policy.memory_layer.contrastive_loss_weight=$C_WEIGHT \
    --policy.memory_layer.contrastive_margin=0.0 \
    --policy.memory_layer.contrastive_query_queue=512 \
    --policy.memory_layer.routing_loss_topk=16 \
    --policy.memory_layer.routing_intra_task_locality_weight=0 \
    --policy.memory_layer.routing_inter_task_separation_weight=$SEP_WEIGHT \
    --policy.memory_layer.routing_query_queue=512
fi
[ -d "$CKPT" ] || { echo "ERROR: warmup finished but checkpoint missing"; exit 1; }

echo "[audit] $AUDIT_RUN"
if [ "$(ls $ROOT_DIR/outputs/train/$AUDIT_RUN/memory_by_task/*.json 2>/dev/null | wc -l)" -ge 10 ]; then
  echo "[audit] already complete - skipping."
else
  bash "$AUDIT_SH" "$CKPT" "$AUDIT_RUN" || echo "[audit] AUDIT FAILED (warmup checkpoint retained; rerun manually)"
fi
python scripts/vla_analysis/vlm_audit_analysis.py "$AUDIT_RUN" 15,16 65536 || true
echo "E44 VLM router warm-up [$ARM_TAG] completed at $(date)"
