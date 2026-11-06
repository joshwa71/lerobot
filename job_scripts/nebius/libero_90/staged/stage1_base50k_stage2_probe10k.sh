#!/bin/bash
# STAGED PRETRAINING protocol (research_log: staged-competence-placement idea, 8 Jul) —
# move ALL core competence into the frozen backbone, make memory a pure residual/adaptation
# substrate. Stage 1: finetune base pi05 (NO memory) on libero_90. Stage 2: attach memory
# and train ONLY the memory modules (values+keys+query/FiLM+gate) against the frozen
# backbone (--policy.train_memory_only=true, new flag) with the sep5 routing recipe.
#
# THIS SCRIPT = stage 1 (50k) -> stage-1 zero-shot floor table (libero_10 @ 50 eps, in-run
# eval at 50k) -> stage 2 PROBE (10k compressed schedule, like every prior probe) ->
# held-out routing audit. Stage 2 full 40k + stage 3 sequential live in
# stage2_full40k_stage3_sequential.sh (ready, launch after reading this audit).
#
# Audit gates (vs the joint-pretrain anchors, same instrument): famIoU <= ~0.28 AND
# core50 >= ~1500 AND query_intra <= ~0.93. The NEW question this probe answers: does the
# sep5 routing pocket reproduce when queries are defined on a FROZEN backbone?
#
# Gradient checkpointing: stage 1 = TRUE (measured requirement: plain pi05 full-backbone
# bs32 OOMs without it, 138.69GiB on step 0, tested 29 Jun — see baselines script header).
# Stage 2 = FALSE (frozen backbone => no backbone grads/optimizer states; rank-2
# values-only no-ckpt at bs32 is the measured sequential precedent). Fallback if stage 2
# OOMs: batch_size=16 + gradient_accumulation_steps=2.

set -eo pipefail
echo "STAGED stage1+probe started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot
PI05_BASE="/home/josh/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30"
DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
AUDIT_SH="$ROOT_DIR/job_scripts/nebius/libero_90/probes/audit_heldout_routing.sh"

STAGE1_RUN=libero_90_pi05_base_nomem_50k
PROBE_RUN=libero_90_pi05_8_10_12_14_frozenbase_probe10k_c0.05_sep5.0_noloc_rq512
AUDIT_RUN=audit_heldout_frozenbase_10k

STAGE1_OUT="$ROOT_DIR/outputs/train/$STAGE1_RUN"
PROBE_OUT="$ROOT_DIR/outputs/train/$PROBE_RUN"
STAGE1_CKPT="$STAGE1_OUT/checkpoints/last/pretrained_model"
PROBE_CKPT="$PROBE_OUT/checkpoints/last/pretrained_model"

export MUJOCO_GL=osmesa
unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
cd "$ROOT_DIR"

###############################################################################
# Stage 1 — base pi05, NO memory, libero_90, 50k. Eval = libero_10 @ 50 eps at
# 50k only => the ZERO-SHOT FLOOR TABLE (what each held-out task falls back to
# when memory is corrupted/absent). Same base args as the E31 baseline (B1)
# except dataset = libero_90 only (libero_10 stays held out).
###############################################################################
echo "=============================================================="
echo "[stage 1] BASE FINETUNE libero_90, no memory, 50k -> $STAGE1_RUN"
echo "=============================================================="
if [ -d "$STAGE1_OUT/checkpoints/050000" ]; then
  echo "[stage 1] final checkpoint exists - skipping."
else
  lerobot-train \
    --policy.path="$PI05_BASE" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$STAGE1_RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_90 \
    --dataset.root="$DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_10 \
    --output_dir="$STAGE1_OUT" \
    --save_freq=50000 \
    --steps=50000 \
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=50 \
    --eval_freq=50000 \
    --log_freq=200 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=50000 \
    --job_name="$STAGE1_RUN" \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --wandb.disable_artifact=true \
    --policy.gradient_checkpointing=true
fi
if [ ! -d "$STAGE1_CKPT" ]; then
  echo "ERROR: stage 1 finished but $STAGE1_CKPT does not exist"; exit 1
fi

###############################################################################
# Stage 2 PROBE — attach memory [8,10,12,14] r2, FREEZE the backbone
# (train_memory_only), sep5 recipe verbatim (c0.05 sample q512 / sep5.0 rq512 /
# noloc / knn36). 10k compressed schedule = comparable to every prior probe.
# No grad ckpt (frozen backbone).
###############################################################################
echo "=============================================================="
echo "[stage 2 probe] FROZEN-BASE memory-only 10k -> $PROBE_RUN"
echo "  from: $STAGE1_CKPT"
echo "=============================================================="
if [ -d "$PROBE_CKPT" ]; then
  echo "[stage 2 probe] checkpoint exists - skipping."
else
  lerobot-train \
    --policy.path="$STAGE1_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$PROBE_RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_90 \
    --dataset.root="$DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_90 \
    --output_dir="$PROBE_OUT" \
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
    --job_name="$PROBE_RUN" \
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
    --policy.memory_layer.contrastive_method=sample \
    --policy.memory_layer.contrastive_loss_weight=0.05 \
    --policy.memory_layer.contrastive_margin=0.0 \
    --policy.memory_layer.contrastive_query_queue=512 \
    --policy.memory_layer.routing_loss_topk=36 \
    --policy.memory_layer.routing_intra_task_locality_weight=0 \
    --policy.memory_layer.routing_inter_task_separation_weight=5.0 \
    --policy.memory_layer.routing_query_queue=512
fi
if [ ! -d "$PROBE_CKPT" ]; then
  echo "ERROR: probe finished but $PROBE_CKPT does not exist"; exit 1
fi

###############################################################################
# Stage 2.5 — held-out routing audit on the probe checkpoint
###############################################################################
echo "=============================================================="
echo "[audit] $AUDIT_RUN"
echo "=============================================================="
if [ "$(ls $ROOT_DIR/outputs/train/$AUDIT_RUN/memory_by_task/*.json 2>/dev/null | wc -l)" -ge 10 ]; then
  echo "[audit] already complete - skipping."
else
  bash "$AUDIT_SH" "$PROBE_CKPT" "$AUDIT_RUN" || echo "[audit] AUDIT FAILED (probe checkpoint retained; rerun audit manually)"
fi

echo "STAGED stage1+probe completed at $(date)"
