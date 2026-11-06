#!/bin/bash
# STAGED PRETRAINING — stage 2 FULL + stage 3. Launch ONLY after the stage-2 probe's
# held-out audit (audit_heldout_frozenbase_10k) clears the gates (famIoU <= ~0.28 AND
# core50 >= ~1500 AND query_intra <= ~0.93); see stage1_base50k_stage2_probe10k.sh.
#
# Stage 2 full: memory-only training (frozen stage-1 backbone, train_memory_only) on
# libero_90, 40k CLEAN schedule (warmup 4000 / decay 40000 honored), sep5 recipe.
# Held-in eval libero_90 @ 4 eps at 20k/40k = "what does memory add ON TOP of the
# frozen backbone" (joint-pretrain anchors for context: control 76.4/81.1, sep5
# 68.1/78.9, [2,2,4,4] 64.2/81.9 — NOT directly comparable: there the backbone
# co-trained; here the delta over stage-1's own held-in is the meaningful number).
# Stage 3: C's sequential config verbatim (beta4 protect + 5000 steps/task +
# top_t 1536, 20 eval eps) => directly comparable to C (44.5) and r2244 (46.5).
#
# Gradient checkpointing FALSE in both stages: backbone frozen throughout (values-only
# no-ckpt at bs32/r2 is the measured sequential precedent). Fallback if OOM:
# batch_size=16 + gradient_accumulation_steps=2.
# Success criteria (pre-registered): no task ends below its stage-1 zero-shot floor
# (stage-1 in-run eval = the floor table), and final >= 50.

set -eo pipefail
echo "STAGED stage2full+stage3 started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot
PRETRAIN_DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
AUDIT_SH="$ROOT_DIR/job_scripts/nebius/libero_90/probes/audit_heldout_routing.sh"

STAGE1_RUN=libero_90_pi05_base_nomem_50k
STAGE2_RUN=libero_90_pi05_8_10_12_14_film_lora_2_frozenbase_c0.05_sep_5.0_noloc_knn_36_rq512_40k
SEQ_RUN=libero_10_sequential_pi05_8_10_12_14_film_lora_2_frozenbase_c0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536_protect_beta4_steps5k
AUDIT_RUN=audit_heldout_frozenbase_40k

STAGE1_CKPT="$ROOT_DIR/outputs/train/$STAGE1_RUN/checkpoints/last/pretrained_model"
STAGE2_OUT="$ROOT_DIR/outputs/train/$STAGE2_RUN"
STAGE2_CKPT="$STAGE2_OUT/checkpoints/last/pretrained_model"
SEQ_OUT="$ROOT_DIR/outputs/train/$SEQ_RUN"

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

if [ ! -d "$STAGE1_CKPT" ]; then echo "ERROR: stage-1 checkpoint missing: $STAGE1_CKPT"; exit 1; fi

###############################################################################
# Stage 2 FULL — frozen-base memory-only 40k, clean schedule
###############################################################################
echo "=============================================================="
echo "[stage 2 full] FROZEN-BASE memory-only 40k -> $STAGE2_RUN"
echo "=============================================================="
if [ -d "$STAGE2_CKPT" ]; then
  echo "[stage 2] checkpoint exists - skipping."
else
  lerobot-train \
    --policy.path="$STAGE1_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$STAGE2_RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_90 \
    --dataset.root="$PRETRAIN_DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_90 \
    --output_dir="$STAGE2_OUT" \
    --save_freq=20000 \
    --steps=40000 \
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
    --job_name="$STAGE2_RUN" \
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
if [ ! -d "$STAGE2_CKPT" ]; then
  echo "ERROR: stage 2 finished but $STAGE2_CKPT does not exist"; exit 1
fi

###############################################################################
# Stage 2.5 — held-out routing audit on the 40k checkpoint (informational)
###############################################################################
echo "=============================================================="
echo "[audit] $AUDIT_RUN"
echo "=============================================================="
if [ "$(ls $ROOT_DIR/outputs/train/$AUDIT_RUN/memory_by_task/*.json 2>/dev/null | wc -l)" -ge 10 ]; then
  echo "[audit] already complete - skipping."
else
  bash "$AUDIT_SH" "$STAGE2_CKPT" "$AUDIT_RUN" || echo "[audit] AUDIT FAILED (continuing to sequential - audit is informational)"
fi

###############################################################################
# Stage 3 — sequential libero_10, C's config (beta4 + 5k steps + top_t 1536).
# Backbone AND router frozen (sequential trainer trains values only). r2 values
# => no grad ckpt (measured precedent). Keep per-task checkpoints (floor-vs-final
# analysis reads them).
###############################################################################
echo "=============================================================="
echo "[stage 3] SEQUENTIAL libero_10 -> $SEQ_RUN"
echo "  from: $STAGE2_CKPT"
echo "=============================================================="
if [ -d "$SEQ_OUT/checkpoints/050000" ]; then
  echo "[stage 3] final checkpoint exists - skipping."
else
  lerobot-sequential-train \
    --policy.path="$STAGE2_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=false \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_10 \
    --dataset.root="$SEQ_DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_10 \
    --output_dir="$SEQ_OUT" \
    --steps=200000 \
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=20 \
    --log_freq=200 \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --job_name="$SEQ_RUN" \
    --online_task_ids='[0,1,2,3,4,5,6,7,8,9]' \
    --online_steps_per_task=5000 \
    --policy.memory_layer.aggregate_usage=false \
    --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}' \
    --save_after_each_task=true \
    --reinit_optimizer_each_task=true \
    --tfidf_enable=true \
    --tfidf_top_t=1536 \
    --use_online_idf_stats=true \
    --idf_exponent=1 \
    --protect_prior_slots=true \
    --protect_beta=4 \
    --memory_value_lr=0.001 \
    --memory_value_lr_end=0.0001 \
    --memory_value_scheduler_type=linear
fi

echo "STAGED stage2full+stage3 completed at $(date)"
