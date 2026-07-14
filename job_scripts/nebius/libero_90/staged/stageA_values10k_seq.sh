#!/bin/bash
# STAGED PROTOCOL final chain (research_log Entry 37): A-PHASE -> re-audit -> SEQUENTIAL.
# From the CERTIFIED warmed router (rwarmup 10k: held-out famIoU 0.145 / core50 2955 /
# q_intra 0.883 — all gates cleared, best prior in project history).
#
# A phase = values-only content fill on libero_90 with the ROUTER FROZEN
# (train_memory_only + freeze_memory_router: values@1e-3 + gate/value_proj/swilu@2.5e-5;
# keys/query_proj untouched -> the audited geometry is preserved exactly; MSE never
# touches the router, by design — see E37 for the option-1 vs option-2 decision).
# Aux losses kept ON as telemetry only (grads dead-end on the frozen router; the in-run
# routing-sim log is the live drift monitor). Held-in eval @10k = seen-task plasticity
# check (reserve trigger: A-phase MSE plateauing >> joint's ~0.13-0.16 => rerun with
# router unfrozen at tiny LR = train_memory_only + --policy.optimizer_lr=2.5e-6).
# Re-audit measures the DEPLOYED geometry (values perturb downstream x -> routing can
# shift a little even with frozen router params; E30 mechanism). Informational.
# Sequential = C's config verbatim (beta4 + 5000 steps/task + top_t 1536, 20 eps).
set -eo pipefail
echo "stageA+seq chain started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
PRETRAIN_DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
AUDIT_SH="$ROOT_DIR/job_scripts/nebius/libero_90/probes/audit_heldout_routing.sh"
WARMED_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_8_10_12_14_frozenbase_rwarmup10k_lr1e-4_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model"
A_RUN=libero_90_pi05_8_10_12_14_frozenbase_rwarmupA_values10k_c0.05_sep5.0_noloc_rq512
SEQ_RUN=libero_10_sequential_pi05_8_10_12_14_frozenbase_rwarmupA_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps5k
AUDIT_RUN=audit_heldout_frozenbase_rwarmupA_10k
A_OUT="$ROOT_DIR/outputs/train/$A_RUN"
A_CKPT="$A_OUT/checkpoints/last/pretrained_model"
SEQ_OUT="$ROOT_DIR/outputs/train/$SEQ_RUN"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$WARMED_CKPT" ] || { echo "ERROR: warmed router checkpoint missing"; exit 1; }

###############################################################################
# A phase — values-only on libero_90, router frozen
###############################################################################
echo "=== [A phase] values-only 10k, ROUTER FROZEN -> $A_RUN ==="
if [ -d "$A_CKPT" ]; then
  echo "[A] checkpoint exists - skipping."
else
  lerobot-train \
    --policy.path="$WARMED_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$A_RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_90 \
    --dataset.root="$PRETRAIN_DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_90 \
    --output_dir="$A_OUT" \
    --save_freq=10000 \
    --steps=10000 \
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=4 \
    --eval_freq=10000 \
    --log_freq=200 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.train_router_only=false \
    --policy.train_memory_only=true \
    --policy.freeze_memory_router=true \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=40000 \
    --job_name="$A_RUN" \
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
[ -d "$A_CKPT" ] || { echo "ERROR: A phase finished but checkpoint missing"; exit 1; }

###############################################################################
# Re-audit — deployed geometry after value training (informational)
###############################################################################
echo "=== [re-audit] $AUDIT_RUN ==="
if [ "$(ls $ROOT_DIR/outputs/train/$AUDIT_RUN/memory_by_task/*.json 2>/dev/null | wc -l)" -ge 10 ]; then
  echo "[re-audit] already complete - skipping."
else
  bash "$AUDIT_SH" "$A_CKPT" "$AUDIT_RUN" || echo "[re-audit] AUDIT FAILED (continuing - informational)"
fi

###############################################################################
# Sequential — C's config verbatim
###############################################################################
echo "=== [sequential] $SEQ_RUN ==="
if [ -d "$SEQ_OUT/checkpoints/050000" ]; then
  echo "[seq] final checkpoint exists - skipping."
else
  lerobot-sequential-train \
    --policy.path="$A_CKPT" \
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
echo "stageA+seq chain completed at $(date)"
