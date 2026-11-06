#!/bin/bash
# E40 4-way batch, ARM 1: affine slots + no gate (the mechanism bet).
#   --policy.memory_layer.lora_slot_bias=true  (per-slot affine: value_i(x) = U V x + b_i)
#   --policy.memory_layer.mem_gated=false      (gate REMOVED — E40 decision 2; the A-phase
#                                               gate was mid-sigmoid-calibrated on libero_90
#                                               and never saw libero_10)
#
# Value-path changes require an A-phase rerun (28 trainable tensors = 32 - 8 gating
# + 4 slot_bias); the warmed router drops in unchanged, and frozen-branch routing is a
# function of backbone+keys+query only, so the post-A audit would be identical to
# audit_heldout_frozenroute_rwarmupB_10k — skipped. Then sequential, C's config verbatim
# + the two flags, first 5 tasks, 20-ep evals + 50-ep final.
#
# NB (15 Jul): RECONSTRUCTION for the git record — this arm was launched on the source
# box 14 Jul (tmux stageC) from a script that lived in gitignored job_scripts/ and never
# propagated. Run names below follow the project convention and may differ from the
# source box's live run; check its outputs/stageC.log before reusing.
set -eo pipefail
echo "E40 arm 1 (affine+nogate) started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
PRETRAIN_DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
WARMED_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_8_10_12_14_frozenbase_rwarmup10k_lr1e-4_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model"
A_RUN=libero_90_pi05_8_10_12_14_frozenroute_rwarmupC_affine_nogate_values10k_c0.05_sep5.0_noloc_rq512
SEQ_RUN=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupC_affine_nogate_c0.05_sep5.0_noloc_rq512_top_t_1536_protect_beta4_steps5k_tasks5
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
# A phase — values-only on libero_90, router frozen, frozen-base routing,
# AFFINE slots + gate off
###############################################################################
echo "=== [A phase] affine+nogate values-only 10k -> $A_RUN ==="
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
    --policy.memory_layer.lora_slot_bias=true \
    --policy.memory_layer.mem_gated=false \
    --policy.memory_layer.contrastive_method=sample \
    --policy.memory_layer.contrastive_loss_weight=0.05 \
    --policy.memory_layer.contrastive_margin=0.0 \
    --policy.memory_layer.contrastive_query_queue=512 \
    --policy.memory_layer.routing_loss_topk=36 \
    --policy.memory_layer.routing_intra_task_locality_weight=0 \
    --policy.memory_layer.routing_inter_task_separation_weight=5.0 \
    --policy.memory_layer.routing_query_queue=512 \
    --policy.memory_layer.use_frozen_base_input_features=true
fi
[ -d "$A_CKPT" ] || { echo "ERROR: A phase finished but checkpoint missing"; exit 1; }

###############################################################################
# Sequential — C's config verbatim + affine/nogate flags, FIRST 5 TASKS
###############################################################################
echo "=== [sequential] $SEQ_RUN ==="
if [ -d "$SEQ_OUT/checkpoints/025000" ]; then
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
    --eval_final_episodes=50 \
    --log_freq=200 \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --job_name="$SEQ_RUN" \
    --online_task_ids='[0,1,2,3,4]' \
    --online_steps_per_task=5000 \
    --policy.memory_layer.aggregate_usage=false \
    --policy.memory_layer.use_frozen_base_input_features=true \
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
echo "E40 arm 1 (affine+nogate) completed at $(date)"
