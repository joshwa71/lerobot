#!/bin/bash
# GRADUATION RUN for the [2,2,4,4] layer-rank prior (research_log Entry 32, cell A).
# = the sep5 graduation recipe (c0.05 / sep5.0 / noloc / rq512 / knn36) with EXACTLY one
# pretrain delta: layer_ranks [2,2,2,2] -> [2,2,4,4] (rank 4 on L12/L14 = 3.62B values;
# probe A audit: famIoU 0.265 = baseline, core50 UP at every layer, q_intra 0.910).
#
# Chain: stage 1 pretrain 40k -> stage 1.5 held-out routing audit -> stage 2 sequential.
# Stage 2 = C's winning config (Entry 30 best, 44.5%): beta4 protection + 5000 steps/task
# + top_t 1536, 20 eval eps (apples-to-apples with the beta4/C family).
# ⚠ RETENTION FLAG (Entry 32): stage-2 per-task checkpoints must NOT be cleaned until the
# rank DiD drift analysis (test 2) is done. Rank-2 DiD baseline = protectB4's checkpoints.
# After landing: overlay the 10 rank-4 points on outputs/analysis/rank2_rto_retention.json
# via scripts/vla_analysis/rto_curve.py (test 1).
#
# Watch during stage 2: L12 channels (the trust ladder flattened at [2,2,4,4] — in-run gate
# L12 >= L14; audit L12 famIoU 0.245 vs P9's 0.210), basket family, env7.

set -eo pipefail
echo "Job started on $(hostname) at $(date)"

ROOT_DIR=/home/josh/lerobot
PI05_BASE="/home/josh/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30"
PRETRAIN_DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
AUDIT_SH="$ROOT_DIR/job_scripts/nebius/libero_90/probes/audit_heldout_routing.sh"

PRETRAIN_RUN=libero_90_pi05_8_10_12_14_film_lora_2244_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k
SEQ_RUN=libero_10_sequential_pi05_8_10_12_14_film_lora_2244_sample_contrastive_0.05_sep_5.0_noloc_knn_36_rq512_40k_top_t_1536_protect_beta4_steps5k
AUDIT_RUN=audit_heldout_ranks2244_40k

PRETRAIN_OUTPUT_DIR="$ROOT_DIR/outputs/train/$PRETRAIN_RUN"
SEQ_OUTPUT_DIR="$ROOT_DIR/outputs/train/$SEQ_RUN"
PRETRAIN_CHECKPOINT="$PRETRAIN_OUTPUT_DIR/checkpoints/last/pretrained_model"

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
# Stage 1 - pretrain on libero_90, layer_ranks [2,2,4,4]
###############################################################################
echo "=============================================================="
echo "[stage 1] PRETRAIN libero_90  [ranks 2,2,4,4 / c=0.05 / sep=5 / noloc / rq512]"
echo "  output: $PRETRAIN_OUTPUT_DIR"
echo "=============================================================="

if [ -d "$PRETRAIN_CHECKPOINT" ]; then
  echo "[stage 1] checkpoint exists - skipping pretrain."
else
  lerobot-train \
    --policy.path="$PI05_BASE" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$PRETRAIN_RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_90 \
    --dataset.root="$PRETRAIN_DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_90 \
    --output_dir="$PRETRAIN_OUTPUT_DIR" \
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
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=40000 \
    --job_name="$PRETRAIN_RUN" \
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
    --policy.memory_layer.contrastive_loss_weight=0.05 \
    --policy.memory_layer.contrastive_margin=0.0 \
    --policy.memory_layer.contrastive_query_queue=512 \
    --policy.memory_layer.routing_loss_topk=36 \
    --policy.memory_layer.routing_intra_task_locality_weight=0 \
    --policy.memory_layer.routing_inter_task_separation_weight=5.0 \
    --policy.memory_layer.routing_query_queue=512
fi

if [ ! -d "$PRETRAIN_CHECKPOINT" ]; then
  echo "ERROR: pretrain finished but $PRETRAIN_CHECKPOINT does not exist"
  exit 1
fi

###############################################################################
# Stage 1.5 - held-out routing audit on the 40k checkpoint (informational)
###############################################################################
echo "=============================================================="
echo "[stage 1.5] 40k HELD-OUT AUDIT -> $AUDIT_RUN"
echo "=============================================================="
if [ "$(ls $ROOT_DIR/outputs/train/$AUDIT_RUN/memory_by_task/*.json 2>/dev/null | wc -l)" -ge 10 ]; then
  echo "[stage 1.5] audit already complete - skipping."
else
  bash "$AUDIT_SH" "$PRETRAIN_CHECKPOINT" "$AUDIT_RUN" || echo "[stage 1.5] AUDIT FAILED (continuing to sequential — audit is informational)"
fi

###############################################################################
# Stage 2 - sequential libero_10, C's config (beta4 + 5k steps + top_t 1536)
###############################################################################
echo "=============================================================="
echo "[stage 2] SEQUENTIAL libero_10  [beta4 protect / 5000 steps/task / top_t 1536 / 20 eps]"
echo "  from: $PRETRAIN_CHECKPOINT"
echo "  output: $SEQ_OUTPUT_DIR"
echo "=============================================================="

# NB gradient_checkpointing=TRUE here, unlike every rank-2 sequential (which ran false).
# At 3.62B values the no-ckpt first forward exceeds the H200 (OOM at 138GiB, 5 Jul):
# rank-4 adds ~5G fp32 params + ~2x the retained LoRA gather activations at L12/L14 on
# top of a config that was already near-capacity at rank-2. The 40k audit ran this exact
# code path (bs32, same ckpt, backward+Adam) WITH ckpt=true and completed - existence
# proof it fits. Grads are mathematically identical (use_reentrant=False, no dropout);
# only cost is ~25% wall-clock. Fallback if recompute-OOM (E27 precedent at rank-2):
# batch_size=16 + gradient_accumulation_steps=2 (E10 rank-4 precedent; TF-IDF/protection
# accumulate correctly across micro-batches).
lerobot-sequential-train \
  --policy.path="$PRETRAIN_CHECKPOINT" \
  --policy.empty_cameras=1 \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --dataset.repo_id=libero_10 \
  --dataset.root="$SEQ_DATASET_ROOT" \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir="$SEQ_OUTPUT_DIR" \
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

echo "Job completed at $(date)"
