# Router warm-up sweeps on the SmolVLA libero_90 base (research_log2 Entry 2), sourced by the configs/wu_*.sh arms.
# Reference: the paper cell's warm-up recipe (summary_research_log 2.3) on a reduced layout of solo sites, then the
# held-out routing audit on libero_10 (as the pi05 audit). Each arm changes one setting from this.
#   HOLD_JID=<base job> cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/wu_<arm>.sh warmup audit
# Queued arms read this file when they start: leave it unchanged until they have run.
WANDB=(--wandb.enable=false)
NUM_WORKERS=8
GPUS="gpu=true,gpu_type=(a100|a100_80|a40|a6000|h100|l40s|rtx6000ada)"   # >= 40 GB
RES_WARMUP="tmem=64G,h_rt=12:00:00,tscratch=200G"   # libero_90: 65 GB data + 65 GB HF datasets cache + checkpoints
RES_AUDIT="tmem=48G,h_rt=8:00:00,tscratch=60G"
RUN_MARGIN_MIN=30

PRETRAIN_DATASET=libero_90
SEQ_DATASET=libero_10
SEQ_TASKS='[0,1,2,3,4,5,6,7,8,9]'
STAGE1_CKPT=$CLUSTER_ROOT/runs/smolvla_l90_base/stage1/checkpoints/050000/pretrained_model

# Reduced layout: three solo (unshared) sites per tower, shallow / mid / deep. With frozen-base routing each site's
# router trains on its own losses from memory-free features, so per-site results carry over to fuller layouts.
# VLM layer 15 is excluded: SmolVLA never reads its output. Everything else as the paper cell.
MEMORY=(
  --policy.memory_layers=true --policy.memory_layer.enabled=true
  --policy.memory_layer.layers='[4,8,14]' --policy.memory_layer.vlm_layers='[5,9,13]'
  --policy.memory_layer.mem_n_keys=256 --policy.memory_layer.vlm_mem_n_keys=256
  --policy.memory_layer.value_type=lora --policy.memory_layer.lora_rank=2 --policy.memory_layer.vlm_lora_rank=2
  --policy.memory_layer.mem_heads=4 --policy.memory_layer.mem_k_dim=512
  --policy.memory_layer.mem_knn=36 --policy.memory_layer.routing_loss_topk=36 --policy.memory_layer.vlm_mem_knn=16
  --policy.memory_layer.mem_gated=true --policy.memory_layer.swilu_projection=true
  --policy.memory_layer.vlm_text_span=200
  --policy.memory_layer.vlm_router_pool=anchored --policy.memory_layer.vlm_router_pool_weights='[1.0,0.5]'
  --policy.memory_layer.expert_anchor_pool=text --policy.memory_layer.expert_anchor_weight=0.4
  --policy.memory_layer.expert_anchor_src_dim=960
  --policy.memory_layer.lang_to_query=false
  --policy.memory_layer.use_frozen_base_input_features=true --policy.memory_layer.frozen_prepass=true
  --policy.memory_layer.memory_lr=0.001 --policy.memory_layer.value_fixed_lr=0.001
  --policy.memory_layer.log_usage=true
)

# Router warm-up: keys + query projections only, values pinned at zero, routing losses only, 10k steps.
WARMUP_STEPS=10000
WARMUP_ARGS=(
  --batch_size=32 --save_freq=5000 --policy.optimizer_lr=1e-4
  --policy.train_router_only=true --policy.memory_layer.router_only_fast=true
  --policy.memory_layer.vlm_route_once=false --policy.memory_layer.aggregate_usage=true
  --policy.memory_layer.contrastive_method=sample --policy.memory_layer.contrastive_loss_weight=0.05
  --policy.memory_layer.contrastive_margin=0.0 --policy.memory_layer.contrastive_query_queue=512
  --policy.memory_layer.routing_intra_task_locality_weight=0
  --policy.memory_layer.routing_inter_task_separation_weight=8.0 --policy.memory_layer.routing_query_queue=512
)

# Held-out routing audit: stream all ten libero_10 tasks through the frozen warm-up checkpoint (value LR 1e-12).
AUDIT_ARGS=(
  --batch_size=8 --online_steps_per_task=400
  --policy.memory_layer.aggregate_usage=false --policy.memory_layer.vlm_route_once=true
  --policy.memory_layer.router_only_fast=false
  --tfidf_enable=true --tfidf_top_t=1536 --use_online_idf_stats=true --idf_exponent=1
  --memory_value_lr=1e-12 --memory_value_lr_end=1e-12 --memory_value_scheduler_type=linear
)
