#!/bin/bash
# E61: shared-pair memory tables smoke suite (stage-1 base, fresh attach, float32,
# small banks). Three invocations:
#   GUARDS  config-level share_groups validation raises (no model work)
#   LEGACY  no share flags, guard-legal layout -> legacy intact (all modules own storage)
#   SHARED  interleaved layout + frozen_prepass + share groups both towers ->
#           full property suite (aliasing, dedupe, grads, mask merge, store sync,
#           strict round-trip with bitwise forward parity)
set -o pipefail
ROOT=/home/josh/lerobot
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

BASE=$ROOT/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'

COMMON="--policy.path=$BASE \
  --policy.dtype=float32 \
  --policy.empty_cameras=1 \
  --policy.gradient_checkpointing=false \
  --policy.normalization_mapping={\"VISUAL\":\"IDENTITY\",\"STATE\":\"MEAN_STD\",\"ACTION\":\"MEAN_STD\"} \
  --policy.train_memory_only=true \
  --policy.freeze_memory_router=true \
  --policy.train_router_only=false \
  --policy.memory_layer.router_only_fast=false \
  --policy.memory_layers=true \
  --policy.memory_layer.enabled=true \
  --policy.memory_layer.memory_only=false \
  --policy.memory_layer.value_type=lora \
  --policy.memory_layer.mem_n_keys=64 \
  --policy.memory_layer.lora_rank=2 \
  --policy.memory_layer.mem_knn=8 \
  --policy.memory_layer.routing_loss_topk=8 \
  --policy.memory_layer.vlm_mem_n_keys=64 \
  --policy.memory_layer.vlm_lora_rank=2 \
  --policy.memory_layer.vlm_mem_knn=8 \
  --policy.memory_layer.vlm_text_span=200 \
  --policy.memory_layer.vlm_router_pool=anchored \
  --policy.memory_layer.vlm_router_pool_weights=[1.0,0.5] \
  --policy.memory_layer.vlm_route_once=true \
  --policy.memory_layer.log_usage=true \
  --policy.memory_layer.mem_heads=4 \
  --policy.memory_layer.mem_k_dim=512 \
  --policy.memory_layer.lang_to_query=false \
  --policy.memory_layer.expert_anchor_pool=text \
  --policy.memory_layer.expert_anchor_weight=0.4 \
  --policy.memory_layer.use_frozen_base_input_features=true \
  --dataset.repo_id=libero_10 \
  --dataset.root=$ROOT/outputs/libero_10 \
  --rename_map=$RENAME \
  --env.type=libero --env.task=libero_10 \
  --output_dir=/tmp/smoke_shared_$$_out \
  --steps=200000 --batch_size=2 --num_workers=2 \
  --online_task_ids=[0] --online_steps_per_task=10 \
  --wandb.enable=false --job_name=smoke_shared \
  --ds_to_env_map_json={\"0\":4,\"1\":6,\"2\":9,\"3\":2,\"4\":7,\"5\":0,\"6\":8,\"7\":1,\"8\":3,\"9\":5}"

echo "=== MODE GUARDS: config validation ==="
rm -rf /tmp/smoke_shared_$$_out
SMOKE_MODE=guards python scripts/vla_analysis/smoke_shared_pairs.py $COMMON \
  --policy.memory_layer.layers=[2,4,6,8] \
  --policy.memory_layer.vlm_layers=[10,12,14,16] \
  --policy.memory_layer.frozen_prepass=false || exit 1

echo "=== MODE LEGACY: no sharing, guard-legal layout ==="
rm -rf /tmp/smoke_shared_$$_out
SMOKE_MODE=legacy python scripts/vla_analysis/smoke_shared_pairs.py $COMMON \
  --policy.memory_layer.layers=[2,4,6,8] \
  --policy.memory_layer.vlm_layers=[10,12,14,16] \
  --policy.memory_layer.frozen_prepass=false || exit 1

echo "=== MODE SHARED: interleaved + prepass + share groups both towers ==="
rm -rf /tmp/smoke_shared_$$_out
SMOKE_MODE=shared python scripts/vla_analysis/smoke_shared_pairs.py $COMMON \
  --policy.memory_layer.layers=[4,6,8,10] \
  --policy.memory_layer.vlm_layers=[5,7,9,11] \
  --policy.memory_layer.share_groups=[[4,6],[8,10]] \
  --policy.memory_layer.vlm_share_groups=[[5,7],[9,11]] \
  --policy.memory_layer.frozen_prepass=true || exit 1

echo "ALL THREE MODES PASS"
