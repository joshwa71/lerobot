#!/bin/bash
# E59: run the frozen_prepass smoke suite on the VM (stage-1 base, fresh attach,
# float32, small banks). Three invocations:
#   A  guard-legal layout (B's spread), prepass OFF at attach -> equivalence suite
#   B  INTERLEAVED layout (expert==vlm [4,6,8,10,12]), prepass ON -> new-property suite
#   C  interleaved WITHOUT prepass -> must FAIL with the placement-guard message
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
  --output_dir=/tmp/smoke_prepass_$$_out \
  --steps=200000 --batch_size=2 --num_workers=2 \
  --online_task_ids=[0] --online_steps_per_task=10 \
  --wandb.enable=false --job_name=smoke_prepass \
  --ds_to_env_map_json={\"0\":4,\"1\":6,\"2\":9,\"3\":2,\"4\":7,\"5\":0,\"6\":8,\"7\":1,\"8\":3,\"9\":5}"

echo "=== MODE A: guard-legal spread, prepass off (equivalence) ==="
rm -rf /tmp/smoke_prepass_$$_out
python scripts/vla_analysis/smoke_frozen_prepass.py $COMMON \
  --policy.memory_layer.layers=[2,4,6,8] \
  --policy.memory_layer.vlm_layers=[10,12,14,16] \
  --policy.memory_layer.frozen_prepass=false || exit 1

echo "=== MODE B: interleaved, prepass on (new property) ==="
rm -rf /tmp/smoke_prepass_$$_out
python scripts/vla_analysis/smoke_frozen_prepass.py $COMMON \
  --policy.memory_layer.layers=[4,6,8,10,12] \
  --policy.memory_layer.vlm_layers=[4,6,8,10,12] \
  --policy.memory_layer.frozen_prepass=true || exit 1

echo "=== MODE C: interleaved WITHOUT prepass must raise the guard ==="
rm -rf /tmp/smoke_prepass_$$_out
if python scripts/vla_analysis/smoke_frozen_prepass.py $COMMON \
  --policy.memory_layer.layers=[4,6,8,10,12] \
  --policy.memory_layer.vlm_layers=[4,6,8,10,12] \
  --policy.memory_layer.frozen_prepass=false 2>&1 | tee /tmp/smoke_prepass_C.log | grep -q "frozen_prepass=true to lift"; then
  echo "[PASS] C guard raises with the lift hint"
else
  echo "[FAIL] C guard did not raise as expected"; exit 1
fi
echo "ALL THREE MODES PASS"
