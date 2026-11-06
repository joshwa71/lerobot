#!/bin/bash
# E68 A1: live-routing smoke suite on the VM (stage-1 base, fresh attach on the PAPER CELL's
# interleaved layout with small banks, float32). Three invocations:
#   L  live flags + allow_nonstationary_routing=true -> the L1..L7 suite (must pass)
#   G  same WITHOUT the switch -> must FAIL at config parse with the placement-guard message
#   X  switch combined with stationary flags -> must FAIL with the "must not be combined" message
set -o pipefail
ROOT=/home/josh/lerobot
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
BASE=$ROOT/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
OUT=/tmp/smoke_live_$$_out
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
  --policy.memory_layer.layers=[4,6,8,10,14,16] \
  --policy.memory_layer.vlm_layers=[5,7,9,11,13,15] \
  --policy.memory_layer.share_groups=[[4,6],[8,10]] \
  --policy.memory_layer.vlm_share_groups=[[5,7],[9,11],[13,15]] \
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
  --dataset.repo_id=libero_10 \
  --dataset.root=$ROOT/outputs/libero_10 \
  --rename_map=$RENAME \
  --env.type=libero --env.task=libero_10 \
  --output_dir=$OUT \
  --steps=200000 --batch_size=2 --num_workers=2 \
  --online_task_ids=[0] --online_steps_per_task=10 \
  --wandb.enable=false --job_name=smoke_live \
  --ds_to_env_map_json={\"0\":4,\"1\":6,\"2\":9,\"3\":2,\"4\":7,\"5\":0,\"6\":8,\"7\":1,\"8\":3,\"9\":5}"

echo "=== MODE L: live flags + switch (suite) ==="
rm -rf $OUT
python scripts/vla_analysis/smoke_live_routing.py $COMMON \
  --policy.memory_layer.use_frozen_base_input_features=false \
  --policy.memory_layer.frozen_prepass=false \
  --policy.memory_layer.allow_nonstationary_routing=true || exit 1

echo "=== MODE G: live flags WITHOUT the switch must raise the placement guard ==="
rm -rf $OUT
python scripts/vla_analysis/smoke_live_routing.py $COMMON \
  --policy.memory_layer.use_frozen_base_input_features=false \
  --policy.memory_layer.frozen_prepass=false \
  --policy.memory_layer.allow_nonstationary_routing=false > /tmp/smoke_live_G.log 2>&1
RC=$?
if [ $RC -ne 0 ] && grep -q "frozen_prepass=true to lift" /tmp/smoke_live_G.log; then
  echo "[PASS] G guard raises without the switch (rc=$RC)"
else
  echo "[FAIL] G guard did not raise as expected (rc=$RC)"; exit 1
fi

echo "=== MODE X: switch combined with stationary flags must raise ==="
rm -rf $OUT
python scripts/vla_analysis/smoke_live_routing.py $COMMON \
  --policy.memory_layer.use_frozen_base_input_features=true \
  --policy.memory_layer.frozen_prepass=true \
  --policy.memory_layer.allow_nonstationary_routing=true > /tmp/smoke_live_X.log 2>&1
RC=$?
if [ $RC -ne 0 ] && grep -q "must not be combined with stationary addressing" /tmp/smoke_live_X.log; then
  echo "[PASS] X switch+stationary raises (rc=$RC)"
else
  echo "[FAIL] X did not raise as expected (rc=$RC)"; exit 1
fi
rm -rf $OUT
echo "ALL THREE LIVE-ROUTING MODES PASS"
