# Plumbing test of the whole chain: every stage for a few steps, small banks, libero_10 everywhere (it stages
# in ~15 min instead of libero_90's ~75). Takes example.sh and overrides; to rerun, delete the run dir:
#   rm -rf /SAN/vision/jo71_vla_wd/runs/smoke_chain && cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/smoke.sh
source "$REPO/cluster/jobs/smolvla/configs/example.sh"
NAME=smoke_chain
NUM_WORKERS=4
GPUS="gpu=true,gpu_type=(a10|a100|a100_80|a40|a6000|h100|l4|l40s|rtx3090|rtx4070ti|rtx4090|rtx6000ada)"
for st in STAGE1 WARMUP AUDIT APHASE SEQ EVAL; do declare "RES_$st=tmem=32G,h_rt=3:00:00,tscratch=40G"; done
RUN_MARGIN_MIN=15

PRETRAIN_DATASET=libero_10
SEQ_TASKS='[0,1]'

STAGE1_STEPS=40
STAGE1_ARGS=(--batch_size=8 --save_freq=20)  # SmolVLA default fine-tune (expert only)
MEMORY+=(--policy.memory_layer.mem_n_keys=32 --policy.memory_layer.vlm_mem_n_keys=32
         --policy.memory_layer.mem_knn=8 --policy.memory_layer.routing_loss_topk=8 --policy.memory_layer.vlm_mem_knn=8)
WARMUP_STEPS=40
WARMUP_ARGS+=(--batch_size=8 --save_freq=20)
AUDIT_ARGS+=(--online_steps_per_task=5 --tfidf_top_t=64)
APHASE_STEPS=40
APHASE_ARGS+=(--batch_size=8 --save_freq=20)
SEQ_STEPS_PER_TASK=20
SEQ_ARGS+=(--batch_size=8 --gradient_accumulation_steps=1 --tfidf_top_t=64 --eval.n_episodes=1 --eval_final_episodes=1)
EVAL_SEEDS=(1000)
EVAL_ARGS=(--env.type=libero --env.task=libero_10 --env.task_ids='[4,6]' --eval.n_episodes=1 --eval.batch_size=1)
