# Experiment config for the SmolVLA staged memory chain. Plain bash, sourced by every job of the chain.
# To try a configuration: copy this file, edit it, submit it:
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/<name>.sh [stage ...]
# Stages, in order: stage1 warmup audit aphase seq tri (default: all; eval = the final checkpoint only). Re-running chain.sh resumes:
# each stage skips itself when complete and otherwise continues from its last synced checkpoint.
#
# Values follow the pi05 sim paper cell (projects/summary_research_log.md section 2) where a knob carries
# over unchanged. The memory LAYOUT is pi05's relative depths on SmolVLA's 16 + 16 layers, without the last VLM
# layer (SmolVLA never reads its output): 11 sites, 7 tables (research_log2 Entry 1).

NAME=example                  # outputs: $CLUSTER_ROOT/runs/$NAME/<stage>; job logs: .../logs
WANDB=(--wandb.enable=false)  # e.g. (--wandb.enable=true --wandb.project=vla-memory)
NUM_WORKERS=8

# Resources per stage: SGE -l string (dip-207-1 is excluded by chain.sh). RUN_MARGIN_MIN before h_rt the
# python process is stopped (SIGABRT + traceback) so the job's exit trap can still sync outputs to /SAN.
GPUS="gpu=true,gpu_type=(a100|a100_80|a40|a6000|h100|l40s|rtx6000ada)"   # >= 40 GB
RES_STAGE1="tmem=64G,h_rt=72:00:00,tscratch=200G"   # libero_90 stages (as the old scripts): 65 GB data + 65 GB HF datasets cache + ckpts
RES_WARMUP="tmem=64G,h_rt=48:00:00,tscratch=200G"
RES_AUDIT="tmem=48G,h_rt=12:00:00,tscratch=60G"
RES_APHASE="tmem=64G,h_rt=48:00:00,tscratch=200G"   # + init 7 GB + 19 GB per checkpoint (values + Adam moments)
RES_SEQ="tmem=64G,h_rt=72:00:00,tscratch=150G"     # libero_10 24 GB with cache + init + one 7 GB checkpoint per task
RES_TRI="tmem=48G,h_rt=24:00:00,tscratch=40G"
RES_EVAL="tmem=48G,h_rt=24:00:00,tscratch=40G"
RUN_MARGIN_MIN=30

# ---- data ($HF_LEROBOT_HOME/<name>) ----
PRETRAIN_DATASET=libero_90
SEQ_DATASET=libero_10
SEQ_TASKS='[0,1,2,3,4,5,6,7,8,9]'   # libero_10 task_index; envs 4,6,9,2,7,0,8,1,3,5
DS_TO_ENV='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'

# ---- stage 1: the shared libero_90 base (configs/l90_base.sh: all weights, the pi05 stage-1 recipe, no memory).
# To run stage 1 inside this chain instead, empty STAGE1_CKPT; STAGE1_FROM / STAGE1_STEPS / STAGE1_ARGS then apply.
STAGE1_CKPT=$CLUSTER_ROOT/runs/smolvla_l90_base/stage1/checkpoints/050000/pretrained_model
STAGE1_FROM=lerobot/smolvla_base
STAGE1_STEPS=50000
STAGE1_ARGS=(
  --batch_size=32 --save_freq=5000
  --policy.train_expert_only=false --policy.freeze_vision_encoder=false
  --policy.scheduler_warmup_steps=4000 --policy.scheduler_decay_steps=50000
)

# ---- memory: attached at the warm-up, carried in the checkpoints from then on ----
MEMORY=(
  --policy.memory_layers=true --policy.memory_layer.enabled=true
  --policy.memory_layer.layers='[4,6,8,10,12,14]' --policy.memory_layer.share_groups='[[4,6],[8,10]]'
  --policy.memory_layer.vlm_layers='[5,7,9,11,13]' --policy.memory_layer.vlm_share_groups='[[5,7],[9,11]]'
  --policy.memory_layer.mem_n_keys=256 --policy.memory_layer.vlm_mem_n_keys=256
  --policy.memory_layer.value_type=lora --policy.memory_layer.lora_rank=2 --policy.memory_layer.vlm_lora_rank=2
  --policy.memory_layer.mem_heads=4 --policy.memory_layer.mem_k_dim=512
  --policy.memory_layer.mem_knn=36 --policy.memory_layer.routing_loss_topk=36 --policy.memory_layer.vlm_mem_knn=16
  --policy.memory_layer.mem_gated=true --policy.memory_layer.swilu_projection=true
  --policy.memory_layer.vlm_text_span=200   # >0: VLM memory on instruction + state tokens; 0: every prefix position
  --policy.memory_layer.vlm_router_pool=anchored --policy.memory_layer.vlm_router_pool_weights='[1.0,0.5]'
  --policy.memory_layer.expert_anchor_pool=text --policy.memory_layer.expert_anchor_weight=0.4
  --policy.memory_layer.expert_anchor_src_dim=960
  --policy.memory_layer.lang_to_query=false
  --policy.memory_layer.use_frozen_base_input_features=true --policy.memory_layer.frozen_prepass=true
  --policy.memory_layer.memory_lr=0.001 --policy.memory_layer.value_fixed_lr=0.001
  --policy.memory_layer.log_usage=true
)

# ---- router warm-up: keys + query projections only, routing losses only ----
WARMUP_STEPS=10000
WARMUP_ARGS=(
  --batch_size=32 --save_freq=2500 --policy.optimizer_lr=1e-4
  --policy.train_router_only=true --policy.memory_layer.router_only_fast=true
  --policy.memory_layer.vlm_route_once=false --policy.memory_layer.aggregate_usage=true
  --policy.memory_layer.contrastive_method=sample --policy.memory_layer.contrastive_loss_weight=0.05
  --policy.memory_layer.contrastive_margin=0.0 --policy.memory_layer.contrastive_query_queue=512
  --policy.memory_layer.routing_intra_task_locality_weight=0
  --policy.memory_layer.routing_inter_task_separation_weight=8.0 --policy.memory_layer.routing_query_queue=512
)

# ---- held-out routing audit: stream SEQ_DATASET through the frozen warm-up checkpoint (value LR 1e-12) ----
AUDIT_TASKS='[0,1,2,3,4,5,6,7,8,9]'   # all ten, whatever SEQ_TASKS is, so certificates stay comparable
AUDIT_ARGS=(
  --batch_size=8 --online_steps_per_task=400
  --policy.memory_layer.aggregate_usage=false --policy.memory_layer.vlm_route_once=true
  --policy.memory_layer.router_only_fast=false
  --tfidf_enable=true --tfidf_top_t=1536 --use_online_idf_stats=true --idf_exponent=1
  --memory_value_lr=1e-12 --memory_value_lr_end=1e-12 --memory_value_scheduler_type=linear
)

# ---- A-phase: values (+ gate / projections) on the pretraining data, router frozen ----
APHASE_STEPS=10000
APHASE_ARGS=(
  --batch_size=32 --save_freq=5000
  --policy.train_router_only=false --policy.memory_layer.router_only_fast=false
  --policy.train_memory_only=true --policy.freeze_memory_router=true
  --policy.memory_layer.vlm_route_once=true
)

# ---- sequential adaptation on SEQ_TASKS: values only ----
SEQ_STEPS_PER_TASK=5000
SEQ_ARGS=(
  --batch_size=16 --gradient_accumulation_steps=2
  --policy.train_router_only=false --policy.memory_layer.router_only_fast=false
  --policy.memory_layer.vlm_route_once=true --policy.memory_layer.aggregate_usage=false
  --policy.n_action_steps=10
  --save_after_each_task=true --reinit_optimizer_each_task=true
  --tfidf_enable=true --tfidf_top_t=3072 --use_online_idf_stats=true --idf_exponent=1
  --protect_prior_slots=true --protect_beta=4 --protect_mode=rank --protect_u_norm=corefrac
  --memory_value_lr=2e-3 --memory_value_lr_end=2e-4 --memory_value_scheduler_type=linear
)  # no --env: no in-run boundary evals (retired on pi05); tri measures every boundary

# ---- evaluation: tri = retention triangle (one job per task boundary, all seeds); eval = final checkpoint, one
# job per seed. SmolVLA executes 10 actions per predicted chunk (research_log2 Entry 1). ----
EVAL_SEEDS=(1000 2000 3000 4000)
EVAL_ARGS=(--env.type=libero --env.task=libero_10 --eval.n_episodes=25 --eval.batch_size=5 --policy.n_action_steps=10)
