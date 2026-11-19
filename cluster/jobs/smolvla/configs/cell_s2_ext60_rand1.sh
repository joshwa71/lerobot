# The pi05 paper cell on SmolVLA, sixty sequential libero_ext tasks in a random order (research_log2 Entry 5; Paper 1
# item 1: scale the sequence to 30-60 new tasks). The sequential stage only, from cell_s2_seq5's A-phase checkpoint
# (SEQ_INIT), i.e. the SmolVLA router warm-up at separation 2 with its values filled on libero_90, exactly as the
# ten-task run (cell_s2_seq10). Settings are cell_s2_seq10's verbatim except:
#   - data: libero_ext (projects/dataset_gen.md; 60 tasks x 50 scripted-expert demos), task_index = env id;
#   - order: a random permutation, random.Random(1).shuffle(list(range(60))), recorded with the task names in
#     configs/ext60_order_rand1.json;
#   - in-run loss matrix: --eval.type=loss scores the paired-noise flow-matching MSE of every task seen so far after
#     each task (16 batches of 16 per task; per-task seeded frames and noise, so rounds pair) ->
#     seq/eval/loss_results.jsonl; the pi05 instrument's loss matrix, computed on the live policy;
#   - logging every 100 steps (was 200), for per-task learning curves;
#   - seq scratch 200G (libero_ext 68 GB + its 68 GB HF datasets cache); only the last boundary checkpoint is copied
#     in on resume, and synced boundary checkpoints are pruned from scratch (run_stage.sh), so all 60 stay on /SAN.
# Statistics for post-hoc analysis, all under runs/<NAME>/seq/:
#   memory_by_task/memory_usage_task_<t>.json   per module, per slot: reads and updates while task t trained
#                                               (routing is stationary, so these are also the task's read profile)
#   checkpoints/<step>/sequential_state.pt      per boundary: protection store u(s), online IDF, loss history
#   checkpoints/<step>/pretrained_model         per boundary (6.6 GB each): for rollout evals of any row/cell
#   eval/loss_results.jsonl                     the loss matrix, one row per boundary
#   logs (runs/<NAME>/logs/*seq*)               loss, grad norm, LR and step timings every 100 steps
# Rollout evaluation (tri) comes after training, from the saved boundary checkpoints: the Paper 1 plan's full
# retention row every fifth task and the just-trained task at every task (EVAL_ARGS below; machinery to follow).
# 5,000 steps per task at ~1.2 s/step is ~1.7 h per task, ~100 h in all: two 72 h seq jobs, the second resuming at
# the first's last complete boundary. Submit:
#   cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/cell_s2_ext60_rand1.sh seq seq seq

NAME=cell_s2_ext60_rand1
WANDB=(--wandb.enable=false)
NUM_WORKERS=8

GPUS="gpu=true,gpu_type=(a100_80|a40|a6000|h100|l40s|rtx6000ada)"   # >= 48 GB
RES_WARMUP="tmem=64G,h_rt=24:00:00,tscratch=200G"   # libero_90: 65 GB data + 65 GB HF datasets cache + checkpoints
RES_AUDIT="tmem=48G,h_rt=12:00:00,tscratch=60G"
RES_APHASE="tmem=64G,h_rt=48:00:00,tscratch=200G"   # + init 7 GB + two 19 GB checkpoints
RES_SEQ="tmem=64G,h_rt=72:00:00,tscratch=200G"      # libero_ext 68 GB + 68 GB cache + init + ~3 checkpoints in flight
RES_TRI="tmem=48G,h_rt=24:00:00,tscratch=40G"
RES_EVAL="tmem=48G,h_rt=24:00:00,tscratch=40G"
RUN_MARGIN_MIN=30

# ---- data ($HF_LEROBOT_HOME/<name>) ----
PRETRAIN_DATASET=libero_90
SEQ_DATASET=libero_ext
# libero_ext task_index in training order (configs/ext60_order_rand1.json); task_index = LIBERO env id
SEQ_TASKS='[26,59,23,9,2,5,18,43,47,22,57,25,19,10,3,11,29,37,15,35,52,53,40,45,34,21,12,50,58,33,39,32,42,20,44,14,17,49,0,38,27,46,1,56,6,13,24,41,30,28,55,31,7,16,4,48,51,54,36,8]'
DS_TO_ENV="{$(for i in $(seq 0 59); do printf '"%d":%d,' "$i" "$i"; done | sed 's/,$//')}"

# ---- stage 1: the shared libero_90 base (configs/l90_base.sh) ----
STAGE1_CKPT=$CLUSTER_ROOT/runs/smolvla_l90_base/stage1/checkpoints/050000/pretrained_model
SEQ_INIT=$CLUSTER_ROOT/runs/cell_s2_seq5/aphase/checkpoints/010000/pretrained_model   # the shared A-phase

# ---- memory: attached at the warm-up, carried in the checkpoints from then on (cell_s2_seq5's; for reference) ----
MEMORY=(
  --policy.memory_layers=true --policy.memory_layer.enabled=true
  --policy.memory_layer.layers='[4,6,8,10,12,14]' --policy.memory_layer.share_groups='[[4,6],[8,10]]'
  --policy.memory_layer.vlm_layers='[5,7,9,11,13]' --policy.memory_layer.vlm_share_groups='[[5,7],[9,11]]'
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

# ---- router warm-up and A-phase: cell_s2_seq5's (not run here; SEQ_INIT is their product) ----
WARMUP_STEPS=10000
APHASE_STEPS=10000

# ---- sequential adaptation on SEQ_TASKS: values only (cell_s2_seq10's, plus the loss matrix and log_freq) ----
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
  --eval.type=loss --eval_loss_n_batches=16
  --log_freq=100
)

# ---- evaluation (after training, from the boundary checkpoints) ----
EVAL_SEEDS=(1000 2000 3000 4000)
EVAL_ARGS=(--env.type=libero --env.task=libero_ext --eval.n_episodes=25 --eval.batch_size=5 --policy.n_action_steps=10)
