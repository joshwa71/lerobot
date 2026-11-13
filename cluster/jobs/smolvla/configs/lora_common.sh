# LoRA baselines on the SmolVLA libero_90 base (research_log2 Entry 4), sourced by the configs/lora_*.sh runs after
# they set RANK and LR. Queued runs read this file when they start: leave it unchanged until they have run.
#   - placement as pi05's baselines: attention q/k/v/o and MLP gate/up/down in both towers (all 16 VLM text layers,
#     all 16 expert layers) plus the state/action/time projections; nothing fully trained beside the adapters.
#   - lora_alpha = r (gain 1); the learning rate is picked by a sweep (configs/lrcal_*.sh).
#   - per task as pi05's specialists and naive sequential rows: 5,000 steps, batch 32, AdamW (wd 0), LR linear to LR/10,
#     optimizer rebuilt per task. Specialists are one-task runs of the same sequential trainer as naive sequential.
#   - evaluation executes 10 actions per chunk (research_log2 Entry 1).
: "${RANK:?set RANK before sourcing lora_common.sh}" "${LR:?set LR before sourcing lora_common.sh}"
LR_END=$(awk -v l="$LR" 'BEGIN { printf "%g", l / 10 }')
WANDB=(--wandb.enable=false)
NUM_WORKERS=8
GPUS="gpu=true,gpu_type=(a100_80|a40|a6000|h100|l40s|rtx6000ada)"   # >= 48 GB: backprop through every layer at batch 32
RES_SPEC="tmem=48G,h_rt=12:00:00,tscratch=40G"
RES_SEQ="tmem=48G,h_rt=48:00:00,tscratch=60G"
RES_TRI="tmem=48G,h_rt=24:00:00,tscratch=40G"
RES_JOINT="tmem=64G,h_rt=48:00:00,tscratch=60G"
RES_EVAL="tmem=48G,h_rt=24:00:00,tscratch=40G"
RUN_MARGIN_MIN=30

SEQ_DATASET=libero_10
SEQ_TASKS='[0,1,2,3,4,5,6,7,8,9]'   # libero_10 task_index; envs 4,6,9,2,7,0,8,1,3,5
DS_TO_ENV='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'
STAGE1_CKPT=$CLUSTER_ROOT/runs/smolvla_l90_base/stage1/checkpoints/050000/pretrained_model

LORA_TARGETS='(model\.vlm_with_expert\.(lm_expert|vlm\.model\.text_model)\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
LORA_ARGS=(--peft.method_type=LORA --peft.r="$RANK" --peft.lora_alpha="$RANK"
           --peft.target_modules="$LORA_TARGETS" --peft.full_training_modules='[]')

# Per-task training (spec: one task; seq with SEQ_INIT: naive sequential)
SEQ_STEPS_PER_TASK=5000
SEQ_ARGS=(
  --batch_size=32 --gradient_accumulation_steps=1
  "${LORA_ARGS[@]}"
  --policy.n_action_steps=10
  --tfidf_enable=false --use_online_idf_stats=false --protect_prior_slots=false
  --memory_value_lr="$LR" --memory_value_lr_end="$LR_END" --memory_value_scheduler_type=linear
  --save_after_each_task=true --reinit_optimizer_each_task=true
)

# Joint training on all ten tasks (multitask LoRA): lerobot-train with the policy's optimizer preset, 50k steps as
# pi05's multitask row. Saved only at the end (lerobot-train has no tested LoRA resume): a failed run restarts.
JOINT_STEPS=50000
JOINT_ARGS=(
  --batch_size=32 --save_freq=50000
  "${LORA_ARGS[@]}"
  --policy.optimizer_lr="$LR" --policy.scheduler_warmup_steps=200
  --policy.scheduler_decay_steps=50000 --policy.scheduler_decay_lr="$LR_END"
  --policy.n_action_steps=10
)

EVAL_SEEDS=(1000 2000 3000 4000)
EVAL_ARGS=(--env.type=libero --env.task=libero_10 --eval.n_episodes=25 --eval.batch_size=5 --policy.n_action_steps=10)
