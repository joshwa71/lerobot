#!/bin/bash
# E41: lr2x + SOFT (grad_scale) protection — the composition bet.
#
# Deltas vs stageB sequential (single mechanism change + the proven amplitude lever):
#   --memory_value_lr=0.002 / end 0.0002    (arm-2 lever: median written-slot displacement
#                                            ranks with rollout conversion; inits 47.4 vs 35.0)
#   --protect_mode=grad_scale               (E41: replace top-t rank-discount with exact
#                                            per-slot LR scaling via post-step blend;
#                                            Adam is invariant to per-row grad scaling, so
#                                            the blend is the correct mechanism)
#   --protect_u_norm=corefrac               (u = counts / core50-boundary count, clip 1;
#                                            fixes the peak-norm degeneracy where u~0.035 at
#                                            the core boundary made beta4 a top-1%-only veto)
#   --protect_beta=4                        (calibrated offline on the lr2x exposure+delta
#                                            structure: e9-block bleed onto e4/e6 kept 48%
#                                            [i.e. 52% suppressed] at 17.6% static write-mass
#                                            cost to e9; beta=8 => 35%/25.6%. Static cost
#                                            overstates real fit loss: the 1536-slot mask
#                                            re-allocates suppressed magnitude to prior-free
#                                            slots. If e6 still bleeds at beta=4 the response
#                                            curve is threshold-y => go 8-16; if rescued, done.)
#
# Mechanism recap (E41 analysis): all four E40 arms share identical exposure topology
# (frozen router); the rollout giveback rides on e9's diluted block writing small deltas
# across 4-6% of prior tasks' read mass — MSE-invisible (+1-2%) but 10-30pp in rollouts on
# marginal tasks, in 4/4 runs. Rank-mode protection provably cannot gate it (diffuse low-u
# tail + high-TF survivors). grad_scale attenuates exactly that write magnitude.
#
# Pre-registered reads: inits >= ~45 (amplitude preserved); e6 drop across e9's block <= 10pp
# (vs -10..-35 in all four arms); e9 init may drop a few pp (it pays the protection bill);
# final >= 42 = new frontier (r2244 42.0 @ matched window).
set -eo pipefail
echo "E41 lr2x+softprotect (grad_scale/corefrac/beta4) started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
A_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_8_10_12_14_frozenroute_rwarmupB_values10k_c0.05_sep5.0_noloc_rq512/checkpoints/last/pretrained_model"
SEQ_RUN=libero_10_sequential_pi05_8_10_12_14_frozenroute_rwarmupB_c0.05_sep5.0_noloc_rq512_top_t_1536_softprotect_cf_beta4_lr2x_steps5k_tasks5
SEQ_OUT="$ROOT_DIR/outputs/train/$SEQ_RUN"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$A_CKPT" ] || { echo "ERROR: stageB A checkpoint missing"; exit 1; }

# 5 tasks x 5000 steps -> final checkpoint 025000
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
    --protect_mode=grad_scale \
    --protect_u_norm=corefrac \
    --memory_value_lr=0.002 \
    --memory_value_lr_end=0.0002 \
    --memory_value_scheduler_type=linear
fi
echo "E41 lr2x+softprotect completed at $(date)"
