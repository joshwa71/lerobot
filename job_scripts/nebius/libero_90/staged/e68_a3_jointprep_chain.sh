#!/bin/bash
# E68 ablation A3 — JOINT router preparation at MATCHED router LR (no separate warm-up).
# Single delta from the paper cell's protocol: instead of (warm-up: routers only, values pinned at
# zero, aux losses only, 10k) -> (A-phase: values only, routers frozen, MSE, 10k), ONE 10k stage
# trains keys + query projections + values + gates + output projections TOGETHER on libero_90 with
# MSE + the same aux losses (c 0.05 / sep 8, queues 512, broadcast), on the frozen stage-1 backbone,
# with the ROUTER GROUP at 1e-4 (= the warm-up's router LR; removes the E35/E36 confound where
# joint routers sat at 2.5e-5 = 40x below the values) and values at memory_lr 1e-3 (= A-phase).
# Stationary routing (prepass) stays ON. Routers are then frozen and the 5-task sequential runs
# C-config verbatim. The held-out audit on the joint checkpoint is recorded (informational) — it is
# part of the result, not a gate. Replaces Table II A (IoU-only, 4-site n=384, LR-confounded).
# Note (E68 addendum): mlp.anchor_proj (W_a) is outside the `.mlp.mem.` name filter, so it stays at
# init here EXACTLY as in the paper cell's warm-up — no anchor confound between A3 and the control.
# Second, minor delta to document: gates/output projections ride in the base group at 1e-4 here
# (A-phase: 2.5e-5); values 1e-3 in both.
set -eo pipefail
export HF_HUB_OFFLINE=1
ROOT_DIR=/home/josh/lerobot
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "E68-A3 joint-preparation chain started on $(hostname) at $(date -u)"

export ARM_TAG=e68a3_jointprep_lr1e-4_merged6x2_e468101416_v579111315_anchor040_sep8_prepass
DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
STAGE1_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model"
AUDIT_SH="$ROOT_DIR/job_scripts/nebius/libero_90/probes/audit_heldout_routing.sh"
J_RUN=libero_90_pi05_jointA10k_${ARM_TAG}          # == the common body's A_RUN for GRAD_TAG=ARM_TAG
J_OUT="$ROOT_DIR/outputs/train/$J_RUN"
J_STEPS=10000; J_SAVE_FREQ=${J_SAVE_FREQ:-5000}
J_FINAL="$J_OUT/checkpoints/$(printf '%06d' "$J_STEPS")/pretrained_model"
AUDIT_RUN=audit_heldout_${ARM_TAG}_10k
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$STAGE1_CKPT" ] || { echo "ERROR: stage-1 checkpoint missing"; exit 1; }

# ---- stage J: joint routers + values, 10k, frozen backbone ----
# = joint_rwarmup_common.sh's command with the deltas marked <<J>>.
joint_phase () {
  lerobot-train \
    --policy.path="$STAGE1_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$J_RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_90 \
    --dataset.root="$DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_90 \
    --output_dir="$J_OUT" \
    --save_freq=$J_SAVE_FREQ \
    --steps=$J_STEPS \
    --batch_size=$1 \
    --gradient_accumulation_steps=$2 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=4 \
    --eval_freq=20000 \
    --log_freq=200 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.train_router_only=false \
    --policy.train_memory_only=true \
    --policy.freeze_memory_router=false \
    --policy.optimizer_lr=1e-4 \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=40000 \
    --job_name="$J_RUN" \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --wandb.disable_artifact=true \
    --policy.gradient_checkpointing=${3:-false} \
    --policy.memory_layers=true \
    --policy.memory_layer.enabled=true \
    --policy.memory_layer.memory_only=false \
    --policy.memory_layer.layers='[4,6,8,10,14,16]' \
    --policy.memory_layer.mem_n_keys=256 \
    --policy.memory_layer.lora_rank=2 \
    --policy.memory_layer.mem_knn=36 \
    --policy.memory_layer.routing_loss_topk=36 \
    --policy.memory_layer.vlm_layers='[5,7,9,11,13,15]' \
    --policy.memory_layer.vlm_mem_n_keys=256 \
    --policy.memory_layer.vlm_lora_rank=2 \
    --policy.memory_layer.vlm_mem_knn=16 \
    --policy.memory_layer.vlm_text_span=200 \
    --policy.memory_layer.vlm_router_pool=anchored \
    --policy.memory_layer.vlm_router_pool_weights='[1.0,0.5]' \
    --policy.memory_layer.vlm_route_once=false \
    --policy.memory_layer.vlm_image_regions=0 \
    --policy.memory_layer.vlm_image_pool_weights='[1.0,0.5]' \
    --policy.memory_layer.router_only_fast=false \
    --policy.memory_layer.use_frozen_base_input_features=true \
    --policy.memory_layer.frozen_prepass=true \
    --policy.memory_layer.share_groups='[[4,6],[8,10]]' \
    --policy.memory_layer.vlm_share_groups='[[5,7],[9,11],[13,15]]' \
    --policy.memory_layer.log_usage=true \
    --policy.memory_layer.aggregate_usage=true \
    --policy.memory_layer.mem_heads=4 \
    --policy.memory_layer.mem_k_dim=512 \
    --policy.memory_layer.value_fixed_lr=0.001 \
    --policy.memory_layer.memory_lr=0.001 \
    --policy.memory_layer.lang_to_query=false \
    --policy.memory_layer.expert_anchor_pool=text \
    --policy.memory_layer.expert_anchor_weight=0.40 \
    --policy.memory_layer.fuse_method=film \
    --policy.memory_layer.embedding_model=all-mpnet-base-v2 \
    --policy.memory_layer.value_type=lora \
    --policy.memory_layer.contrastive_method=sample \
    --policy.memory_layer.contrastive_loss_weight=0.05 \
    --policy.memory_layer.contrastive_margin=0.0 \
    --policy.memory_layer.contrastive_query_queue=512 \
    --policy.memory_layer.routing_intra_task_locality_weight=0 \
    --policy.memory_layer.routing_inter_task_separation_weight=8.0 \
    --policy.memory_layer.routing_query_queue=512
}
joint_phase_resume () { lerobot-train --resume=true --config_path="$1"; }
J_LADDER="${J_LADDER:-8:4:false,4:8:false,8:4:true,4:8:true}"
if [ -d "$J_FINAL" ]; then
  echo "[J] final checkpoint exists - skipping."
else
  J_PARTIAL=$({ ls -d "$J_OUT"/checkpoints/[0-9]*/pretrained_model/train_config.json 2>/dev/null || true; } | sort | tail -1)   # `|| true`: under set -e/pipefail a no-match ls (rc 2) would silently abort
  if [ -n "$J_PARTIAL" ]; then
    echo "[J] RESUMING from $J_PARTIAL"
    joint_phase_resume "$J_PARTIAL" || { echo "ERROR: J resume failed - NOT wiping $J_OUT"; exit 1; }
  else
    ok=0
    for rung in ${J_LADDER//,/ }; do
      IFS=: read -r rb ra rc <<< "$rung"
      echo "[J] rung: bs=$rb accum=$ra grad_ckpt=$rc"
      if joint_phase "$rb" "$ra" "$rc"; then ok=1; break; fi
      if ls -d "$J_OUT"/checkpoints/[0-9]*/pretrained_model >/dev/null 2>&1; then
        echo "[J] rung failed AFTER a periodic checkpoint - not VRAM; aborting (relaunch resumes)."; exit 1
      fi
      echo "[J] rung failed before any checkpoint (treating as VRAM) - wiping and trying next rung"
      rm -rf "$J_OUT"
    done
    [ "$ok" = 1 ] || { echo "ERROR: all J_LADDER rungs failed"; exit 1; }
  fi
fi
[ -d "$J_FINAL" ] || { echo "ERROR: J finished but final checkpoint missing"; exit 1; }
[ -e "$J_OUT/checkpoints/last" ] || ln -sfn "$(printf '%06d' "$J_STEPS")" "$J_OUT/checkpoints/last"

# ---- stage J-audit: held-out routing audit on the JOINT checkpoint (informational, part of the result) ----
if [ "$(ls $ROOT_DIR/outputs/train/$AUDIT_RUN/memory_by_task/*.json 2>/dev/null | wc -l)" -ge 10 ]; then
  echo "[J-audit] already complete - skipping."
else
  AUDIT_BS=8 AUDIT_STEPS=400 bash "$AUDIT_SH" "$J_FINAL" "$AUDIT_RUN" || echo "[J-audit] AUDIT FAILED (non-fatal; rerun manually)"
fi
python scripts/vla_analysis/vlm_audit_analysis.py "$AUDIT_RUN" 5,7,9,11,13,15 65536 vlm || true
python scripts/vla_analysis/vlm_audit_analysis.py "$AUDIT_RUN" 4,6,8,10,14,16 65536 expert || true

# ---- stage B: 5-task sequential, routers frozen, C-config verbatim ----
export WARM_RUN=libero_90_pi05_jointwarm10k_merged6x2_e468101416_v579111315_anchor040_sep8_prepass  # unused: stage A is skipped
export GRAD_TAG=$ARM_TAG
export SEQ_EXTRA_ARGS="--policy.freeze_memory_router=true"
export SEQ_TOP_T=3072
export SEQ_VALUE_LR=0.002
export SEQ_VALUE_LR_END=0.0002
export SEQ_PROTECT_UNORM=corefrac
export SEQ_RUN=libero_10_seq5_jw_e68a3_jointprep_lr1e-4_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
export SEQ_LADDER="${SEQ_LADDER:-8:4:false,4:8:false,8:4:true,4:8:true}"
source "$SCRIPT_DIR/joint_aphase_seq5_common.sh"
echo "E68-A3 joint-preparation chain COMPLETE at $(date -u)"
