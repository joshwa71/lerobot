#!/bin/bash
# E45 pooled-router chain — COMMON BODY (sourced by arm wrappers, which set ARM_TAG /
# POOL_MODE / POOL_W / C_WEIGHT / SEP_WEIGHT before sourcing this file).
#
# The E44 sweep failed structurally: per-token routing over the state-as-text sub-span
# routes on shared digit vocabulary (state-region famIoU 0.38-0.40 vs instruction
# 0.21-0.22; task signal at state positions = 0.11-0.13x their within-task variance,
# querystats probe). This build keys the state region on POOLED, RMS-normalized region
# means (vlm_router_pool): instruction positions keep per-token routing; every position
# from the ", State:" boundary shares one per-sample key. Value path stays per-position.
#
# Chain: warm-up 10k (router-only, aux losses) -> held-out audit -> gate (permissive:
# kill only on hard collapse/sprawl; morning reads decide) -> merge with the warmed
# EXPERT n256 bank (certificate famIoU 0.145) -> joint A-phase 10k (values both towers,
# routers frozen, frozen-base expert routing) -> e4 1-task plasticity probe (5k, C-config,
# 20-ep eval). Chunk probes run centrally in the morning.
#
# Prereqs on this box: stage-1 ckpt, outputs/libero_90 + outputs/libero_10, and the two
# shipped files under outputs/analysis/e44/: expert_bank_n256_rwarmup10k.safetensors +
# expert_bank_n256_config.json (scp'd from the base box; see log Entry 45).
set -eo pipefail
echo "E45 pooled-router chain [$ARM_TAG] (pool=$POOL_MODE w=$POOL_W c=$C_WEIGHT sep=$SEP_WEIGHT) started on $(hostname) at $(date)"
ROOT_DIR=/home/josh/lerobot
DATASET_ROOT="$ROOT_DIR/outputs/libero_90"
SEQ_DATASET_ROOT="$ROOT_DIR/outputs/libero_10"
AUDIT_SH="$ROOT_DIR/job_scripts/nebius/libero_90/probes/audit_heldout_routing.sh"
STAGE1_CKPT="$ROOT_DIR/outputs/train/libero_90_pi05_base_nomem_50k/checkpoints/last/pretrained_model"
BANK="$ROOT_DIR/outputs/analysis/e44/expert_bank_n256_rwarmup10k.safetensors"
BANK_CFG="$ROOT_DIR/outputs/analysis/e44/expert_bank_n256_config.json"
RUN=libero_90_pi05_vlm1516_pool_${ARM_TAG}_rwarmup10k_n256_r2_knn16_rq512
AUDIT_RUN=audit_heldout_vlmpool_${ARM_TAG}_10k
OUT="$ROOT_DIR/outputs/train/$RUN"
CKPT="$OUT/checkpoints/last/pretrained_model"
MERGED="$ROOT_DIR/outputs/train/${RUN}_MERGED/pretrained_model"
A_RUN=libero_90_pi05_exp8-14n256_vlm1516_pool_${ARM_TAG}_A10k
A_OUT="$ROOT_DIR/outputs/train/$A_RUN"
A_CKPT="$A_OUT/checkpoints/last/pretrained_model"
E4_RUN=libero_10_e4probe_pool_${ARM_TAG}_beta4_topt1536_steps5k
E4_OUT="$ROOT_DIR/outputs/train/$E4_RUN"
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$STAGE1_CKPT" ] || { echo "ERROR: stage-1 checkpoint missing"; exit 1; }
[ -f "$BANK" ] || { echo "ERROR: expert bank file missing ($BANK) - scp it from the base box"; exit 1; }
[ -f "$BANK_CFG" ] || { echo "ERROR: expert bank config missing ($BANK_CFG)"; exit 1; }

# ---------- stage 1: VLM router warm-up ----------
if [ -d "$CKPT" ]; then
  echo "[warmup] checkpoint exists - skipping."
else
  lerobot-train \
    --policy.path="$STAGE1_CKPT" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_90 \
    --dataset.root="$DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_90 \
    --output_dir="$OUT" \
    --save_freq=10000 \
    --steps=10000 \
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=4 \
    --eval_freq=20000 \
    --log_freq=200 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.train_router_only=true \
    --policy.optimizer_lr=1e-4 \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=40000 \
    --job_name="$RUN" \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --wandb.disable_artifact=true \
    --policy.gradient_checkpointing=false \
    --policy.memory_layers=true \
    --policy.memory_layer.enabled=true \
    --policy.memory_layer.memory_only=false \
    --policy.memory_layer.layers='[]' \
    --policy.memory_layer.vlm_layers='[15,16]' \
    --policy.memory_layer.vlm_mem_n_keys=256 \
    --policy.memory_layer.vlm_lora_rank=2 \
    --policy.memory_layer.vlm_mem_knn=16 \
    --policy.memory_layer.vlm_text_span=200 \
    --policy.memory_layer.vlm_router_pool="$POOL_MODE" \
    --policy.memory_layer.vlm_router_pool_weights="$POOL_W" \
    --policy.memory_layer.log_usage=true \
    --policy.memory_layer.aggregate_usage=true \
    --policy.memory_layer.mem_heads=4 \
    --policy.memory_layer.mem_k_dim=512 \
    --policy.memory_layer.value_fixed_lr=0.001 \
    --policy.memory_layer.memory_lr=0.001 \
    --policy.memory_layer.lang_to_query=false \
    --policy.memory_layer.value_type=lora \
    --policy.memory_layer.contrastive_method=sample \
    --policy.memory_layer.contrastive_loss_weight=$C_WEIGHT \
    --policy.memory_layer.contrastive_margin=0.0 \
    --policy.memory_layer.contrastive_query_queue=512 \
    --policy.memory_layer.routing_loss_topk=16 \
    --policy.memory_layer.routing_intra_task_locality_weight=0 \
    --policy.memory_layer.routing_inter_task_separation_weight=$SEP_WEIGHT \
    --policy.memory_layer.routing_query_queue=512
fi
[ -d "$CKPT" ] || { echo "ERROR: warmup finished but checkpoint missing"; exit 1; }

# ---------- stage 2: held-out audit + region-split probe ----------
if [ "$(ls $ROOT_DIR/outputs/train/$AUDIT_RUN/memory_by_task/*.json 2>/dev/null | wc -l)" -ge 10 ]; then
  echo "[audit] already complete - skipping."
else
  bash "$AUDIT_SH" "$CKPT" "$AUDIT_RUN" || echo "[audit] AUDIT FAILED (warmup checkpoint retained; rerun manually)"
fi
python scripts/vla_analysis/vlm_audit_analysis.py "$AUDIT_RUN" 15,16 65536 || true
mkdir -p outputs/analysis/e45
ARM="$ARM_TAG" OUT="$ROOT_DIR/outputs/analysis/e45/subspan_${ARM_TAG}.json" \
python scripts/vla_analysis/probe_subspan.py \
  --policy.path="$CKPT" \
  --policy.empty_cameras=1 --policy.dtype=bfloat16 \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --dataset.repo_id=libero_10 --dataset.root="$SEQ_DATASET_ROOT" \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
  --output_dir="$ROOT_DIR/outputs/train/_subspan_tmp_${ARM_TAG}" \
  --steps=1 --batch_size=8 --wandb.enable=false --job_name=subspan_${ARM_TAG} \
  --online_task_ids='[0,1,2,3,4,5,6,7,8,9]' --online_steps_per_task=1 --save_checkpoint=false \
  || echo "[subspan] probe failed (non-fatal)"

# Permissive overnight gate: kill ONLY on hard collapse or hard sprawl — mid-band results
# proceed to the A-phase/e4 probe (the morning decision instrument). famIoU semantics on
# pooled arms are palette-dominated (shared-key multiplicity), so strict E44 gates do not
# transfer; the subspan JSON is the region-resolved read.
python - "$ROOT_DIR/outputs/train/$AUDIT_RUN/vlm_audit_summary.json" <<'EOF' || { echo "GATE: HARD FAIL - chain stopped"; exit 1; }
import json, sys
d = json.load(open(sys.argv[1]))
effs = [v["effnum"] for k, v in d.items() if isinstance(v, dict)]
fams = [d.get("L15_famIoU"), d.get("L16_famIoU")]
collapse = min(effs) <= 100
sprawl = all(f is not None and f >= 0.45 for f in fams)
print(f"[gate] min effnum {min(effs):.0f}  famIoU {fams}  collapse={collapse} sprawl={sprawl}")
sys.exit(1 if (collapse or sprawl) else 0)
EOF

# ---------- stage 3: merge the warmed expert bank ----------
if [ -f "$MERGED/model.safetensors" ]; then
  echo "[merge] exists - skipping."
else
  python scripts/vla_analysis/merge_banks.py "$CKPT" "$BANK_CFG" "$BANK" "$MERGED"
fi

# ---------- stage 4: joint A-phase (values both towers, routers frozen) ----------
a_phase () {
  lerobot-train \
    --policy.path="$MERGED" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/$A_RUN" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id=libero_90 \
    --dataset.root="$DATASET_ROOT" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' \
    --env.type=libero \
    --env.task=libero_90 \
    --output_dir="$A_OUT" \
    --save_freq=10000 \
    --steps=10000 \
    --batch_size=$1 \
    --gradient_accumulation_steps=$2 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=4 \
    --eval_freq=20000 \
    --log_freq=200 \
    --policy.train_router_only=false \
    --policy.train_memory_only=true \
    --policy.freeze_memory_router=true \
    --policy.memory_layer.use_frozen_base_input_features=true \
    --policy.optimizer_lr=2.5e-5 \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=40000 \
    --job_name="$A_RUN" \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --wandb.disable_artifact=true \
    --policy.gradient_checkpointing=false
}
if [ -d "$A_CKPT" ]; then
  echo "[A-phase] checkpoint exists - skipping."
else
  echo "[A-phase] launching at bs32 (fallback bs16 x accum2 on failure)"
  a_phase 32 1 || { echo "[A-phase] bs32 failed - retrying bs16 x accum2"; rm -rf "$A_OUT"; a_phase 16 2; }
fi
[ -d "$A_CKPT" ] || { echo "ERROR: A-phase finished but checkpoint missing"; exit 1; }

# ---------- stage 5: e4 1-task plasticity probe (the decision instrument) ----------
if [ -d "$E4_OUT/checkpoints/005000" ]; then
  echo "[e4] final checkpoint exists - skipping."
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
    --output_dir="$E4_OUT" \
    --steps=200000 \
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --num_workers=8 \
    --eval.batch_size=1 \
    --eval.n_episodes=20 \
    --log_freq=200 \
    --wandb.enable=true \
    --wandb.project=vla-memory \
    --job_name="$E4_RUN" \
    --online_task_ids='[0]' \
    --online_steps_per_task=5000 \
    --policy.train_router_only=false \
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
    --memory_value_lr=0.001 \
    --memory_value_lr_end=0.0001 \
    --memory_value_scheduler_type=linear
fi
echo "E45 pooled-router chain [$ARM_TAG] COMPLETE at $(date)"
