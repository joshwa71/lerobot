#!/bin/bash
# E65 REAL-WORLD (WidowX AI) chain — shared environment. Sourced by every rw_* script.
# Realworld DUPLICATE of the libero_90/staged conventions; the LIBERO scripts are untouched.
#
# Dataset contract (built by scripts/vla_analysis/realworld/build_rw_split.sh from the
# 20-task pool outputs/realworld_all_tasks):
#   RW_PRETRAIN_ROOT  the 15-task pretrain split  (stage-1 base finetune, router warm-up, A-phase)
#   RW_SEQ_ROOT       the 5-task held-out split, task_index 0..4 IN SEQUENTIAL ORDER
#                     (held-out routing audit + sequential adaptation)
# Both are LeRobot v3.0 datasets: observation.images.{cam_high,cam_wrist} (480x640, 2 real
# cams + pi05's empty slots via --policy.empty_cameras=1), 7-D joint-position state/action,
# 30 fps. NOTHING here constructs a simulator: lerobot-train stages run --eval_freq=0 with no
# --env.*; the sequential stage runs --eval.type=loss (the realworld-E3 forgetting-matrix
# instrument). Rollout-based instruments (4-seed campaigns, harvest bank) have no real-world
# analogue — the robot evals are Josh's.
#
# SMOKE=1: every stage at 6 steps, run names prefixed _smoke_, wandb off, audit 2 steps x bs4,
# loss-eval 2 batches. Exercises the full code path on the real datasets in ~15 min.
ROOT_DIR=/home/josh/lerobot
RW_TAG=${RW_TAG:-v5}
RW_PRETRAIN_ROOT=${RW_PRETRAIN_ROOT:-$ROOT_DIR/outputs/realworld_pretrain_${RW_TAG}}
RW_SEQ_ROOT=${RW_SEQ_ROOT:-$ROOT_DIR/outputs/realworld_seq_${RW_TAG}}
RW_PRETRAIN_ID=${RW_PRETRAIN_ID:-$(basename "$RW_PRETRAIN_ROOT")}
RW_SEQ_ID=${RW_SEQ_ID:-$(basename "$RW_SEQ_ROOT")}
RW_SEQ_TASK_IDS=${RW_SEQ_TASK_IDS:-[0,1,2,3,4]}
RW_N_SEQ=$(echo "$RW_SEQ_TASK_IDS" | tr -d '[] ' | awk -F, '{print NF}')
# famIoU family pairs in SEQ task ids, e.g. "1-3" (INFORMATIONAL — the gate is bg-first, E59)
RW_FAMILY=${RW_FAMILY:-}
RW_RENAME_MAP='{"observation.images.cam_high":"observation.images.base_0_rgb","observation.images.cam_wrist":"observation.images.left_wrist_0_rgb"}'
RW_NORM_MAP='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
# pinned pi05_base snapshot (the E31 / stage-1 base)
PI05_BASE=${PI05_BASE:-/home/josh/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30}

SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
  RUN_PREFIX=_smoke_; WANDB=false
  S1_STEPS=6; S1_SAVE=6; WARM_STEPS=6; A_STEPS=6; SEQ_STEPS=6
  AUDIT_BS=${AUDIT_BS:-4}; AUDIT_STEPS=${AUDIT_STEPS:-2}; EVAL_LOSS_NB=2
else
  RUN_PREFIX=; WANDB=${WANDB:-true}
  S1_STEPS=50000; S1_SAVE=10000; WARM_STEPS=10000; A_STEPS=10000; SEQ_STEPS=5000
  AUDIT_BS=${AUDIT_BS:-8}; AUDIT_STEPS=${AUDIT_STEPS:-400}; EVAL_LOSS_NB=${EVAL_LOSS_NB:-20}
fi
STAGE1_RUN=${STAGE1_RUN:-realworld_${RW_TAG}_pi05_base_nomem_50k}
STAGE1_OUT="$ROOT_DIR/outputs/train/${RUN_PREFIX}${STAGE1_RUN}"
STAGE1_CKPT="$STAGE1_OUT/checkpoints/last/pretrained_model"

export ROOT_DIR RW_TAG RW_PRETRAIN_ROOT RW_SEQ_ROOT RW_PRETRAIN_ID RW_SEQ_ID RW_SEQ_TASK_IDS RW_N_SEQ \
       RW_FAMILY RW_RENAME_MAP RW_NORM_MAP PI05_BASE SMOKE RUN_PREFIX WANDB \
       S1_STEPS S1_SAVE WARM_STEPS A_STEPS SEQ_STEPS AUDIT_BS AUDIT_STEPS EVAL_LOSS_NB \
       STAGE1_RUN STAGE1_OUT STAGE1_CKPT
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT_DIR"
[ -d "$RW_PRETRAIN_ROOT/meta" ] || { echo "ERROR: pretrain dataset missing: $RW_PRETRAIN_ROOT"; exit 1; }
[ -d "$RW_SEQ_ROOT/meta" ] || { echo "ERROR: sequential dataset missing: $RW_SEQ_ROOT"; exit 1; }
echo "[rw_env] tag=$RW_TAG smoke=$SMOKE pretrain=$RW_PRETRAIN_ROOT seq=$RW_SEQ_ROOT ids=$RW_SEQ_TASK_IDS (n=$RW_N_SEQ) family='${RW_FAMILY}' HEAD=$(git rev-parse --short HEAD 2>/dev/null)"
