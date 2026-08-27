#!/usr/bin/env bash
# Tier-1 held-out-set selection: task geometry of the WidowX 20-task pool under a frozen
# pi05 checkpoint (default: raw pi05_base — the real stage-1 does not exist until the
# hold-out is chosen; probes RANK, never veto — E45/E49).
#
#   CKPT=<pretrained_model dir> TAG=<name> ./run_rw_task_geometry.sh
#
# Outputs: outputs/analysis/realworld/task_geometry_<TAG>.{json,npz} + geom_<TAG>.log
# Rank locally with: python scripts/vla_analysis/realworld/rank_heldout_subsets_rw.py <json>
set -euo pipefail

ROOT=/home/josh/lerobot
# pinned pi05_base snapshot (the E31/stage-1 base; the other cached snapshot 7de663 is not the one the project uses)
CKPT=${CKPT:-/home/josh/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30}
DATA=${DATA:-$ROOT/outputs/realworld_all_tasks}
REPO_ID=${REPO_ID:-realworld_all_tasks}
TAG=${TAG:-pi05base}
OUTD=$ROOT/outputs/analysis/realworld
mkdir -p "$OUTD"
exec > >(tee -a "$OUTD/geom_${TAG}.log") 2>&1
echo "[geom-run] $(date -u +%FT%TZ) ckpt=$CKPT data=$DATA tag=$TAG"

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT"

SCRATCH="$OUTD/_scratch_geom_${TAG}_$$"
OUT_JSON="$OUTD/task_geometry_${TAG}.json" OUT_NPZ="$OUTD/task_geometry_${TAG}.npz" \
python scripts/vla_analysis/realworld/probe_task_geometry_rw.py \
  --policy.path="$CKPT" \
  --policy.dtype=bfloat16 \
  --policy.empty_cameras=1 \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATA" \
  --rename_map='{"observation.images.cam_high":"observation.images.base_0_rgb","observation.images.cam_wrist":"observation.images.left_wrist_0_rgb"}' \
  --output_dir="$SCRATCH" \
  --online_task_ids='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]' \
  --wandb.enable=false
rm -rf "$SCRATCH"
echo "[geom-run] $(date -u +%FT%TZ) GEOM-RUN-DONE"
