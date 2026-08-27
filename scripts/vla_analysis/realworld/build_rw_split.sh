#!/usr/bin/env bash
# Build the real-world pretrain / sequential split from the 20-task WidowX pool (CPU only, VM).
#
#   HELDOUT="0,10,16,7,1" TAG=v5 [DROP_EPISODES="[548]"] bash build_rw_split.sh
#
# HELDOUT = pool task ids IN SEQUENTIAL ORDER (they become seq task_index 0..4 in that order);
# pretrain = every other pool task, in pool order. Steps:
#   1. drop degenerate episodes from the pool -> outputs/realworld_all_tasks_clean (skipped if present)
#   2. split the clean pool into per-task parts (existing split_dataset_by_task.py)
#   3. patch each part's task table to its single task (rw_split_tools.py patch-tasks) so that
#   4. merge_datasets.py renumbers contiguously IN MERGE ORDER: pretrain -> realworld_pretrain_$TAG,
#      seq (HELDOUT order) -> realworld_seq_$TAG
#   5. verify both (contiguous ids, data/table consistency, totals) + write the split manifest
# Outputs never overwrite: the script refuses to run if a destination exists.
set -euo pipefail
ROOT=/home/josh/lerobot
POOL=${POOL:-$ROOT/outputs/realworld_all_tasks}
CLEAN=${CLEAN:-$ROOT/outputs/realworld_all_tasks_clean}
TAG=${TAG:?set TAG (e.g. v5)}
HELDOUT=${HELDOUT:?set HELDOUT="a,b,c,d,e" (pool ids in sequential order)}
DROP_EPISODES=${DROP_EPISODES:-"[548]"}
PARTS=$ROOT/outputs/_rw_split_parts_${TAG}
PRE=$ROOT/outputs/realworld_pretrain_${TAG}
SEQ=$ROOT/outputs/realworld_seq_${TAG}
TOOLS=$ROOT/scripts/vla_analysis/realworld/rw_split_tools.py
export HF_HUB_OFFLINE=1
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd "$ROOT"
for d in "$PRE" "$SEQ" "$PARTS"; do [ -e "$d" ] && { echo "ERROR: $d exists - refusing to overwrite"; exit 1; }; done

# 1. clean pool
if [ -d "$CLEAN/meta" ]; then
  echo "[split] clean pool exists: $CLEAN"
else
  echo "[split] dropping episodes $DROP_EPISODES from $POOL -> $CLEAN"
  python src/lerobot/scripts/lerobot_edit_dataset.py \
    --repo_id realworld_all_tasks --root "$POOL" \
    --new_repo_id realworld_all_tasks_clean --new_root "$CLEAN" \
    --operation.type delete_episodes --operation.episode_indices "$DROP_EPISODES"
fi
python "$TOOLS" verify "$CLEAN" 20

# 2. per-task parts
echo "[split] splitting $CLEAN into per-task parts under $PARTS"
mkdir -p "$PARTS"
python src/lerobot/scripts/split_dataset_by_task.py --src_root "$CLEAN" --dst_root_parent "$PARTS"

# 3. single-task tables
for p in "$PARTS"/libero_task_*; do
  i=${p##*_}
  python "$TOOLS" patch-tasks "$p" "$i"
done

# 4. merge in order
IFS=, read -r -a HO <<< "$HELDOUT"
seq_srcs=(); for i in "${HO[@]}"; do seq_srcs+=("$PARTS/libero_task_$i"); done
pre_srcs=()
for p in $(ls -d "$PARTS"/libero_task_* | sort -t_ -k3 -n); do
  i=${p##*_}; skip=0
  for h in "${HO[@]}"; do [ "$h" = "$i" ] && skip=1; done
  [ $skip = 0 ] && pre_srcs+=("$p")
done
echo "[split] pretrain sources (${#pre_srcs[@]}): ${pre_srcs[*]##*/}"
echo "[split] seq sources (${#seq_srcs[@]}, in order): ${seq_srcs[*]##*/}"
python src/lerobot/scripts/merge_datasets.py --sources "${pre_srcs[@]}" --target "$PRE"
python src/lerobot/scripts/merge_datasets.py --sources "${seq_srcs[@]}" --target "$SEQ"

# 5. verify + manifest
python "$TOOLS" verify "$PRE" "${#pre_srcs[@]}"
python "$TOOLS" verify "$SEQ" "${#seq_srcs[@]}"
mkdir -p "$ROOT/outputs/analysis/realworld"
python "$TOOLS" manifest "$ROOT/outputs/analysis/realworld/split_manifest_${TAG}.json" "$CLEAN" "$PRE" "$SEQ" "$HELDOUT"
rm -rf "$PARTS"
echo "[split] DONE tag=$TAG pretrain=$PRE seq=$SEQ"
