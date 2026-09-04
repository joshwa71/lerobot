#!/bin/bash
# E66 post-landing (Josh, 1 Sep: "on E66-DONE, the 4-seed campaign is the next step").
# 4-seed retention row on the PARAMETER-MATCHED naive sequential LoRA r=1216/a=304
# (2.6806B added, -0.12% vs our 2.6837B memory) after 10 tasks x 5,000 steps.
#
# Instrument is IDENTICAL to every comparator row (E63 stage 3, the naive r512 foil):
# all ten envs, 25 episodes, 4 paired seeds 1000/2000/3000/4000, vec-batched bs13.
# Comparators: ours merged6x2 10-task 65.1 / specialists 63.7 / multitask-LoRA-10 53.2 /
# naive seq-LoRA r512 (852M) 9.7.
# PRE-REGISTERED: expect collapse in the r512 band; >~20 would be the surprise that
# needs explaining before it goes in the paper.
#
# VRAM: training needed bs8 (r=1216 on 162 vision modules), but eval is inference-only
# with no stored activations, so bs13 should hold as it did for r512. Falls back to
# bs8 then bs4 if the rung OOMs before writing its output.
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e66_seeds.log
exec >> "$LOG" 2>&1
echo "=== E66 seeds queue started $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

E66=$ROOT/outputs/train/libero_10_seq10_naive_lora_r1216_a304_paramatched_steps5k
OUT=$ROOT/outputs/analysis/e60/seeds_naive10_paramatched_r1216.json
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
ALL10='[4,6,9,2,7,0,8,1,3,5]'

[ -d "$E66/checkpoints/050000/pretrained_model" ] || { echo "[e66-seeds] ERROR: final checkpoint missing"; exit 1; }
if [ -f "$OUT" ]; then echo "[e66-seeds] row exists - nothing to do."; echo "E66-SEEDS-DONE"; exit 0; fi

ok=0
for bs in 13 8 4; do
  echo "[e66-seeds] 4-seed campaign, eval.batch_size=$bs $(date -u)"
  CAMP_SEEDS="1000,2000,3000,4000" CAMP_TAG=naive10_paramatched_r1216 CAMP_OUT=$OUT \
  python scripts/vla_analysis/eval_seeds_campaign.py \
    --policy.path="$E66/checkpoints/050000/pretrained_model" \
    --policy.dtype=bfloat16 --policy.use_peft=true \
    --env.type=libero --env.task=libero_10 --env.task_ids="$ALL10" \
    --rename_map="$RENAME" \
    --eval.batch_size=$bs --eval.n_episodes=25 \
    --seed=1000 --output_dir=/tmp/camp_e66_paramatched \
    && { ok=1; break; }
  if [ -f "$OUT" ]; then echo "[e66-seeds] failed AFTER writing output - not VRAM; aborting."; exit 1; fi
  echo "[e66-seeds] bs=$bs failed before any output (treating as VRAM) - next rung"
done
[ "$ok" = 1 ] || { echo "[e66-seeds] ERROR: all rungs failed"; exit 1; }
echo "=== E66 SEEDS COMPLETE $(date -u) ==="
echo "E66-SEEDS-DONE"
