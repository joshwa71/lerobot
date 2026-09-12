#!/usr/bin/env bash
# E69 spot queue (nebius-spot, H200), v2 (Josh, 12 Sep 26 18:00 UK: replay baseline DROPPED in favour
# of the paper cell under the REVERSED task order — Codex's order-robustness point). Hands-free once
# A3's 5-task sequential (unit e68-a3) is COMPLETE:
#   1. the reversed-order 10-task chain (e69_seq10_reversed_merged6x2.sh; resumable per boundary);
#   2. its FINAL-ROW 4-seed campaign: all ten envs x 25 eps x seeds 1000/2000/3000/4000, bs13
#      (= the control's seeds_merged6x2.json instrument) -> outputs/analysis/e60/seeds_merged6x2_rev10.json.
# A3's triangle runs on nebius2 after the naive chain (queue there), NOT here.
set -uo pipefail
ROOT=/home/josh/lerobot
cd "$ROOT" || exit 1
A3_SEQ=$ROOT/outputs/train/libero_10_seq5_jw_e68a3_jointprep_lr1e-4_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
A3_LOG=$ROOT/outputs/e68/a3_train.log
REV_RUN=libero_10_seq10rev_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
OUT=$ROOT/outputs/analysis/e60/seeds_merged6x2_rev10.json
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
POLL=${POLL:-300}
say(){ echo "[e69-spot] $* $(date -u +%H:%M:%SZ)"; }
echo "=== E69 SPOT QUEUE v2 START $(date -u) (reversed-order paper cell) ==="
while true; do
  if ! systemctl is-active e68-a3 >/dev/null 2>&1 && [ -d "$A3_SEQ/checkpoints/025000/pretrained_model" ] \
     && grep -q "E68-A3 joint-preparation chain COMPLETE" "$A3_LOG" 2>/dev/null; then
    say "A3 sequential complete - GPU released"; break
  fi
  sleep "$POLL"
done
FINAL=$ROOT/outputs/train/$REV_RUN/checkpoints/050000/pretrained_model
if [ -d "$FINAL" ]; then say "reversed chain already complete"; else
  say "reversed-order chain: $REV_RUN"
  bash job_scripts/nebius/libero_90/staged/e69_seq10_reversed_merged6x2.sh >> "$ROOT/outputs/e69/rev10_train.log" 2>&1
  [ -d "$FINAL" ] || { say "reversed chain ended without checkpoints/050000 (relaunch resumes)"; echo "E69-REV10-FAIL"; exit 1; }
fi
echo "E69-REV10-CHAIN-DONE"
if [ -f "$OUT" ]; then say "final row exists"; else
  say "final-row 4-seed campaign (10 envs x 25 eps x 4 seeds)"
  source /home/josh/miniforge3/etc/profile.d/conda.sh; conda activate lerobot-memory-updated
  export MUJOCO_GL=osmesa; unset DISPLAY; export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
  ok=0
  for bs in 13 8 4; do
    CAMP_SEEDS="1000,2000,3000,4000" CAMP_TAG=merged6x2_rev10 CAMP_OUT=$OUT \
    python scripts/vla_analysis/eval_seeds_campaign.py \
      --policy.path="$FINAL" --policy.dtype=bfloat16 \
      --env.type=libero --env.task=libero_10 --env.task_ids="[4,6,9,2,7,0,8,1,3,5]" \
      --rename_map="$RENAME" --eval.batch_size=$bs --eval.n_episodes=25 --seed=1000 \
      --output_dir=/tmp/camp_merged6x2_rev10 >> "$ROOT/outputs/e69/rev10_campaign.log" 2>&1 && { ok=1; break; }
    [ -f "$OUT" ] && { say "campaign failed AFTER writing output - not VRAM; aborting"; break; }
    say "campaign bs=$bs failed before any output (treating as VRAM) - next rung"
  done
fi
if [ -f "$OUT" ]; then echo "E69-REV10-DONE"; exit 0; fi
say "final row missing"; echo "E69-REV10-FAIL"; exit 1
