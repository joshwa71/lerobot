#!/bin/bash
# E62 noise-arm landing battery + 4-seed row, gated on the e62-vnoise unit
# exiting (NB is-active exits nonzero for inactive — capture with || true).
# Battery: MSE matrix + jitter (vnoise run) -> harvest rescore TAG=vnoise05x
# (the redundancy arbiter: spec/succ Q4 vs merged6x2's 0.332) -> 4-seed
# campaign row (the recipe-decision instrument per the arm's pre-registration).
set -o pipefail
ROOT=/home/josh/lerobot
LOG=$ROOT/outputs/e62_vnoise_battery.log
exec >> "$LOG" 2>&1
echo "=== e62-vnoise battery: waiting on e62-vnoise $(date -u) ==="
while true; do
  st=$(systemctl is-active e62-vnoise 2>/dev/null) || true
  [ "$st" = "active" ] || [ "$st" = "activating" ] || break
  sleep 300
done
echo "=== e62-vnoise battery: arm exited (state=$st) — starting $(date -u) ==="
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

SEQ=libero_10_seq5_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_vnoise05x_steps5k
RD=$ROOT/outputs/train/$SEQ
SP_AN=$ROOT/outputs/analysis/e62
mkdir -p $SP_AN
# guard: only proceed if the final checkpoint exists (a crashed arm should not
# trigger a battery on a partial run)
if [ ! -d "$RD/checkpoints/025000/pretrained_model" ]; then
  echo "[battery] FINAL CHECKPOINT MISSING — arm did not complete; aborting"
  exit 1
fi

COMMON_ARGS=(
  --policy.empty_cameras=1 --policy.dtype=bfloat16
  --policy.gradient_checkpointing=false
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
  --dataset.repo_id=libero_10 --dataset.root="$ROOT/outputs/libero_10"
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
  --env.type=libero --env.task=libero_10
  --steps=200000 --batch_size=32 --num_workers=4
  --online_task_ids='[0,1,2,3,4]' --online_steps_per_task=5000
  --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'
  --wandb.enable=false
)

export MSEMAT_RUN_DIR=$RD MSEMAT_STEPS=005000,010000,015000,020000,025000 MSEMAT_OUT=$SP_AN/mse_matrix_vnoise05x.jsonl
python scripts/vla_analysis/mse_matrix2.py \
  --policy.path="$RD/checkpoints/005000/pretrained_model" \
  "${COMMON_ARGS[@]}" \
  --output_dir=$SP_AN/msemat_out_vnoise05x --job_name=msemat_vnoise05x \
  || echo "[battery] msemat FAILED"

export PROBE_RUN_DIR=$RD PROBE_CKPTS="t0:025000,t3:025000,t4:025000" PROBE_OUT=$SP_AN/probe_jitter_vnoise05x.jsonl PROBE_SWAP_SLOTS=1
python scripts/vla_analysis/probe_jitter.py \
  --policy.path="$RD/checkpoints/025000/pretrained_model" \
  "${COMMON_ARGS[@]}" \
  --output_dir=$SP_AN/jitter_out_vnoise05x --job_name=jitter_vnoise05x \
  || echo "[battery] jitter FAILED"

SP=$ROOT/outputs/analysis/e56_offtrail
BRUN_MBT=$ROOT/outputs/train/libero_10_seq5_jw_layermax_A_anchor040_sep8_nofilm_beta4corefrac_topt3072_lr2x_steps5k/memory_by_task/memory_usage_task_4.json
TAG=vnoise05x
if [ ! -f "$SP/chunks_$TAG.npz" ]; then
  rm -rf "$SP/score_out_$TAG"
  SCORE_HARVESTS=$SP/harv_B,$SP/harv_spec SCORE_OUT_DIR=$SP SCORE_TAG=$TAG \
  SCORE_SEEDS=4 SCORE_TASK=4 SCORE_DEMO_N=120 SCORE_FEAT_LAYER=4 \
  python scripts/vla_analysis/probe_offtrail_score.py \
    --policy.path="$RD/checkpoints/025000/pretrained_model" "${COMMON_ARGS[@]}" \
    --output_dir="$SP/score_out_$TAG" --job_name=offtrail_score_$TAG \
    || echo "[battery] rescore FAILED"
fi
REPORT_DIR=$SP REPORT_TAG_A=$TAG REPORT_TAG_B=spec_e7 \
REPORT_HARVESTS=$SP/harv_B,$SP/harv_spec \
REPORT_MBT=$BRUN_MBT REPORT_TASKKEY=task_4 \
REPORT_OUT=$SP/offtrail_e7_$TAG.jsonl \
python scripts/vla_analysis/probe_offtrail_report.py || echo "[battery] report FAILED"
echo "=== E62-VNOISE BATTERY COMPLETE $(date -u) ==="

# ---- 4-seed campaign row (the recipe-decision instrument) ----
export CAMP_SEEDS="1000,2000,3000,4000"
export CAMP_TAG=vnoise05x
export CAMP_OUT=$ROOT/outputs/analysis/e60/seeds_vnoise05x.json
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
if [ ! -f "$CAMP_OUT" ]; then
  python scripts/vla_analysis/eval_seeds_campaign.py \
    --policy.path="$RD/checkpoints/025000/pretrained_model" \
    --policy.dtype=bfloat16 \
    --env.type=libero --env.task=libero_10 --env.task_ids="[4,6,9,2,7]" \
    --rename_map="$RENAME" \
    --eval.batch_size=13 --eval.n_episodes=25 \
    --seed=1000 \
    --output_dir=/tmp/camp_vnoise05x \
    || echo "[FAIL] vnoise05x campaign"
fi
echo "=== E62-VNOISE SEEDS COMPLETE $(date -u) ==="
