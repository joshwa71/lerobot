#!/bin/bash
# E67 EVAL queue (VM-side; unit e67-eval). Runs alongside e67-train: every 10 min it tries the
# retention-triangle rows whose boundary checkpoints exist (skip-guarded per row) for RETAIN then
# O-LoRA, gated on >= GATE_FREE_MIB of free VRAM so the training unit is never starved, and once a
# chain has all 10 boundaries it runs that chain's loss-drift matrix (dense / adapter-swap).
# Exits with E67-EVAL-DONE when 20 rows + 2 matrices exist.
set -uo pipefail
ROOT=/home/josh/lerobot
GATE_FREE_MIB=${GATE_FREE_MIB:-45000}
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
A=$ROOT/outputs/analysis/e67; mkdir -p $A
RENAME='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
NORM='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
RET=$ROOT/outputs/train/libero_10_seq10_retain_a05_fullft_steps5k
OLO=$ROOT/outputs/train/libero_10_seq10_olora_r64_a16_lam05_steps5k
STEPS=005000,010000,015000,020000,025000,030000,035000,040000,045000,050000
say(){ echo "[e67-eval] $* $(date -u +%H:%M:%SZ)"; }
rows(){ ls $A/seeds_tri_$1_b*.json 2>/dev/null | wc -l; }
done_tasks(){ python3 -c "import json;print(json.load(open('$1'))['completed_tasks'])" 2>/dev/null || echo 0; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
echo "=== E67 EVAL QUEUE START $(date -u) (gate ${GATE_FREE_MIB} MiB free) ==="
while true; do
  GATE_FREE_MIB=$GATE_FREE_MIB bash scripts/baselines/run_baseline_triangle.sh retain 2>&1 | grep -v "exists - skipping\|checkpoint .* missing - skipping"
  GATE_FREE_MIB=$GATE_FREE_MIB bash scripts/baselines/run_baseline_triangle.sh olora  2>&1 | grep -v "exists - skipping\|checkpoint .* missing - skipping"
  # drift matrices once a chain is complete (dense: strict full reload; olora: adapter swap)
  MR=$A/mse_matrix_retain10_a05.jsonl
  if [ "$(done_tasks $RET/retain_state/progress.json)" = "10" ] && [ "$(grep -c '^{' $MR 2>/dev/null || echo 0)" -lt 10 ]; then
    while [ "$(free_mib)" -lt "$GATE_FREE_MIB" ]; do sleep 120; done
    say "RETAIN dense loss matrix"; rm -f "$MR"; rm -rf $A/out_msemat_retain
    MSEMAT_RUN_DIR=$RET MSEMAT_STEPS=$STEPS MSEMAT_TASKS=0,1,2,3,4,5,6,7,8,9 MSEMAT_OUT=$MR \
    python scripts/baselines/mse_matrix_dense.py --policy.path=$RET/checkpoints/005000/pretrained_model \
      --policy.empty_cameras=1 --policy.dtype=bfloat16 --policy.gradient_checkpointing=false --policy.normalization_mapping="$NORM" \
      --dataset.repo_id=libero_10 --dataset.root=$ROOT/outputs/libero_10 --rename_map="$RENAME" \
      --output_dir=$A/out_msemat_retain --steps=50000 --batch_size=32 --num_workers=4 \
      --online_task_ids='[0,1,2,3,4,5,6,7,8,9]' --online_steps_per_task=5000 --wandb.enable=false --job_name=e67_msemat_retain \
      && say "RETAIN matrix done ($(grep -c '^{' $MR) rows)" || say "RETAIN matrix FAILED"
  fi
  MO=$A/mse_matrix_olora10_r64.jsonl
  if [ "$(done_tasks $OLO/olora_state/progress.json)" = "10" ] && [ "$(grep -c '^{' $MO 2>/dev/null || echo 0)" -lt 10 ]; then
    while [ "$(free_mib)" -lt "$GATE_FREE_MIB" ]; do sleep 120; done
    say "O-LoRA adapter-swap loss matrix"; rm -f "$MO"; rm -rf $A/out_msemat_olora
    MSEMAT_RUN_DIR=$OLO MSEMAT_STEPS=$STEPS MSEMAT_TASKS=0,1,2,3,4,5,6,7,8,9 MSEMAT_OUT=$MO \
    python scripts/vla_analysis/mse_matrix_peft.py --policy.path=$OLO/checkpoints/005000/pretrained_model --policy.use_peft=true \
      --policy.empty_cameras=1 --policy.dtype=bfloat16 --policy.gradient_checkpointing=false --policy.normalization_mapping="$NORM" \
      --dataset.repo_id=libero_10 --dataset.root=$ROOT/outputs/libero_10 --rename_map="$RENAME" \
      --output_dir=$A/out_msemat_olora --steps=50000 --batch_size=32 --num_workers=4 \
      --online_task_ids='[0,1,2,3,4,5,6,7,8,9]' --online_steps_per_task=5000 --tfidf_enable=false --wandb.enable=false --job_name=e67_msemat_olora \
      && say "O-LoRA matrix done ($(grep -c '^{' $MO) rows)" || say "O-LoRA matrix FAILED"
  fi
  nr=$(rows retain10_a05); no=$(rows olora10_r64)
  mr=$(grep -c '^{' $MR 2>/dev/null || echo 0); mo=$(grep -c '^{' $MO 2>/dev/null || echo 0)
  say "status: retain rows $nr/10 matrix $mr/10 | olora rows $no/10 matrix $mo/10"
  if [ "$nr" = "10" ] && [ "$no" = "10" ] && [ "$mr" -ge 10 ] && [ "$mo" -ge 10 ]; then echo "E67-EVAL-DONE"; exit 0; fi
  sleep 600
done
