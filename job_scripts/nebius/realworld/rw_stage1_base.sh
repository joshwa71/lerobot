#!/bin/bash
# Stage 1 (base competence): full finetune of pi05_base on the real-world PRETRAIN split,
# NO memory. Sourced by rw_merged6x2_full_chain.sh after rw_env.sh.
#
# Realworld duplicate of the stage-1 half of libero_90/staged/stage1_base50k_stage2_probe10k.sh
# (= the E31 base recipe): bf16, full backbone, warmup 4k / decay 50k, pi05 default LR,
# effective batch 32, 50k steps. Deltas:
#   - no --env.* / no in-run eval (the LIBERO stage-1 ran the 50-ep zero-shot floor table at 50k;
#     the real-world zero-shot floor is measured on the robot)  -> --eval_freq=0
#   - --policy.push_to_hub=false (raw pi05_base config carries no repo_id)
#   - preemption-safe: save_freq 10000 + --resume from train_config.json (E60-add-6/7 pattern)
#   - VRAM ladder S1_LADDER, default rung 1 = the E60-add-7 measured winner bs8 x acc4 no-ckpt
#     (2.20 s/step, 92G) -> bs16 x acc2 + ckpt -> bs32 + ckpt (the E31 rung)
S1_LADDER=${S1_LADDER:-"8:4:false,16:2:true,32:1:true"}
S1_FINAL="$STAGE1_OUT/checkpoints/$(printf '%06d' "$S1_STEPS")"

stage1_fresh () {
  lerobot-train \
    --policy.path="$PI05_BASE" \
    --policy.empty_cameras=1 \
    --policy.dtype=bfloat16 \
    --policy.repo_id="outputs/train/${RUN_PREFIX}${STAGE1_RUN}" \
    --policy.push_to_hub=false \
    --policy.normalization_mapping="$RW_NORM_MAP" \
    --dataset.repo_id="$RW_PRETRAIN_ID" \
    --dataset.root="$RW_PRETRAIN_ROOT" \
    --rename_map="$RW_RENAME_MAP" \
    --output_dir="$STAGE1_OUT" \
    --save_freq=$S1_SAVE \
    --steps=$S1_STEPS \
    --batch_size=$1 \
    --gradient_accumulation_steps=$2 \
    --num_workers=8 \
    --eval_freq=0 \
    --log_freq=200 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=50000 \
    --job_name="${RUN_PREFIX}${STAGE1_RUN}" \
    --wandb.enable=$WANDB \
    --wandb.project=vla-memory \
    --wandb.disable_artifact=true \
    --policy.gradient_checkpointing=$3
}
stage1_resume () {
  lerobot-train --resume=true \
    --config_path="$STAGE1_OUT/checkpoints/last/pretrained_model/train_config.json" \
    --batch_size=$1 --gradient_accumulation_steps=$2 --policy.gradient_checkpointing=$3
}

echo "=============================================================="
echo "[stage1] BASE FINETUNE $RW_PRETRAIN_ID, no memory, $S1_STEPS steps -> ${RUN_PREFIX}${STAGE1_RUN}"
echo "=============================================================="
if [ -d "$S1_FINAL" ]; then
  echo "[stage1] final checkpoint exists - skipping."
else
  ok=0
  for rung in ${S1_LADDER//,/ }; do
    IFS=: read -r rb ra rc <<< "$rung"
    if [ -f "$STAGE1_OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
      echo "[stage1] RESUMING from $(readlink -f "$STAGE1_OUT/checkpoints/last") at bs=$rb accum=$ra ckpt=$rc"
      if stage1_resume "$rb" "$ra" "$rc"; then ok=1; break; fi
      echo "[stage1] resume rung bs=$rb accum=$ra failed - trying next rung"
      continue
    fi
    echo "[stage1] fresh start at bs=$rb accum=$ra ckpt=$rc"
    if stage1_fresh "$rb" "$ra" "$rc"; then ok=1; break; fi
    if ls -d "$STAGE1_OUT"/checkpoints/[0-9]* >/dev/null 2>&1; then
      echo "[stage1] rung failed AFTER a checkpoint was written - not a VRAM failure; aborting."; exit 1
    fi
    echo "[stage1] rung failed before any checkpoint (treating as VRAM) - wiping and trying next rung"
    rm -rf "$STAGE1_OUT"
  done
  [ "$ok" = 1 ] || { echo "ERROR: all S1_LADDER rungs failed"; exit 1; }
fi
[ -d "$STAGE1_CKPT" ] || { echo "ERROR: stage 1 finished but $STAGE1_CKPT does not exist"; exit 1; }
echo "[stage1] checkpoint: $STAGE1_CKPT"
