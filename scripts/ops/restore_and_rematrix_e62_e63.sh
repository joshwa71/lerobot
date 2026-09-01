#!/bin/bash
# E65 add-24 (Josh, 1 Sep): recompute the E62 (5-task) and E63 (10-task) sim MSE forgetting matrices
# with the FIXED shared-table loader (add-16: the old `".mlp.mem.slot_" in k` filter loaded 10 of 14
# slot tensors, leaving the (8,10) expert and (9,11) VLM storages frozen at the first checkpoint =>
# both matrices UNDER-REPORT forgetting). LOCAL script — the checkpoints live on the desk PC's cold
# drive and must be pushed back to the VM (measured upload 13.4 MB/s, ~6h for both runs).
#
# Stages, each skip-guarded:
#   1. rsync cold -> VM for the two runs (weights only; training_state was pruned at archival and is
#      not needed — the matrix reads slot tensors from pretrained_model/model.safetensors).
#   2. verify each transfer (rsync -aHc --dry-run zero itemized changes + du -sb byte-exact).
#   3. run mse_matrix2.py on the VM for each (5 and 10 checkpoints), writing NEW report files
#      alongside the originals; the pre-fix numbers stay in the log (add-16) for the record.
# The VM copies are left in place afterwards for inspection; cold retains the authoritative copy.
#
# LAUNCH DETACHED:  nohup setsid bash scripts/ops/restore_and_rematrix_e62_e63.sh >/dev/null 2>&1 &
set -uo pipefail
COLD=/media/josh/Backup/memory-models
VM=nebius-spot
VMBASE=/home/josh/lerobot/outputs/train
LOG=$COLD/_rematrix_e62_e63.log
E62=libero_10_seq5_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
E63=libero_10_seq10_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k

say () { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$LOG"; }

push () {   # <run dir name>
  local c=$1
  if ssh -o BatchMode=yes "$VM" "[ -d $VMBASE/$c/checkpoints ]" 2>/dev/null; then
    local nvm ncold
    nvm=$(ssh -o BatchMode=yes "$VM" "ls -d $VMBASE/$c/checkpoints/[0-9]* 2>/dev/null | wc -l")
    ncold=$(ls -d "$COLD/$c"/checkpoints/[0-9]* 2>/dev/null | wc -l)
    if [ "$nvm" = "$ncold" ]; then say "[skip-push] $c already on VM ($nvm ckpts)"; return 0; fi
  fi
  say "=== push $c ($(du -sh "$COLD/$c" | cut -f1)) ==="
  local ok=0
  for a in 1 2 3; do
    rsync -aH --partial --timeout=300 "$COLD/$c" "$VM:$VMBASE/" >> "$LOG" 2>&1 && { ok=1; break; }
    say "[retry $a] $c"; sleep 60
  done
  [ "$ok" = 1 ] || { say "FAIL-PUSH $c"; return 1; }
  local vout vrc delta cold_b vm_b
  vout=$(rsync -aHc --dry-run --itemize-changes "$COLD/$c" "$VM:$VMBASE/" 2>/dev/null); vrc=$?
  if [ "$vrc" -ne 0 ]; then delta=999; else delta=$(printf '%s\n' "$vout" | grep -c '^[<>ch]'); true; delta=${delta:-999}; fi
  cold_b=$(du -sb "$COLD/$c" | cut -f1)
  vm_b=$(ssh -o BatchMode=yes "$VM" "du -sb $VMBASE/$c | cut -f1")
  if [ "$delta" = "0" ] && [ -n "$vm_b" ] && [ "$cold_b" = "$vm_b" ]; then
    say "[verified] $c ($vm_b bytes)"; return 0
  fi
  say "FAIL-VERIFY $c delta=$delta cold=$cold_b vm=$vm_b"; return 1
}

matrix () {   # <run dir name> <steps csv> <first ckpt> <tag>
  local c=$1 steps=$2 first=$3 tag=$4
  say "=== matrix $tag ==="
  ssh -o BatchMode=yes "$VM" "cd /home/josh/lerobot && \
    source /home/josh/miniforge3/etc/profile.d/conda.sh && conda activate lerobot-memory-updated && \
    export MUJOCO_GL=osmesa TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 && unset DISPLAY && \
    mkdir -p outputs/analysis/e65_rematrix && \
    MSEMAT_RUN_DIR=$VMBASE/$c MSEMAT_STEPS='$steps' \
    MSEMAT_OUT=outputs/analysis/e65_rematrix/mse_matrix_${tag}_FIXED.jsonl \
    python scripts/vla_analysis/mse_matrix2.py \
      --policy.path=$VMBASE/$c/checkpoints/$first/pretrained_model \
      --policy.empty_cameras=1 --policy.dtype=bfloat16 --policy.gradient_checkpointing=false \
      --policy.normalization_mapping='{\"VISUAL\":\"IDENTITY\",\"STATE\":\"MEAN_STD\",\"ACTION\":\"MEAN_STD\"}' \
      --dataset.repo_id=libero_10 --dataset.root=/home/josh/lerobot/outputs/libero_10 \
      --rename_map='{\"observation.images.image\":\"observation.images.base_0_rgb\",\"observation.images.image2\":\"observation.images.left_wrist_0_rgb\"}' \
      --env.type=libero --env.task=libero_10 \
      --output_dir=outputs/analysis/e65_rematrix/out_${tag} \
      --steps=200000 --batch_size=32 --num_workers=4 \
      --online_task_ids='$5' --online_steps_per_task=5000 \
      --ds_to_env_map_json='{\"0\":4,\"1\":6,\"2\":9,\"3\":2,\"4\":7,\"5\":0,\"6\":8,\"7\":1,\"8\":3,\"9\":5}' \
      --wandb.enable=false --job_name=rematrix_${tag}" >> "$LOG" 2>&1 \
    && say "[matrix-done] $tag" || say "FAIL-MATRIX $tag"
}

say "=== rematrix started (upload ~13.4 MB/s; E62 102G + E63 204G) ==="
push "$E62" && matrix "$E62" "005000,010000,015000,020000,025000" 005000 e62_seq5 "[0,1,2,3,4]"
push "$E63" && matrix "$E63" "005000,010000,015000,020000,025000,030000,035000,040000,045000,050000" 005000 e63_seq10 "[0,1,2,3,4,5,6,7,8,9]"
say "=== rematrix done | VM: $(ssh -o BatchMode=yes "$VM" 'df -h /home/josh | tail -1') ==="
echo "REMATRIX-DONE"
