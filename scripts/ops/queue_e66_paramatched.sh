#!/bin/bash
# E66 queue: wait for the E63 rematrix to release the GPU, smoke the parameter-matched wrapper
# (fatal on failure), then run it. VM-side; launch under systemd-run.
# Gate: the E63 corrected matrix file has all 10 rows, OR a 3h timeout (so a rematrix failure
# cannot block this indefinitely), AND the GPU is free.
set -uo pipefail
ROOT=/home/josh/lerobot
# systemd gives a bare environment: activate conda before ANY inline python (the wrappers do their
# own activation, but stage 0 below calls python directly). 1 Sep: this was missing and stage 0 died
# with `python: command not found`.
source /home/josh/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-memory-updated
cd $ROOT
export MUJOCO_GL=osmesa; unset DISPLAY
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "=== E66 QUEUE START $(date -u) ==="   # heartbeat slices the log from the LAST marker
W=$ROOT/job_scripts/nebius/baselines/naive_seq_lora_r1216_paramatched_10task.sh
M=$ROOT/outputs/analysis/e65_rematrix/mse_matrix_e63_seq10_FIXED.jsonl
say(){ echo "[e66] $* $(date -u +%H:%M:%SZ)"; }

# stage 0: redo the E63 corrected matrix over ALL TEN tasks (the first rerun set MSEMAT_TASKS to
# its 0-4 default, so only the front five were scored). Checkpoints are still on the VM.
E63=$ROOT/outputs/train/libero_10_seq10_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k
if [ "$(python3 -c "import json;print(len(json.loads(open('"'"'$M'"'"').readline())['"'"'per_task'"'"']))" 2>/dev/null || echo 0)" -lt 10 ]; then
  say "E63 matrix has <10 tasks per row - recomputing over all ten"
  rm -rf $ROOT/outputs/analysis/e65_rematrix/out_e63_all10; MTMP=$M.all10.tmp; rm -f "$MTMP"
  MSEMAT_RUN_DIR=$E63   MSEMAT_STEPS=005000,010000,015000,020000,025000,030000,035000,040000,045000,050000   MSEMAT_TASKS=0,1,2,3,4,5,6,7,8,9 MSEMAT_OUT=$M   python scripts/vla_analysis/mse_matrix2.py     --policy.path=$E63/checkpoints/005000/pretrained_model     --policy.empty_cameras=1 --policy.dtype=bfloat16 --policy.gradient_checkpointing=false     --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'     --dataset.repo_id=libero_10 --dataset.root=$ROOT/outputs/libero_10     --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'     --env.type=libero --env.task=libero_10     --output_dir=$ROOT/outputs/analysis/e65_rematrix/out_e63_all10     --steps=200000 --batch_size=32 --num_workers=4     --online_task_ids='[0,1,2,3,4,5,6,7,8,9]' --online_steps_per_task=5000     --ds_to_env_map_json='{"0":4,"1":6,"2":9,"3":2,"4":7,"5":0,"6":8,"7":1,"8":3,"9":5}'     --wandb.enable=false --job_name=rematrix_e63_all10 \
    && { mv "$MTMP" "$M"; say "E63 all-10 matrix done"; } || say "E63 all-10 matrix FAILED (non-fatal)"
fi

say "waiting for the GPU to be free"
end=$(( $(date +%s) + 10800 ))
while [ "$(date +%s)" -lt "$end" ]; do
  rows=$(grep -c '^{' "$M" 2>/dev/null || echo 0)
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${rows:-0}" -ge 10 ] && [ "${gpu:-0}" -lt 2000 ] && { say "E63 matrix complete ($rows rows), GPU free"; break; }
  sleep 120
done
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "${gpu:-0}" -ge 2000 ]; then say "GPU still busy (${gpu}MiB) after the wait - aborting rather than contending"; exit 1; fi

say "SMOKE (2 tasks x 20 steps) - checks the r1216 adapter builds, wraps and fits in VRAM"
if SMOKE=1 bash "$W"; then
  say "smoke OK"
  rm -rf $ROOT/outputs/train/smoke_naive_lora_r1216_paramatched
else
  say "SMOKE FAILED - not launching the full run"; echo "E66-SMOKE-FAIL"; exit 1
fi

say "full run: 10 tasks x 5,000 steps"
bash "$W" || { say "RUN FAILED (self-resuming on relaunch)"; echo "E66-RUN-FAIL"; exit 1; }
say "done"
echo "E66-DONE"
