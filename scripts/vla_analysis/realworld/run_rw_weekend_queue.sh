#!/bin/bash
# E65 WEEKEND QUEUE (Josh, 29 Aug 26) — runs unattended behind the arm-4 chain on nebius-spot.
#   0 wait for RW-CHAIN-DONE + rw-chain inactive (defensive; the rw-weekend bootstrap unit already waited
#     and git-pulled with the chain unit stopped, per CLAUDE.md 9.8)
#   1 landing battery on the arm-4 sequential (JIT_T=0,1,3,4: task 1 added to the jitter grid)
#   2 SMOKES of the two new LoRA wrappers (GPU free by now) — a failure STOPS the queue
#     (prints RW-WEEKEND-SMOKE-FAIL; the heartbeat then alerts instead of relaunching)
#   3 r64 LoRA specialists t0..t4 (rw_loraft_specialists_r64.sh) -> own-task MSE via mse_matrix_peft.py
#   4 r64 naive sequential LoRA (rw_naive_seq_lora_r64.sh, self-resuming) -> adapter-swap MSE matrix
#     + in-run / msemat reports (rw_matrix_report.py)
#   5 RW-WEEKEND-DONE only when specialists 5/5 + naive final exist; otherwise RW-WEEKEND-INCOMPLETE
#     (exit 1 -> the heartbeat relaunches; every stage is skip-guarded on its artifacts)
# Launched by the rw-weekend bootstrap (scripts/ops/heartbeat_rw_chain.sh relaunch_weekend);
# stdout/stderr -> outputs/rw_weekend_v5.log.  DRYRUN=1: print the plan + the wrappers' commands, no GPU.
set -o pipefail
source /home/josh/lerobot/job_scripts/nebius/realworld/rw_env.sh
DRYRUN=${DRYRUN:-0}
LORA_R=${LORA_R:-64}; LORA_ALPHA=${LORA_ALPHA:-16}; export LORA_R LORA_ALPHA
ARM=${ARM_TAG:-merged6x2_e468101416_v579111315_anchor030_pool1010_sep1_c100_prepass}
SEQ_RUN=${RW_SEQ_RUN:-${RUN_PREFIX}realworld_${RW_TAG}_seq${RW_N_SEQ}_jw_${ARM}_beta4corefrac_topt3072_lr2x_steps5k}
SP=${SP:-$ROOT_DIR/outputs/analysis/realworld/${RUN_PREFIX}e65}; mkdir -p "$SP"
CHAIN_LOG=$ROOT_DIR/outputs/${RUN_PREFIX}rw_chain_${RW_TAG}.log
BAT_LOG=$ROOT_DIR/outputs/${RUN_PREFIX}rw_battery_${RW_TAG}.log
RWD=${RWD:-$ROOT_DIR/job_scripts/nebius/realworld}   # overridable for a dry run from a throwaway clone
VA=${VA:-$ROOT_DIR/scripts/vla_analysis}
SPEC_ROOT=$ROOT_DIR/outputs/train/${RUN_PREFIX}rw_${RW_TAG}_loraft_baseline_r${LORA_R}
NAIVE_RUN=${RUN_PREFIX}realworld_${RW_TAG}_seq${RW_N_SEQ}_naive_lora_r${LORA_R}_a${LORA_ALPHA}_steps5k
NAIVE_DIR=$ROOT_DIR/outputs/train/$NAIVE_RUN
SEQ_FINAL=$(printf '%06d' $((RW_N_SEQ * SEQ_STEPS)))
TASK_IDS=$(echo "$RW_SEQ_TASK_IDS" | tr -d '[] ' | tr , ' ')
stage() { echo "[weekend] $1 $(date -u +%H:%M:%SZ)"; }
peft_msemat() {  # <run_dir> <steps csv> <first ckpt> <out jsonl> <scratch dir> <job>
  rm -rf "$5"
  MSEMAT_RUN_DIR=$1 MSEMAT_STEPS=$2 MSEMAT_OUT=$4 python "$VA/mse_matrix_peft.py" \
    --policy.path="$1/checkpoints/$3/pretrained_model" --policy.use_peft=true \
    --policy.empty_cameras=1 --policy.dtype=bfloat16 --policy.gradient_checkpointing=false \
    --policy.normalization_mapping="$RW_NORM_MAP" \
    --dataset.repo_id="$RW_SEQ_ID" --dataset.root="$RW_SEQ_ROOT" --rename_map="$RW_RENAME_MAP" \
    --output_dir="$5" --steps=200000 --batch_size=32 --num_workers=4 \
    --online_task_ids="$RW_SEQ_TASK_IDS" --online_steps_per_task=$SEQ_STEPS \
    --wandb.enable=false --job_name="$6"
}
echo "=== RW WEEKEND QUEUE started $(date -u) arm=$ARM seq=$SEQ_RUN r=$LORA_R/a$LORA_ALPHA HEAD=$(git rev-parse --short HEAD) dryrun=$DRYRUN ==="
if [ "$DRYRUN" = "1" ]; then
  echo "[weekend] plan: battery(JIT_T=0,1,3,4 on $SEQ_RUN) -> smokes -> specialists ($SPEC_ROOT) -> spec MSE -> naive ($NAIVE_RUN, final $SEQ_FINAL) -> naive MSE + reports"
  DRYRUN=1 SMOKE=1 bash "$RWD/rw_loraft_specialists_r64.sh" && DRYRUN=1 SMOKE=1 bash "$RWD/rw_naive_seq_lora_r64.sh" \
    && DRYRUN=1 bash "$RWD/rw_loraft_specialists_r64.sh" && DRYRUN=1 bash "$RWD/rw_naive_seq_lora_r64.sh" \
    && { echo "RW-WEEKEND-DRYRUN-OK"; exit 0; }
  echo "RW-WEEKEND-DRYRUN-FAIL"; exit 1
fi
# ---- 0. chain must be complete (RW-CHAIN-DONE) and its unit stopped ----
stage "wait-chain"
until grep -q "RW-CHAIN-DONE" "$CHAIN_LOG" 2>/dev/null && ! systemctl is-active --quiet rw-chain; do sleep 60; done
[ -d "$ROOT_DIR/outputs/train/$SEQ_RUN/checkpoints/$SEQ_FINAL" ] \
  || { echo "[weekend] ERROR: seq final $SEQ_FINAL missing in $SEQ_RUN"; echo "RW-WEEKEND-BOOTSTRAP-FAIL"; exit 1; }
# ---- 1. battery ----
if [ -f "$SP/mse_matrix_merged6x2_report.json" ] && grep -q "RW-BATTERY-DONE" "$BAT_LOG" 2>/dev/null; then
  stage "battery-skip (done)"
else
  stage "battery"
  JIT_T=0,1,3,4 RW_SEQ_RUN=$SEQ_RUN bash "$VA/realworld/run_rw_battery.sh" || echo "[weekend] battery FAILED"
  if grep -q "RW-BATTERY-DONE" "$BAT_LOG" 2>/dev/null; then stage "battery-done"; else echo "[weekend] WARNING: battery did not print RW-BATTERY-DONE (see $BAT_LOG)"; fi
fi
# ---- 2. smokes (fatal) ----
if [ -f "$SP/weekend_smoke_ok" ]; then
  stage "smoke-skip (ok before)"
else
  stage "smoke"
  if SMOKE=1 bash "$RWD/rw_loraft_specialists_r64.sh" && SMOKE=1 bash "$RWD/rw_naive_seq_lora_r64.sh"; then
    touch "$SP/weekend_smoke_ok"; stage "smoke-ok"
    rm -rf "$ROOT_DIR/outputs/train/_smoke_rw_${RW_TAG}_loraft_baseline_r${LORA_R}" \
           "$ROOT_DIR/outputs/train/_smoke_realworld_${RW_TAG}_seq2_naive_lora_r${LORA_R}_a${LORA_ALPHA}_steps6"
  else
    echo "[weekend] SMOKE FAILED - stopping the queue (fix locally, push, pull with the unit stopped, relaunch)"
    echo "RW-WEEKEND-SMOKE-FAIL"; exit 1
  fi
fi
# ---- 3. specialists + own-task MSE ----
stage "specialists"
bash "$RWD/rw_loraft_specialists_r64.sh" || echo "[weekend] specialists FAILED (a relaunch reruns the missing tasks)"
n_spec=$(ls -d "$SPEC_ROOT"/task*/checkpoints/005000 2>/dev/null | wc -l)
stage "specialists-done $n_spec/$RW_N_SEQ"
stage "spec-mse"
for T in $TASK_IDS; do
  RD=$SPEC_ROOT/task$T
  [ -d "$RD/checkpoints/005000" ] || { echo "[weekend] spec t$T: no final ckpt - MSE skipped"; continue; }
  if [ -f "$SP/mse_spec_r${LORA_R}.jsonl" ] && grep -q "\"run\": \"task$T\"" "$SP/mse_spec_r${LORA_R}.jsonl"; then
    echo "[weekend] spec t$T MSE exists - skipping"; continue
  fi
  peft_msemat "$RD" 005000 005000 "$SP/mse_spec_r${LORA_R}.jsonl" "$SP/msemat_out_spec_r${LORA_R}_t$T" "msemat_spec_r${LORA_R}_t$T" \
    || echo "[weekend] spec t$T MSE FAILED"
done
# ---- 4. naive sequential + matrix ----
stage "naive"
bash "$RWD/rw_naive_seq_lora_r64.sh" || echo "[weekend] naive FAILED (self-resuming on relaunch)"
if [ -d "$NAIVE_DIR/checkpoints/$SEQ_FINAL" ]; then
  stage "naive-mse"
  steps=$(for i in $(seq 1 "$RW_N_SEQ"); do printf '%06d,' $((i * SEQ_STEPS)); done); steps=${steps%,}
  if [ -f "$SP/mse_matrix_naive_r${LORA_R}.jsonl" ] && [ "$(wc -l < "$SP/mse_matrix_naive_r${LORA_R}.jsonl")" -ge "$RW_N_SEQ" ]; then
    echo "[weekend] naive matrix exists - skipping"
  else
    rm -f "$SP/mse_matrix_naive_r${LORA_R}.jsonl"
    peft_msemat "$NAIVE_DIR" "$steps" "$(printf '%06d' "$SEQ_STEPS")" "$SP/mse_matrix_naive_r${LORA_R}.jsonl" "$SP/msemat_out_naive_r${LORA_R}" "msemat_naive_r${LORA_R}" \
      || echo "[weekend] naive MSE FAILED"
  fi
  OUT=$SP/inrun_matrix_naive_r${LORA_R}.json python "$VA/realworld/rw_matrix_report.py" inrun "$NAIVE_DIR" "$SEQ_STEPS" || echo "[weekend] naive inrun report FAILED"
  OUT=$SP/mse_matrix_naive_r${LORA_R}_report.json python "$VA/realworld/rw_matrix_report.py" msemat "$SP/mse_matrix_naive_r${LORA_R}.jsonl" "$SEQ_STEPS" || echo "[weekend] naive msemat report FAILED"
else
  echo "[weekend] naive final $SEQ_FINAL missing - matrix skipped"
fi
# ---- 5. verdict ----
n_spec=$(ls -d "$SPEC_ROOT"/task*/checkpoints/005000 2>/dev/null | wc -l)
if [ "$n_spec" -ge "$RW_N_SEQ" ] && [ -d "$NAIVE_DIR/checkpoints/$SEQ_FINAL" ]; then
  stage "done"; echo "=== RW WEEKEND QUEUE COMPLETE $(date -u) ==="; echo "RW-WEEKEND-DONE"; exit 0
fi
echo "[weekend] INCOMPLETE: specialists $n_spec/$RW_N_SEQ, naive final $([ -d "$NAIVE_DIR/checkpoints/$SEQ_FINAL" ] && echo yes || echo no) - exiting 1 for relaunch"
echo "RW-WEEKEND-INCOMPLETE"; exit 1
