#!/bin/bash
# E65 WEEKEND QUEUE (Josh, 29-30 Aug) — runs unattended behind the arm-4 chain on nebius-spot.
# Order per Josh (30 Aug): "play this one out then batteries then topt=1536 after then batteries
# and baselines."
#   0 wait for RW-CHAIN-DONE + rw-chain inactive (defensive; the rw-weekend bootstrap already
#     waited and git-pulled with the chain unit stopped, per CLAUDE.md 9.8)
#   1 battery A on the arm-4 topt3072 sequential (JIT_T=0,1,3,4: task 1 added to the jitter grid)
#   2 TOP_T RERUN: the sequential stage ONLY, single delta SEQ_TOP_T=1536, reusing the existing
#     A-phase checkpoint (stage-1/warm-up/audit/A-phase all skip on their guards). Rationale in
#     E65 add-14: on RW batches task 1 reads ~3.8-6.8k distinct slots/site, so k=min(top_t,n_read)
#     lets corefrac's zero-score core slots into the write mask (mask saturation); 1536 clears the
#     low tail of that distribution at every site.
#   3 battery B on the rerun (BATTERY_TAG=merged6x2_topt1536)
#   4 SMOKES of the two LoRA wrappers — a failure STOPS the queue (RW-WEEKEND-SMOKE-FAIL)
#   5 r64 LoRA specialists t0..t4 -> own-task MSE (mse_matrix_peft.py)
#   6 r64 naive sequential LoRA (self-resuming) -> adapter-swap MSE matrix + reports
#   7 RW-WEEKEND-DONE only when both batteries, the rerun, specialists 5/5 and the naive final
#     exist; otherwise RW-WEEKEND-INCOMPLETE (exit 1 -> the heartbeat relaunches; every stage is
#     skip-guarded on its artifacts).
# Launched by the rw-weekend bootstrap (heartbeat relaunch_weekend); log outputs/rw_weekend_v5.log.
# DRYRUN=1: print the plan + the wrappers' commands, no GPU.
set -o pipefail
source /home/josh/lerobot/job_scripts/nebius/realworld/rw_env.sh
DRYRUN=${DRYRUN:-0}
LORA_R=${LORA_R:-64}; LORA_ALPHA=${LORA_ALPHA:-16}; export LORA_R LORA_ALPHA
ARM=${ARM_TAG:-merged6x2_e468101416_v579111315_anchor030_pool1010_sep1_c100_prepass}
SEQ_RUN_A=${RW_SEQ_RUN:-${RUN_PREFIX}realworld_${RW_TAG}_seq${RW_N_SEQ}_jw_${ARM}_beta4corefrac_topt3072_lr2x_steps5k}
RERUN_TOP_T=${RERUN_TOP_T:-1536}
SEQ_RUN_B=${RUN_PREFIX}realworld_${RW_TAG}_seq${RW_N_SEQ}_jw_${ARM}_beta4corefrac_topt${RERUN_TOP_T}_lr2x_steps5k
SP=${SP:-$ROOT_DIR/outputs/analysis/realworld/${RUN_PREFIX}e65}; mkdir -p "$SP"
CHAIN_LOG=$ROOT_DIR/outputs/${RUN_PREFIX}rw_chain_${RW_TAG}.log
BAT_LOG=$ROOT_DIR/outputs/${RUN_PREFIX}rw_battery_${RW_TAG}.log
RWD=${RWD:-$ROOT_DIR/job_scripts/nebius/realworld}
VA=${VA:-$ROOT_DIR/scripts/vla_analysis}
SPEC_ROOT=$ROOT_DIR/outputs/train/${RUN_PREFIX}rw_${RW_TAG}_loraft_baseline_r${LORA_R}
NAIVE_RUN=${RUN_PREFIX}realworld_${RW_TAG}_seq${RW_N_SEQ}_naive_lora_r${LORA_R}_a${LORA_ALPHA}_steps5k
NAIVE_DIR=$ROOT_DIR/outputs/train/$NAIVE_RUN
SEQ_FINAL=$(printf '%06d' $((RW_N_SEQ * SEQ_STEPS)))
TASK_IDS=$(echo "$RW_SEQ_TASK_IDS" | tr -d '[] ' | tr , ' ')
stage() { echo "[weekend] $1 $(date -u +%H:%M:%SZ)"; }

run_battery () {   # <seq run> <battery tag>
  if [ -f "$SP/mse_matrix_$2_report.json" ]; then stage "battery-skip ($2 done)"; return 0; fi
  stage "battery:$2"
  JIT_T=0,1,3,4 RW_SEQ_RUN=$1 BATTERY_TAG=$2 bash "$VA/realworld/run_rw_battery.sh" \
    || echo "[weekend] battery $2 FAILED"
  [ -f "$SP/mse_matrix_$2_report.json" ] && stage "battery-done:$2" \
    || echo "[weekend] WARNING: battery $2 produced no report (see $BAT_LOG)"
}

peft_msemat () {   # <run_dir> <steps csv> <first ckpt> <out jsonl> <scratch dir> <job>
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

echo "=== RW WEEKEND QUEUE v3 started $(date -u) arm=$ARM r=$LORA_R/a$LORA_ALPHA rerun_top_t=$RERUN_TOP_T HEAD=$(git rev-parse --short HEAD) dryrun=$DRYRUN ==="
echo "[weekend] order: battery(topt3072) -> rerun(topt$RERUN_TOP_T) -> battery(topt$RERUN_TOP_T) -> smokes -> specialists -> naive"
if [ "$DRYRUN" = "1" ]; then
  echo "[weekend] A=$SEQ_RUN_A"; echo "[weekend] B=$SEQ_RUN_B"; echo "[weekend] specialists=$SPEC_ROOT"; echo "[weekend] naive=$NAIVE_RUN (final $SEQ_FINAL)"
  DRYRUN=1 SMOKE=1 bash "$RWD/rw_loraft_specialists_r64.sh" && DRYRUN=1 SMOKE=1 bash "$RWD/rw_naive_seq_lora_r64.sh" \
    && { echo "RW-WEEKEND-DRYRUN-OK"; exit 0; }
  echo "RW-WEEKEND-DRYRUN-FAIL"; exit 1
fi

# ---- 0. the arm-4 chain must be complete and its unit stopped ----
stage "wait-chain"
until grep -q "RW-CHAIN-DONE" "$CHAIN_LOG" 2>/dev/null && ! systemctl is-active --quiet rw-chain; do sleep 60; done
[ -d "$ROOT_DIR/outputs/train/$SEQ_RUN_A/checkpoints/$SEQ_FINAL" ] \
  || { echo "[weekend] ERROR: seq final $SEQ_FINAL missing in $SEQ_RUN_A"; echo "RW-WEEKEND-BOOTSTRAP-FAIL"; exit 1; }

# ---- 1. battery A (topt3072) ----
run_battery "$SEQ_RUN_A" merged6x2

# ---- 2. top_t rerun: sequential stage only, single delta ----
if [ -d "$ROOT_DIR/outputs/train/$SEQ_RUN_B/checkpoints/$SEQ_FINAL" ]; then
  stage "rerun-skip (final exists)"
else
  stage "rerun:top_t=$RERUN_TOP_T"
  RW_TAG=$RW_TAG RW_FAMILY=${RW_FAMILY:-0-4,3-4} ARM_TAG=$ARM \
  SEP_W=1.0 CONTRASTIVE_W=1.0 EXPERT_ANCHOR_W=0.30 VLM_POOL_W='[1.0,1.0]' \
  SKIP_GATE=1 SEQ_TOP_T=$RERUN_TOP_T SEQ_RUN=$SEQ_RUN_B \
    bash "$RWD/rw_merged6x2_full_chain.sh" || echo "[weekend] RERUN FAILED (resumes on relaunch)"
  [ -d "$ROOT_DIR/outputs/train/$SEQ_RUN_B/checkpoints/$SEQ_FINAL" ] && stage "rerun-done" \
    || echo "[weekend] WARNING: rerun final $SEQ_FINAL missing"
fi

# ---- 3. battery B (the rerun) ----
[ -d "$ROOT_DIR/outputs/train/$SEQ_RUN_B/checkpoints/$SEQ_FINAL" ] \
  && run_battery "$SEQ_RUN_B" merged6x2_topt${RERUN_TOP_T} \
  || echo "[weekend] battery B skipped (no rerun final)"

# ---- 4. smokes (fatal) ----
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

# ---- 5. specialists + own-task MSE ----
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

# ---- 6. naive sequential + matrix ----
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

# ---- 7. verdict ----
n_spec=$(ls -d "$SPEC_ROOT"/task*/checkpoints/005000 2>/dev/null | wc -l)
have_b=$([ -d "$ROOT_DIR/outputs/train/$SEQ_RUN_B/checkpoints/$SEQ_FINAL" ] && echo 1 || echo 0)
if [ "$n_spec" -ge "$RW_N_SEQ" ] && [ -d "$NAIVE_DIR/checkpoints/$SEQ_FINAL" ] && [ "$have_b" = 1 ] \
   && [ -f "$SP/mse_matrix_merged6x2_report.json" ] && [ -f "$SP/mse_matrix_merged6x2_topt${RERUN_TOP_T}_report.json" ]; then
  stage "done"; echo "=== RW WEEKEND QUEUE COMPLETE $(date -u) ==="; echo "RW-WEEKEND-DONE"; exit 0
fi
echo "[weekend] INCOMPLETE: batteryA $([ -f "$SP/mse_matrix_merged6x2_report.json" ] && echo y || echo n), rerun $have_b, batteryB $([ -f "$SP/mse_matrix_merged6x2_topt${RERUN_TOP_T}_report.json" ] && echo y || echo n), specialists $n_spec/$RW_N_SEQ, naive $([ -d "$NAIVE_DIR/checkpoints/$SEQ_FINAL" ] && echo y || echo n) - exiting 1 for relaunch"
echo "RW-WEEKEND-INCOMPLETE"; exit 1
