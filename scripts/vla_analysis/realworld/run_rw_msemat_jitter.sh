#!/bin/bash
# E65 real-world landing battery, GPU part — realworld duplicate of run_e62_msemat_jitter.sh.
# 1. MSE forgetting matrix: every per-task checkpoint x every seq task, paired-noise, seed 0
#    (mse_matrix_rw.py == mse_matrix2.py numerics at the defaults 16 batches x bs32);
# 2. jitter/OOD grid on the FINAL checkpoint (t0/t3/t4 x clean / state@0.1 / state@0.2 /
#    image@0.05, swap-slots; E52 convention) via the unchanged probe_jitter.py.
# Deltas vs the sim launcher: no --env.* / --ds_to_env_map_json / MUJOCO (no simulator),
# dataset = the RW SEQ split, every arg from rw_env.sh. SMOKE=1 -> the _smoke_ run, MINI jitter,
# msemat 2 batches x bs4 (so it can run beside a training job).
# Overridable for dry runs: RW_SEQ_RUN, MSEMAT_STEPS, JIT_CKPTS, PROBE_SWAP_SLOTS, FIRST, FINAL.
set -eo pipefail
source /home/josh/lerobot/job_scripts/nebius/realworld/rw_env.sh
SCRIPTS=$ROOT_DIR/scripts/vla_analysis
BATTERY_TAG=${BATTERY_TAG:-merged6x2}
RW_SEQ_RUN=${RW_SEQ_RUN:-${RUN_PREFIX}realworld_${RW_TAG}_seq${RW_N_SEQ}_jw_merged6x2_e468101416_v579111315_prepass_beta4corefrac_topt3072_lr2x_steps5k}
RD=$ROOT_DIR/outputs/train/$RW_SEQ_RUN
SP=${SP:-$ROOT_DIR/outputs/analysis/realworld/${RUN_PREFIX}e65}
mkdir -p "$SP"
[ -d "$RD/checkpoints" ] || { echo "ERROR: $RD/checkpoints missing"; exit 1; }
steps=(); for i in $(seq 1 "$RW_N_SEQ"); do steps+=("$(printf '%06d' $((i*SEQ_STEPS)))"); done
FIRST=${FIRST:-${steps[0]}}; FINAL=${FINAL:-${steps[$((${#steps[@]}-1))]}}
MSEMAT_STEPS=${MSEMAT_STEPS:-$(IFS=,; echo "${steps[*]}")}
JIT_T=${JIT_T:-"0,3,4"}
if [ -z "${JIT_CKPTS:-}" ]; then
  JIT_CKPTS=$(for t in ${JIT_T//,/ }; do printf "t%s:%s," "$t" "$FINAL"; done); JIT_CKPTS=${JIT_CKPTS%,}
fi
COMMON_ARGS=(
  --policy.empty_cameras=1 --policy.dtype=bfloat16 --policy.gradient_checkpointing=false
  --policy.push_to_hub=false
  --policy.normalization_mapping="$RW_NORM_MAP"
  --dataset.repo_id="$RW_SEQ_ID" --dataset.root="$RW_SEQ_ROOT"
  --rename_map="$RW_RENAME_MAP"
  --steps=200000 --batch_size=32 --num_workers=4
  --online_task_ids="$RW_SEQ_TASK_IDS" --online_steps_per_task=$SEQ_STEPS
  --wandb.enable=false
)
if [ "$SMOKE" = "1" ]; then export MSEMAT_NB=${MSEMAT_NB:-2} MSEMAT_BS=${MSEMAT_BS:-4} MSEMAT_NW=${MSEMAT_NW:-2} MINI=${MINI:-1}; fi
echo "[rw-battery] run=$RW_SEQ_RUN steps=$MSEMAT_STEPS jitter=$JIT_CKPTS swap_slots=${PROBE_SWAP_SLOTS:-1} out=$SP"
# the probes' own scratch dirs (TrainPipelineConfig.validate refuses an existing output_dir)
rm -rf "$SP/msemat_out_${BATTERY_TAG}" "$SP/jitter_out_${BATTERY_TAG}"
export MSEMAT_RUN_DIR=$RD MSEMAT_STEPS MSEMAT_OUT=$SP/mse_matrix_${BATTERY_TAG}.jsonl \
       MSEMAT_TASKS=$(echo "$RW_SEQ_TASK_IDS" | tr -d '[] ')
python $SCRIPTS/realworld/mse_matrix_rw.py \
  --policy.path="$RD/checkpoints/$FIRST/pretrained_model" "${COMMON_ARGS[@]}" \
  --output_dir="$SP/msemat_out_${BATTERY_TAG}" --job_name=msemat_${BATTERY_TAG}
echo "=== MSE matrix done ==="
export PROBE_RUN_DIR=$RD PROBE_CKPTS="$JIT_CKPTS" PROBE_OUT=$SP/probe_jitter_${BATTERY_TAG}.jsonl \
       PROBE_SWAP_SLOTS=${PROBE_SWAP_SLOTS:-1}
python $SCRIPTS/probe_jitter.py \
  --policy.path="$RD/checkpoints/$FINAL/pretrained_model" "${COMMON_ARGS[@]}" \
  --output_dir="$SP/jitter_out_${BATTERY_TAG}" --job_name=jitter_${BATTERY_TAG}
echo "=== RW probe battery (msemat + jitter) COMPLETE ==="
