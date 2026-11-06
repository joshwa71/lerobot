#!/bin/bash
# One stage of the SmolVLA staged memory chain. Submitted by cluster/jobs/smolvla/chain.sh with
#   -v RUNS=<run root>,STAGE=<stage>,RUN_LIMIT=<seconds>[,SEED=<eval seed>]
# The run root holds config.sh (the experiment config snapshot) and one directory per stage.
# Scratch pattern (CLAUDE.md 9.4): dataset, starting checkpoint and HF configs are copied to node
# scratch; outputs live on scratch and are synced to $RUNS/<stage> every 30 min and on exit. Each stage
# exits at once when already complete, and otherwise resumes from its last synced checkpoint.
#$ -S /bin/bash
#$ -j y
set -eo pipefail
source /SAN/vision/jo71_vla_wd/lerobot/cluster/env.sh
source "$RUNS/config.sh"
export HF_HUB_OFFLINE=1 PYTHONFAULTHANDLER=1
CTR=$REPO/cluster/bin/ctr
RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
VLM=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
echo "== $NAME/$STAGE${SEED:+ seed $SEED} on $(hostname), job $JOB_ID, $(date), GPU ${CUDA_VISIBLE_DEVICES:-none}," \
    "repo $(cd "$REPO" && /usr/bin/git rev-parse --short HEAD)"

# Python runs under a limit RUN_MARGIN_MIN short of h_rt (SGE's h_rt kill is SIGKILL, which skips the exit
# trap and its final sync); SIGABRT leaves a faulthandler traceback.
run() { timeout -s ABRT "$RUN_LIMIT" "$CTR" "$@"; }
steps_done() {  # <out dir>: training step of checkpoints/last, 0 if none
    local f=$1/checkpoints/last/training_state/training_step.json
    if [ -f "$f" ]; then tr -cd '0-9' < "$f"; else echo 0; fi
}
ckpt() { echo "$RUNS/$1/checkpoints/last/pretrained_model"; }
require_train_done() {  # <stage> <steps>
    [ "$(steps_done "$RUNS/$1")" -ge "$2" ] || { echo "ERROR: stage $1 is not complete ($(steps_done "$RUNS/$1")/$2 steps)"; exit 1; }
}
n_seq_tasks=$(( $(echo "$SEQ_TASKS" | tr -cd ',' | wc -c) + 1 ))
seq_final=$(printf '%06d' $((n_seq_tasks * SEQ_STEPS_PER_TASK)))
seq_done() { [ "$(readlink "$RUNS/seq/checkpoints/last")" = "$seq_final" ] && [ -f "$RUNS/seq/checkpoints/last/sequential_state.pt" ]; }

# lerobot-train stage: fresh from <init> (checkpoint dir or hub id), or resumed from its own last checkpoint
train_stage() {  # <stage> <dataset> <steps> <init> [lerobot-train args...]
    local stage=$1 dataset=$2 steps=$3 init=$4 data args; shift 4
    if [ "$(steps_done "$RUNS/$stage")" -ge "$steps" ]; then echo "== $stage already complete"; return 0; fi
    scratch_setup
    cd "$SCRATCH"
    data=$(scratch_copy "$HF_LEROBOT_HOME/$dataset")
    scratch_hub_configs "$VLM"
    scratch_outputs "$RUNS/$stage"
    if [ -L "$SCRATCH_OUT/checkpoints/last" ]; then
        echo "== resuming $stage at step $(steps_done "$SCRATCH_OUT")"
        args=(--config_path="$SCRATCH_OUT/checkpoints/last/pretrained_model/train_config.json" --resume=true)
    else
        rm -rf "$SCRATCH_OUT"  # lerobot-train starts fresh only in a new directory
        if [ -d "$init" ]; then init=$(scratch_copy "$init" init_model); else scratch_hub_models "$init"; fi
        args=(--policy.path="$init" --policy.load_vlm_weights=false --policy.push_to_hub=false --rename_map="$RENAME" "$@")
    fi
    run lerobot-train "${args[@]}" --output_dir="$SCRATCH_OUT" --dataset.repo_id="$dataset" --dataset.root="$data" \
        --steps="$steps" --num_workers="$NUM_WORKERS" "${WANDB[@]}"
}

case "$STAGE" in
stage1)
    if [ -n "$STAGE1_CKPT" ]; then echo "== STAGE1_CKPT is set ($STAGE1_CKPT): nothing to do"; exit 0; fi
    train_stage stage1 "$PRETRAIN_DATASET" "$STAGE1_STEPS" "$STAGE1_FROM" "${STAGE1_ARGS[@]}"
    ;;
warmup)
    if [ -z "$STAGE1_CKPT" ]; then require_train_done stage1 "$STAGE1_STEPS"; fi
    train_stage warmup "$PRETRAIN_DATASET" "$WARMUP_STEPS" "${STAGE1_CKPT:-$(ckpt stage1)}" "${MEMORY[@]}" "${WARMUP_ARGS[@]}"
    ;;
audit)
    if [ -f "$RUNS/audit/AUDIT_DONE" ]; then echo "== audit already complete"; exit 0; fi
    require_train_done warmup "$WARMUP_STEPS"
    scratch_setup
    cd "$SCRATCH"
    data=$(scratch_copy "$HF_LEROBOT_HOME/$SEQ_DATASET")
    init=$(scratch_copy "$(ckpt warmup)" init_model)
    scratch_hub_configs "$VLM"
    scratch_outputs "$RUNS/audit"
    rm -rf "$SCRATCH_OUT"
    run lerobot-sequential-train --policy.path="$init" --rename_map="$RENAME" \
        --dataset.repo_id="$SEQ_DATASET" --dataset.root="$data" --output_dir="$SCRATCH_OUT" \
        --online_task_ids="$SEQ_TASKS" --save_checkpoint=false --save_after_each_task=false \
        --reinit_optimizer_each_task=true --num_workers="$NUM_WORKERS" --wandb.enable=false "${AUDIT_ARGS[@]}"
    # Per-tower routing summary (core50 / effnum per task, family and background IoU) for the gate
    read -r EL EB VL VB < <(run python -c 'import json, sys
m = json.load(open(sys.argv[1]))["memory_layer"]
print(",".join(map(str, m["layers"])) or "-", m["mem_n_keys"] ** 2, ",".join(map(str, m["vlm_layers"])) or "-", m["vlm_mem_n_keys"] ** 2)' "$init/config.json")
    [ "$EL" = - ] || run python "$REPO/scripts/vla_analysis/vlm_audit_analysis.py" "$SCRATCH_OUT" "$EL" "$EB" expert
    [ "$VL" = - ] || run python "$REPO/scripts/vla_analysis/vlm_audit_analysis.py" "$SCRATCH_OUT" "$VL" "$VB" vlm
    date > "$SCRATCH_OUT/AUDIT_DONE"
    ;;
aphase)
    require_train_done warmup "$WARMUP_STEPS"
    train_stage aphase "$PRETRAIN_DATASET" "$APHASE_STEPS" "$(ckpt warmup)" "${APHASE_ARGS[@]}"
    ;;
seq)
    if seq_done; then echo "== seq already complete"; exit 0; fi
    require_train_done aphase "$APHASE_STEPS"
    scratch_setup
    cd "$SCRATCH"
    data=$(scratch_copy "$HF_LEROBOT_HOME/$SEQ_DATASET")
    scratch_hub_configs "$VLM"
    scratch_outputs "$RUNS/seq"
    if [ -f "$SCRATCH_OUT/checkpoints/last/sequential_state.pt" ]; then
        echo "== resuming seq after step $(readlink "$SCRATCH_OUT/checkpoints/last")"
        args=(--policy.path="$SCRATCH_OUT/checkpoints/last/pretrained_model" --resume_sequential=true)
    else
        rm -rf "$SCRATCH_OUT"
        args=(--policy.path="$(scratch_copy "$(ckpt aphase)" init_model)")
    fi
    run lerobot-sequential-train "${args[@]}" --rename_map="$RENAME" \
        --dataset.repo_id="$SEQ_DATASET" --dataset.root="$data" --output_dir="$SCRATCH_OUT" \
        --online_task_ids="$SEQ_TASKS" --online_steps_per_task="$SEQ_STEPS_PER_TASK" --ds_to_env_map_json="$DS_TO_ENV" \
        --num_workers="$NUM_WORKERS" "${WANDB[@]}" "${SEQ_ARGS[@]}"
    ;;
eval)
    if [ -f "$RUNS/eval/seeds_$SEED.json" ]; then echo "== eval seed $SEED already complete"; exit 0; fi
    seq_done || { echo "ERROR: seq is not complete"; exit 1; }
    scratch_setup
    cd "$SCRATCH"
    model=$(scratch_copy "$(ckpt seq)" model)
    scratch_hub_configs "$VLM"
    scratch_outputs "$RUNS/eval"
    mkdir -p "$SCRATCH_OUT"
    export CAMP_SEEDS=$SEED CAMP_TAG="${NAME}_seq_final" CAMP_OUT="$SCRATCH_OUT/seeds_$SEED.json"
    run python "$REPO/scripts/vla_analysis/eval_seeds_campaign.py" --policy.path="$model" --rename_map="$RENAME" \
        "${EVAL_ARGS[@]}"
    ;;
*)
    echo "unknown STAGE=$STAGE"; exit 1
    ;;
esac
echo "== $NAME/$STAGE done $(date)"
