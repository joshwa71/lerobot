#!/bin/bash
# One stage of the SmolVLA staged memory chain. Submitted by cluster/jobs/smolvla/chain.sh with
#   -v RUNS=<run root>,STAGE=<stage>,RUN_LIMIT=<seconds>[,SEED=<eval seed>][,BLOCK=<triangle row>][,TASK=<task_index>]
# The run root holds config.sh (the experiment config snapshot) and one directory per stage.
# Scratch pattern (CLAUDE.md 9.4): dataset, starting checkpoint and HF models are copied to node
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
VLM=HuggingFaceTB/SmolVLM2-500M-Video-Instruct  # loaded by every stage: load_vlm_weights=true, HF's path and dtypes
echo "== $NAME/$STAGE${SEED:+ seed $SEED}${BLOCK:+ row $BLOCK}${TASK:+ task $TASK} on $(hostname), job $JOB_ID, $(date), GPU ${CUDA_VISIBLE_DEVICES:-none}," \
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
seq_envs() {  # <n>: LIBERO env ids of the first n SEQ_TASKS (through DS_TO_ENV), comma-separated
    local t ids=()
    for t in $(echo "$SEQ_TASKS" | tr -d '[] ' | tr ',' ' ' | cut -d' ' -f1-"$1"); do
        ids+=("$(echo "$DS_TO_ENV" | grep -o "\"$t\": *[0-9]*" | grep -o '[0-9]*$')")
    done
    (IFS=,; echo "${ids[*]}")
}
task_env() { echo "$DS_TO_ENV" | grep -o "\"$1\": *[0-9]*" | grep -o '[0-9]*$'; }  # <task_index>: its LIBERO env id

# Policy arguments for a checkpoint already on scratch. A LoRA checkpoint (adapter_config.json) records the path
# of its base model, which was a scratch path of the job that trained it: point it at a fresh scratch copy of the
# stage-1 base, and load it through PEFT.
peft_args() {  # <checkpoint dir on scratch> -> POLICY_ARGS
    POLICY_ARGS=(--policy.path="$1")
    [ -f "$1/adapter_config.json" ] || return 0
    local base
    base=$(scratch_copy "$STAGE1_CKPT" base_model)
    run python -c 'import json, sys
p, base = sys.argv[1:3]
c = json.load(open(p)); c["base_model_name_or_path"] = base
json.dump(c, open(p, "w"), indent=2)' "$1/adapter_config.json" "$base"
    POLICY_ARGS+=(--policy.use_peft=true)
}

# lerobot-train stage: fresh from <init> (checkpoint dir or hub id), or resumed from its own last checkpoint
train_stage() {  # <stage> <dataset> <steps> <init> [lerobot-train args...]
    local stage=$1 dataset=$2 steps=$3 init=$4 data args; shift 4
    if [ "$(steps_done "$RUNS/$stage")" -ge "$steps" ]; then echo "== $stage already complete"; return 0; fi
    # a checkpoint path that is missing (e.g. its run failed) fails here, not after an hour of data staging
    case $init in /*) [ -d "$init" ] || { echo "ERROR: starting checkpoint $init does not exist"; exit 1; } ;; esac
    scratch_setup
    cd "$SCRATCH"
    data=$(scratch_copy "$HF_LEROBOT_HOME/$dataset")
    scratch_hub_models "$VLM"
    scratch_outputs "$RUNS/$stage"
    if [ -L "$SCRATCH_OUT/checkpoints/last" ]; then
        echo "== resuming $stage at step $(steps_done "$SCRATCH_OUT")"
        args=(--config_path="$SCRATCH_OUT/checkpoints/last/pretrained_model/train_config.json" --resume=true)
    else
        rm -rf "$SCRATCH_OUT"  # lerobot-train starts fresh only in a new directory
        if [ -d "$init" ]; then init=$(scratch_copy "$init" init_model); else scratch_hub_models "$init"; fi
        args=(--policy.path="$init" --policy.push_to_hub=false --rename_map="$RENAME" "$@")
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
    scratch_hub_models "$VLM"
    scratch_outputs "$RUNS/audit"
    rm -rf "$SCRATCH_OUT"
    run lerobot-sequential-train --policy.path="$init" --rename_map="$RENAME" \
        --dataset.repo_id="$SEQ_DATASET" --dataset.root="$data" --output_dir="$SCRATCH_OUT" \
        --online_task_ids="${AUDIT_TASKS:-$SEQ_TASKS}" --save_checkpoint=false --save_after_each_task=false \
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
    # Starts from SEQ_INIT when the config sets one (e.g. naive sequential LoRA from the stage-1 base, or a longer
    # sequence from another run's A-phase), otherwise from this run's A-phase.
    if seq_done; then echo "== seq already complete"; exit 0; fi
    if [ -n "${SEQ_INIT:-}" ]; then
        [ -d "$SEQ_INIT" ] || { echo "ERROR: SEQ_INIT $SEQ_INIT does not exist"; exit 1; }
    else
        require_train_done aphase "$APHASE_STEPS"
    fi
    scratch_setup
    cd "$SCRATCH"
    data=$(scratch_copy "$HF_LEROBOT_HOME/$SEQ_DATASET")
    scratch_hub_models "$VLM"
    scratch_outputs "$RUNS/seq"
    if [ -f "$SCRATCH_OUT/checkpoints/last/sequential_state.pt" ]; then
        echo "== resuming seq after step $(readlink "$SCRATCH_OUT/checkpoints/last")"
        peft_args "$SCRATCH_OUT/checkpoints/last/pretrained_model"
        args=("${POLICY_ARGS[@]}" --resume_sequential=true)
    else
        rm -rf "$SCRATCH_OUT"
        args=(--policy.path="$(scratch_copy "${SEQ_INIT:-$(ckpt aphase)}" init_model)" --policy.push_to_hub=false)
    fi
    run lerobot-sequential-train "${args[@]}" --rename_map="$RENAME" \
        --dataset.repo_id="$SEQ_DATASET" --dataset.root="$data" --output_dir="$SCRATCH_OUT" \
        --online_task_ids="$SEQ_TASKS" --online_steps_per_task="$SEQ_STEPS_PER_TASK" --ds_to_env_map_json="$DS_TO_ENV" \
        --num_workers="$NUM_WORKERS" "${WANDB[@]}" "${SEQ_ARGS[@]}"
    ;;
eval)
    # One seed of EVAL_ARGS on the envs of all SEQ_TASKS: the final sequential checkpoint, or EVAL_MODEL when the
    # config sets one (e.g. the zero-shot floor of the stage-1 base). An --env.task_ids in EVAL_ARGS overrides the envs.
    if [ -f "$RUNS/eval/seeds_$SEED.json" ]; then echo "== eval seed $SEED already complete"; exit 0; fi
    if [ -n "${EVAL_MODEL:-}" ]; then
        src=$EVAL_MODEL tag=$NAME
        [ -d "$src" ] || { echo "ERROR: EVAL_MODEL $src does not exist"; exit 1; }
    elif [ "${EVAL_FROM:-seq}" = joint ]; then
        require_train_done joint "$JOINT_STEPS"
        src=$(ckpt joint) tag=${NAME}_joint
    else
        seq_done || { echo "ERROR: seq is not complete"; exit 1; }
        src=$(ckpt seq) tag=${NAME}_seq_final
    fi
    scratch_setup
    cd "$SCRATCH"
    peft_args "$(scratch_copy "$src" model)"
    scratch_hub_models "$VLM"
    scratch_outputs "$RUNS/eval"
    mkdir -p "$SCRATCH_OUT"
    export CAMP_SEEDS=$SEED CAMP_TAG=$tag CAMP_OUT="$SCRATCH_OUT/seeds_$SEED.json"
    run python "$REPO/scripts/vla_analysis/eval_seeds_campaign.py" "${POLICY_ARGS[@]}" --rename_map="$RENAME" \
        --env.task_ids="[$(seq_envs "$n_seq_tasks")]" "${EVAL_ARGS[@]}"
    ;;
tri)
    # Retention-triangle row BLOCK: the checkpoint after sequential task BLOCK on the envs of tasks 1..BLOCK,
    # EVAL_ARGS episodes x every EVAL_SEEDS seed (the pi05 instrument: 25 episodes x 4 paired seeds per cell).
    if [ -f "$RUNS/tri/b$BLOCK.json" ]; then echo "== triangle row $BLOCK already complete"; exit 0; fi
    ck=$RUNS/seq/checkpoints/$(printf '%06d' $((BLOCK * SEQ_STEPS_PER_TASK)))
    # sequential_state.pt is written after the boundary checkpoint, so its presence means the boundary is complete
    [ -f "$ck/sequential_state.pt" ] || { echo "ERROR: sequential boundary $BLOCK is not complete ($ck)"; exit 1; }
    scratch_setup
    cd "$SCRATCH"
    peft_args "$(scratch_copy "$ck/pretrained_model" model)"
    scratch_hub_models "$VLM"
    scratch_outputs "$RUNS/tri"
    mkdir -p "$SCRATCH_OUT"
    export CAMP_SEEDS=$(IFS=,; echo "${EVAL_SEEDS[*]}") CAMP_TAG="${NAME}_tri_b$BLOCK" CAMP_OUT="$SCRATCH_OUT/b$BLOCK.json"
    run python "$REPO/scripts/vla_analysis/eval_seeds_campaign.py" "${POLICY_ARGS[@]}" --rename_map="$RENAME" \
        "${EVAL_ARGS[@]}" --env.task_ids="[$(seq_envs "$BLOCK")]"
    ;;
spec)
    # Per-task specialist for task_index TASK: the sequential trainer on that task alone (SEQ_ARGS, from SEQ_INIT or
    # the stage-1 base), then its own env, EVAL_ARGS episodes x every EVAL_SEEDS seed. Outputs: spec/t<TASK>/.
    if [ -f "$RUNS/spec/t$TASK/eval.json" ]; then echo "== specialist $TASK already complete"; exit 0; fi
    final=$(printf '%06d' "$SEQ_STEPS_PER_TASK")
    scratch_setup
    cd "$SCRATCH"
    scratch_hub_models "$VLM"
    scratch_outputs "$RUNS/spec/t$TASK"
    if [ ! -f "$SCRATCH_OUT/train/checkpoints/$final/sequential_state.pt" ]; then
        rm -rf "$SCRATCH_OUT/train"  # one short task: an interrupted specialist retrains from the start
        data=$(scratch_copy "$HF_LEROBOT_HOME/$SEQ_DATASET")
        run lerobot-sequential-train --policy.path="$(scratch_copy "${SEQ_INIT:-$STAGE1_CKPT}" init_model)" \
            --policy.push_to_hub=false --rename_map="$RENAME" --dataset.repo_id="$SEQ_DATASET" --dataset.root="$data" \
            --output_dir="$SCRATCH_OUT/train" --online_task_ids="[$TASK]" --online_steps_per_task="$SEQ_STEPS_PER_TASK" \
            --ds_to_env_map_json="$DS_TO_ENV" --num_workers="$NUM_WORKERS" "${WANDB[@]}" "${SEQ_ARGS[@]}"
    fi
    peft_args "$(scratch_copy "$SCRATCH_OUT/train/checkpoints/$final/pretrained_model" model)"
    export CAMP_SEEDS=$(IFS=,; echo "${EVAL_SEEDS[*]}") CAMP_TAG="${NAME}_spec_t$TASK" CAMP_OUT="$SCRATCH_OUT/eval.json"
    run python "$REPO/scripts/vla_analysis/eval_seeds_campaign.py" "${POLICY_ARGS[@]}" --rename_map="$RENAME" \
        "${EVAL_ARGS[@]}" --env.task_ids="[$(task_env "$TASK")]"
    ;;
joint)
    # Joint training on every SEQ_DATASET episode from the stage-1 base: JOINT_ARGS give all weights (the joint
    # fine-tune) or --peft.* (one multitask adapter). Evaluate with the eval stage and EVAL_FROM=joint.
    train_stage joint "$SEQ_DATASET" "$JOINT_STEPS" "$STAGE1_CKPT" "${JOINT_ARGS[@]}"
    ;;
*)
    echo "unknown STAGE=$STAGE"; exit 1
    ;;
esac
echo "== $NAME/$STAGE done $(date)"
