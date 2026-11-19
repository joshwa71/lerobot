#!/bin/bash
# Submit the SmolVLA staged memory chain for one experiment config (template: configs/example.sh).
# On a login node:
#   cd /SAN/vision/jo71_vla_wd/lerobot && cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/<name>.sh [stage ...]
# Stages (default: stage1 warmup audit aphase seq tri). Each is one job held on the previous stage's job(s).
# tri is the retention triangle, one job per sequential task boundary (row k: the checkpoint after task k on
# the envs of tasks 1..k, every EVAL_SEEDS seed); its last row is the final evaluation. eval is one job per
# EVAL_SEEDS entry on the final checkpoint, or on EVAL_MODEL (e.g. a zero-shot floor). Re-running resumes:
# complete stages exit at once. Baselines: spec (per-task specialists, SPEC_GROUP tasks per job, env or config), joint (all
# SEQ_DATASET episodes, then eval with EVAL_FROM=joint); naive sequential is seq with SEQ_INIT and --peft.* args.
# The config is snapshotted to $CLUSTER_ROOT/runs/<NAME>/config.sh, and the jobs read that copy. A different
# config under an existing NAME is refused (use a new NAME, or FORCE=1 to replace the snapshot).
# HOLD_JID=<job id[,...]> makes the first stage wait for jobs outside this chain (e.g. the stage-1 base).
set -eo pipefail
source /SAN/vision/jo71_vla_wd/lerobot/cluster/env.sh
[ -f "${1:-}" ] || { echo "usage: $0 <config.sh> [stage ...]"; exit 1; }
CONFIG=$(readlink -f "$1"); shift
source "$CONFIG"
RUNS=$CLUSTER_ROOT/runs/$NAME
mkdir -p "$RUNS/logs"
if [ -f "$RUNS/config.sh" ] && ! cmp -s "$CONFIG" "$RUNS/config.sh" && [ "${FORCE:-0}" != 1 ]; then
    echo "ERROR: $RUNS/config.sh differs from $CONFIG:"; diff "$RUNS/config.sh" "$CONFIG" || true; exit 1
fi
cp "$CONFIG" "$RUNS/config.sh"

stages=("$@")
[ ${#stages[@]} -gt 0 ] || stages=(stage1 warmup audit aphase seq tri)
hold=${HOLD_JID:-}
# Nodes to avoid (CLAUDE.md 9.4 rule 9): dip-207-1 froze on /SAN reads; dip-207-2 GPU 6 failed or hung LIBERO evals;
# strangepork-205-1 has a broken /scratch0 (19 Nov 26: "mkdir: cannot create directory /scratch0: File exists")
EXCLUDE_HOSTS=${EXCLUDE_HOSTS:-dip-207-1*|dip-207-2*|strangepork-205-1*}
for st in "${stages[@]}"; do
    res_var=RES_${st^^}
    res=${!res_var}
    [ -n "$res" ] || { echo "ERROR: no $res_var in $CONFIG"; exit 1; }
    hrt=$(echo "$res" | grep -o 'h_rt=[0-9:]*' | cut -d= -f2)
    limit=$(( $(echo "$hrt" | awk -F: '{print $1 * 3600 + $2 * 60 + $3}') - RUN_MARGIN_MIN * 60 ))
    # eval: one job per EVAL_SEEDS entry; tri: one per sequential task boundary (triangle row); spec: one per
    # SPEC_GROUP SEQ_TASKS entries (default 1; specialists run back to back in a job); else one job
    jobs=("")
    [ "$st" = eval ] && jobs=("${EVAL_SEEDS[@]/#/SEED=}")
    [ "$st" = tri ] && jobs=($(seq -f 'BLOCK=%g' 1 $(( $(echo "$SEQ_TASKS" | tr -cd ',' | wc -c) + 1 ))))
    [ "$st" = spec ] && jobs=($(echo "$SEQ_TASKS" | tr -d '[] ' | tr ',' '\n' | xargs -n "${SPEC_GROUP:-1}" | tr ' ' ':' | sed 's/^/TASKS=/'))
    ids=()
    for job in "${jobs[@]}"; do
        suffix=${job#*=}
        ids+=("$(qsub -terse -N "${NAME}_${st}${job:+_${suffix//:/-}}" -o "$RUNS/logs/" -l "$res" -l "$GPUS" -l "h=!($EXCLUDE_HOSTS)" \
            ${hold:+-hold_jid "$hold"} -v "RUNS=$RUNS,STAGE=$st,RUN_LIMIT=$limit${job:+,$job}" \
            "$REPO/cluster/jobs/smolvla/run_stage.sh")")
    done
    echo "$st: ${ids[*]}"
    hold=$(IFS=,; echo "${ids[*]}")
done
echo "logs: $RUNS/logs/"
