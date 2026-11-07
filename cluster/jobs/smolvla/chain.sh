#!/bin/bash
# Submit the SmolVLA staged memory chain for one experiment config (template: configs/example.sh).
# On a login node:
#   cd /SAN/vision/jo71_vla_wd/lerobot && cluster/jobs/smolvla/chain.sh cluster/jobs/smolvla/configs/<name>.sh [stage ...]
# Stages (default all): stage1 warmup audit aphase seq eval. Each is one job held on the previous stage's
# job(s); eval is one job per EVAL_SEEDS entry. Re-running resumes: complete stages exit at once.
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
[ ${#stages[@]} -gt 0 ] || stages=(stage1 warmup audit aphase seq eval)
hold=${HOLD_JID:-}
for st in "${stages[@]}"; do
    res_var=RES_${st^^}
    res=${!res_var}
    [ -n "$res" ] || { echo "ERROR: no $res_var in $CONFIG"; exit 1; }
    hrt=$(echo "$res" | grep -o 'h_rt=[0-9:]*' | cut -d= -f2)
    limit=$(( $(echo "$hrt" | awk -F: '{print $1 * 3600 + $2 * 60 + $3}') - RUN_MARGIN_MIN * 60 ))
    seeds=("")
    [ "$st" = eval ] && seeds=("${EVAL_SEEDS[@]}")
    ids=()
    for seed in "${seeds[@]}"; do
        ids+=("$(qsub -terse -N "${NAME}_${st}${seed:+_$seed}" -o "$RUNS/logs/" -l "$res" -l "$GPUS" -l 'h=!dip-207-1*' \
            ${hold:+-hold_jid "$hold"} -v "RUNS=$RUNS,STAGE=$st,RUN_LIMIT=$limit,SEED=$seed" \
            "$REPO/cluster/jobs/smolvla/run_stage.sh")")
    done
    echo "$st: ${ids[*]}"
    hold=$(IFS=,; echo "${ids[*]}")
done
echo "logs: $RUNS/logs/"
