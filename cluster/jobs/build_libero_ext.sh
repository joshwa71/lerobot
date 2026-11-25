#!/bin/bash
# Render the libero_ext expert demos into a LeRobot dataset (scripts/datasets/build_libero_ext.py):
#   $HF_LEROBOT_HOME/libero_ext_demos  (npz per task, from scripts/libero_tasks/generate_demos.py on the desk PC)
#   -> $HF_LEROBOT_HOME/libero_ext     (libero_10's format: two 256x256 cameras, 8-D state, 7-D action)
# Needs a GPU for EGL rendering. The per-task datasets are built on node scratch and synced to
# $HF_LEROBOT_HOME/_build/libero_ext every 30 min and on exit, so a re-submitted job skips the tasks already
# built; the aggregate is made on scratch, copied to /SAN and renamed into place. Exclude the bad nodes
# (CLAUDE.md 9.4 rule 9):
#   qsub -l 'h=!(dip-207-1*|dip-207-2*)' cluster/jobs/build_libero_ext.sh
# Also writes $HF_LEROBOT_HOME/libero_ext_learn4: suite tasks LEARN_TASKS only (task_index 0-3), for the
# learnability check (configs/ext_learn_spec_r128.sh), copied to /SAN before the full dataset.
# Versions: V=<suffix> reads libero_ext_demos$V and writes libero_ext$V / libero_ext_learn4$V (build dir
# _build/libero_ext$V); SUBSET_ONLY=1 renders only the demos present (e.g. just LEARN_TASKS) and stops after the
# subset (named LEARN_NAME$V, default libero_ext_learn4$V). E.g. a learnability retest of new demos:
#   qsub -l 'h=!(...)' -v V=_v2,SUBSET_ONLY=1,LEARN_TASKS=0:31:42:59 cluster/jobs/build_libero_ext.sh
#$ -S /bin/bash
#$ -N build_libero_ext
# One gpu.q slot (each slot is a GPU there, so -pe smp N would need N free GPUs on one host); the render
# workers use the node's other CPUs (jobs are not pinned). No h_vmem: EGL/CUDA reserve large virtual memory.
#$ -l tmem=20G,h_rt=12:00:00,tscratch=200G
#$ -l gpu=true,gpu_type=(a10|a100|a100_80|a40|a6000|h100|l4|l40s|rtx3090|rtx4070ti|rtx4090|rtx6000ada)
#$ -R y
#$ -j y
#$ -o /SAN/vision/jo71_vla_wd/logs/
set -euo pipefail
source /SAN/vision/jo71_vla_wd/lerobot/cluster/env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONFAULTHANDLER=1
D=$HF_LEROBOT_HOME
# a list passed through `qsub -v` must use ':' (qsub splits -v on commas: 25 Nov 26 a comma list arrived as its
# first entry alone)
LEARN_TASKS=${LEARN_TASKS:-0:31:42:59}
LEARN_TASKS=${LEARN_TASKS//:/,}
V=${V:-}
SUBSET_ONLY=${SUBSET_ONLY:-0}
FULL=libero_ext$V LEARN=${LEARN_NAME:-libero_ext_learn4}$V
PARTIAL=()
[ "$SUBSET_ONLY" = 1 ] && PARTIAL=(--partial)
echo "== $(hostname) $(date) job=$JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} nproc=$(nproc)"
cd "$REPO/scripts/datasets"
git log -1 --format='== repo %h %s'
if [ -f "$D/$LEARN/meta/info.json" ] && { [ "$SUBSET_ONLY" = 1 ] || [ -f "$D/$FULL/meta/info.json" ]; }; then
    echo "== $D/$FULL and $LEARN already exist; nothing to do"
    exit 0
fi
scratch_setup
DEMOS=$(scratch_copy "$D/libero_ext_demos$V")
scratch_outputs "$D/_build/libero_ext$V"
mkdir -p "$SCRATCH_OUT"
WORK=$SCRATCH_OUT/tasks
echo "== per-task datasets $(date)"
timeout -s ABRT 8h "$REPO/cluster/bin/ctr" python build_libero_ext.py --stage tasks --demos "$DEMOS" \
    --work "$WORK" --out "$SCRATCH/$FULL" --workers 3 ${PARTIAL[@]+"${PARTIAL[@]}"}
scratch_outputs_close  # all per-task datasets on /SAN before scratch copies are dropped
if [ ! -f "$D/$LEARN/meta/info.json" ]; then
    echo "== learnability subset (suite tasks $LEARN_TASKS) $(date)"
    timeout -s ABRT 1h "$REPO/cluster/bin/ctr" python build_libero_ext.py --stage subset --subset "$LEARN_TASKS" \
        --repo-id "$LEARN" --demos "$DEMOS" --work "$WORK" --out "$SCRATCH/$LEARN" --stats-workers 4 \
        ${PARTIAL[@]+"${PARTIAL[@]}"}
    timeout 2h rsync -a --timeout=600 "$SCRATCH/$LEARN/" "$D/.$LEARN.tmp/"
    mv -T "$D/.$LEARN.tmp" "$D/$LEARN"
    rm -rf "$SCRATCH/$LEARN"
fi
if [ "$SUBSET_ONLY" = 1 ] || [ -f "$D/$FULL/meta/info.json" ]; then
    echo "== done $(date)"
    exit 0
fi
echo "== aggregate + stats $(date)"
timeout -s ABRT 3h "$REPO/cluster/bin/ctr" python build_libero_ext.py --stage aggregate --demos "$DEMOS" \
    --work "$WORK" --out "$SCRATCH/$FULL" --drop-work --stats-workers 4
echo "== copy to /SAN $(date)"
timeout 4h rsync -a --timeout=600 "$SCRATCH/$FULL/" "$D/.$FULL.tmp/"
mv -T "$D/.$FULL.tmp" "$D/$FULL"
echo "== done $(date): $D/$FULL"
