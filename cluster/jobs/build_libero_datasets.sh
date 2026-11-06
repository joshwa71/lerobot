#!/bin/bash
# Rebuild the LIBERO training datasets into $HF_LEROBOT_HOME (= /SAN/.../data) from their hub sources
# (downloaded on a login node by cluster/download_assets.sh). CPU-only.
#   libero_10 / libero_goal / libero_object / libero_spatial  <- HuggingFaceVLA/libero
#   libero_90                                                  <- jesbu1/libero_90_lerobot
# WHICH=suites or WHICH=90 builds just one part (default: both):
#   qsub [-v WHICH=90] cluster/jobs/build_libero_datasets.sh
#$ -S /bin/bash
#$ -N build_libero_data
#$ -l tmem=8G,h_vmem=8G,h_rt=8:00:00
#$ -pe smp 8
#$ -R y
#$ -j y
#$ -o /SAN/vision/jo71_vla_wd/logs/
set -euo pipefail
source /SAN/vision/jo71_vla_wd/lerobot/cluster/env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONFAULTHANDLER=1
WHICH=${WHICH:-all}
D=$HF_LEROBOT_HOME
echo "== $(hostname) $(date) job=$JOB_ID slots=$NSLOTS WHICH=$WHICH"
cd "$REPO/scripts/datasets"
git log -1 --format='== repo %h %s'

if [ "$WHICH" = all ] || [ "$WHICH" = suites ]; then
    echo "== suites $(date)"
    ctr python build_libero_suites.py --src "$D/HuggingFaceVLA/libero" --out "$D" --work "$D/_build/libero_suites"
fi
if [ "$WHICH" = all ] || [ "$WHICH" = 90 ]; then
    echo "== libero_90 $(date)"
    ctr python build_libero_90.py --src "$D/jesbu1/libero_90_lerobot" --work "$D/_build/libero_90" \
        --out "$D/libero_90" --workers "$NSLOTS"
fi
echo "== done $(date)"
