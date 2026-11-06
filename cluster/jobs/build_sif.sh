#!/bin/bash
# Convert the docker-saved image tar (built on the desk PC from cluster/Dockerfile) into the
# Singularity image used by every job. CPU-heavy (squashfs), so it runs on a compute node.
#   qsub -v TAG=2026-11-06 cluster/jobs/build_sif.sh
#$ -S /bin/bash
#$ -N build_sif
#$ -l tmem=6G,h_vmem=6G,h_rt=2:00:00,tscratch=30G
#$ -pe smp 8
#$ -R y
#$ -j y
#$ -o /SAN/vision/jo71_vla_wd/logs/
set -euo pipefail
source /SAN/vision/jo71_vla_wd/lerobot/cluster/env.sh
echo "build_sif on $(hostname) at $(date), TAG=$TAG"

SCRATCH=/scratch0/$USER/$JOB_ID
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
export SINGULARITY_TMPDIR=$SCRATCH SINGULARITY_CACHEDIR=$SCRATCH/cache

TAR=$CLUSTER_ROOT/env/lerobot-cluster_$TAG.tar
OUT=$CLUSTER_ROOT/env/lerobot-cluster_$TAG.sif
singularity build "$SCRATCH/image.sif" "docker-archive://$TAR"
cp "$SCRATCH/image.sif" "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
ln -sfn "$(basename "$OUT")" "$SIF"
ls -la "$CLUSTER_ROOT/env"
echo "done at $(date)"
