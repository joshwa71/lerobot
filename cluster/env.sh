# Shared paths and helpers for running this repo on the UCL CS cluster.
#   source /SAN/vision/jo71_vla_wd/lerobot/cluster/env.sh
#   ctr <cmd> [args...]     # run <cmd> inside the container, with the venv on PATH
#
# The host is CentOS 7 (glibc 2.17); Python, torch and everything else only work inside the
# Singularity image (see cluster/Dockerfile). Never `pip install` on the host.

export CLUSTER_ROOT=/SAN/vision/jo71_vla_wd
export REPO=$CLUSTER_ROOT/lerobot
export SIF=$CLUSTER_ROOT/env/lerobot-cluster.sif   # symlink to the current dated image
export VENV=$CLUSTER_ROOT/env/venv

# uv: build the venv from uv.lock with the container's Python, cache on /SAN (home is 50G)
export UV_PROJECT_ENVIRONMENT=$VENV
export UV_CACHE_DIR=$CLUSTER_ROOT/env/uv-cache
export UV_PYTHON=/usr/bin/python3.12
export UV_PYTHON_DOWNLOADS=never
export UV_HTTP_TIMEOUT=300

# Caches and data. Datasets resolve as $HF_LEROBOT_HOME/<repo_id>.
export HF_HOME=$CLUSTER_ROOT/hf
export HF_LEROBOT_HOME=$CLUSTER_ROOT/data
export LIBERO_CONFIG_PATH=$CLUSTER_ROOT/env/libero
export WANDB_DIR=$CLUSTER_ROOT/logs
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false

# Variables forwarded into the container (it runs with --cleanenv so host PATH/LD_LIBRARY_PATH
# from /share/apps can't leak in). Add to this list rather than dropping --cleanenv.
_CTR_VARS="CLUSTER_ROOT REPO VENV UV_PROJECT_ENVIRONMENT UV_CACHE_DIR UV_PYTHON UV_PYTHON_DOWNLOADS
UV_HTTP_TIMEOUT HF_HOME HF_LEROBOT_HOME HF_HUB_OFFLINE HF_DATASETS_OFFLINE LIBERO_CONFIG_PATH
WANDB_DIR WANDB_MODE MUJOCO_GL MUJOCO_EGL_DEVICE_ID PYOPENGL_PLATFORM TOKENIZERS_PARALLELISM
CUDA_VISIBLE_DEVICES JOB_ID SGE_TASK_ID NSLOTS OMP_NUM_THREADS TMPDIR"

ctr() {
    local v binds=$CLUSTER_ROOT nv=()
    for v in $_CTR_VARS; do
        if [ -n "${!v+x}" ]; then export "SINGULARITYENV_$v=${!v}"; fi
    done
    export SINGULARITYENV_PREPEND_PATH=$VENV/bin
    [ -d /scratch0 ] && binds="$binds,/scratch0"
    command -v nvidia-smi >/dev/null 2>&1 && nv=(--nv)
    # ${nv[@]+...}: the host's bash 4.2 treats an empty array as unbound under `set -u`
    singularity exec ${nv[@]+"${nv[@]}"} --cleanenv -B "$binds" "$SIF" "$@"
}
