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
CUDA_VISIBLE_DEVICES JOB_ID SGE_TASK_ID NSLOTS OMP_NUM_THREADS TMPDIR PYTHONFAULTHANDLER
SCRATCH HF_DATASETS_CACHE HF_HUB_CACHE WANDB_CACHE_DIR WANDB_DATA_DIR TORCH_HOME
TORCH_NCCL_BLOCKING_WAIT TORCH_NCCL_ASYNC_ERROR_HANDLING NCCL_P2P_DISABLE CAMP_SEEDS CAMP_TAG CAMP_OUT"

# Render LIBERO on the GPU SGE assigned. robosuite indexes EGL devices by this id; NVIDIA EGL devices
# come first and in nvidia-smi order (Mesa's follow), which is the numbering SGE uses.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}; fi

# Fills _CTR_CMD with the `singularity exec ...` prefix and exports the forwarded variables.
_ctr_prepare() {
    local v f binds=$CLUSTER_ROOT
    for v in $_CTR_VARS; do
        if [ -n "${!v+x}" ]; then export "SINGULARITYENV_$v=${!v}"; fi
    done
    export SINGULARITYENV_PREPEND_PATH=$VENV/bin
    [ -d /scratch0 ] && binds="$binds,/scratch0"
    _CTR_CMD=(singularity exec --cleanenv)
    if command -v nvidia-smi >/dev/null 2>&1; then
        _CTR_CMD+=(--nv)
        # Singularity 3.8's --nv library list predates driver 530+, where libnvidia-eglcore needs
        # libnvidia-gpucomp. Without it every NVIDIA EGL device fails to initialise (MUJOCO_GL=egl).
        for f in /usr/lib64/libnvidia-gpucomp.so.*; do
            [ -e "$f" ] && binds="$binds,$f:/opt/nvextra/$(basename "$f")"
        done
        export SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs:/opt/nvextra
    fi
    _CTR_CMD+=(-B "$binds" "$SIF")
}

# --- Node-local scratch for jobs (request it: #$ -l tscratch=<n>G) --------------------------------
# The pattern of the old cluster job scripts (CLAUDE.md 9.4): copy in, work on scratch, copy out.
# Reads from /SAN can stall for 20+ minutes (seen 6 Nov 26), and training random-reads the HF datasets
# arrow cache (memory-mapped), so dataset, base model, caches and outputs all live in the per-job dir
# /scratch0/$USER/$JOB_ID. Outputs are synced to /SAN periodically and on exit, then scratch is removed.
# Software (container image, venv, code) is read from /SAN, as the old scripts did with their conda env.
#   scratch_setup
#   DATA=$(scratch_copy "$HF_LEROBOT_HOME/libero_10")       # -> --dataset.root=$DATA
#   scratch_hub_models lerobot/smolvla_base HuggingFaceTB/SmolVLM2-500M-Video-Instruct
#   scratch_outputs "$CLUSTER_ROOT/runs/<run>"             # -> --output_dir=$SCRATCH_OUT
scratch_setup() {
    export SCRATCH=/scratch0/$USER/$JOB_ID
    mkdir -p "$SCRATCH"/{tmp,hf_datasets,data,hub,wandb,torch,outputs}
    export TMPDIR=$SCRATCH/tmp HF_DATASETS_CACHE=$SCRATCH/hf_datasets TORCH_HOME=$SCRATCH/torch
    export WANDB_DIR=$SCRATCH/wandb WANDB_CACHE_DIR=$SCRATCH/wandb/cache WANDB_DATA_DIR=$SCRATCH/wandb/data
    trap _scratch_finish EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    echo "scratch: $SCRATCH ($(df -h /scratch0 | awk 'NR==2 {print $4}') free)"
}

# Keep a run's outputs on scratch, synced to /SAN every SYNC_EVERY seconds (default 1800) and again on
# exit. Copies any existing outputs of the run in from /SAN first, so a resumed job (new job, new
# scratch) continues from the last synced checkpoint; if there are none, $SCRATCH_OUT is left
# uncreated for lerobot-train (which refuses to start a fresh run in an existing dir). A hard kill (h_rt) loses at most one interval.
# The sync never deletes on /SAN (unlike the old scripts' rsync --delete, which a resumed job whose
# scratch lacked earlier checkpoints would have wiped). Sets SCRATCH_OUT. Call it directly, not in
# $(...): a subshell would hide the sync state from the exit trap.
scratch_outputs() {
    _OUT_DEST=$1
    _OUT=$SCRATCH/outputs/$(basename "$1")
    if [ -d "$_OUT_DEST" ]; then
        timeout 60m rsync -a --exclude '.*.tmp' "$_OUT_DEST/" "$_OUT/" >&2 || { echo "scratch_outputs: copy-in of $_OUT_DEST failed" >&2; return 1; }
    fi
    # don't create $_OUT for a new run: lerobot-train refuses to start fresh in an existing dir
    mkdir -p "$_OUT_DEST" "$(dirname "$_OUT")"
    (
        trap - EXIT TERM INT
        while sleep "${SYNC_EVERY:-1800}"; do
            [ -d "$_OUT" ] || continue
            echo "[$(date '+%F %T')] syncing $_OUT -> $_OUT_DEST" >&2
            _sync_out >&2 || echo "outputs sync failed or stalled" >&2
        done
    ) >&2 &
    _OUT_SYNC_PID=$!
    export SCRATCH_OUT=$_OUT
}

# Outputs -> /SAN, with checkpoints/last moved only after its target has arrived in full: note where
# `last` points, copy everything else, then set `last` on /SAN by an atomic rename. An rsync cut off
# part-way (job killed) can leave a half-copied checkpoint directory on /SAN, but never a `last`
# pointing at one. Hidden temp dirs of a checkpoint save in progress (.<step>.tmp) are skipped.
_sync_out() {
    local tgt=""
    [ -L "$_OUT/checkpoints/last" ] && tgt=$(readlink "$_OUT/checkpoints/last")
    timeout 30m rsync -a --exclude '/checkpoints/last' --exclude '.*.tmp' "$_OUT/" "$_OUT_DEST/" || return 1
    if [ -n "$tgt" ]; then
        ln -sfn "$tgt" "$_OUT_DEST/checkpoints/.last.tmp" && mv -T "$_OUT_DEST/checkpoints/.last.tmp" "$_OUT_DEST/checkpoints/last"
    fi
}

_scratch_finish() {
    local rc=$?
    set +e
    if [ -n "${_OUT_SYNC_PID:-}" ]; then kill "$_OUT_SYNC_PID" 2>/dev/null; wait "$_OUT_SYNC_PID" 2>/dev/null; fi
    if [ -n "${_OUT:-}" ] && [ -d "$_OUT" ]; then
        echo "[$(date '+%F %T')] final sync $_OUT -> $_OUT_DEST"
        local try
        for try in 1 2 3; do _sync_out && break; done
    fi
    rm -rf "$SCRATCH"
    exit "$rc"
}

# Copy a directory to scratch ($SCRATCH/data/<name>, default its basename), retrying a copy that
# stalls; prints the scratch path. A stall is no data for 10 min (rsync --timeout), not a wall-clock
# cap: libero_90 (65 GB) alone takes ~75 min at /SAN's ~15 MB/s. Retries continue where they stopped.
scratch_copy() {
    local src=$1 dst=$SCRATCH/data/${2:-$(basename "$1")} try
    for try in 1 2 3; do
        timeout 4h rsync -a --timeout=600 "$src/" "$dst/" && { echo "$dst"; return 0; }
        echo "scratch_copy: attempt $try of $src failed or stalled" >&2
    done
    return 1
}

# Copy hub models from $HF_HOME/hub to scratch and point HF_HUB_CACHE there.
scratch_hub_models() {
    local repo d try
    for repo in "$@"; do
        d=models--${repo//\//--}
        for try in 1 2 3; do
            timeout 20m rsync -a "$HF_HOME/hub/$d/" "$SCRATCH/hub/$d/" && break
            echo "scratch_hub_models: attempt $try of $repo failed or stalled" >&2
            [ "$try" = 3 ] && return 1
        done
    done
    export HF_HUB_CACHE=$SCRATCH/hub
}

# Like scratch_hub_models but configs/tokenizers only (no weights): for the VLM backbone when the
# weights come from a checkpoint (--policy.load_vlm_weights=false).
scratch_hub_configs() {
    local repo d try
    for repo in "$@"; do
        d=models--${repo//\//--}
        for try in 1 2 3; do
            timeout 10m rsync -aL --exclude onnx --exclude '*.safetensors' --exclude '*.bin' \
                "$HF_HOME/hub/$d/snapshots" "$HF_HOME/hub/$d/refs" "$SCRATCH/hub/$d/" && break
            echo "scratch_hub_configs: attempt $try of $repo failed or stalled" >&2
            [ "$try" = 3 ] && return 1
        done
    done
    export HF_HUB_CACHE=$SCRATCH/hub
}

# In scripts and shells. Under `timeout`, xargs etc. (which can't run shell functions) use the
# executable form instead, cluster/bin/ctr, which execs singularity so signals reach the container.
ctr() {
    _ctr_prepare
    "${_CTR_CMD[@]}" "$@"
}
