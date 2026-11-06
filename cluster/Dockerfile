# Container userland for running this repo on the UCL CS cluster.
#
# Why: login and compute nodes run CentOS 7 (glibc 2.17), which is too old for the torch 2.10
# wheels pinned in uv.lock. This image supplies a modern glibc, Python 3.12, FFmpeg (for torchcodec)
# and EGL (for headless MuJoCo/LIBERO rendering). It deliberately holds NO Python packages: the venv
# lives on /SAN and is built from uv.lock by cluster/setup_env.sh, inside this container.
#
# Build here (desk PC), then convert to a Singularity image on the cluster:
#   docker build -t lerobot-cluster:<tag> cluster/
#   docker save lerobot-cluster:<tag> -o lerobot-cluster_<tag>.tar     # copy to $CLUSTER_ROOT/env/
#   qsub cluster/jobs/build_sif.sh                                     # tar -> .sif on a compute node
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        build-essential cmake pkg-config git curl ca-certificates rsync \
        ffmpeg \
        libegl1 libgl1 libgles2 libglvnd0 libosmesa6 libglib2.0-0 \
        libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# `singularity exec --nv` binds the host's libEGL_nvidia but not the glvnd vendor file that tells
# libEGL to use it; without this, MUJOCO_GL=egl silently falls back to (or fails on) Mesa.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /usr/local/bin/
