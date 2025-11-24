# ROCm 7.14 ships ROCm only as pip wheels (rocm-sdk-core, no /opt/rocm). Its bundled
# ROCr 1.21 has a known bug where rocr::core::Runtime::AsyncEventsLoop busy-spins one
# full CPU core forever after the first GPU op - see ROCm/TheRock#7051. No env var
# works around it (HSA_ENABLE_INTERRUPT / ROCR_VISIBLE_DEVICES / HSA_ENABLE_MWAITX all
# tried). 7.2.4 is the newest release-channel build before the gap (nothing exists
# between 7.2.4 and 7.14) and still offers torch 2.10.
#
# RE-TESTED 2026-08-21 against rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0
# (torch 2.12.0+rocm7.14.0, hip 7.14.60850). STILL BROKEN: 4.99s of process CPU
# over a 5s idle window after the first GPU op, i.e. one core pinned. 7.2.4 scores
# 0.00s on the same test. Do not upgrade for this.
# The VRAM that a ComfyUI run never gives back is NOT this: on both 7.2.4 and 7.14,
# allocating and freeing 8 GB through torch returns it to the OS exactly (-0.10 GB
# drift). Whatever holds ~19.6 GB after a run sits above torch, so a ROCm bump
# cannot fix it.
#
# The base is an ARG so 7.14 can be re-tested without editing this file:
#   docker build --build-arg BASE_IMAGE=docker.io/rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0 \
#                -t comfyui-rocm714 .
# Note 7.14 has no /opt/rocm, so the ROCM_HOME/PATH/LD_LIBRARY_PATH block below is
# inert there (harmless: the wheels put their libs on the default search path).
ARG BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
FROM ${BASE_IMAGE}

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROCM_HOME=/opt/rocm
ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    nano \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /workspace

# Create virtual environment and install dependencies
# We use --system-site-packages to inherit PyTorch from the base image
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python packages (do not override base-image ROCm PyTorch)
RUN --mount=source=requirements.txt,target=/tmp/requirements.txt \
    --mount=source=manager_requirements.txt,target=/tmp/manager_requirements.txt \
    pip install --no-cache -r /tmp/requirements.txt && \
    pip install --no-cache -r /tmp/manager_requirements.txt

# Install any requirements.txt found under custom_nodes (if present).
# Notes:
# - This runs at build time, so it only sees custom_nodes that are in the build context.
# - If you bind-mount custom_nodes at runtime, those requirements will NOT be installed by this step.
RUN --mount=source=custom_nodes,target=/tmp/custom_nodes \
    find /tmp/custom_nodes -type f -iname "requirements.txt" -print0 | \
    xargs -0 -r -n 1 sh -c 'echo "Installing $1" && pip install --no-cache -r "$1"' sh;

# Expose ComfyUI default port
EXPOSE 8188

# The container runs as the host user (see `user:` in docker-compose.yml) so that
# files written into the bind-mounted workspace are not root-owned. That user has no
# passwd entry, so HOME must be set explicitly and pre-created: a named volume mounted
# over this path inherits the ownership set here when it is first created.
ARG UID=1000
ARG GID=1000
ENV HOME=/home/comfy
RUN mkdir -p ${HOME}/.triton/cache && chown -R ${UID}:${GID} ${HOME}

# ComfyUI-Manager fetches the ComfyUI repo on startup (update_policy in
# user/__manager/config.ini). This repo's `origin` is git@github.com:..., so that
# fetch shells out to ssh; with no known_hosts it prompts "The authenticity of host
# 'github.com' can't be established" and, because compose allocates a tty, blocks
# startup forever.
#
# Written to /etc/ssh/ssh_known_hosts, NOT ~/.ssh: HOME=/home/comfy is a named volume
# that only inherits image content when it is first created, so a key baked into the
# home directory would be invisible to any already-existing volume.
#
# The scanned key is verified against GitHub's published ed25519 fingerprint rather
# than trusted blindly - ssh-keyscan alone is trust-on-first-use against whatever the
# build-time network returns. Update this constant if GitHub ever rotates the key:
#   https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints
RUN set -eu; \
    ssh-keyscan -t ed25519 github.com > /tmp/gh_known_hosts 2>/dev/null; \
    ssh-keygen -lf /tmp/gh_known_hosts | grep -q 'SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU'; \
    mkdir -p /etc/ssh; \
    cat /tmp/gh_known_hosts >> /etc/ssh/ssh_known_hosts; \
    rm -f /tmp/gh_known_hosts; \
    chmod 644 /etc/ssh/ssh_known_hosts
