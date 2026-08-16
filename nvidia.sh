#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${NVIDIA_IMAGE:-$ROOT/containers/nvidia/nvidia.sif}
RUNTIME_ROOT=${RUNTIME_ROOT:-$ROOT/.container-runtime/nvidia}

mkdir -p \
    "$RUNTIME_ROOT/tmp" \
    "$RUNTIME_ROOT/cache/huggingface" \
    "$RUNTIME_ROOT/cache/torch" \
    "$RUNTIME_ROOT/cache/matplotlib"

args=(
    --nv
    --cleanenv
    --bind "$ROOT:/workspace:rw"
    --bind "$RUNTIME_ROOT:/runtime:rw"
    --pwd /workspace/base
    --env "PROJECT_ACCELERATOR=cuda"
    --env "PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"
    --env "TMPDIR=/runtime/tmp"
    --env "TMP=/runtime/tmp"
    --env "TEMP=/runtime/tmp"
    --env "XDG_CACHE_HOME=/runtime/cache"
    --env "HF_HOME=/runtime/cache/huggingface"
    --env "TORCH_HOME=/runtime/cache/torch"
    --env "MPLCONFIGDIR=/runtime/cache/matplotlib"
)

if [[ -n "${DATA_BIND:-}" ]]; then
    args+=(--bind "$DATA_BIND")
elif [[ -d /data ]]; then
    args+=(--bind "/data:/data:ro")
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    args+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi

exec apptainer exec "${args[@]}" "$IMAGE" "$@"
