#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${GAUDI_IMAGE:-$ROOT/containers/gaudi/gaudi.sif}
RUNTIME_ROOT=${RUNTIME_ROOT:-$ROOT/.container-runtime/gaudi}

mkdir -p \
    "$RUNTIME_ROOT/tmp" \
    "$RUNTIME_ROOT/cache/huggingface" \
    "$RUNTIME_ROOT/cache/torch" \
    "$RUNTIME_ROOT/cache/matplotlib" \
    "$RUNTIME_ROOT/habana_logs"

args=(
    --intel-hpu
    --cleanenv
    --bind "$ROOT:/workspace:rw"
    --bind "$RUNTIME_ROOT:/runtime:rw"
    --bind "$RUNTIME_ROOT/habana_logs:/var/log/habana_logs:rw"
    --pwd /workspace/base
    --env "PROJECT_ACCELERATOR=hpu"
    --env "PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"
    --env "TMPDIR=/runtime/tmp"
    --env "TMP=/runtime/tmp"
    --env "TEMP=/runtime/tmp"
    --env "XDG_CACHE_HOME=/runtime/cache"
    --env "HF_HOME=/runtime/cache/huggingface"
    --env "TORCH_HOME=/runtime/cache/torch"
    --env "MPLCONFIGDIR=/runtime/cache/matplotlib"
    --env "HABANA_LOGS=/runtime/habana_logs"
    --env "PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE:-0}"
    --env "OMPI_MCA_btl_vader_single_copy_mechanism=none"
)

if [[ -n "${HABANA_VISIBLE_MODULES:-}" ]]; then
    args+=(--env "HABANA_VISIBLE_MODULES=$HABANA_VISIBLE_MODULES")
elif [[ -n "${HABANA_VISIBLE_DEVICES:-}" ]]; then
    args+=(--env "HABANA_VISIBLE_DEVICES=$HABANA_VISIBLE_DEVICES")
else
    echo "No scheduler-provided Habana visibility variable found" >&2
    exit 2
fi

exec apptainer exec "${args[@]}" "$IMAGE" "$@"
