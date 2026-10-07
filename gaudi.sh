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

# --cleanenv keeps the container environment the same on every run, so card selection reaches the container
# only through the HABANA_VISIBLE_* variables forwarded below. Measured on Sol (2026-10-04): Slurm exports only
# SLURM_JOB_GPUS and the device cgroup does not hide the other cards (a 1-card job saw all 8). So gaudi.sh maps
# the allocated gres index to the Habana module ID with the host's hl-smi, and refuses rather than run with every
# card visible. Assumption: gres index == hl-smi `index`. Evidence: the smoke job (64599859) was allocated index 4,
# an idle card, while 0-3 and 6 were busy; the smoke test's hl-smi snapshot during training confirms it. A
# HABANA_VISIBLE_* already set on the host wins and is forwarded as is. gaudi_check_cards.sh then verifies the count.
if [[ -z "${HABANA_VISIBLE_MODULES:-}" && -z "${HABANA_VISIBLE_DEVICES:-}" && -n "${SLURM_JOB_GPUS:-}" ]]; then
    refuse() { echo "gaudi.sh: $*; refusing to run with every card visible" >&2; exit 1; }
    # the allocated indices: trimmed, non-empty, each a canonical integer (awk would equate 04 and 4.0 with 4)
    ids=$(printf '%s\n' "$SLURM_JOB_GPUS" | tr ',' '\n' | awk '{ gsub(/[[:space:]]/, "") } $0 != "" { print; if ($0 !~ /^(0|[1-9][0-9]*)$/) bad = 1 } END { exit bad }') &&
        [[ -n $ids ]] || refuse "SLURM_JOB_GPUS=$SLURM_JOB_GPUS is not a list of plain card indices"
    # index=module_id pairs from hl-smi: keep only rows whose first two comma-separated fields are integers
    map=$(hl-smi -Q index,module_id -f csv | awk -F, '{ gsub(/[[:space:]]/, "") } $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ { print $1 "=" $2 }') ||
        refuse "cannot map SLURM_JOB_GPUS=$SLURM_JOB_GPUS to Habana modules (hl-smi is missing or failed)"
    mods=
    while read -r id; do
        m=$(printf '%s\n' "$map" | awk -F= -v i="$id" '$1 == i && !s++ { print $2 }')
        [[ -n $m ]] || refuse "cannot map SLURM_JOB_GPUS=$SLURM_JOB_GPUS to Habana modules (hl-smi lists no module_id for index $id)"
        mods="${mods:+$mods,}$m"
    done <<< "$ids"
    export HABANA_VISIBLE_MODULES=$mods
    echo "gaudi.sh: SLURM_JOB_GPUS=$SLURM_JOB_GPUS -> HABANA_VISIBLE_MODULES=$mods (hl-smi index->module_id map)" >&2
fi

args=(
    --cleanenv
    --bind "$ROOT:/workspace:rw"
    --bind "$RUNTIME_ROOT:/runtime:rw"
    --bind /dev/shm:/dev/shm
    --bind "$RUNTIME_ROOT/habana_logs:/var/log/habana_logs"
    --pwd /workspace/base
    --env "PROJECT_ACCELERATOR=hpu"
    --env "PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE:-1}"
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

for var in HABANA_VISIBLE_MODULES HABANA_VISIBLE_DEVICES OMP_NUM_THREADS; do
    if [[ -n "${!var:-}" ]]; then
        args+=(--env "$var=${!var}")
    fi
done

exec apptainer exec "${args[@]}" "$IMAGE" "$@"
