#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIRECTORY/../common.sh"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <seed> [nvidia|gaudi]" >&2
    exit 2
fi

seed="$1"
container="${2:-nvidia}"

ark_launch_experiment \
    "$container" \
    vindr_cxr_chestxray14_cyclic \
    "$seed" \
    true \
    --data_set VinDrCXR \
    --data_set ChestXray14
