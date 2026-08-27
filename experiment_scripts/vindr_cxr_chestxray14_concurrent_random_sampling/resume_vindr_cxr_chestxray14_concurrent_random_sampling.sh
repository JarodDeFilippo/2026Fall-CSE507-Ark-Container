#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIRECTORY/../common.sh"

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <nvidia|gaudi> <seed>" >&2
    exit 2
fi

ark_launch_experiment \
    "$1" \
    vindr_cxr_chestxray14_concurrent_random_sampling \
    "$2" \
    true \
    --data_set VinDrCXR \
    --data_set ChestXray14 \
    --training_strategy joint \
    --joint_sampling proportional
