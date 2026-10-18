#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIRECTORY/../common.sh"

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <seed> <nvidia|gaudi> <experiment_name> <in_container_weights_path>" >&2
    exit 2
fi

seed="$1"
container="$2"
experiment_name="$3"
weights="$4"

# Pilot gate probe: exactly the frozen-backbone recipe of all4_cyclic_dinov3_frozen (arm D): --pretrain_epochs 5, --warmup-epochs 2, same cosine,
# so the gate compares a pilot probe with arm D like with like (fewer probe cycles would run a different warmup / cosine shape and bias the
# comparison toward arm D). Only the run name and the weights differ: it starts from a pilot's exported backbone, not the stock DINOv3 weights.
# The run dir is <experiment_name>_seed_<seed>, so every probe passes its own name. ark_launch_experiment appends these arguments after
# common.sh's defaults (training_args+=("$@")) and optparse keeps the last occurrence, so the --pretrained_weights below wins over
# ARK_PRETRAINED_WEIGHTS. <in_container_weights_path> is the path inside the container (/workspace = this repo root).
ark_launch_experiment \
    "$container" \
    "$experiment_name" \
    "$seed" \
    false \
    --data_set VinDrCXR \
    --data_set ChestXray14 \
    --data_set MIMIC \
    --data_set CheXpert \
    --freeze_backbone true \
    --opt adamw \
    --weight-decay 0.01 \
    --clip-grad 3.0 \
    --warmup-lr 1e-6 \
    --min-lr 1e-6 \
    --pretrain_epochs 5 \
    --warmup-epochs 2 \
    --pretrained_weights "$weights"
