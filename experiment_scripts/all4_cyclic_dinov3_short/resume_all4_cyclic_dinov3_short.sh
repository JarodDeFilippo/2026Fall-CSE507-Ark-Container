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

# --pretrain_epochs is the cycle count (cosine LR, teacher momentum, training loop); the last occurrence wins over
# common.sh's 200. main_ark.py's --epochs is parsed but never read, so it is not passed.
# Warmup cycles come from ARK_WARMUP_EPOCHS (common.sh default 20 is too long for 40 cycles; run_d1_8c_gaudi.sh sets 5). --warmup-epochs 5 is appended for the same reason, so the 5-of-40 warmup holds even without the wrapper's ARK_WARMUP_EPOCHS.
ark_launch_experiment \
    "$container" \
    all4_cyclic_dinov3_short \
    "$seed" \
    true \
    --data_set VinDrCXR \
    --data_set ChestXray14 \
    --data_set MIMIC \
    --data_set CheXpert \
    --opt adamw \
    --weight-decay 0.01 \
    --layer-decay 0.75 \
    --patch-embed-lr-mult 0.2 \
    --clip-grad 3.0 \
    --warmup-lr 1e-6 \
    --min-lr 1e-6 \
    --pretrain_epochs 40 \
    --warmup-epochs 5
