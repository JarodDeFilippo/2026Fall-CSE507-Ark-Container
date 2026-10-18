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

# Frozen-backbone linear probe: --freeze_backbone trains only the projector and the per-dataset heads. It takes a value
# (--freeze_backbone true); main_ark.py refuses a bare flag instead of letting vararg_callback_bool swallow the next option.
# No --layer-decay / --patch-embed-lr-mult (engine.py refuses them together with --freeze_backbone).
# --pretrain_epochs is the cycle count (cosine LR, teacher momentum, training loop); the last occurrence wins over
# common.sh's 200. main_ark.py's --epochs is parsed but never read, so it is not passed.
# Warmup cycles come from ARK_WARMUP_EPOCHS (common.sh default 20 is too long for 5 cycles; run_d1_8d_gaudi.sh sets 2). --warmup-epochs 2 is appended for the same reason, so the 2-of-5 warmup holds even without the wrapper's ARK_WARMUP_EPOCHS.
ark_launch_experiment \
    "$container" \
    all4_cyclic_dinov3_frozen \
    "$seed" \
    true \
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
    --warmup-epochs 2
