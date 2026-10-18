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

# D-long: arm D's recipe (all4_cyclic_dinov3_frozen) run for 50 cycles instead of 5; the run dir is all4_cyclic_dinov3_frozen_long_seed_100.
# Frozen-backbone linear probe: --freeze_backbone trains only the projector and the per-dataset heads. It takes a value
# (--freeze_backbone true); main_ark.py refuses a bare flag instead of letting vararg_callback_bool swallow the next option.
# No --layer-decay / --patch-embed-lr-mult (engine.py refuses them together with --freeze_backbone).
# --pretrain_epochs is the cycle count (cosine LR, teacher momentum, training loop): 50 here; the last occurrence wins over
# common.sh's 200. main_ark.py's --epochs is parsed but never read, so it is not passed.
# Warmup cycles come from ARK_WARMUP_EPOCHS (common.sh default is 20; chain/armD_long.sh sets 2, as arm D does). --warmup-epochs 2 is appended so the 2-of-50 warmup holds even without the phase script's ARK_WARMUP_EPOCHS.
ark_launch_experiment \
    "$container" \
    all4_cyclic_dinov3_frozen_long \
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
    --pretrain_epochs 50 \
    --warmup-epochs 2
