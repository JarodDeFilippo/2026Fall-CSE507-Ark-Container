#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIRECTORY/../common.sh"

if [[ $# -ne 10 ]]; then
    echo "Usage: $0 <sweep_id> <seed> <nvidia|gaudi> <adamw|momentum> <weight_decay> <layer_decay> <patch_embed_lr_mult> <warmup_cycles> <cycles> <normalization>" >&2
    echo "The peak LR is not an argument: it comes from ARK_LR, as in the other experiments (chain/sweep_run.sh exports it from the arm's row)." >&2
    exit 2
fi
if [[ -z "${ARK_LR:-}" ]]; then
    echo "ARK_LR (the peak learning rate) is not set; common.sh would silently fall back to 0.01" >&2
    exit 2
fi

sweep_id="$1"
seed="$2"
container="$3"
opt="$4"
weight_decay="$5"
layer_decay="$6"
patch_embed_lr_mult="$7"
warmup_cycles="$8"
cycles="$9"
normalization="${10}"

# Step VIII fine-tuning sweep: ONE parameterized pair for every arm (the arms are the rows of sweep/wave1.tsv, read by chain/sweep_run.sh,
# which calls this). The run dir is all4_cyclic_dinov3_sweep_<sweep_id>_seed_<seed>.
# Fixed for every arm: the four data sets in this order, --clip-grad 3.0, --warmup-lr 1e-6, --min-lr 1e-6. The phase script's ARK_* env fixes
# the rest (batch 200 global, test augment off, workers) and the arm's LR (ARK_LR), --eval_every (ARK_EVAL_EVERY), --eval_cycle0 (ARK_EVAL_CYCLE0),
# model, init and weights; the seed is the second argument (the row's seed column).
# ark_launch_experiment appends these arguments after common.sh's defaults and optparse keeps the last occurrence, so --opt, --pretrain_epochs
# (the cycle count: cosine LR, teacher momentum, training loop) and --warmup-epochs below win over common.sh's sgd / 200 / ARK_WARMUP_EPOCHS.
# main_ark.py's --epochs is parsed but never read, so it is not passed.
# --opt momentum is plain momentum SGD (engine.py: "sgd" is the Nesterov variant); its momentum is main_ark.py's --momentum default 0.9, so
# no --momentum is passed. --layer-decay / --patch-embed-lr-mult (0 = the patch embedding does not train) select engine.py's grouped
# optimizer, which needs --model vit_base_dinov3.
# --normalization is passed on as given (imagenet is main_ark.py's default, what every other DINOv3 arm uses, and what the Stage A SSL transforms apply).
# chain/sweep_run.sh accepts only imagenet in the table for now: with --train_augment true (the default) the dataset loaders hard-code ImageNet mean/std for
# training and --normalization only reaches the val/test transforms, so any other preset would normalize train and val/test differently.
ark_launch_experiment \
    "$container" \
    "all4_cyclic_dinov3_sweep_${sweep_id}" \
    "$seed" \
    false \
    --data_set VinDrCXR \
    --data_set ChestXray14 \
    --data_set MIMIC \
    --data_set CheXpert \
    --opt "$opt" \
    --weight-decay "$weight_decay" \
    --layer-decay "$layer_decay" \
    --patch-embed-lr-mult "$patch_embed_lr_mult" \
    --clip-grad 3.0 \
    --warmup-lr 1e-6 \
    --min-lr 1e-6 \
    --pretrain_epochs "$cycles" \
    --warmup-epochs "$warmup_cycles" \
    --normalization "$normalization"
