#!/usr/bin/env bash
#SBATCH --job-name=d1-6-eval
#SBATCH -A grp_jliang12
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 04:00:00
#SBATCH -G a100:1
#SBATCH --constraint=a100_80
#SBATCH -c 16
#SBATCH --mem=64G
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <run_dir_name> <cycle> [dataset ...]  (default datasets: VinDrCXR ChestXray14 MIMIC CheXpert; cycle is zero-padded for you)" >&2
    exit 2
fi
run="$1"
cycle=$(printf '%04d' "$((10#$2))")
shift 2
if [[ $# -eq 0 ]]; then
    set -- VinDrCXR ChestXray14 MIMIC CheXpert
fi
dataset_args=()
for dataset in "$@"; do
    dataset_args+=(--data_set "$dataset")
done
cd /scratch/$USER/2026Fall-CSE507-Ark-Container
exec ./nvidia.sh python main_ark.py \
    --mode test \
    --test_augment true \
    --pretrained_weights "/workspace/${run}/models/weights/epoch_${cycle}" \
    "${dataset_args[@]}" \
    --model swin_base \
    --projector_features 1376 \
    --device cuda
