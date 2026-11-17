#!/bin/bash
#SBATCH -J bbox-compare
#SBATCH -A class_bme50769722fall2026
#SBATCH -p public,htc
#SBATCH -q class
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
set -euo pipefail
[[ $# == 5 ]] || { echo 'Usage: sbatch run_localization_compare.sh DATASET TAG MODEL WEIGHTS NORMALIZATION' >&2; exit 2; }
DATASET=$1; TAG=$2; MODEL=$3; WEIGHTS=$4; NORM=$5
[[ $DATASET == tbx11k || $DATASET == candidptx ]] || exit 2
[[ $TAG =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || exit 2
SEED=${SLURM_ARRAY_TASK_ID:-0}
ARK_ROOT=${ARK_ROOT:-/scratch/$USER/2026Fall-CSE507-Ark-Container}
cd "$ARK_ROOT"
export NVIDIA_IMAGE=${NVIDIA_IMAGE:-$ARK_ROOT/containers/nvidia/nvidia.sif}
export DATA_BIND=${DATA_BIND:-/data:/data:ro,/scratch/jdefili4/2026Fall-CSE507-Ark-Container:/jarod:ro}
exec ./nvidia.sh python -m tbx_localization.compare_detectors --dataset "$DATASET" --model "$MODEL" --weights "$WEIGHTS" --normalization "$NORM" --seed "$SEED" --epochs 10 --batch 4 --manifest /workspace/base/tbx_localization/splits/tbx11k_train_selection_seed100.json --out "/workspace/results/bbox512/$TAG/$DATASET/seed$SEED"
