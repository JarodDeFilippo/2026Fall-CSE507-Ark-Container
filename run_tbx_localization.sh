#!/bin/bash
#SBATCH -J tbx-detect-plain90
#SBATCH -A class_bme50769722fall2026
#SBATCH -p public
#SBATCH -q class
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
set -euo pipefail
cd /scratch/jcuesta/BME/DINO-X-plain-control
export NVIDIA_IMAGE=/scratch/jcuesta/BME/2026Fall-CSE507-Ark-Container/containers/nvidia/nvidia.sif
exec ./nvidia.sh python -m tbx_localization.train_detector --weights /workspace/ssl_dinov3_cxr_constlr_nogram_plain_seed_100/export/teacher_backbone_epoch_0090.pth --out /workspace/results/tbx_localization/plain_e0090_seed100 --epochs 10
