#!/bin/bash
#SBATCH -J bbox-smoke
#SBATCH -A class_bme50769722fall2026
#SBATCH -p public,htc
#SBATCH -q class
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
set -euo pipefail
cd /scratch/jcuesta/BME/DINO-X-plain-control
export NVIDIA_IMAGE=/scratch/jcuesta/BME/2026Fall-CSE507-Ark-Container/containers/nvidia/nvidia.sif
for dataset in tbx11k candidptx; do
 ./nvidia.sh python -m tbx_localization.smoke_detector --dataset "$dataset" --model vit_base_dinov3 --weights /workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth --normalization imagenet
done
