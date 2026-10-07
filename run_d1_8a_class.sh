#!/usr/bin/env bash
#SBATCH --job-name=d1-8a
#SBATCH -A class_cse49478170fall2026
#SBATCH -p public,htc
#SBATCH -q class
#SBATCH -t 07:30:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH -c 8
#SBATCH --mem=64G
export ARK_BATCH_SIZE=200
# Meta's distilled-student lr 2e-4 under its sqrt_wrt_1024 rule (lr *= 4*sqrt(batch/1024)) at batch 200 = 3.5e-4
export ARK_LR=0.00035
export ARK_TEST_AUGMENT=false
export ARK_EVAL_EVERY=5
export ARK_WORKERS=8
export ARK_MODEL=vit_base_dinov3
export ARK_INIT=dinov3
export ARK_PRETRAINED_WEIGHTS=/workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
cd /scratch/$USER/2026Fall-CSE507-Ark-Container
# fail fast (seconds, not an 8 h slot) if the DINOv3 checkpoint is not in place yet;
# ARK_PRETRAINED_WEIGHTS is the in-container path (/workspace = this repo root)
HOST_WEIGHTS="${ARK_PRETRAINED_WEIGHTS/#\/workspace\//./}"
if [[ ! -s "$HOST_WEIGHTS" ]]; then
  echo "ERROR: DINOv3 weights not found at $HOST_WEIGHTS (host path for $ARK_PRETRAINED_WEIGHTS); requeue after the download lands" >&2
  exit 1
fi
./experiment_scripts/all4_cyclic_dinov3_adamw/start_all4_cyclic_dinov3_adamw.sh 100
