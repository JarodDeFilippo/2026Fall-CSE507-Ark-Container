#!/usr/bin/env bash
#SBATCH --job-name=d1-6c-r
#SBATCH -A class_cse49478170fall2026
#SBATCH -p public,htc
#SBATCH -q class
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --constraint="a100_80|h100"
#SBATCH -c 32
#SBATCH --mem=128G
export ARK_BATCH_SIZE=200
export ARK_LR=0.01
export ARK_TEST_AUGMENT=false
export ARK_EVAL_EVERY=5
export ARK_WORKERS=8
cd /scratch/$USER/2026Fall-CSE507-Ark-Container
./experiment_scripts/all4_concurrent_random_sampling/resume_all4_concurrent_random_sampling.sh 100
