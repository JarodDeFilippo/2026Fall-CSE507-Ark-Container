#!/usr/bin/env bash
#SBATCH --job-name=d1-6-r
#SBATCH -A grp_jliang12
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80
#SBATCH -c 32
#SBATCH --mem=128G
export ARK_BATCH_SIZE=200
export ARK_LR=0.01
export ARK_TEST_AUGMENT=false
export ARK_EVAL_EVERY=5
export ARK_WORKERS=8
cd /scratch/$USER/2026Fall-CSE507-Ark-Container
./experiment_scripts/all4_concurrent_random_sampling/resume_all4_concurrent_random_sampling.sh 100
