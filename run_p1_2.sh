#!/usr/bin/env bash
#SBATCH --job-name=p1-2
#SBATCH -q class
#SBATCH -p public,htc
#SBATCH --time=04:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
export ARK_BATCH_SIZE=40
export ARK_LR=0.002
export ARK_TEST_AUGMENT=false
export ARK_EVAL_EVERY=5
cd /scratch/$USER/2026Fall-CSE507-Ark-Container
./experiment_scripts/vindr_cxr_chestxray14_cyclic/start_vindr_cxr_chestxray14_cyclic.sh 100
