#!/usr/bin/env bash
#SBATCH --job-name=d1-8a-g
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=256G
export ARK_BATCH_SIZE=200
# global batch; engine.py divides by world size, so 50 per card on 4 cards
# Meta's distilled-student lr 2e-4 under its sqrt_wrt_1024 rule (lr *= 4*sqrt(batch/1024)) at batch 200 = 3.5e-4
export ARK_LR=0.00035
export ARK_TEST_AUGMENT=false
export ARK_EVAL_EVERY=5
# 16 data workers per rank: 8 left B data-bound on 4 cards (64617617: DT ~0.8 s of ~1.1 s steps, 32 workers ~68% CPU, 29/72 cores); 4 x 16 = 64 workers fit -c 72
export ARK_WORKERS=16
export ARK_MODEL=vit_base_dinov3
export ARK_INIT=dinov3
export ARK_PRETRAINED_WEIGHTS=/workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail fast (seconds, not an 8 h slot) if the DINOv3 checkpoint is not in place yet;
# ARK_PRETRAINED_WEIGHTS is the in-container path (/workspace = this repo root)
HOST_WEIGHTS="${ARK_PRETRAINED_WEIGHTS/#\/workspace\//./}"
if [[ ! -s "$HOST_WEIGHTS" ]]; then
  echo "ERROR: DINOv3 weights not found at $HOST_WEIGHTS (host path for $ARK_PRETRAINED_WEIGHTS); requeue after the download lands" >&2
  exit 1
fi
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before 4 ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
./experiment_scripts/all4_cyclic_dinov3_adamw/start_all4_cyclic_dinov3_adamw.sh 100 gaudi
