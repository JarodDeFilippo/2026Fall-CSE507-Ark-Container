#!/usr/bin/env bash
#SBATCH --job-name=d1-8d-g
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=256G
export ARK_BATCH_SIZE=200
# global batch; engine.py divides by world size, so 50 per card on 4 cards
# frozen-backbone linear probe (--freeze_backbone true in the frozen start script): only the projector and heads train, so no layer-wise decay and no patch-embed multiplier
export ARK_LR=7e-5
export ARK_TEST_AUGMENT=false
export ARK_EVAL_EVERY=1
# 5-cycle schedule (--pretrain_epochs 5 is in the frozen start script): 2 warmup cycles
export ARK_WARMUP_EPOCHS=2
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
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
./experiment_scripts/all4_cyclic_dinov3_frozen/start_all4_cyclic_dinov3_frozen.sh 100 gaudi
