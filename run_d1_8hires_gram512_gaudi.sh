#!/usr/bin/env bash
#SBATCH --job-name=d1-8hires-g512
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=256G
# DINO-X ConstLR 512 px, Gram teacher 512 px: the twin of the `hires` chain phase (chain/hires.sh, Gram teacher 1024 px, run dir ssl_dinov3_cxr_constlr512_seed_100) with only the Gram size changed,
# on its own 4 Gaudi cards and into its own run dir, ssl_dinov3_cxr_constlr512_gram512_seed_100. chain/hires.sh owns the recipe, so none of it is repeated here; the two settings on the last line are its
# env overrides (HIRES_GRAM replaces chain/hires_gram_size.txt, which is then not read). It shares chain/main_lr.txt, chain/hires_stop_after.txt and chain/hires_batch_per_gpu.txt with the 1024 run.
# Re-submitting this same script continues the run from its last epoch checkpoint; to extend it, raise chain/hires_stop_after.txt and re-submit. The log is the Slurm .out: there is no chain/log file.
# NOT a chain phase: it is submitted on its own and must never be added to chain/phases.txt.
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
HIRES_GRAM=512 HIRES_RUN_SUFFIX=_gram512 exec ./chain/hires.sh
