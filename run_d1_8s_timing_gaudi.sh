#!/usr/bin/env bash
#SBATCH --job-name=d1-8s-t
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=256G
# Standalone timing job for the Stage A SSL trainer (optional; the chain's phases run inside their own allocation).
# Submit from the repo root, or set ARK_ROOT to it.
cd "${ARK_ROOT:-${SLURM_SUBMIT_DIR:?sbatch from the repo root or set ARK_ROOT}}" || exit 1
source ./chain/ssl_common.sh
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
# The first 200 steps of the --epochs 40 schedule (the trainer's default), on 4 cards at batch 64 each. --print_freq 10 logs the
# first step alone (compile time) and then a row every 10 steps (steady state). Stopping at --max_steps also exercises the
# checkpoint/export save path (ckpt/maxsteps_*.pth). ssl_launch passes --resume, which is harmless here: no epoch checkpoint is
# ever written in 200 steps.
ssl_launch ssl_dinov3_cxr_timing_seed_100 --max_steps 200 --print_freq 10
