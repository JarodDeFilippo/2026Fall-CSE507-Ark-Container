#!/usr/bin/env bash
#SBATCH --job-name=d1-8-nogram-smoke
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 00:40:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=128G
# 128G: measured 2026-10-26 on the running no-Gram trunk (4 cards, 72 CPUs, job 66633863): proportional set size of all its processes 53.3 GB; sstat MaxRSS reports the request itself because page cache is charged to the job cgroup; 256G could not share a Gaudi node with another lab's 1-card jobs that allocate 511 of 515 GB; Sol's own floor is 2 GB per CPU, it accepts a lower request with a notice.
# Standalone timing job for the Stage A SSL trainer (optional; the chain's phases run inside their own allocation).
# Submit from the repo root, or set ARK_ROOT to it.
cd "${ARK_ROOT:-${SLURM_SUBMIT_DIR:?sbatch from the repo root or set ARK_ROOT}}" || exit 1
source ./chain/ssl_common.sh
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
# Smoke run for the no-Gram trunk's recipe on 4 cards (Jarod, 2026-10-26): the trunk's launch line (chain/constlr.sh with CONSTLR_NOGRAM=1:
# DINOv3 recipe --drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm, --no_gram, flat LR from chain/main_lr.txt's value 2e-3, the
# --epochs 1000 ceiling) stopped after 20 steps in a throwaway run dir. Stopping at --max_steps exercises the checkpoint save path
# (ckpt/maxsteps_*.pth), which on 4 ranks gathers every rank's RNG state with dist.all_gather_object (f2dcb79): the first time that collective runs
# on HCCL. Expected log: a `recipe (...)` line naming the three settings, `local_cls_norm: initialized as a copy of norm.weight / norm.bias`,
# `Gram: OFF (--no_gram)`, 20 step rows, then the maxsteps save with no error. ssl_launch passes --resume, harmless here.
ssl_launch ssl_dinov3_cxr_nogram_smoke_seed_100 --drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm --no_gram --lr 0.002 --min_lr 0.002 --no_lr_scaling --epochs 1000 --max_steps 20 --print_freq 5
