#!/usr/bin/env bash
#SBATCH --job-name=d1-8-branch
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=128G
# 128G: measured 2026-10-26 on the running no-Gram trunk (4 cards, 72 CPUs, job 66633863): proportional set size of all its processes 53.3 GB; sstat MaxRSS reports the request itself because page cache is charged to the job cgroup; 256G could not share a Gaudi node with another lab's 1-card jobs that allocate 511 of 515 GB; Sol's own floor is 2 GB per CPU, it accepts a lower request with a notice.
# DINO-X ConstLR 224, Gram-anchored branch: the no-Gram trunk (run_d1_8_constlr_nogram_gaudi.sh, run dir ssl_dinov3_cxr_constlr_nogram_seed_100) continued EXACTLY from its epoch-K checkpoint (student, EMA
# teacher, optimizer, step, epoch, schedule position, data order, RNG) with ONE change: the Gram teacher on, anchored to that checkpoint's EMA teacher backbone (ssl_pretrain.py --branch_from). Gram weight 2.0
# and Gram teacher 448: the trainer's defaults, as in constlr. The trunk keeps running; the branch is a new run dir, so it never writes to the trunk's.
# The trunk's line carries DINOv3's --drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm (2026-10-26), so a branch passes them too (--branch_from requires them to match the source), and
# neither is an exact ConstLR-224 control.
# Usage: sbatch run_d1_8_branch_gram_gaudi.sh <K> <STOP> [EVERY] [COUNT]   (EVERY default 10000, COUNT default 3: Meta's 10k x 3, DINOv3 Appendix C)
#   K = the trunk epoch to branch from (its ckpt/epoch_<K %04d>.pth must exist when the branch first starts: the trunk keeps every 10th epoch and its last 3); STOP = the epoch the branch stops after (above K, at most the 1000 ceiling);
#   EVERY / COUNT = the Gram refreshes: COUNT of them, EVERY steps apart, the first one EVERY steps after the branch step.
# Refresh steps are absolute global steps: K x the trunk's epoch_length (read from its args.json, where the trainer writes it) + i x EVERY, i = 1..COUNT. The two planned runs, at the trunk's epoch_length 2666 (4 cards):
#   sbatch run_d1_8_branch_gram_gaudi.sh 10 40          -> run dir ssl_dinov3_cxr_constlr_nogram_e0010_gram_every10000x3_seed_100, Gram refreshes after steps 36660,46660,56660
#   sbatch run_d1_8_branch_gram_gaudi.sh 10 40 2666 30  -> run dir ssl_dinov3_cxr_constlr_nogram_e0010_gram_every2666x30_seed_100, one refresh per epoch for 30 epochs (the professor's idea)
# chain/constlr.sh owns the recipe (the LR file, --min_lr = --lr, --no_lr_scaling, the 1000 ceiling, ssl_launch); the setting on the last line is its CONSTLR_BRANCH_* override. It validates everything and prints
# the `run:` and `gram refresh steps:` lines before launching (see its header).
# Extend a branch: re-submit with a larger STOP and the SAME K / EVERY / COUNT (they name the run dir and fix the refresh list, which --resume locks); it resumes from its last epoch checkpoint, and the source
# checkpoint is no longer needed then.
# Branches of an Ark-teacher trunk (run_d1_8_constlr_arkteacher_gaudi.sh: an epoch = a cycle, every epoch checkpoint kept; CONSTLR_ARK_LOCK picks the arm, 0 when unset):
#   sbatch --export=ALL,CONSTLR_BRANCH_ARK=1 run_d1_8_branch_gram_gaudi.sh 1 40 2665 30                   -> from ssl_dinov3_cxr_constlr_arkteacher_seed_100 (no lock passes, 2665 steps a cycle) into
#                                                                                                         ssl_dinov3_cxr_constlr_arkteacher_e0001_gram_every2665x30_seed_100
#   sbatch --export=ALL,CONSTLR_BRANCH_ARK=1,CONSTLR_ARK_LOCK=1 run_d1_8_branch_gram_gaudi.sh 1 40 3997 30 -> from ssl_dinov3_cxr_constlr_arkteacher_lock_seed_100 (lock-release, 3997 steps a cycle) into
#                                                                                                         ssl_dinov3_cxr_constlr_arkteacher_lock_e0001_gram_every3997x30_seed_100
# (EVERY = the trunk's epoch_length, read from its args.json: the refreshes fall on the next COUNT cycle boundaries.) The two variables reach chain/constlr.sh through the job's environment; it checks that
# the trunk's args.json records the lock arm its run dir stands for (the branch inherits it) and keeps the trunk's mode flags (--cyclic_ark_teacher --heads per_dataset, the lock flag,
# --keep_all_checkpoints) on the branch's line: the trunk's line with --no_gram replaced by the branch pair, as for the no-Gram trunk. Pass the same variables on every re-submit of such a branch (without
# them the same arguments name another trunk's branch, a different run dir).
# NOT a chain phase: it is submitted on its own and must never be added to chain/phases.txt. The log is the Slurm .out (no chain/log file, no chain/done marker); its first lines must show the `run:` line, the
# `gram refresh steps:` line, the trainer's `Gram refresh scheduled after step ...` lines and its `branched from ...` line (a re-submit: `resumed from ...`).
export PT_HPU_LAZY_MODE=1
[[ $# -ge 2 && $# -le 4 ]] || { echo "ERROR: usage: sbatch run_d1_8_branch_gram_gaudi.sh <K> <STOP> [EVERY=10000] [COUNT=3]; got $# arguments" >&2; exit 1; }
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
CONSTLR_BRANCH_EPOCH=$1 CONSTLR_BRANCH_STOP=$2 CONSTLR_BRANCH_EVERY=${3:-10000} CONSTLR_BRANCH_COUNT=${4:-3} exec ./chain/constlr.sh
