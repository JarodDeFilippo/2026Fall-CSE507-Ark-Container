#!/usr/bin/env bash
#SBATCH --job-name=d1-8-newdata512
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=256G
# DINO-X Ark teacher, 512 px, new data: a NEW run, ssl_dinov3_cxr_newdata512_arkteacher_seed_100, started from the finished 224 px Ark-teacher trunk (ssl_dinov3_cxr_constlr_arkteacher_seed_100, its epoch-200 checkpoint)
# and trained at 512 px on seven datasets: the trunk's MIMIC, CheXpert, ChestXray14 and VinDrCXR plus PadChest, TAIXRay and ReXGradient (14 heads; the three new datasets' heads start as copies of MIMIC's).
# chain/newdata512.sh owns the recipe, so none of it is repeated here (its header lists the flags, the LR rule and the checks). What this wrapper adds is the mount of the new datasets: EXTRA_BINDS (gaudi.sh)
# binds /scratch/$USER/datasets-new, where TAIXRay and ReXGradient live, read-only next to /data; PadChest is under /data/jliang12/jpang12/dataset/PadChest/image_zips/ already.
# 4 cards by default. 8 cards: sbatch --gres=gpu:hl225:8 -c 144 --mem=480G run_d1_8_newdata512_arkteacher_gaudi.sh (the same script; the command line overrides the #SBATCH lines above). The cards and the LR adapt: chain/newdata512.sh
# counts the allocated cards, takes the global batch G = per-card batch x cards and lets the trainer's sqrt_wrt_1024 rule scale the LR to it, so the effective LR changes with G (it prints the base LR, the effective LR, G and the
# cards before the launch). The 480G is run_d1_8sw8_gaudi.sh's assumption, not a measured figure.
# Settings, read from the repo root by chain/newdata512.sh, cluster-only (never in the repo): chain/main_lr.txt (constlr's absolute LR, as in every other SSL run), chain/newdata512_stop_after.txt (the epoch = cycle to stop
# after, a positive integer of at most 1000; create it before submitting with echo <epochs> > chain/newdata512_stop_after.txt) and chain/newdata512_batch_per_gpu.txt (the per-card batch: no file, no launch). A missing, empty or invalid
# one fails the run with an ERROR before anything is launched, as does a missing epoch-200 checkpoint of the 224 trunk while the new run has no epoch checkpoint of its own yet.
# Stop / extend / resume: the stop epoch is the number in chain/newdata512_stop_after.txt. To run further, raise it and re-submit this same script; to stop earlier, lower it or scancel (a kill mid-cycle loses at most that cycle:
# checkpoints are written at cycle ends only). A re-submit resumes from the last epoch checkpoint (--resume locks the data sets and their order, the crops, the batch, the LR, the ceiling, the mode and --init_new_heads_from,
# not the stop epoch), and then needs no epoch-200 checkpoint any more.
# Disk: a checkpoint holds 14 heads of 23.1M parameters (student and teacher) plus AdamW's two moments, about 6.5 GB (the 224 trunk's 8 heads: 4.3 GB), and none is pruned (--keep_all_checkpoints): about 6.9 GB a cycle with its
# export, 690 GB per 100 cycles in the run dir.
# base/datasets_config.yaml: the TAIXRay and ReXGradient entries (data_dir, train_list) are filled in once their on-disk layout under /scratch/$USER/datasets-new is known, and their lists written (a header row, the image path in
# column 0, relative to data_dir); until then the trainer stops at the start, on the missing list. PadChest's list ships in the repo, base/dataset/PadChest/padchest_train.csv.
# NOT a chain phase: it is submitted on its own and must never be added to chain/phases.txt. The log is the Slurm .out (no chain/log file, no chain/done marker); its first lines show the `run:` and `LR:` lines of
# chain/newdata512.sh, the trainer's `Gram: OFF (--no_gram)`, `heads: per_dataset`, `crops: global 512, local 224 (x8), no Gram teacher (global crops cut at 512, no resize)`, `initialized from ... loaded strictly but for the new
# datasets' heads` and `new heads: ReXGradient, TAIXRay, PadChest ...` lines, and the `cycle (--cyclic_ark_teacher): ...` line with the seven datasets' image counts and the steps a cycle.
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
EXTRA_BINDS=/scratch/$USER/datasets-new:/scratch/$USER/datasets-new:ro exec ./chain/newdata512.sh
