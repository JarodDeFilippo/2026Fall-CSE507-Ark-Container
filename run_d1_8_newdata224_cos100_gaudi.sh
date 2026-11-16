#!/usr/bin/env bash
#SBATCH --job-name=d1-8-newdata224-cos100
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:8
#SBATCH -c 144
#SBATCH --mem=480G
# DINO-X Ark teacher with a cosine schedule: a NEW run, ssl_dinov3_cxr_newdata224_cos100_tiedhead_arkteacher_seed_100, started from the stock DINOv3 ViT-B/16 (no start checkpoint) and trained for 100 cycles on seven datasets: MIMIC, CheXpert, ChestXray14 and VinDrCXR
# plus PadChest, TAIXRay and ReXGradient, with one tied DINO + iBOT head shared by all of them, student crops of 224 px cut from a 512 px squash, a whole-image teacher target at 448 px, a 10-cycle LR warm-up then a cosine from 2e-3 to 1e-6, and Ark+'s augmentation.
# TAIXRay is read from the authors' native-resolution files (base/datasets_config_taix_original.yaml). chain/newdata224_cos100.sh owns the recipe, so none of it is repeated here (its header lists the flags, the schedules and the checks). What this wrapper adds is the mount of
# the new datasets: EXTRA_BINDS (gaudi.sh) binds /scratch/$USER/datasets-new, where TAIXRay and ReXGradient live, read-only next to /data; PadChest is under /data/jliang12/jpang12/dataset/PadChest/image_zips/ already.
# 8 cards, and no other count: the global batch (Sinkhorn-Knopp centering and the DINO / iBOT statistics are computed over all ranks) is 64 x 8 = 512, the LR is absolute and a cycle is 2380 steps at that batch, so another card count would change all three; chain/newdata224_cos100.sh
# refuses one. --mem=480G is run_d1_8sw8_gaudi.sh's 8-card request, not measured for this run.
# Before the first launch, on Sol: unpack TAIXRay's native files once (sbatch base/dataset/TAIXRay/taix_unpack_original.sh train), and run the smoke, run_d1_8_newdata224_cos100_smoke_gaudi.sh (sbatch it, no arguments: does the run's launch line fit at 64 per card
# on 8 cards, and how fast it steps and how long the loader makes the ranks wait, per dataset). There are no settings files to create: the run trains all 100 cycles of its schedule, with no stop epoch.
# Two links in a row: the partition limit is 7 days, so submit the run and, behind it, the same script again; the second starts when the first ends, however it ends, and resumes from the last epoch checkpoint (a kill or a time limit mid-cycle loses at most that cycle; if the first
# link already trained all 100 cycles, the second finds nothing to do and ends):
#   A=$(sbatch --parsable run_d1_8_newdata224_cos100_gaudi.sh)
#   sbatch -d afterany:$A run_d1_8_newdata224_cos100_gaudi.sh
# To stop earlier, scancel (checkpoints are written at cycle ends only: a kill mid-cycle loses at most that cycle); to continue, re-submit this same script (--resume continues from the last epoch checkpoint and locks every setting, so the re-submit must launch the same line).
# Disk: a checkpoint holds the student and the teacher (the backbone, about 86M parameters, and one head of 23.1M each) plus AdamW's two moments, about 1.7 GB by parameter count (an estimate, not measured), and none is pruned (--keep_all_checkpoints).
# NOT a chain phase: it is submitted on its own and must never be added to chain/phases.txt. The log is the Slurm .out (no chain/log file, no chain/done marker); its first lines show the `run:` line of chain/newdata224_cos100.sh and the trainer's `recipe (student, ...`,
# `iBOT head: none, the iBOT loss uses the DINO head ...` and `Gram: OFF (--no_gram) ...` lines, `crops: global 224, local 96 (x8), no Gram teacher (global crops cut at 224, no resize); pre-resize 512x512 ...`, `whole-image teacher (--teacher_whole_image_size) ...`, the
# `cycle (--cyclic_ark_teacher): ...` line with the seven datasets' image counts and the steps a cycle (2380 if the lists have the plan's sizes) and `epoch_length ... steps, ... steps in the 100-epoch schedule, stopping after epoch 100`.
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
EXTRA_BINDS=/scratch/$USER/datasets-new:/scratch/$USER/datasets-new:ro exec ./chain/newdata224_cos100.sh
