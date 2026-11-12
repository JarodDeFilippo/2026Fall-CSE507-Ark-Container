#!/usr/bin/env bash
#SBATCH --job-name=d1-8-newdata512-cropsrc
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:8
#SBATCH -c 144
#SBATCH --mem=480G
# DINO-X Ark teacher, 512 px, new data, crops cut from the source image: a NEW run, ssl_dinov3_cxr_newdata512_cropsrc_arkteacher_seed_100, that is run_d1_8_newdata512_arkteacher_gaudi.sh's run with the crops cut from the decoded image
# instead of from an image squashed to 512 x 512 (no --pre_resize, the JPEG decode side held at 1024 with --decode_min_side), TAIXRay read from the authors' native-resolution files (base/datasets_config_taix_original.yaml) and a constant
# base LR of 1e-4, 1.4142e-4 effective at batch 128 (LR_BASE in chain/newdata512_cropsrc.sh; the run it copies used a base of 0.001, 1.4142e-3 effective). Started from the
# finished 224 px Ark-teacher trunk (ssl_dinov3_cxr_constlr_arkteacher_seed_100, its epoch-200 checkpoint) and trained at 512 px on seven datasets: the trunk's MIMIC, CheXpert, ChestXray14 and VinDrCXR plus PadChest, TAIXRay and
# ReXGradient (14 heads; the three new datasets' heads start as copies of MIMIC's). chain/newdata512_cropsrc.sh owns the run (its header lists exactly what differs from chain/newdata512.sh, and how to start from a checkpoint of the 224 px
# seven-dataset run, chain/newdata224.sh, instead), so none of it is repeated here. What this wrapper adds is the mount of the new datasets: EXTRA_BINDS (gaudi.sh) binds /scratch/$USER/datasets-new, where TAIXRay and ReXGradient live,
# read-only next to /data; PadChest is under /data/jliang12/jpang12/dataset/PadChest/image_zips/ already.
# 8 cards by default, as the run it copies (16 per card, global batch 128). 4 cards: sbatch --gres=gpu:hl225:4 -c 72 --mem=256G run_d1_8_newdata512_cropsrc_arkteacher_gaudi.sh (the same script; the command line overrides the #SBATCH lines
# above). The cards and the LR adapt: chain/newdata512_cropsrc.sh counts the allocated cards, takes the global batch G = per-card batch x cards and lets the trainer's sqrt_wrt_1024 rule scale the LR to it, so the effective LR changes with G
# (it prints the base LR, the effective LR, G and the cards before the launch). The 480G is run_d1_8sw8_gaudi.sh's assumption, not a measured figure.
# Settings, read from the repo root by chain/newdata512_cropsrc.sh, cluster-only (never in the repo): chain/newdata512_cropsrc_stop_after.txt (the epoch = cycle to stop after, a positive integer of at most 1000;
# echo 1000 > chain/newdata512_cropsrc_stop_after.txt means no early stop) and chain/newdata512_cropsrc_batch_per_gpu.txt (the per-card batch, 16 as the run it copies: echo 16 > chain/newdata512_cropsrc_batch_per_gpu.txt; no file, no
# launch). A missing, empty or invalid one fails the run with an ERROR before anything is launched, as does a missing start checkpoint while the new run has no epoch checkpoint of its own yet. chain/main_lr.txt is not read: the LR is
# LR_BASE in the script. Before the first launch, on Sol: unpack TAIXRay's native files once (sbatch base/dataset/TAIXRay/taix_unpack_original.sh train).
# Two links in a row, as the 512 px run: the partition limit is 7 days, so submit the run and, behind it, the same script again; the second starts when the first ends, however it ends, and resumes from the last epoch checkpoint (a kill or a
# time limit mid-cycle loses at most that cycle; if the first link already reached the stop epoch, the second finds nothing to do and ends):
#   A=$(sbatch --parsable run_d1_8_newdata512_cropsrc_arkteacher_gaudi.sh)
#   sbatch -d afterany:$A run_d1_8_newdata512_cropsrc_arkteacher_gaudi.sh
# To start from a checkpoint of the 224 px seven-dataset run instead of the trunk's epoch 200, export CROPSRC_INIT and CROPSRC_NEW_HEADS_FROM=none before BOTH submissions (sbatch passes the environment on; a resume compares them with the
# checkpoint's); chain/newdata512_cropsrc.sh's header has the values.
# To stop earlier, lower the number in chain/newdata512_cropsrc_stop_after.txt (or scancel; a kill mid-cycle loses at most that cycle: checkpoints are written at cycle ends only); to run further, raise it and re-submit this same script.
# Disk: a checkpoint holds 14 heads of 23.1M parameters (student and teacher) plus AdamW's two moments, about 6.5 GB, and none is pruned (--keep_all_checkpoints): about 6.9 GB a cycle with its export, 690 GB per 100 cycles in the run dir.
# NOT a chain phase: it is submitted on its own and must never be added to chain/phases.txt. The log is the Slurm .out (no chain/log file, no chain/done marker); its first lines show the `run:` and `LR:` lines of
# chain/newdata512_cropsrc.sh, the trainer's `Gram: OFF (--no_gram)`, `heads: per_dataset`, `crops: global 512, local 224 (x8), no Gram teacher (global crops cut at 512, no resize); JPEG decode min side 1024`, `initialized from ... loaded
# strictly but for the new datasets' heads` and `new heads: ReXGradient, TAIXRay, PadChest ...` lines, and the `cycle (--cyclic_ark_teacher): ...` line with the seven datasets' image counts and the steps a cycle.
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
EXTRA_BINDS=/scratch/$USER/datasets-new:/scratch/$USER/datasets-new:ro exec ./chain/newdata512_cropsrc.sh
