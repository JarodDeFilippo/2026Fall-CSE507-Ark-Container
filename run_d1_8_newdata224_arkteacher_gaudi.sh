#!/usr/bin/env bash
#SBATCH --job-name=d1-8-newdata224
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=256G
# DINO-X Ark teacher, 224 px, new data: a NEW run, ssl_dinov3_cxr_newdata224_arkteacher_seed_100, started from the finished 224 px Ark-teacher trunk (ssl_dinov3_cxr_constlr_arkteacher_seed_100, its epoch-200 checkpoint)
# and trained with the trunk's recipe, unchanged, on seven datasets: the trunk's MIMIC, CheXpert, ChestXray14 and VinDrCXR plus PadChest, TAIXRay and ReXGradient (14 heads; the three new datasets' heads start as copies of MIMIC's).
# TAIXRay is read from the authors' native-resolution files (base/datasets_config_taix_original.yaml). chain/newdata224.sh owns the recipe, so none of it is repeated here (its header lists the flags, what a fresh start restarts
# and the checks). What this wrapper adds is the mount of the new datasets: EXTRA_BINDS (gaudi.sh) binds /scratch/$USER/datasets-new, where TAIXRay and ReXGradient live, read-only next to /data; PadChest is under
# /data/jliang12/jpang12/dataset/PadChest/image_zips/ already.
# 4 cards, as the trunk, and no other count: the global batch (Sinkhorn-Knopp centering and the DINO / iBOT statistics are computed over all ranks) is 64 x 4 = 256 and the LR is the trunk's absolute one, so another card count would change
# both; chain/newdata224.sh refuses one. --mem=256G is run_d1_8_newdata512_arkteacher_gaudi.sh's 4-card request, not measured for this run (the trunk's 128G was measured on four datasets, 53 GB; the seven include native-size 16-bit PNGs).
# Before the first launch, on Sol: unpack TAIXRay's native files once (sbatch base/dataset/TAIXRay/taix_unpack_original.sh train), and create the settings file chain/newdata224_stop_after.txt next to chain/main_lr.txt (0.002, the trunk's):
# both cluster-only, never in the repo, so an rsync cannot overwrite a raised value. The stop file holds the epoch (= cycle) to stop after, a positive integer of at most 1000; echo 1000 > chain/newdata224_stop_after.txt means no early
# stop: each 7-day job runs to its wall and the next resumes. A cycle is 4763 steps (the trunk's was 2665).
# Two links in a row, as the 512 px run: the partition limit is 7 days, so submit the run and, behind it, the same script again; the second starts when the first ends, however it ends, and resumes from the last epoch checkpoint
# (a kill or a time limit mid-cycle loses at most that cycle; if the first link already reached the stop epoch, the second finds nothing to do and ends):
#   A=$(sbatch --parsable run_d1_8_newdata224_arkteacher_gaudi.sh)
#   sbatch -d afterany:$A run_d1_8_newdata224_arkteacher_gaudi.sh
# To stop earlier, lower the number in chain/newdata224_stop_after.txt (or scancel); to run further, raise it and re-submit this same script.
# Disk: a checkpoint holds 14 heads of 23.1M parameters (student and teacher) plus AdamW's two moments, about 6.5 GB, and none is pruned (--keep_all_checkpoints): about 6.9 GB a cycle with its export, 690 GB per 100 cycles in
# the run dir.
# NOT a chain phase: it is submitted on its own and must never be added to chain/phases.txt. The log is the Slurm .out (no chain/log file, no chain/done marker); its first lines show the `run:` and `LR:` lines of
# chain/newdata224.sh, the trainer's `Gram: OFF (--no_gram)`, `heads: per_dataset`, `crops: global 224, local 96 (x8), no Gram teacher (global crops cut at 448 and resized to 224, as in the Gram runs); JPEG decode min side 1024`,
# `initialized from ... loaded strictly but for the new datasets' heads` and `new heads: ReXGradient, TAIXRay, PadChest ...` lines, and the `cycle (--cyclic_ark_teacher): ...` line with the seven datasets' image counts and the steps a cycle.
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
EXTRA_BINDS=/scratch/$USER/datasets-new:/scratch/$USER/datasets-new:ro exec ./chain/newdata224.sh
