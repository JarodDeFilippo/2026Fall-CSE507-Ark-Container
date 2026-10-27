#!/usr/bin/env bash
#SBATCH --job-name=d1-8-arkteacher
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=256G
# DINO-X ConstLR 224, no Gram, Ark teacher: the Ark+ / Foundation X teacher transplanted into DINO (Jarod, 2026-10-26). It is the no-Gram trunk (run_d1_8_constlr_nogram_gaudi.sh, run dir
# ssl_dinov3_cxr_constlr_nogram_seed_100) with the trainer's --cyclic_ark_teacher mode on: the same launch line (ssl_launch's stock DINOv3 ViT-B/16 init, chain/main_lr.txt as the absolute peak LR, --min_lr = --lr,
# --no_lr_scaling, the --epochs 1000 ceiling, 64 per card on 4 cards = global batch 256, seed 100, 224 / 96 crops cut at 448 and resized to 224, DINOv3's --drop_path_rate 0.3 --rope_rescale_coords 2
# --untie_local_cls_norm, --no_gram) plus --cyclic_ark_teacher --heads per_dataset --switch_lock_steps 100 --keep_all_checkpoints. Jarod's reasons, the Ark+ / FX recipe: the datasets are trained in turn, largest
# to smallest (MIMIC 1441, CheXpert 872, ChestXray14 294, VinDrCXR 58 steps at 256: a cycle of 2665 steps is one epoch), every batch from one dataset; the EMA teacher is not updated per step but blended
# teacher <- 0.80 x teacher + 0.20 x student (backbone and all heads) at every dataset-epoch boundary, and scores clean views (the global crops' crop + flip, no color jitter or blur); one DINO and one iBOT head
# per dataset (the batch's pair is used); the backbone is at LR 0 for the first 100 steps after every dataset switch while the heads adapt (the first cycle is frozen anyway, and VinDrCXR's 58-step epoch is
# locked throughout); Gram off; every epoch checkpoint is kept so that a Gram branch can start from any cycle, cycle 1 included (sbatch --export=ALL,CONSTLR_BRANCH_ARK=1 run_d1_8_branch_gram_gaudi.sh
# <K> <STOP> [EVERY] [COUNT]; EVERY 2665 refreshes at cycle boundaries). chain/constlr.sh owns the recipe, so none of it is repeated here; the settings on the last line are its env overrides (CONSTLR_NOGRAM=1
# CONSTLR_ARK=1: run dir ssl_dinov3_cxr_constlr_arkteacher_seed_100, stop epoch from chain/constlr_arkteacher_stop_after.txt).
# Disk: a checkpoint holds 8 heads of 23.1M parameters (student and teacher) plus AdamW's two moments, about 4.3 GB (2.1 GB with shared heads), and none is pruned: about 4.7 GB a cycle with its export,
# 470 GB per 100 cycles in the run dir. A kill mid-cycle loses at most that cycle (about 33 min): checkpoints are written at cycle ends only.
# Stop epoch (= cycle): chain/constlr_arkteacher_stop_after.txt, cluster-only (never in the repo, so an rsync cannot overwrite it), validated as constlr.sh validates its own. To extend: raise that file and
# re-submit this same script; it resumes from its last epoch checkpoint (--resume locks the mode, its five settings, no_gram, the LR and the ceiling, not the stop epoch).
# 4 cards, not 8: the global batch (Sinkhorn-Knopp centering and the DINO / iBOT statistics are computed over all ranks) and the per-card batch (KoLeo is computed per card) both match ConstLR 224.
# NOT a chain phase: it is submitted on its own and must never be added to chain/phases.txt. The log is the Slurm .out (no chain/log file, no chain/done marker); its first lines must show the
# `run: ssl_dinov3_cxr_constlr_arkteacher_seed_100 (no Gram teacher ...` line, the trainer's `Gram: OFF (--no_gram)`, `heads: per_dataset`, `teacher views: clean` and `cycle (--cyclic_ark_teacher): ...
# MIMIC 368960 images / 1441 steps, CheXpert 223414 images / 872 steps, ChestXray14 75312 images / 294 steps, VinDrCXR 15000 images / 58 steps; 2665 steps a cycle` lines, the VinDrCXR NOTE, and then one
# `dataset switch after step ...` line per dataset epoch and one `teacher blend after step ...` line per boundary.
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
CONSTLR_NOGRAM=1 CONSTLR_ARK=1 exec ./chain/constlr.sh
