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
# --untie_local_cls_norm, --no_gram) plus --cyclic_ark_teacher --heads per_dataset <lock flag> --keep_all_checkpoints. Jarod's reasons, the Ark+ / FX recipe: the datasets are trained in turn, largest to
# smallest, every batch from one dataset, a cycle over the four being one epoch; the EMA teacher is not updated per step but frozen through every pass and blended teacher <- 0.80 x teacher + 0.20 x student
# (backbone and all heads) after each dataset's release pass (its full pass), and it scores clean views (the global crops' crop + flip, no color jitter or blur); one DINO and one iBOT head per dataset
# (the batch's pair is used); Gram off; every epoch checkpoint is kept so that a Gram branch can start from any cycle, cycle 1 included.
# The lock arm is a per-run ablation, chosen at launch with CONSTLR_ARK_LOCK (chain/constlr.sh; 0 when unset):
#   sbatch run_d1_8_constlr_arkteacher_gaudi.sh                               -> 0: no lock passes, each dataset's release pass alone (MIMIC 1441, CheXpert 872, ChestXray14 294, VinDrCXR 58
#                                                                                steps at 64 x 4: 2665 a cycle, about 33 min); run dir ssl_dinov3_cxr_constlr_arkteacher_seed_100,
#                                                                                stop file chain/constlr_arkteacher_stop_after.txt
#   sbatch --export=ALL,CONSTLR_ARK_LOCK=1 run_d1_8_constlr_arkteacher_gaudi.sh -> 1: FX lock-release, per dataset a lock pass over a random half (backbone LR 0, heads train: MIMIC 720, CheXpert 436,
#                                                                                ChestXray14 147, VinDrCXR 29 steps) before its release pass: 3997 a cycle, about 50 min; run dir
#                                                                                ssl_dinov3_cxr_constlr_arkteacher_lock_seed_100, stop file chain/constlr_arkteacher_lock_stop_after.txt
# The two arms never share a run dir or a stop file, so they can run side by side and a resume never flips the lock (the trainer also locks lock_half_pass on --resume). Pass CONSTLR_ARK_LOCK=1 on every
# re-submit of the lock arm: without it this script launches the other arm's run dir. Gram branches: sbatch --export=ALL,CONSTLR_BRANCH_ARK=1[,CONSTLR_ARK_LOCK=1] run_d1_8_branch_gram_gaudi.sh <K> <STOP>
# [EVERY] [COUNT] (EVERY = the trunk's epoch_length refreshes at cycle boundaries; the branch inherits the trunk's lock from its args.json, its run dir carries the trunk's _lock token).
# chain/constlr.sh owns the recipe, so none of it is repeated here; the settings on the last line are its env overrides (CONSTLR_NOGRAM=1 CONSTLR_ARK=1).
# Disk: a checkpoint holds 8 heads of 23.1M parameters (student and teacher) plus AdamW's two moments, about 4.3 GB (2.1 GB with shared heads), and none is pruned: about 4.7 GB a cycle with its export,
# 470 GB per 100 cycles in the run dir. A kill mid-cycle loses at most that cycle: checkpoints are written at cycle ends only.
# Stop epoch (= cycle): the arm's stop file, cluster-only (never in the repo, so an rsync cannot overwrite it), validated as constlr.sh validates its own. To extend: raise that file and re-submit this same
# script (with the same CONSTLR_ARK_LOCK); it resumes from its last epoch checkpoint (--resume locks the mode, its settings, no_gram, the LR and the ceiling, not the stop epoch).
# 4 cards, not 8: the global batch (Sinkhorn-Knopp centering and the DINO / iBOT statistics are computed over all ranks) and the per-card batch (KoLeo is computed per card) both match ConstLR 224.
# NOT a chain phase: it is submitted on its own and must never be added to chain/phases.txt. The log is the Slurm .out (no chain/log file, no chain/done marker); its first lines must show the
# `run: ssl_dinov3_cxr_constlr_arkteacher[_lock]_seed_100 (no Gram teacher ...` line naming the lock arm, the trainer's `Gram: OFF (--no_gram)`, `heads: per_dataset`, `teacher views: clean` and
# `cycle (--cyclic_ark_teacher): ... MIMIC 368960 images: [lock 720 + ]release 1441 steps, ...; 2665 [3997] steps a cycle` lines, and then one `dataset pass after step ...` line per pass and one
# `teacher blend after step ...` line per release pass.
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
CONSTLR_NOGRAM=1 CONSTLR_ARK=1 exec ./chain/constlr.sh
