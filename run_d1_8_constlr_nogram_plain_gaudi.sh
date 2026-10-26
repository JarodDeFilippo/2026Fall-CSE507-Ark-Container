#!/usr/bin/env bash
#SBATCH --job-name=d1-8-nogram-plain
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=256G
# DINO-X ConstLR 224, no Gram, plain: the one-variable control for the hypothesis "Gram anchoring from step 0 capped the probe" (Jarod, 2026-10-26). It is the `constlr` chain phase (chain/constlr.sh, run dir
# ssl_dinov3_cxr_constlr_seed_100) with ONE variable changed: the same launch line (ssl_launch's stock DINOv3 ViT-B/16 init, chain/main_lr.txt as the absolute peak LR, --min_lr = --lr, --no_lr_scaling, the
# --epochs 1000 ceiling, 64 per card on 4 cards = global batch 256, seed 100, 224 / 96 crops still cut at 448 and resized to 224 exactly as in the Gram run) with --no_gram in place of --gram_refresh_steps
# 10000,20000,30000: no Gram teacher is built, loaded or run, there is no Gram loss and there are no refreshes. Unlike the no-Gram trunk (run_d1_8_constlr_nogram_gaudi.sh, run dir ssl_dinov3_cxr_constlr_nogram_seed_100)
# it does NOT pass DINOv3's training-recipe flags (--drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm), so --no_gram is the only difference from the Gram run. chain/constlr.sh owns the recipe, so none
# of it is repeated here; the setting on the last line is its env override (CONSTLR_NOGRAM=1 CONSTLR_PLAIN=1: run dir ssl_dinov3_cxr_constlr_nogram_plain_seed_100, stop epoch from
# chain/constlr_nogram_plain_stop_after.txt instead of chain/constlr_stop_after.txt).
# Plan: one 7-day job, probed every 10 epochs at 224 px and compared with the Gram run's frozen-probe means (mean of the four datasets, test mAUC x100) at the same epochs: 81.18 / 81.53 / 81.61 / 81.58 at epochs
# 10 / 20 / 30 / 40. Intended for a group member to submit; nothing branches from it (constlr.sh refuses CONSTLR_PLAIN with CONSTLR_BRANCH_*). The checkout it runs in, /scratch/$USER/2026Fall-CSE507-Ark-Container, needs
# the stock DINOv3 weights, chain/main_lr.txt holding the Gram run's LR (2e-3: any other value would be a second variable) and the stop file below.
# 4 cards, not 8: the global batch (Sinkhorn-Knopp centering and the DINO / iBOT statistics are computed over all ranks) and the per-card batch (KoLeo is computed per card) both match ConstLR 224.
# Stop epoch: chain/constlr_nogram_plain_stop_after.txt, cluster-only (never in the repo, so an rsync cannot overwrite it), validated as constlr.sh validates its own; at least 40 to reach the last comparison epoch
# (ConstLR 224 took about 58 min per epoch on 4 cards WITH the Gram teacher, so 100 epochs is at most ~97 h of the 168 h). To extend: raise that file and re-submit this same script; it resumes from its last
# epoch checkpoint (--resume locks no_gram, the LR and the ceiling, not the stop epoch).
# NOT a chain phase: it is submitted on its own and must never be added to chain/phases.txt. The log is the Slurm .out (no chain/log file, no chain/done marker); its first lines must show the
# `run: ssl_dinov3_cxr_constlr_nogram_plain_seed_100 (no Gram teacher ...` line and the trainer's `Gram: OFF (--no_gram): no Gram teacher built or loaded, ...` line, and neither a `Gram refresh scheduled` line nor the trainer's `recipe (...)` line.
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
CONSTLR_NOGRAM=1 CONSTLR_PLAIN=1 exec ./chain/constlr.sh
