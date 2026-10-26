#!/usr/bin/env bash
#SBATCH --job-name=d1-8-nogram
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=256G
# DINO-X ConstLR 224, no Gram: the trunk of the Gram-anchored runs (decided 2026-10-23 by Jarod, with his professor's input). It is the `constlr` chain phase (chain/constlr.sh, run dir
# ssl_dinov3_cxr_constlr_seed_100) with ONE variable changed: the same launch line (ssl_launch's stock DINOv3 ViT-B/16 init, chain/main_lr.txt as the absolute peak LR, --min_lr = --lr, --no_lr_scaling, the
# --epochs 1000 ceiling, 64 per card on 4 cards = global batch 256, seed 100, 224 / 96 crops still cut at 448 and resized to 224 exactly as in the Gram run) with --no_gram in place of --gram_refresh_steps
# 10000,20000,30000: no Gram teacher is built, loaded or run, there is no Gram loss and there are no refreshes. chain/constlr.sh owns the recipe, so none of it is repeated here; the setting on the last line
# is its env override (CONSTLR_NOGRAM=1: run dir ssl_dinov3_cxr_constlr_nogram_seed_100, stop epoch from chain/constlr_nogram_stop_after.txt instead of chain/constlr_stop_after.txt).
# Plan: one 7-day job, probed every 10 epochs at 224 px. When its probes approach / establish a baseline (provisionally: beat Stage A's 82.0 frozen-probe mean), Gram-anchored branches start FROM its checkpoints
# (ckpt/epoch_NNNN.pth, anchor = that checkpoint's EMA teacher) while the trunk keeps running; the trainer's checkpoint retention keeps every 10th epoch.
# 4 cards, not 8: the global batch (Sinkhorn-Knopp centering and the DINO / iBOT statistics are computed over all ranks) and the per-card batch (KoLeo is computed per card) both match ConstLR 224.
# Stop epoch: chain/constlr_nogram_stop_after.txt, cluster-only (never in the repo, so an rsync cannot overwrite it), validated as constlr.sh validates its own; 100 for this job (ConstLR 224 took about 58 min
# per epoch on 4 cards WITH the Gram teacher, so 100 epochs is at most ~97 h of the 168 h). To extend: raise that file and re-submit this same script; it resumes from its last epoch checkpoint (--resume locks
# no_gram, the LR and the ceiling, not the stop epoch).
# NOT a chain phase: it is submitted on its own and must never be added to chain/phases.txt. The log is the Slurm .out (no chain/log file, no chain/done marker); its first lines must show the
# `run: ssl_dinov3_cxr_constlr_nogram_seed_100 (no Gram teacher ...` line and the trainer's `Gram: OFF (--no_gram): no Gram teacher built or loaded, ...` line, and no `Gram refresh scheduled` line.
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
CONSTLR_NOGRAM=1 exec ./chain/constlr.sh
