#!/usr/bin/env bash
#SBATCH --job-name=d1-8-arkteacher-smoke
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=256G
# Standalone timing job for the Stage A SSL trainer (optional; the chain's phases run inside their own allocation).
# Submit from the repo root, or set ARK_ROOT to it.
cd "${ARK_ROOT:-${SLURM_SUBMIT_DIR:?sbatch from the repo root or set ARK_ROOT}}" || exit 1
source ./chain/ssl_common.sh
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
# Smoke run for the Ark-teacher trunk on 4 cards (Jarod, 2026-10-26): the trunk's launch line (chain/constlr.sh with CONSTLR_NOGRAM=1 CONSTLR_ARK=1: DINOv3 recipe --drop_path_rate 0.3
# --rope_rescale_coords 2 --untie_local_cls_norm, --no_gram, --cyclic_ark_teacher --heads per_dataset --switch_lock_steps 100 --keep_all_checkpoints, flat LR from chain/main_lr.txt's value 2e-3, the
# --epochs 1000 ceiling) stopped after 1500 steps in a throwaway run dir: past MIMIC's 1441-step dataset epoch (at 64 x 4), so the first dataset switch, its lock and the first teacher blend all run on HCCL,
# and the --max_steps save (ckpt/maxsteps_0001500.pth: 8 heads, the per-rank RNG gather) follows. Expected log, in this order:
#   cycle (--cyclic_ark_teacher): ... MIMIC 368960 images / 1441 steps, CheXpert 223414 images / 872 steps, ChestXray14 75312 images / 294 steps, VinDrCXR 15000 images / 58 steps; 2665 steps a cycle (= one epoch)
#   (with the `heads: per_dataset`, `teacher views: clean` and VinDrCXR `NOTE:` lines around it)
#   dataset switch after step 0: start -> MIMIC (cycle 0, dataset 1/4, 1441 steps): backbone LR 0 for its first 100 steps (--switch_lock_steps 100; a no-op in this frozen cycle)
#   step rows tagged | MIMIC, then
#   teacher blend after step 1441 (end of MIMIC's epoch, cycle 0, dataset 1/4): teacher <- 0.8 x teacher + 0.2 x student, backbone and all heads
#   dataset switch after step 1441: MIMIC -> CheXpert (cycle 0, dataset 2/4, 872 steps): backbone LR 0 for its first 100 steps (--switch_lock_steps 100; a no-op in this frozen cycle)
#   step rows tagged | CheXpert, then
#   --max_steps 1500 reached at step 1500
# with no error. ssl_launch passes --resume, harmless here.
ssl_launch ssl_dinov3_cxr_arkteacher_smoke_seed_100 --drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm --no_gram --cyclic_ark_teacher --heads per_dataset --switch_lock_steps 100 --keep_all_checkpoints --lr 0.002 --min_lr 0.002 --no_lr_scaling --epochs 1000 --max_steps 1500
