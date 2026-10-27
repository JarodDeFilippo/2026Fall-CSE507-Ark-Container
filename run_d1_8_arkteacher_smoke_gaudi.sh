#!/usr/bin/env bash
#SBATCH --job-name=d1-8-arkteacher-smoke
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 01:15:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=256G
# Standalone timing job for the Stage A SSL trainer (optional; the chain's phases run inside their own allocation).
# Submit from the repo root, or set ARK_ROOT to it.
cd "${ARK_ROOT:-${SLURM_SUBMIT_DIR:?sbatch from the repo root or set ARK_ROOT}}" || exit 1
source ./chain/ssl_common.sh
# the lock arm (see below), checked before the cards are touched
LOCK=${CONSTLR_ARK_LOCK:-0}
[[ $LOCK == 0 || $LOCK == 1 ]] || { echo "ERROR: CONSTLR_ARK_LOCK holds '$LOCK', which is not empty, 0 or 1" >&2; exit 1; }
TOKEN=  LOCK_FLAG=--no_lock_half_pass
[[ $LOCK == 0 ]] || { TOKEN=_lock; LOCK_FLAG=--lock_half_pass; }
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
# Smoke run for the Ark-teacher trunk on 4 cards (Jarod, 2026-10-26): the trunk's launch line (chain/constlr.sh with CONSTLR_NOGRAM=1 CONSTLR_ARK=1: DINOv3 recipe --drop_path_rate 0.3
# --rope_rescale_coords 2 --untie_local_cls_norm, --no_gram, --cyclic_ark_teacher --heads per_dataset <lock flag> --keep_all_checkpoints, flat LR from chain/main_lr.txt's value 2e-3, the --epochs 1000
# ceiling) stopped after 2300 steps in a throwaway run dir. The lock arm is CONSTLR_ARK_LOCK as for the trunk (0 when unset; sbatch --export=ALL,CONSTLR_ARK_LOCK=1 for 1), in the run dir as there:
# ssl_dinov3_cxr_arkteacher_smoke_seed_100 for 0, ssl_dinov3_cxr_arkteacher_lock_smoke_seed_100 for 1. 2300 steps cross the first dataset boundary in either arm (at 64 x 4): 0 runs MIMIC's release pass
# (1441 steps), then CheXpert's; 1 runs MIMIC's lock pass (720) and release pass (1441), then CheXpert's lock pass (from step 2161). So the first teacher blend, the first switch and (1) the lock pass's
# LR-0 backbone all run on HCCL, and the --max_steps save (ckpt/maxsteps_0002300.pth: 8 heads, the per-rank RNG gather) follows. Expected log, in this order:
#   cycle (--cyclic_ark_teacher): ... MIMIC 368960 images: [lock 720 + ]release 1441 steps, CheXpert 223414 images: [lock 436 + ]release 872 steps, ...; 2665 [3997] steps a cycle (= one epoch)
#   (with the `heads: per_dataset` and `teacher views: clean` lines around it)
#   1 only: dataset pass after step 0: MIMIC lock (cycle 0, dataset 1/4, a random half, 184480 of 368960 images, 720 steps): backbone frozen (LR 0, no optimizer step), heads train
#   dataset pass after step 0 [720]: MIMIC release (cycle 0, dataset 1/4, all 368960 images, 1441 steps): backbone and heads train; the backbone at LR 0 in this frozen cycle
#   step rows tagged | MIMIC:lock (1 only), then | MIMIC:release
#   teacher blend after step 1441 [2161] (end of MIMIC's release pass, cycle 0, dataset 1/4): teacher <- 0.8 x teacher + 0.2 x student, backbone and all heads
#   dataset pass after step 1441 [2161]: CheXpert release [lock] (cycle 0, dataset 2/4, ...)
#   step rows tagged | CheXpert:release [CheXpert:lock], then
#   --max_steps 2300 reached at step 2300
# with no error. ssl_launch passes --resume, harmless here.
ssl_launch ssl_dinov3_cxr_arkteacher${TOKEN}_smoke_seed_100 --drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm --no_gram --cyclic_ark_teacher --heads per_dataset "$LOCK_FLAG" --keep_all_checkpoints --lr 0.002 --min_lr 0.002 --no_lr_scaling --epochs 1000 --max_steps 2300
