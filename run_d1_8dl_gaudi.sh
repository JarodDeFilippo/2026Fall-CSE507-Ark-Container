#!/usr/bin/env bash
#SBATCH --job-name=d1-8dl-g1
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:1
#SBATCH -c 18
#SBATCH --mem=64G
# D-long: arm D's frozen-backbone linear probe on the stock DINOv3 weights, 50 cycles, on ONE Gaudi card. chain/armD_long.sh owns the ARK_*
# recipe, the weights guard and the start-or-resume logic, so none of that is repeated here.
# Global batch 200 (ARK_BATCH_SIZE in chain/armD_long.sh) = 200 per card on this one card; engine.py divides it by the world size (arm D: 50 per card on 4).
# Re-submitting this same script continues the run from its latest checkpoint (a finished run just reports "armD_long already done" and exits 0).
# The 3-day limit above is only a default: override it at submission, e.g. sbatch -t 1-12:00:00 run_d1_8dl_gaudi.sh
# The maintenance reservation (Oct 12 08:00 to Oct 15 15:00) means Slurm will not start a job whose limit crosses it: a job that cannot end
# before Oct 12 08:00 waits until Oct 15 15:00, so shorten -t to start sooner and re-submit afterwards to continue.
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated card before the rank starts
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
exec ./chain/armD_long.sh
