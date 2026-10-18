#!/usr/bin/env bash
#SBATCH --job-name=d1-8sw
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:8
#SBATCH -c 144
#SBATCH --mem=480G
# --mem: 480G is an assumption, not a measured figure: the Gaudi node RAM is not recorded in the repo notes (check `scontrol show node gaudi001` RealMemory and keep this at <= 90% of it).
# One arm of the Step VIII fine-tuning sweep (DINOv3 ViT-B/16, Ark cyclic training) on EIGHT Gaudi cards: run_d1_8sw_gaudi.sh with 8 cards, for an arm whose per-card batch does not fit at 2 (the input-size arms).
# usage, from the repo root: sbatch --job-name=d1-8sw-<id> --dependency=singleton run_d1_8sw8_gaudi.sh <sweep id, a row of sweep/wave1.tsv>
# (sweep/submit_wave.sh does this for a whole wave). A manual resubmit must use the same --job-name=d1-8sw-<id> --dependency=singleton: Slurm then runs at most one
# job of that name per user at a time, whichever of run_d1_8sw_gaudi.sh / run_d1_8sw4_gaudi.sh / run_d1_8sw8_gaudi.sh it runs, so a resubmit while the arm's previous link is queued or running waits for it instead of starting a second trainer on the same
# run dir. The #SBATCH job name above is only the fallback for a bare sbatch, which has no such guard.
# chain/sweep_run.sh owns the row lookup, the ARK_* recipe, the weights guard and the start-or-resume logic, so none of that is repeated here.
# Global batch 200 (ARK_BATCH_SIZE in chain/sweep_run.sh) = 25 per card on these 8 cards; engine.py divides it by the world size.
# 16 data workers per rank (ARK_WORKERS in chain/sweep_run.sh): 8 ranks x 16 = 128 workers fit the 144 cores requested above (18 per card).
# Re-submitting the same sweep id continues that run from its latest checkpoint (a finished run just reports "sweep_run <id> already done" and exits 0).
# The 1-day limit above is only a default: override it at submission, e.g. sbatch -t 12:00:00 run_d1_8sw8_gaudi.sh S18
# The maintenance reservation (Oct 12 08:00 to Oct 15 15:00) means Slurm will not start a job whose limit crosses it: with this 1-day limit nothing
# starts after Oct 11 08:00 until Oct 15 15:00, so shorten -t to start sooner (the job must end before Oct 12 08:00) and re-submit afterwards to continue.
[[ $# -eq 1 ]] || { echo "usage: sbatch run_d1_8sw8_gaudi.sh <sweep id, a row of sweep/wave1.tsv>" >&2; exit 2; }
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
exec ./chain/sweep_run.sh "$1"
