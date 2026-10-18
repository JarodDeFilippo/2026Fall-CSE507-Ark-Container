#!/usr/bin/env bash
#SBATCH --job-name=d1-8ssl-probe
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:1
#SBATCH -c 18
#SBATCH --mem=64G
# SSL memory probe for the Stage A trainer (base/ssl_pretrain.py: student + EMA teacher + Gram teacher, 2 global + 8 local + 2 Gram teacher crops per image)
# on ONE Gaudi card: for each crop-size triple global:local:gram, the largest per-card batch that fits and the step time (base/probe_ssl_memory.py). It
# decides chain/hires_batch_per_gpu.txt and chain/hires_gram_size.txt (1024 or 512) for chain/hires.sh. Random data, no weights, no run dir: the only file
# written is probe_ssl_memory.json in the repo root (/workspace/probe_ssl_memory.json in the container), rewritten after every measurement, so a time-out
# keeps the rows done so far.
# usage, from the repo root: sbatch run_probe_ssl_gaudi.sh [probe options]
#   defaults: --sizes 224:96:448,448:192:896,512:224:1024,512:224:512 --batches 4,8,16,32,64 --steps 3
#   e.g. a triple alone (no state carried over from the others): sbatch run_probe_ssl_gaudi.sh --sizes 512:224:1024 --batches 4,8,16
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated card before the probe starts
./gaudi_check_cards.sh || exit 1
exec ./gaudi.sh python /workspace/base/probe_ssl_memory.py --out /workspace/probe_ssl_memory.json "$@"
