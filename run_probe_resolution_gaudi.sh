#!/usr/bin/env bash
#SBATCH --job-name=d1-8res-probe
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:1
#SBATCH -c 18
#SBATCH --mem=64G
# Input-resolution probe for the DINOv3 ViT-B/16 fine-tune (Step VIII sweep model) on ONE Gaudi card: for each input size, the largest per-card batch that
# fits and the step time (base/probe_resolution_memory.py). Random data, no weights, no run dir: the only file written is probe_resolution_memory.json in
# the repo root (/workspace/probe_resolution_memory.json in the container), rewritten after every measurement, so a time-out keeps the rows done so far.
# usage, from the repo root: sbatch run_probe_resolution_gaudi.sh [probe options]
#   defaults: --input_sizes 224,448,512,768 --batches 8,16,25,32,50,100 --steps 3; e.g. sbatch run_probe_resolution_gaudi.sh --input_sizes 768 --batches 8,16,25
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated card before the probe starts
./gaudi_check_cards.sh || exit 1
exec ./gaudi.sh python /workspace/base/probe_resolution_memory.py --out /workspace/probe_resolution_memory.json "$@"
