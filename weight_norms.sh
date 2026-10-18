#!/usr/bin/env bash
#SBATCH --job-name=weight-norms
#SBATCH -A grp_jliang12
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 00:30:00
# The script is CPU-only, but nvidia.sh runs `apptainer exec --nv`, which needs a GPU node, so keep one A100.
#SBATCH -G a100:1
#SBATCH --constraint=a100_80
#SBATCH -c 4
#SBATCH --mem=32G
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch -p htc weight_norms.sh <run_dir_name> <cycle> [<cycle> ...]  (writes <run_dir_name>/weight_norms.csv)" >&2
    exit 2
fi
run="$1"
shift
cd /scratch/$USER/2026Fall-CSE507-Ark-Container
# nvidia.sh binds this repo root to /workspace and starts in /workspace/base (as eval_d1_6.sh relies on)
exec ./nvidia.sh python /workspace/base/weight_norms.py "/workspace/${run}" "$@" \
    --out "/workspace/${run}/weight_norms.csv"
