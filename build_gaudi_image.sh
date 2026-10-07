#!/usr/bin/env bash
#SBATCH --job-name=gaudi-build
#SBATCH -A grp_jliang12
#SBATCH -p htc
#SBATCH -q public
#SBATCH -t 04:00:00
#SBATCH -c 10
#SBATCH --mem=32G
set -euo pipefail
cd /scratch/$USER/2026Fall-CSE507-Ark-Container
export APPTAINER_TMPDIR=/scratch/$USER/.apptainer/tmp APPTAINER_CACHEDIR=/scratch/$USER/.apptainer/cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
# build to a temp name, then rename: a failed rebuild never clobbers an image a running job uses
apptainer build --fakeroot --force containers/gaudi/.gaudi.building.sif containers/gaudi/apptainer.def
mv -f containers/gaudi/.gaudi.building.sif containers/gaudi/gaudi.sif
ls -la containers/gaudi/gaudi.sif
apptainer exec containers/gaudi/gaudi.sif grep -iE '^(torch|torchvision|numpy|timm|albumentations|habana)' /opt/pip-freeze.txt
