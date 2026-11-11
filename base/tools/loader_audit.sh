#!/usr/bin/env bash
#SBATCH --job-name=loader-audit
#SBATCH -A grp_jliang12
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -o /scratch/jdefili4/datasets-new/loader-audit.%j.out
# Loader audit of the 512 px new-data run (chain/newdata512.sh): shows what the SSL training loader actually feeds the model for each of its seven pretraining datasets, so a human can see whether the
# high-bit-depth images come out flat or dark and how much real detail the 512 px crops hold. It runs base/tools/loader_audit.py, which calls the SSL data code itself (ssl_dinov3/data.py: decode, per-image
# rescale of 16-bit images, --pre_resize 512, DataAugmentationCXR with the run's crop settings) on 20 images per dataset, seeded (--seed 0). CPU only: no --gres.
# Submit, from the repo root on Sol (the copy at /scratch/jdefili4/2026Fall-CSE507-Ark-Container must hold base/tools/loader_audit.py):   sbatch base/tools/loader_audit.sh
# Anything after the script name goes to the tool, e.g. sbatch base/tools/loader_audit.sh --n 50 --datasets PadChest TAIXRay   (the tool's docstring lists its options; they apply to this run only).
# Output: /scratch/jdefili4/datasets-new/loader_audit/<job id>/ (mode 700): stats.csv, and <dataset>_whole.png and <dataset>_crops.png per dataset. The job's stdout, /scratch/jdefili4/datasets-new/loader-audit.<job id>.out,
# holds one summary line per dataset: the number of images, the median p99-p1 and how many of them are under 64 (a flat render), the median source size, and the size of a global crop in pre-resized pixels.
# stats.csv has one row per sampled image: `tile` is its number on the two sheets (their labels), `list_index` its 0-based position in the dataset's train list, which does not count the header row of the csv
# lists (the tool's docstring gives the rule for finding the line in the list file); an error that names a sampled image gives its list index and path.
# The Slurm log (the #SBATCH -o file) is created by Slurm before this script runs, with the submitter's umask, and an error for an unreadable image puts that image's path in it: the script's first action is chmod 600 of it (the
# path is the #SBATCH -o line's, built from SLURM_JOB_ID). If that cannot be done it only warns on stderr and the job goes on (an sbatch -o on the command line replaces the path; the warning then names a file that is not there).
# The tool writes only into a new or an empty directory and refuses a non-empty --out before it does any work. This script creates the per-job directory empty, which is allowed, so a job restarted or
# requeued under the same job id (its earlier attempt's files are already there) fails at once with "is not empty" in the Slurm log instead of mixing two runs' output: move or delete that directory first, or sbatch a fresh job.
# DO NOT COPY THE SHEETS (or stats.csv, which lists image paths) INTO ANY REPO OR SHARED DIRECTORY: they are dataset images, and ReXGradient's data-use agreement forbids sharing them. They stay under
# datasets-new (chmod 700 here) and are looked at there; the tool itself refuses an --out inside a repo checkout or under a results/ directory.
# The container, the way every CPU job of this repo runs one (run_emb_fit_class.sh, weight_norms.sh): nvidia.sh binds the repo root as /workspace and starts in /workspace/base. DATA_BIND (nvidia.sh) REPLACES its
# default /data mount, so /data is listed again (read-only: the four older datasets and PadChest live under /data/jliang12/jpang12/dataset/); datasets-new (TAIXRay, ReXGradient) is mounted at the same path, as
# run_d1_8_newdata512_arkteacher_gaudi.sh mounts it for the run, but read-write here, because the output directory is inside it. The image is the A100 one (torch 2.11.0, Pillow 11.3.0, numpy 2.2.6); the
# trainer runs in the Gaudi one (Habana's torch, the same Pillow and numpy pins).
# nvidia.sh passes --nv: check_hf_dinov3_parity.sh reports CPU jobs running it fine without a GPU (htc, 2026-10-22), weight_norms.sh asks for a GPU because of it. This job has not run on public: if apptainer
# refuses to start without a GPU, resubmit with sbatch -p htc.
set -euo pipefail
LOG=/scratch/jdefili4/datasets-new/loader-audit.${SLURM_JOB_ID:?SLURM_JOB_ID is not set: submit this with sbatch base/tools/loader_audit.sh}.out
chmod 600 "$LOG" || echo "WARNING: could not make the Slurm log $LOG owner-only (chmod 600 failed above); it can carry an image path" >&2
ROOT=/scratch/jdefili4/2026Fall-CSE507-Ark-Container
NEW=/scratch/jdefili4/datasets-new
OUT=$NEW/loader_audit/$SLURM_JOB_ID
umask 077
mkdir -p "$OUT"
chmod 700 "$NEW/loader_audit" "$OUT"
cd "$ROOT"
DATA_BIND="/data:/data:ro,$NEW:$NEW:rw" "$ROOT/nvidia.sh" python /workspace/base/tools/loader_audit.py --out "$OUT" "$@"
