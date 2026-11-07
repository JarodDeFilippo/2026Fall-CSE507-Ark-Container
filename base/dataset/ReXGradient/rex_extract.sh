#!/bin/bash
#SBATCH -J rex-extract
#SBATCH -A grp_jliang12
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 1-00:00:00
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -o /scratch/jdefili4/datasets-new/rex-extract.%j.out
# ReXGradient-160K on Sol: unpacks the image parts and writes the SSL train list, rexgradient_train.csv, that base/datasets_config.yaml's ReXGradient entry points at.
# Run once, from anywhere (every path is absolute): sbatch base/dataset/ReXGradient/rex_extract.sh
# The parts deid_png.part0* are one zstd-compressed tar (the dataset README says plain tar; `file` says Zstandard), hence zstd -dc. Inside:
# deid_png/<PatientID>/<AccessionNumber>/studies/<StudyInstanceUid>/series/<SeriesUID>/instances/<SOP>.png
# The list holds the images of the studies in metadata/train_metadata.csv: a header row "Path", then each image's path relative to the dataset directory (starting deid_png/), which
# ssl_dinov3/data.py image_paths joins to data_dir. It stays on Sol: the data-use agreement forbids publishing a directory of the data.
# Expected (the dataset README), the last line printed: 140,000 train studies, 238,968 train images, 273,004 images in all.
set -euo pipefail
set +u  # the module scripts read unset variables
module load mamba/latest  # zstd and python3
set -u
cd /scratch/jdefili4/datasets-new/ReXGradient-160K
cat deid_png.part0* | zstd -dc | tar -xf -
find deid_png -name '*.png' | LC_ALL=C sort > all_images.txt
python3 - <<'PY'
import csv

with open("metadata/train_metadata.csv", newline="") as f:
    train = {row["StudyInstanceUid"] for row in csv.DictReader(f)}
with open("all_images.txt") as f:
    paths = f.read().splitlines()
kept = []
for p in paths:
    if "/studies/" not in p:
        raise SystemExit("no /studies/ segment in " + p)
    if p.split("/studies/", 1)[1].split("/", 1)[0] in train:  # the study is the path segment after /studies/
        kept.append(p)
with open("rexgradient_train.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Path"])
    w.writerows([p] for p in kept)
print(f"train studies {len(train)}, train images {len(kept)}, all images {len(paths)}")
PY
