#!/bin/bash
#SBATCH -J taix-unpack
#SBATCH -A grp_jliang12
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 1-00:00:00
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -o /scratch/jdefili4/datasets-new/taix-unpack.%j.out
# TAIX-Ray on Sol: unpacks the train split's images to PNGs under png512/ (base/datasets_config.yaml's TAIXRay data_dir) and writes the SSL train list, taix_train.csv, next to it.
# The `default` config of TLAIM/TAIX-Ray on the Hugging Face hub is the authors' 512 px version (62 GB, all splits); data/train-*.parquet was pulled with
# `hf_pull.sh TLAIM/TAIX-Ray "data/train-*"`, so this writes the train split only.
# Run once, from anywhere (every path is absolute): sbatch base/dataset/TAIXRay/taix_unpack.sh. After a time limit, run it again: PNGs already written are skipped.
# Each row's Image holds the file bytes: written as they are when they are a PNG (checked by the signature), else decoded by PIL and saved as PNG. taix_train.csv has a header row "Path",
# then one <UID>.png per row, relative to png512/ (which image_paths joins to data_dir).
# Printed: each shard's row count, then the total with how many PNGs were byte-copied, re-encoded or already there. The total is the number of rows in taix_train.csv (header aside).
set -euo pipefail
cd /scratch/jdefili4/datasets-new/TAIX-Ray
mkdir -p png512
# the env's interpreter by path: sbatch inherits the submitting shell, and with hf already active there "source activate hf" is a no-op while "module load" re-prepends the base bin, so a bare python is the wrong interpreter
"$HOME"/.conda/envs/hf/bin/python - <<'PY'
import csv
import glob
import io
import os

import pyarrow.parquet as pq

PNG = b"\x89PNG\r\n\x1a\n"
shards = sorted(glob.glob("data/train-*.parquet"))
if not shards:
    raise SystemExit("no data/train-*.parquet in " + os.getcwd())
uids = []
copied = reencoded = skipped = 0
for shard in shards:
    pf = pq.ParquetFile(shard)
    names = pf.schema_arrow.names
    if "UID" not in names or "Image" not in names:
        raise SystemExit(f"{shard}: needs the columns UID and Image, has {names}")
    rows = 0
    for batch in pf.iter_batches(batch_size=64, columns=["UID", "Image"]):
        for row in batch.to_pylist():
            uid, b = row["UID"], row["Image"]["bytes"]
            uids.append(uid)
            rows += 1
            out = f"png512/{uid}.png"
            if os.path.exists(out):  # a relaunch after a time limit
                skipped += 1
                continue
            tmp = out + ".tmp"  # written whole or not at all: a killed job leaves no half PNG that a relaunch would skip
            if b[:8] == PNG:
                with open(tmp, "wb") as f:
                    f.write(b)
                copied += 1
            else:
                from PIL import Image
                Image.open(io.BytesIO(b)).save(tmp, format="PNG")
                reencoded += 1
            os.replace(tmp, out)
    print(f"{shard}: {rows} rows", flush=True)
if len(set(uids)) != len(uids):
    raise SystemExit(f"{len(uids) - len(set(uids))} repeated UIDs: png512/<UID>.png would hold one image for several rows")
with open("taix_train.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Path"])
    w.writerows([f"{uid}.png"] for uid in uids)
print(f"total {len(uids)} rows: {copied} byte-copies, {reencoded} re-encoded, {skipped} already there")
PY
