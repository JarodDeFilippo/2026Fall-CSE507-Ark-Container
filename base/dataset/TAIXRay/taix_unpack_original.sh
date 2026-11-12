#!/bin/bash
#SBATCH -J taix-unpack-original
#SBATCH -A grp_jliang12
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 1-00:00:00
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -o /scratch/jdefili4/datasets-new/taix-unpack-original.%j.out
# TAIX-Ray on Sol, native resolution: unpacks one split of the authors' `original` config of TLAIM/TAIX-Ray (variable image size, 1.2 TB in all: original/train-*.parquet, original/val-*, original/test-*) to PNGs under png_original/<split>/ and writes that split's list, taix_original_<split>.csv, next to png_original/. taix_unpack.sh's 512 px images (png512/), taix_train.csv and the label files are never read or written here.
# The split is the one argument: sbatch base/dataset/TAIXRay/taix_unpack_original.sh train (val and test likewise); anything else exits 2. Run from anywhere (the dataset folder is an absolute path). After a time limit or a kill, submit the same line again: PNGs already written are skipped. 1 day is more than a pass needs; the limit is only a guard.
# Each row's Image holds the file bytes: written as they are when they are a PNG (checked by the signature), so the size and the bit depth stay what they were stored at, else decoded by PIL and saved as PNG, as taix_unpack.sh does. Every image goes to <UID>.png.tmp first and is renamed when whole, so a killed job leaves no half PNG under a name that a relaunch would skip (a leftover .tmp is overwritten); a file under its final name is skipped only if it starts as a PNG (signature and IHDR).
# The list has a header row "Path", then one <split>/<UID>.png per row, relative to png_original/ (which ssl_dinov3/data.py image_paths joins to data_dir), in shard then row order, LF line endings. It is written last, whole (to .tmp, then renamed), after every shard was read and the rows counted are the paper's number: it exists only for a split that is complete.
# Checks before any file is written: the shards are the whole split (each is named <split>-NNNNN-of-MMMMM.parquet, all with one MMMMM, and the NNNNN are exactly 0 to MMMMM-1: a download cut short is refused, with the missing ordinals named), the filesystem of the dataset folder has at least the split's parquet files + 10% free (the PNGs an earlier run wrote count as free, so a rerun is not refused for its own output), and every shard is readable and has the columns UID and Image. Per row: a Split column, if the shards have one, must say the split (a val or test image must not reach the pretraining list), the UID must not be empty or hold a '/', the image bytes must be there. After the last shard: no UID twice.
# Memory: a shard (about 1 GB) is read 8 rows at a time (iter_batches) with buffered file reads (buffer_size), never whole: the working set is a few native-size images and a read buffer, far under the 2 GB an Arrow binary array holds at most (taix_unpack.sh reads 64 rows a batch, a size for 512 px images).
# Printed: each shard's row count, then the total with how many PNGs were byte-copied, re-encoded or already there (their sum is the number of rows in the list), the paper's row count beside the one found (MISMATCH if they differ: once everything is printed the exit is not 0 and no list is written; the PNGs stay and a rerun skips them), the min / median / max of the images' widths and heights read from the PNG headers (IHDR, no decode, every image), how many have a long side under 1024, and their bit depths.
# TAIX_ROOT (default /scratch/jdefili4/datasets-new/TAIX-Ray) and TAIX_PYTHON (default the hf env's interpreter) override the dataset folder and the interpreter, for testing and for another group member's own folder (with sbatch -o for the log). TAIX_PAPER_ROWS (default the paper's count for the split) overrides the row count the list is held to, for testing.
set -euo pipefail
[[ $# == 1 && ( $1 == train || $1 == val || $1 == test ) ]] || { echo "usage: sbatch base/dataset/TAIXRay/taix_unpack_original.sh train|val|test" >&2; exit 2; }
TAIX_ROOT="${TAIX_ROOT:-/scratch/jdefili4/datasets-new/TAIX-Ray}"
TAIX_PYTHON="${TAIX_PYTHON:-$HOME/.conda/envs/hf/bin/python}"
cd "$TAIX_ROOT"
# the env's interpreter by path: sbatch inherits the submitting shell, and with hf already active there "source activate hf" is a no-op while "module load" re-prepends the base bin, so a bare python is the wrong interpreter
"$TAIX_PYTHON" -u - "$1" <<'PY'
import csv
import glob
import io
import os
import re
import shutil
import statistics
import struct
import sys
from collections import Counter

import pyarrow.parquet as pq

PNG = b"\x89PNG\r\n\x1a\n"
PAPER = {"train": 137593, "val": 34860, "test": 42928}  # images per split: Table 2 of the data descriptor, as taix_unpack_eval.sh's PAPER
split = sys.argv[1]
paper = int(os.environ.get("TAIX_PAPER_ROWS", PAPER[split]))  # TAIX_PAPER_ROWS: for testing
shards = sorted(glob.glob(f"original/{split}-*.parquet"))
if not shards:
    raise SystemExit(f"no original/{split}-*.parquet in " + os.getcwd())
named = [re.fullmatch(rf"{split}-([0-9]{{5}})-of-([0-9]{{5}})\.parquet", os.path.basename(s)) for s in shards]
for shard, m in zip(shards, named):  # the names say how many shards the split has: a download cut short is refused here, before anything is written
    if not m or int(m[1]) >= int(m[2]):
        raise SystemExit(f"{shard}: not named {split}-NNNNN-of-MMMMM.parquet with NNNNN below MMMMM")
totals = {int(m[2]) for m in named}
if len(totals) > 1:
    raise SystemExit(f"the {split} shards declare different totals, {sorted(totals)}")
total = totals.pop()
gone = sorted(set(range(total)) - {int(m[1]) for m in named})  # names are distinct and each ordinal is below the total, so no gone ordinal means all of 0..total-1 are there once
if gone:
    raise SystemExit(f"the {split} shard names declare {total} shards, original/ holds {len(shards)}: {len(gone)} missing, the first ordinals " + ", ".join(f"{i:05d}" for i in gone[:5]))
out_dir, list_file = f"png_original/{split}", f"taix_original_{split}.csv"
size = sum(os.path.getsize(s) for s in shards)
free = shutil.disk_usage(".").free
if os.path.isdir(out_dir):  # an earlier run's PNGs are space already spent: they count as free
    free += sum(e.stat().st_size for e in os.scandir(out_dir) if e.name.endswith(".png"))
if free < 1.1 * size:
    raise SystemExit(f"not enough room in {os.getcwd()}: {free / 1e9:.0f} GB free (counting the {split} PNGs already written), the {len(shards)} {split} shards are {size / 1e9:.0f} GB, and shards + 10% = {1.1 * size / 1e9:.0f} GB are needed")
columns = {}
for shard in shards:  # every shard's columns are checked before anything is written
    try:
        columns[shard] = pq.ParquetFile(shard).schema_arrow.names
    except Exception as e:  # e.g. a cut-off download
        raise SystemExit(f"{shard}: not a readable parquet file: {e}")
    if "UID" not in columns[shard] or "Image" not in columns[shard]:
        raise SystemExit(f"{shard}: needs the columns UID and Image, has {columns[shard]}")


def png_header(path):
    """(width, height, bit depth, colour type) from the IHDR of a PNG file (26 bytes read, no decode); None if the file is absent or does not start as a PNG."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
    except OSError:
        return None
    if len(head) < 26 or head[:8] != PNG or head[12:16] != b"IHDR":
        return None
    return (*struct.unpack(">II", head[16:24]), head[24], head[25])


os.makedirs(out_dir, exist_ok=True)
uids, sizes, depths = [], [], Counter()
copied = reencoded = skipped = 0
for shard in shards:
    pf = pq.ParquetFile(shard, buffer_size=1 << 23)
    rows = 0
    for batch in pf.iter_batches(batch_size=8, columns=["UID", "Image"] + (["Split"] if "Split" in columns[shard] else [])):
        for row in batch.to_pylist():
            uid, image = row["UID"], row["Image"]
            b = image.get("bytes") if isinstance(image, dict) else image
            if not uid or "/" in uid:
                raise SystemExit(f"{shard}: UID {uid!r} is empty or holds a '/'")
            if row.get("Split", split) != split:
                raise SystemExit(f"{shard}: {uid} is in the {row['Split']} split, not {split}")
            if not b:
                raise SystemExit(f"{shard}: {uid} has no image bytes")
            uids.append(uid)
            rows += 1
            out = f"{out_dir}/{uid}.png"
            header = png_header(out)
            if header:  # a relaunch after a time limit
                skipped += 1
            else:
                tmp = out + ".tmp"  # written whole or not at all: a killed job leaves no half PNG that a relaunch would skip
                if b[:8] == PNG:
                    with open(tmp, "wb") as f:
                        f.write(b)
                    copied += 1
                else:
                    from PIL import Image
                    try:
                        Image.open(io.BytesIO(b)).save(tmp, format="PNG")
                    except Exception as e:
                        raise SystemExit(f"{shard}: {uid} is not a PNG and PIL cannot read it: {e}")
                    reencoded += 1
                header = png_header(tmp)
                if not header:
                    raise SystemExit(f"{shard}: {uid} does not start as a PNG (no IHDR chunk first)")
                os.replace(tmp, out)
            sizes.append(header[:2])
            depths[header[2:]] += 1
    print(f"{shard}: {rows} rows", flush=True)
if not uids:
    raise SystemExit(f"no rows in the {len(shards)} {split} shards")
if len(set(uids)) != len(uids):
    raise SystemExit(f"{len(uids) - len(set(uids))} repeated UIDs: {out_dir}/<UID>.png would hold one image for several rows")
print(f"total {len(uids)} rows: {copied} byte-copies, {reencoded} re-encoded, {skipped} already there")
print(f"{split} rows: found {len(uids)}, paper {paper}" + ("" if len(uids) == paper else "  MISMATCH"))
widths, heights = zip(*sizes)
for what, values in (("width", widths), ("height", heights)):
    print(f"image {what} (IHDR): min {min(values)}, median {statistics.median(values):g}, max {max(values)}")
print(f"{sum(max(s) < 1024 for s in sizes)} of {len(sizes)} images have a long side under 1024")
print("bit depth, colour type: " + ", ".join(f"{n} x {d}-bit type {c}" for (d, c), n in depths.most_common()))
if len(uids) != paper:  # a split short of rows (or over) is not the split: no list, the PNGs stay
    raise SystemExit(f"{list_file} not written: {len(uids)} rows found, the paper has {paper} for the {split} split; the PNGs written so far stay in {out_dir}/ and a rerun skips them")
with open(list_file + ".tmp", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["Path"])
    w.writerows([f"{split}/{uid}.png"] for uid in uids)
os.replace(list_file + ".tmp", list_file)
print(f"wrote {list_file}: {len(uids)} rows")
PY
