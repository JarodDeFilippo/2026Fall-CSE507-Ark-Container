#!/bin/bash
#SBATCH -J taix-unpack-eval
#SBATCH -A grp_jliang12
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 1-00:00:00
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -o /scratch/jdefili4/datasets-new/taix-unpack-eval.%j.out
# TAIX-Ray on Sol, evaluation side (a linear probe on the official test split): unpacks the val and test splits' images to PNGs under png512/, next to the train PNGs taix_unpack.sh wrote, and writes one label file per split beside it: taix_train_labels.csv, taix_val_labels.csv, taix_test_labels.csv.
# Val and test are held out of pretraining: taix_train.csv (written by taix_unpack.sh) stays the only pretraining list. This script never writes it and opens it for nothing but a read-only check against the train split: the pretraining job's data fingerprint depends on that file.
# The splits come from each row's Split column (train, val, test), not from the shard names: the val and test shards were pulled into the same data/ folder as the train ones (hf_pull.sh TLAIM/TAIX-Ray) and their file-name prefix is not relied on.
# Run once, from anywhere (the dataset folder is an absolute path), after the val and test shards are in data/: sbatch base/dataset/TAIXRay/taix_unpack_eval.sh. A rerun is safe (after a time limit, or once more shards have arrived): PNGs already written are skipped and the label files are rewritten whole.
# TAIX_ROOT (default /scratch/jdefili4/datasets-new/TAIX-Ray) and TAIX_PYTHON (default the hf env's interpreter) override the dataset folder and the interpreter, for testing.
# Nothing is written until the checks pass. Pass 1 reads only the metadata columns of every shard (and, read-only, taix_train.csv), prints each shard's row count per split and then the paper's counts (Table 2) beside the ones found (MISMATCH on any that differ: printed, not fatal), and exits, before any PNG, folder or label file is written, on: a shard that is unreadable or lacks a column, a UID that is empty or holds a '/', a Split other than train, val, test, a UID repeated across the splits, a PatientID in more than one split (the paper splits by patient), a split with no rows, or a taix_train.csv that is not the train split's <UID>.png set.
# A split with no rows (shards not pulled yet) stops the run for the splits that are there too: their images are not written either, so a run that is too early leaves the dataset folder as it found it.
# Pass 2 then writes the val and test images: a row's Image holds the file bytes, written as they are when they are a PNG (checked by the signature), else decoded by PIL and saved as PNG, as taix_unpack.sh does; only the shards that hold val or test rows have their Image column read, and train rows get no image written, only a check that png512/<UID>.png exists. A val or test row without image bytes exits here, after the images before it (each a whole file under a checked UID, skipped on a rerun) and before any label file.
# The label files come last, each whole or not at all. Each has a header row, then one row per image of its split, in shard order then row order: Path (<UID>.png, relative to png512/), UID, Split, Fold, PatientID, PhysicianID, StudyDate, Age, Sex, HeartSize, PulmonaryCongestion, PleuralEffusion_Right, PleuralEffusion_Left, PulmonaryOpacities_Right, PulmonaryOpacities_Left, Atelectasis_Right, Atelectasis_Left. A null is an empty field.
# Printed in pass 2: each shard's val/test image count as it is done, then per split its rows, distinct patients and, for val and test, how many PNGs were byte-copied, re-encoded or already there (for train, how many PNGs are missing); then the label files written.
set -euo pipefail
TAIX_ROOT="${TAIX_ROOT:-/scratch/jdefili4/datasets-new/TAIX-Ray}"
TAIX_PYTHON="${TAIX_PYTHON:-$HOME/.conda/envs/hf/bin/python}"
cd "$TAIX_ROOT"
# the env's interpreter by path: sbatch inherits the submitting shell, and with hf already active there "source activate hf" is a no-op while "module load" re-prepends the base bin, so a bare python is the wrong interpreter
"$TAIX_PYTHON" -u - <<'PY'
import csv
import glob
import io
import os
from collections import Counter, namedtuple

import pyarrow.parquet as pq

PNG = b"\x89PNG\r\n\x1a\n"
SPLITS = ("train", "val", "test")
EVAL = ("val", "test")  # the splits whose images are written here
PAPER = {"train": (137593, 30543), "val": (34860, 7636), "test": (42928, 9545)}  # rows, patients: Table 2
COLS = ["UID", "Split", "Fold", "PatientID", "PhysicianID", "StudyDate", "Age", "Sex", "HeartSize", "PulmonaryCongestion",
        "PleuralEffusion_Right", "PleuralEffusion_Left", "PulmonaryOpacities_Right", "PulmonaryOpacities_Left",
        "Atelectasis_Right", "Atelectasis_Left"]  # the label files' columns after Path
Row = namedtuple("Row", COLS)
shards = sorted(glob.glob("data/*.parquet"))
if not shards:
    raise SystemExit("no data/*.parquet in " + os.getcwd())
for shard in shards:  # every shard's columns are checked before a row is read
    try:
        names = pq.read_schema(shard).names
    except Exception as e:  # e.g. a cut-off download
        raise SystemExit(f"{shard}: not a readable parquet file: {e}")
    missing = [c for c in COLS + ["Image"] if c not in names]
    if missing:
        raise SystemExit(f"{shard}: lacks the columns {missing}, has {names}")
# pass 1: the metadata columns of every shard, then the checks; nothing is written until all of them have passed
rows = {s: [] for s in SPLITS}  # Split value -> its Rows, in shard then row order
holds_eval = []  # the shards with val or test rows: the only ones whose Image column pass 2 reads
for shard in shards:
    cols = pq.ParquetFile(shard).read(columns=COLS).to_pydict()  # a few thousand rows, no Image
    shard_rows = [Row(*vals) for vals in zip(*(cols[c] for c in COLS))]
    bad = [r.UID for r in shard_rows if not r.UID or "/" in r.UID]  # the UID is a file name in pass 2
    if bad:
        raise SystemExit(f"{shard}: {len(bad)} UIDs are empty or hold a '/', first {bad[:3]}")
    per = Counter(r.Split for r in shard_rows)
    parts = ", ".join(f"{k} {n}" for k, n in sorted(per.items(), key=lambda kv: str(kv[0])))
    print(f"{shard}: {len(shard_rows)} rows ({parts})")
    for r in shard_rows:
        rows.setdefault(r.Split, []).append(r)
    if any(s in per for s in EVAL):
        holds_eval.append(shard)
patients = {s: dict.fromkeys(r.PatientID for r in rows[s]) for s in SPLITS}  # in first-seen order, so the "first" in a message is the same every run
for s in SPLITS:
    for what, found, paper in (("rows", len(rows[s]), PAPER[s][0]), ("patients", len(patients[s]), PAPER[s][1])):
        print(f"{s} {what}: found {found}, paper {paper}" + ("" if found == paper else "  MISMATCH"))
# the checks: each exits before anything is written
odd = {k: len(v) for k, v in rows.items() if k not in SPLITS}
if odd:
    raise SystemExit(f"Split values other than train, val, test: {odd}")
uids = Counter(r.UID for s in SPLITS for r in rows[s])
dup = [u for u, n in uids.items() if n > 1]
if dup:
    raise SystemExit(f"{len(dup)} UIDs occur more than once across the splits, first {dup[:3]}: png512/<UID>.png would hold one image for several rows")
home = {}  # PatientID -> the splits it sits in
for s in SPLITS:
    for p in patients[s]:
        home.setdefault(p, []).append(s)
shared = {p: ss for p, ss in home.items() if len(ss) > 1}
if shared:
    raise SystemExit(f"{len(shared)} PatientIDs sit in more than one split, first {list(shared.items())[:5]}")
empty = [s for s in SPLITS if not rows[s]]
if empty:
    raise SystemExit(f"no rows for the split {', '.join(empty)} in {os.getcwd()}/data (shards not pulled yet?)")
if os.path.exists("taix_train.csv"):
    with open("taix_train.csv", newline="") as f:  # read-only: the pretraining job's data fingerprint depends on this file
        have = Counter(r[0] for r in list(csv.reader(f))[1:] if r)
    want = Counter(f"{r.UID}.png" for r in rows["train"])
    if have != want:
        extra, lack = have - want, want - have
        raise SystemExit(f"taix_train.csv is not the train split's <UID>.png set: {sum(extra.values())} rows only in the csv (first {list(extra)[:3]}), {sum(lack.values())} only in the split (first {list(lack)[:3]})")
    print(f"taix_train.csv matches the train split: {sum(want.values())} rows")
else:
    print("no taix_train.csv here: not checked")
# pass 2: the only place a PNG is written
os.makedirs("png512", exist_ok=True)
stat = {s: Counter() for s in SPLITS}
stat["train"]["missing"] = sum(1 for r in rows["train"] if not os.path.exists(f"png512/{r.UID}.png"))
for shard in holds_eval:
    n = 0
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=64, columns=["UID", "Split", "Image"]):
        for rec in batch.to_pylist():
            split, uid, b = rec["Split"], rec["UID"], (rec["Image"] or {}).get("bytes")
            if split not in EVAL:
                continue
            n += 1
            if not b:
                raise SystemExit(f"{shard}: {uid} has no image bytes")
            out = f"png512/{uid}.png"
            if os.path.exists(out):  # a relaunch after a time limit
                stat[split]["skipped"] += 1
                continue
            tmp = out + ".tmp"  # written whole or not at all: a killed job leaves no half PNG that a relaunch would skip
            if b[:8] == PNG:
                with open(tmp, "wb") as f:
                    f.write(b)
                stat[split]["copied"] += 1
            else:
                from PIL import Image
                Image.open(io.BytesIO(b)).save(tmp, format="PNG")
                stat[split]["reencoded"] += 1
            os.replace(tmp, out)
    print(f"{shard}: {n} images done")
for s in SPLITS:
    c = stat[s]
    imgs = f"{c['missing']} PNGs missing" if s == "train" else f"{c['copied']} byte-copied, {c['reencoded']} re-encoded, {c['skipped']} already there"
    print(f"{s}: {len(rows[s])} rows, {len(patients[s])} distinct patients, {imgs}")
labels = {s: f"taix_{s}_labels.csv" for s in SPLITS}
for s in SPLITS:  # all three are written whole before any is put in place
    with open(labels[s] + ".tmp", "w", newline="") as f:
        w = csv.writer(f)  # a None is written as an empty field
        w.writerow(["Path"] + COLS)
        w.writerows([f"{r.UID}.png", *r] for r in rows[s])
for s in SPLITS:
    os.replace(labels[s] + ".tmp", labels[s])
    print(f"wrote {labels[s]}: {len(rows[s])} rows")
PY
