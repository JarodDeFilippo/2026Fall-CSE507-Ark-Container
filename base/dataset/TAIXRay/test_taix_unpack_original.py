"""CPU check of taix_unpack_original.sh on a tiny fake `original` config: a 16-bit PNG, an 8-bit PNG, a non-PNG (16-bit TIFF) and a wide PNG, in two train shards and one val shard.
pyarrow is not installed in the project venv, so the script runs against a stand-in for pyarrow.parquet (put first on PYTHONPATH) that implements the calls the script makes (ParquetFile(source, buffer_size=),
.schema_arrow.names, .iter_batches(batch_size=, columns=), batch.to_pylist()) over pickled rows in files named like the shards (<split>-NNNNN-of-MMMMM.parquet, as the script requires). What is checked is the
script's own logic (the argument, the checks before any write, the byte copy and the re-encode, the temp name and rename, the resume, the list file, the printed numbers, the row count held to the paper's: the
fakes hold a few rows, so a run meant to finish sets TAIX_PAPER_ROWS to its own count), not pyarrow's reading of a real parquet file or the buffer_size effect on memory.
Run from anywhere: python base/dataset/TAIXRay/test_taix_unpack_original.py   (needs PIL and bash; no Slurm, no cluster data). The test_ function is also collected by pytest."""
import io
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

from PIL import Image

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taix_unpack_original.sh")
STAND_IN = '''
import pickle


class _Batch:
    def __init__(self, rows):
        self.rows = rows

    def to_pylist(self):
        return self.rows


class ParquetFile:
    def __init__(self, source, *, buffer_size=0):
        with open(source, "rb") as f:
            self.names, self.rows = pickle.load(f)
        self.schema_arrow = type("Schema", (), {"names": self.names})

    def iter_batches(self, batch_size=65536, row_groups=None, columns=None, use_threads=True):
        for i in range(0, len(self.rows), batch_size):
            yield _Batch([{c: row[c] for c in columns} for row in self.rows[i:i + batch_size]])
'''


def encode(mode, size, values, fmt):
    image = Image.new(mode, size)
    image.putdata(values)
    out = io.BytesIO()
    image.save(out, format=fmt)
    return out.getvalue()


PNG16 = encode("I;16", (7, 5), [i * 1800 for i in range(35)], "PNG")  # 7 x 5, 16-bit grey
PNG8 = encode("L", (4, 6), list(range(24)), "PNG")  # 4 x 6, 8-bit grey
TIFF16 = encode("I;16", (3, 2), [0, 1000, 65535, 4, 5, 6], "TIFF")  # 3 x 2, not a PNG
WIDE = encode("L", (1100, 3), [0] * 3300, "PNG")  # long side over 1024
read = lambda path: open(path, "rb").read()


def fake_root(tmp, shards, names=("UID", "Image", "Split", "Age")):
    """A dataset folder with original/<name> per shard: rows of (UID, image bytes or None, Split)."""
    root = tempfile.mkdtemp(dir=tmp)
    os.makedirs(os.path.join(root, "original"))
    for name, rows in shards.items():
        with open(os.path.join(root, "original", name), "wb") as f:
            pickle.dump((list(names), [dict(UID=u, Image={"bytes": b, "path": None}, Split=s, Age=60) for u, b, s in rows]), f)
    return root


def test_unpack_original():
    with tempfile.TemporaryDirectory() as tmp:
        stand_in = os.path.join(tmp, "stand_in")
        os.makedirs(os.path.join(stand_in, "pyarrow"))
        open(os.path.join(stand_in, "pyarrow", "__init__.py"), "w").close()
        open(os.path.join(stand_in, "pyarrow", "parquet.py"), "w").write(STAND_IN)

        def run(root, *args, rows=None):
            env = dict(os.environ, TAIX_ROOT=root, TAIX_PYTHON=sys.executable, PYTHONPATH=stand_in)
            env.pop("TAIX_PAPER_ROWS", None)  # unless rows is given, the script holds the split to the paper's own count
            if rows is not None:
                env["TAIX_PAPER_ROWS"] = str(rows)
            return subprocess.run(["bash", SCRIPT, *args], cwd=tmp, env=env, capture_output=True, text=True)

        shards = {"train-00000-of-00002.parquet": [("a", PNG16, "train"), ("b", PNG8, "train")], "train-00001-of-00002.parquet": [("c", TIFF16, "train"), ("d", WIDE, "train")],
                  "val-00000-of-00001.parquet": [("v", PNG8, "val")]}
        root = fake_root(tmp, shards)
        p = lambda *parts: os.path.join(root, *parts)
        os.makedirs(p("png512"))
        keep = {"png512/x.png": b"x", "taix_train.csv": b"Path\r\nx.png\r\n", "taix_train_labels.csv": b"Path,UID\r\n"}  # the 512 px side: never touched
        for name, content in keep.items():
            open(p(name), "wb").write(content)
        for args in ((), ("trian",), ("png512",), ("train", "val")):  # one argument, one of the three splits
            result = run(root, *args)
            assert result.returncode == 2 and "usage" in result.stderr and not os.path.exists(p("png_original")), (args, result)

        result = run(root, "train", rows=4)
        assert result.returncode == 0, result.stderr
        first = result.stdout
        for line in ("original/train-00000-of-00002.parquet: 2 rows", "original/train-00001-of-00002.parquet: 2 rows", "total 4 rows: 3 byte-copies, 1 re-encoded, 0 already there",
                     "train rows: found 4, paper 4\n", "image width (IHDR): min 3, median 5.5, max 1100", "image height (IHDR): min 2, median 4, max 6",
                     "3 of 4 images have a long side under 1024", "bit depth, colour type: 2 x 16-bit type 0, 2 x 8-bit type 0", "wrote taix_original_train.csv: 4 rows"):
            assert line in first, (line, first)
        assert "val-00000" not in first and "MISMATCH" not in first  # only the split asked for is read; the row count is the one asked for
        assert read(p("png_original/train/a.png")) == PNG16 and read(p("png_original/train/b.png")) == PNG8 and read(p("png_original/train/d.png")) == WIDE  # PNG bytes as they were
        c = Image.open(p("png_original/train/c.png"))  # the non-PNG: decoded and saved as PNG, the 16-bit values kept
        assert c.format == "PNG" and c.mode == "I;16" and c.size == (3, 2) and list(c.getdata()) == [0, 1000, 65535, 4, 5, 6]
        assert sorted(os.listdir(p("png_original/train"))) == ["a.png", "b.png", "c.png", "d.png"] and not os.path.exists(p("png_original/val"))  # no temp file left, no other split
        assert read(p("taix_original_train.csv")) == b"Path\ntrain/a.png\ntrain/b.png\ntrain/c.png\ntrain/d.png\n"  # header, paths relative to png_original/, LF only
        assert {name: read(p(name)) for name in keep} == keep

        # a rerun skips everything: no image is written again, the same numbers and list
        stamps = {n: os.stat(p("png_original/train", n)).st_mtime_ns for n in os.listdir(p("png_original/train"))}
        result = run(root, "train", rows=4)
        assert result.returncode == 0 and "total 4 rows: 0 byte-copies, 0 re-encoded, 4 already there" in result.stdout, result
        assert stamps == {n: os.stat(p("png_original/train", n)).st_mtime_ns for n in os.listdir(p("png_original/train"))}
        assert "image width (IHDR): min 3, median 5.5, max 1100" in result.stdout and read(p("taix_original_train.csv")).count(b"\n") == 5  # the sizes of images already there are read too

        # what a killed run leaves: a half-written temp file (a.png never got its name) and a final name that is not a PNG; neither counts as done
        os.remove(p("png_original/train/a.png"))
        open(p("png_original/train/a.png.tmp"), "wb").write(PNG16[:40])  # cut after its IHDR: it would look like a PNG if the temp name counted
        open(p("png_original/train/b.png"), "wb").write(PNG8[:10])
        result = run(root, "train", rows=4)
        assert result.returncode == 0 and "total 4 rows: 2 byte-copies, 0 re-encoded, 2 already there" in result.stdout, result
        assert read(p("png_original/train/a.png")) == PNG16 and read(p("png_original/train/b.png")) == PNG8 and not os.path.exists(p("png_original/train/a.png.tmp"))

        # another split: its own folder and list
        result = run(root, "val", rows=1)
        assert result.returncode == 0 and read(p("png_original/val/v.png")) == PNG8 and read(p("taix_original_val.csv")) == b"Path\nval/v.png\n", result

        # refused before a PNG or a folder is written: a shard that is not a parquet file, one without the Image column, no shard at all, too little room (a sparse shard of twice the free space), and a shard set that is
        # not the whole split: the middle one gone, the last one gone, two totals, a stray name beside a complete set, an ordinal past the total
        one = [("a", PNG8, "train")]
        for label, shard_map, names, words in (("garbage", None, None, ("train-00000-of-00001.parquet", "not a readable parquet file")),
                                               ("columns", {"train-00000-of-00001.parquet": one}, ("UID", "Split"), ("needs the columns UID and Image", "'UID', 'Split'")),
                                               ("none", {}, None, ("no original/train-*.parquet",)),
                                               ("room", {"train-00000-of-00002.parquet": one}, None, ("not enough room", "GB free")),
                                               ("middle", {"train-00000-of-00003.parquet": one, "train-00002-of-00003.parquet": one}, None,
                                                ("the train shard names declare 3 shards, original/ holds 2", "1 missing, the first ordinals 00001")),
                                               ("last", {"train-00000-of-00002.parquet": one}, None, ("the train shard names declare 2 shards, original/ holds 1", "1 missing, the first ordinals 00001")),
                                               ("totals", {"train-00000-of-00002.parquet": one, "train-00001-of-00003.parquet": one}, None, ("the train shards declare different totals, [2, 3]",)),
                                               ("stray", {"train-00000-of-00002.parquet": one, "train-00001-of-00002.parquet": one, "train-00001-of-00002 (1).parquet": one}, None,
                                                ("original/train-00001-of-00002 (1).parquet: not named train-NNNNN-of-MMMMM.parquet",)),
                                               ("past", {"train-00000-of-00001.parquet": one, "train-00001-of-00001.parquet": one}, None, ("original/train-00001-of-00001.parquet: not named", "NNNNN below MMMMM"))):
            bad = fake_root(tmp, shard_map or {}, *([names] if names else []))
            if label == "garbage":
                open(os.path.join(bad, "original", "train-00000-of-00001.parquet"), "wb").write(b"PAR1 cut off")
            if label == "room":
                with open(os.path.join(bad, "original", "train-00001-of-00002.parquet"), "wb") as f:
                    f.truncate(shutil.disk_usage(tmp).free * 2)
            result = run(bad, "train")
            assert result.returncode == 1 and all(w in result.stderr for w in words), (label, result.stderr)
            assert not os.path.exists(os.path.join(bad, "png_original")) and not os.path.exists(os.path.join(bad, "taix_original_train.csv")), label
        bad = fake_root(tmp, {"train-00000-of-00008.parquet": one, "train-00007-of-00008.parquet": one})  # six gone: the message gives the first five
        result = run(bad, "train")
        assert result.returncode == 1 and "declare 8 shards, original/ holds 2: 6 missing, the first ordinals 00001, 00002, 00003, 00004, 00005" in result.stderr and "00006" not in result.stderr, result.stderr
        assert not os.path.exists(os.path.join(bad, "png_original")) and not os.path.exists(os.path.join(bad, "taix_original_train.csv"))

        # refused on a row: the list is not written (images before the bad row stay, whole)
        for label, rows, words in (("split", [("a", PNG8, "train"), ("w", PNG8, "test")], ("w is in the test split, not train",)),
                                   ("slash", [("a/b", PNG8, "train")], ("holds a '/'",)),
                                   ("empty", [("a", None, "train")], ("a has no image bytes",)),
                                   ("repeat", [("a", PNG8, "train"), ("a", PNG8, "train")], ("1 repeated UIDs",)),
                                   ("unreadable", [("a", b"neither", "train")], ("a is not a PNG and PIL cannot read it",))):
            bad = fake_root(tmp, {"train-00000-of-00001.parquet": rows})
            result = run(bad, "train")
            assert result.returncode == 1 and all(w in result.stderr for w in words), (label, result.stderr)
            assert not os.path.exists(os.path.join(bad, "taix_original_train.csv")) and not any(n.endswith(".tmp") for n in os.listdir(os.path.join(bad, "png_original", "train"))), label

        # a row count that is not the paper's (the 4 rows against the real 137593, against 5, against 3): everything is read, written and printed, but no list is, and the exit is 1; the PNGs stay, and a rerun
        # with the right count skips them and writes the list
        for rows, paper in ((None, 137593), (5, 5), (3, 3)):
            short = fake_root(tmp, shards)
            s = lambda *parts: os.path.join(short, *parts)
            result = run(short, "train", rows=rows)
            assert result.returncode == 1 and "total 4 rows: 3 byte-copies, 1 re-encoded, 0 already there" in result.stdout and "bit depth, colour type: " in result.stdout, result
            assert f"train rows: found 4, paper {paper}  MISMATCH" in result.stdout and f"taix_original_train.csv not written: 4 rows found, the paper has {paper} for the train split" in result.stderr, result
            assert "stay in png_original/train/ and a rerun skips them" in result.stderr, result.stderr
            assert sorted(os.listdir(s("png_original/train"))) == ["a.png", "b.png", "c.png", "d.png"] and not os.path.exists(s("taix_original_train.csv")) and not os.path.exists(s("taix_original_train.csv.tmp"))
            result = run(short, "train", rows=4)
            assert result.returncode == 0 and "total 4 rows: 0 byte-copies, 0 re-encoded, 4 already there" in result.stdout, result
            assert read(s("taix_original_train.csv")) == b"Path\ntrain/a.png\ntrain/b.png\ntrain/c.png\ntrain/d.png\n"


if __name__ == "__main__":
    test_unpack_original()
    print("OK test_unpack_original")
