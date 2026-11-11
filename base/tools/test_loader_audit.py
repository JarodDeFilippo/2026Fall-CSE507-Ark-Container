"""CPU check of loader_audit.py on a tiny fake dataset: an 8-bit PNG, a full-range 16-bit PNG, and a 16-bit PNG whose content sits in a narrow band next to two outlier pixels.
Run from anywhere: python base/tools/test_loader_audit.py   (needs torch, torchvision, numpy, PIL, PyYAML; no Slurm, no cluster data). The test_ functions are also collected by pytest.
The expectations for the three images restate the loader's rescale as ssl_dinov3/data.py:110-112 writes it (per-image min-max for 16-bit images, convert("L") for 8-bit ones): this test documents
what the loader really does, and fails, at case (c), if that ever changes. It also pins the output-directory rule: --out is new or empty (anything else is refused, nothing touched), and ends at mode 700; and that stats.csv is written atomically (a hidden temporary file, fsynced, renamed); and that each row's tile and list_index are what the docstring says; and that --out is judged by identity, not spelling; and that the batch script restricts its Slurm log."""
import contextlib
import csv
import errno
import io
import os
import random
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True  # importing the tool and the trainer's modules must not leave .pyc files behind
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
REPO = os.path.dirname(BASE)
TOOL = os.path.join(HERE, "loader_audit.py")
sys.path[:0] = [HERE, BASE]

import numpy as np
import torch
import yaml
from PIL import Image

import loader_audit
from ssl_dinov3.data import IMAGENET_MEAN, IMAGENET_STD, DataAugmentationCXR, SSLImageDataset, image_paths

W, H = 96, 80  # the fake images: width x height, so the squash to 512 upsamples them
G, L, P = 512, 224, 512  # global crop, local crop, pre_resize: the run's
PAD, LABEL_H = 4, 14  # the sheets' layout (loader_audit.PAD / LABEL_H), restated so a change of the layout fails here
RAMP = np.tile(np.linspace(0, 1, W), (H, 1))
COLUMNS = ["dataset", "tile", "list_index", "path", "src_mode", "src_dtype", "src_bits", "src_width", "src_height",
           "raw_min", "raw_p1", "raw_p50", "raw_p99", "raw_max",
           "rescaled_min", "rescaled_p1", "rescaled_p50", "rescaled_p99", "rescaled_max", "rescaled_mean", "rescaled_std", "p99_minus_p1", "fraction_at_0", "fraction_at_255"]


def fake_images():
    narrow = (30000 + RAMP * 400).astype(np.uint16)  # a 400-level band of the 65536 levels
    narrow[0, 0], narrow[0, 1] = 0, 65535  # two outlier pixels: the darkest and the brightest value a 16-bit file holds
    return {"a8.png": (40 + RAMP * 160).astype(np.uint8),  # 8-bit, content 40..200
            "b16.png": (RAMP * 65535).astype(np.uint16),  # 16-bit, the full range
            "c16.png": narrow}


def fake_dataset(tmp, padchest=("a8.png", "b16.png", "c16.png")):
    """(image dir, config path): PadChest through a list next to the config (relative), TAIXRay through an absolute one; both data_dirs are wrong on purpose, so only an override reaches the images."""
    images = os.path.join(tmp, "images")
    os.makedirs(images)
    for name, array in fake_images().items():
        Image.fromarray(array).save(os.path.join(images, name))
    os.makedirs(os.path.join(tmp, "cfg"))
    for path, names in ((os.path.join(tmp, "cfg", "padchest.csv"), padchest), (os.path.join(tmp, "taix.csv"), ("b16.png", "c16.png"))):
        with open(path, "w") as f:
            f.write("Path\n" + "".join(name + "\n" for name in names))
    config = os.path.join(tmp, "cfg", "datasets_config.yaml")
    with open(config, "w") as f:
        yaml.safe_dump({"PadChest": {"data_dir": "/nonexistent/padchest", "train_list": "padchest.csv"},
                        "TAIXRay": {"data_dir": "/nonexistent/taix", "train_list": os.path.join(tmp, "taix.csv")}}, f)
    return images, config


def argv_for(out, config, images, *extra, datasets=("PadChest", "TAIXRay"), n=3):
    return ["--datasets", *datasets, "--n", str(n), "--seed", "0", "--out", out, "--config", config] + [x for d in datasets for x in ("--data_root_override", f"{d}={images}")] + list(extra)


def cli(out, config, images, *extra, **kwargs):
    argv = argv_for(out, config, images, *extra, **kwargs)
    return subprocess.run([sys.executable, "-B", TOOL] + argv, capture_output=True, text=True), argv


def run_main(argv):
    """main(argv) in this process, its report lines swallowed."""
    with contextlib.redirect_stdout(io.StringIO()):
        loader_audit.main(argv)


def exit_of(call, *args):
    """What call(*args) exits with (the message of loader_audit's fail(), or argparse's 2); fails when it returns. Its report lines and usage text are swallowed."""
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        try:
            call(*args)
        except SystemExit as error:
            return error.code
    raise AssertionError("%s%s returned" % (call.__name__, args))


def expected_whole(array):
    """What enters the augmentation, from the file's own array: convert("L") for 8 bits, per-image min-max to 0..255 (truncated) for 16, then the bicubic squash to P x P."""
    if array.dtype == np.uint8:
        grey = array
    else:
        a = array.astype(np.float64)
        grey = ((a - a.min()) / (a.max() - a.min()) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(grey).resize((P, P), Image.BICUBIC))


def mode_of(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def snapshot(directory):
    """{file name: bytes} of everything in a directory."""
    return {name: open(os.path.join(directory, name), "rb").read() for name in sorted(os.listdir(directory))}


def read_stats(out):
    with open(os.path.join(out, "stats.csv"), newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS, reader.fieldnames
        return list(reader)


def test_audit_end_to_end():
    arrays = fake_images()
    with tempfile.TemporaryDirectory() as tmp:
        images, config = fake_dataset(tmp)
        out = os.path.join(tmp, "out")
        done, argv = cli(out, config, images)  # n 3: PadChest's list holds 3 images, TAIXRay's only 2
        assert done.returncode == 0, done.stdout + done.stderr
        assert sorted(os.listdir(out)) == ["PadChest_crops.png", "PadChest_whole.png", "TAIXRay_crops.png", "TAIXRay_whole.png", "stats.csv"], os.listdir(out)
        assert oct(os.stat(out).st_mode & 0o777) == "0o700", oct(os.stat(out).st_mode)  # the sheets hold dataset images
        rows = read_stats(out)
        padchest, taix = [r for r in rows if r["dataset"] == "PadChest"], [r for r in rows if r["dataset"] == "TAIXRay"]
        assert len(rows) == 5 and [r["tile"] for r in padchest] == ["0", "1", "2"] and [r["tile"] for r in taix] == ["0", "1"], [(r["dataset"], r["tile"]) for r in rows]
        # list_index is the true 0-based position in the list, drawn here in random.Random(0).sample's order: PadChest's list is a8, b16, c16 (--n 3 takes all three), TAIXRay's b16, c16 (--n 3 takes both)
        assert [(r["path"], r["list_index"]) for r in padchest] == [("b16.png", "1"), ("c16.png", "2"), ("a8.png", "0")], padchest
        assert [(r["path"], r["list_index"]) for r in taix] == [("c16.png", "1"), ("b16.png", "0")], taix
        assert sorted(r["path"] for r in padchest) == sorted(arrays) and sorted(r["path"] for r in taix) == ["b16.png", "c16.png"]
        assert "TAIXRay: the list holds 2 images, fewer than --n 3: auditing all of them" in done.stdout, done.stdout  # the cap is said, not silent

        by = {r["path"]: r for r in padchest}
        for name, row in by.items():  # every column against the file itself and the loader's rescale, restated above
            array, exp = arrays[name], expected_whole(arrays[name])
            assert (int(row["src_width"]), int(row["src_height"])) == (W, H) and row["src_dtype"] == array.dtype.name and int(row["src_bits"]) == 8 * array.dtype.itemsize, row
            assert (float(row["raw_min"]), float(row["raw_max"])) == (array.min(), array.max())
            assert all(abs(float(row["raw_" + k]) - v) < 1e-2 for k, v in zip(("p1", "p50", "p99"), np.percentile(array, [1, 50, 99]))), row
            p1, p50, p99 = np.percentile(exp, [1, 50, 99])
            for key, value in dict(rescaled_min=exp.min(), rescaled_p1=p1, rescaled_p50=p50, rescaled_p99=p99, rescaled_max=exp.max(), rescaled_mean=exp.mean(), rescaled_std=exp.std(),
                                   p99_minus_p1=p99 - p1).items():
                assert abs(float(row[key]) - value) < 2e-3, (name, key, row[key], value)
            assert abs(float(row["fraction_at_0"]) - (exp == 0).mean()) < 1e-5 and abs(float(row["fraction_at_255"]) - (exp == 255).mean()) < 1e-5, row
        # (a) 8-bit: no rescale, the content stays at 40..200
        a = by["a8.png"]
        assert (a["src_mode"], a["src_bits"], a["raw_min"], a["raw_max"]) == ("L", "8", "40", "200"), a
        assert float(a["rescaled_min"]) >= 39 and float(a["rescaled_max"]) <= 201 and 140 < float(a["p99_minus_p1"]) < 170, a
        # (b) 16-bit, full range: min-max leaves it full range
        b = by["b16.png"]
        assert b["src_mode"] in ("I;16", "I") and (b["raw_min"], b["raw_max"]) == ("0", "65535") and float(b["p99_minus_p1"]) > 200, b
        # (c) 16-bit, a 400-level band plus two outlier pixels: the loader's rescale is per-image min-max (ssl_dinov3/data.py:111-112), not a percentile clip, so the two outliers set lo = 0 and
        # hi = 65535 and the band lands on about 1.6 of the 255 output levels (p1 = 116, p99 = 118): the whole image is rendered flat, which p99_minus_p1 reports
        c = by["c16.png"]
        assert (c["raw_min"], c["raw_max"]) == ("0", "65535") and 30000 <= float(c["raw_p1"]) and float(c["raw_p99"]) <= 30400, c
        assert float(c["p99_minus_p1"]) < 4 and 100 < float(c["rescaled_p50"]) < 135, c
        # the summary line: one per dataset; only (c) is below 64; the crop arithmetic for scale (0.25, 1.0) at pre_resize 512 and output 512
        lines = [line for line in done.stdout.splitlines() if line.startswith("PadChest: n=3 |")]
        assert len(lines) == 1 and ", 1 of 3 below 64 | source median 96x80 | global crop at scale 0.25/0.625/1 = 256/405/512 px of the 512x512 image, upsampled to 512 by x2.00/x1.26/x1.00" in lines[0], lines

        # the sheets: sizes, and the pixels of the tiles
        whole, crops = np.asarray(Image.open(os.path.join(out, "PadChest_whole.png"))), np.asarray(Image.open(os.path.join(out, "PadChest_crops.png")))
        assert Image.open(os.path.join(out, "PadChest_whole.png")).size == (PAD + 3 * (P + PAD), PAD + (P + LABEL_H + PAD)) == (1552, 534)
        assert Image.open(os.path.join(out, "PadChest_crops.png")).size == (PAD + 2 * (G + PAD) + 4 * (L + PAD), PAD + 3 * (G + LABEL_H + PAD)) == (1948, 1594)  # 3 images: 3 rows
        assert Image.open(os.path.join(out, "TAIXRay_whole.png")).size == (PAD + 2 * (P + PAD), PAD + (P + LABEL_H + PAD)) == (1036, 534)
        assert Image.open(os.path.join(out, "TAIXRay_crops.png")).size == (1948, PAD + 2 * (G + LABEL_H + PAD)) == (1948, 1064)
        for row in padchest:  # tile i of the whole-image sheet is the real loader's output for that row's file, as it enters the augmentation
            i, x, y = int(row["tile"]), PAD + int(row["tile"]) * (P + PAD), PAD + LABEL_H
            real = np.asarray(SSLImageDataset([os.path.join(images, row["path"])], lambda img: img, P, P)[0][0])
            assert np.array_equal(whole[y:y + P, x:x + P], real), i
            assert np.array_equal(real, expected_whole(arrays[row["path"]])), i
        # the crops of tile 0: the real augmentation, seeded as the tool seeds it (torch.manual_seed(--seed) before the dataset's first image), normalization undone, at native size
        first = SSLImageDataset([os.path.join(images, padchest[0]["path"])], lambda img: img, P, P)[0][0]
        aug = DataAugmentationCXR(local_crops_number=8, global_crops_size=G, local_crops_size=L, gram_teacher_crops_size=G, emit_gram_teacher_crops=False, teacher_no_distortions=True)
        torch.manual_seed(0)
        made = aug(first)
        mean, std = (torch.tensor(v).view(3, 1, 1) for v in (IMAGENET_MEAN, IMAGENET_STD))
        undo = lambda crop: np.rint(((crop * std + mean)[0].clamp(0, 1) * 255).numpy()).astype(int)
        tiles = [(PAD, G, made["global_crops"][0]), (PAD + (G + PAD), G, made["global_crops"][1]), (PAD + 2 * (G + PAD) + 3 * (L + PAD), L, made["local_crops"][3])]  # global 1, global 2, local 4
        for x, side, crop in tiles:
            assert crop.shape == (3, side, side)
            assert np.abs(crops[PAD + LABEL_H:PAD + LABEL_H + side, x:x + side].astype(int) - undo(crop)).max() <= 1, (x, side)

        # deterministic: the same arguments give the same csv and the same four sheets, byte for byte (into a directory of its own: the tool refuses one that already holds the first run's files)
        again = os.path.join(tmp, "again")
        done2, _ = cli(again, config, images)
        assert done2.returncode == 0, done2.stderr
        for name in sorted(os.listdir(out)):
            assert open(os.path.join(out, name), "rb").read() == open(os.path.join(again, name), "rb").read(), name


def test_audit_refusals():
    with tempfile.TemporaryDirectory() as tmp:
        images, config = fake_dataset(tmp, padchest=("a8.png", "gone.png"))
        # a listed image that is not there: non-zero exit, the file named, no stats.csv
        out = os.path.join(tmp, "out")
        done, _ = cli(out, config, images, datasets=("PadChest",), n=2)
        assert done.returncode != 0 and "cannot open" in done.stderr and "gone.png" in done.stderr, (done.returncode, done.stderr)
        assert not os.path.exists(os.path.join(out, "stats.csv")), os.listdir(out)
        # --out under a results/ directory: non-zero exit, and nothing is created
        results = os.path.join(tmp, "results")
        done, _ = cli(os.path.join(results, "audit"), config, images, datasets=("PadChest",), n=1)
        assert done.returncode != 0 and "results/" in done.stderr and not os.path.exists(results), (done.returncode, done.stderr)
        # the same guard in-process: results in any case, the repo checkout itself and everything under it are refused; a fresh directory elsewhere passes
        for bad, words in ((os.path.join(tmp, "Results", "x"), "results/ directory"), (os.path.join(tmp, "a", "results"), "results/ directory"),
                           (REPO, "inside the repo checkout"), (os.path.join(REPO, "base", "tools", "out"), "inside the repo checkout")):
            message = str(exit_of(loader_audit.check_out, bad))
            assert "--out" in message and words in message, (bad, message)
        assert str(loader_audit.check_out(os.path.join(tmp, "fine"))) == os.path.realpath(os.path.join(tmp, "fine"))
        assert not os.path.exists(os.path.join(REPO, "base", "tools", "out"))
        # a missing image root, an override for a dataset that is not audited, --pre_resize 0, then a missing list; none leaves an output directory behind
        argv = ["--datasets", "PadChest", "--out", os.path.join(tmp, "o2"), "--config", config]
        message = exit_of(loader_audit.main, argv + ["--data_root_override", "PadChest=" + os.path.join(tmp, "nope")])
        assert "PadChest: image root" in message and "is missing" in message, message
        assert "--data_root_override" in exit_of(loader_audit.main, argv + ["--data_root_override", "TAIXRay=" + images])
        assert exit_of(loader_audit.main, argv + ["--data_root_override", "PadChest=" + images, "--pre_resize", "0"]) == 2
        os.remove(os.path.join(tmp, "cfg", "padchest.csv"))
        message = exit_of(loader_audit.main, argv + ["--data_root_override", "PadChest=" + images])
        assert "PadChest: train list" in message and "is missing" in message, message
        assert not os.path.exists(os.path.join(tmp, "o2")), "a refused run left its output directory behind"


def test_audit_output_directory():
    """--out is new or empty and ends at exactly 700; a directory with anything in it is refused before any work, and nothing in it, nor its mode, changes."""
    with tempfile.TemporaryDirectory() as tmp:
        images, config = fake_dataset(tmp)
        # (a) an existing empty directory with a permissive mode is accepted and ends at 700
        empty = os.path.join(tmp, "empty")
        os.mkdir(empty)
        os.chmod(empty, 0o755)
        done, _ = cli(empty, config, images, datasets=("PadChest",))
        assert done.returncode == 0, done.stdout + done.stderr
        assert mode_of(empty) == 0o700 and sorted(os.listdir(empty)) == ["PadChest_crops.png", "PadChest_whole.png", "stats.csv"], (oct(mode_of(empty)), os.listdir(empty))
        # (c) a second run into that directory, which is no longer empty, is refused (non-zero, naming the directory), and everything in it stays as it was
        before = snapshot(empty)
        done, _ = cli(empty, config, images, datasets=("PadChest",))
        assert done.returncode != 0 and "is not empty" in done.stderr and "new or empty directory" in done.stderr and os.path.realpath(empty) in done.stderr, (done.returncode, done.stderr)
        assert done.stdout == "" and snapshot(empty) == before and mode_of(empty) == 0o700, "the refused run did work or changed the directory"
        # (b) a directory holding one unrelated file, mode 755: refused before any work; the file, the listing and the mode are untouched; no stats.csv, no sheets
        busy = os.path.join(tmp, "busy")
        os.mkdir(busy)
        os.chmod(busy, 0o755)
        note = os.path.join(busy, "notes.txt")
        with open(note, "w") as f:
            f.write("keep me")
        stamp = os.stat(note).st_mtime_ns
        done, _ = cli(busy, config, images, datasets=("PadChest",))
        assert done.returncode != 0 and "is not empty" in done.stderr and os.path.realpath(busy) in done.stderr, (done.returncode, done.stderr)
        assert done.stdout == "" and os.listdir(busy) == ["notes.txt"] and open(note).read() == "keep me" and os.stat(note).st_mtime_ns == stamp and mode_of(busy) == 0o755
        # the rule in-process: a file is no output directory; the path policy answers before the emptiness rule (this results/ directory is not empty either)
        afile = os.path.join(tmp, "afile")
        open(afile, "w").close()
        assert "not a directory" in str(exit_of(loader_audit.check_out, afile))
        os.makedirs(os.path.join(tmp, "pol", "results"))
        open(os.path.join(tmp, "pol", "results", "x.txt"), "w").close()
        assert "is under a results/ directory" in str(exit_of(loader_audit.check_out, os.path.join(tmp, "pol", "results")))
        # a new directory, parents included, ends at 700; so does one made under a umask that leaves mkdir(mode=0o700) alone at 500 (only the explicit chmod gives 700 there)
        loader_audit.make_out(Path(tmp) / "p" / "q" / "leaf")
        assert mode_of(os.path.join(tmp, "p", "q", "leaf")) == 0o700
        old = os.umask(0o277)
        try:
            loader_audit.make_out(Path(tmp) / "strict")
        finally:
            os.umask(old)
        assert mode_of(os.path.join(tmp, "strict")) == 0o700
        # a mode that does not stick (a filesystem that ignores chmod) is an error, not a silent 755
        stuck = os.path.join(tmp, "stuck")
        os.mkdir(stuck)
        os.chmod(stuck, 0o755)
        real_chmod, os.chmod = os.chmod, lambda *args, **kwargs: None
        try:
            message = str(exit_of(loader_audit.make_out, Path(stuck)))
        finally:
            os.chmod = real_chmod
        assert "is mode 755 after chmod 700" in message, message


LONG = 12  # images in the long fake lists
DRAW_SEED0_N5 = [6, 11, 0, 4, 7]  # random.Random(0).sample(range(12), 5), CPython 3.12: the draw the tool has always made for --seed 0 --n 5 over a 12-image list
DRAW_SEED3_ALL = [3, 9, 8, 2, 5, 4, 11, 6, 0, 10, 7, 1]  # random.Random(3).sample(range(12), 12): --seed 3 --n 20 over that list takes all 12, in this order, not in list order


def long_dataset(tmp):
    """(image dir, config path): LONG small distinct 8-bit images p00.png .., listed by PadChest (a csv with a header row, in name order) and by ChestXray14 (no header, a `name label label` line per image,
    in reverse name order, so the two lists put different images at the same list index). Both data_dirs are wrong on purpose, as in fake_dataset."""
    images, cfg = os.path.join(tmp, "long_images"), os.path.join(tmp, "long_cfg")
    os.makedirs(images)
    os.makedirs(cfg)
    names = ["p%02d.png" % i for i in range(LONG)]
    for i, name in enumerate(names):
        Image.fromarray((RAMP[:32, :40] * (100 + 10 * i)).astype(np.uint8)).save(os.path.join(images, name))
    with open(os.path.join(cfg, "padchest.csv"), "w") as f:
        f.write("Path\n" + "".join(name + "\n" for name in names))
    with open(os.path.join(cfg, "xray.txt"), "w") as f:
        f.write("".join("%s 0 1\n" % name for name in reversed(names)))
    config = os.path.join(cfg, "datasets_config.yaml")
    with open(config, "w") as f:
        yaml.safe_dump({"PadChest": {"data_dir": "/nonexistent/padchest", "train_list": "padchest.csv"},
                        "ChestXray14": {"data_dir": "/nonexistent/xray", "train_list": "xray.txt"}}, f)
    return images, config


def test_audit_tile_and_list_index():
    """tile is the position on the sheets (0 .. n-1, in sheet order); list_index is the 0-based index in the list as image_paths reads it, which does not count a csv list's header row: the image is on line
    list_index + 2 of a csv list, list_index + 1 of a headerless one. The draw is the one the tool made before list_index was kept."""
    with tempfile.TemporaryDirectory() as tmp:
        images, config = long_dataset(tmp)
        cfg = os.path.join(tmp, "long_cfg")
        csv_list, txt_list = os.path.join(cfg, "padchest.csv"), os.path.join(cfg, "xray.txt")
        csv_lines, txt_lines = open(csv_list).read().splitlines(), open(txt_list).read().splitlines()  # 1-based line L of the file is lines[L - 1]
        assert len(csv_lines) == LONG + 1 and len(txt_lines) == LONG
        # (a) a list longer than --n: the tile column is 0..n-1 in sheet order, and every list_index points at the file line that holds the row's path
        out = os.path.join(tmp, "n5")
        run_main(argv_for(out, config, images, datasets=("PadChest", "ChestXray14"), n=5))
        rows = read_stats(out)
        by = {name: [r for r in rows if r["dataset"] == name] for name in ("PadChest", "ChestXray14")}
        assert len(rows) == 10 and all(len(selected) == 5 for selected in by.values())
        for selected in by.values():
            assert [r["tile"] for r in selected] == ["0", "1", "2", "3", "4"]
            assert len({r["list_index"] for r in selected}) == 5 and all(0 <= int(r["list_index"]) < LONG for r in selected)
        for r in by["PadChest"]:  # csv list, header row skipped by the parser: list_index i is on 1-based line i + 2
            assert csv_lines[int(r["list_index"]) + 2 - 1].split(",")[0] == r["path"], r
        for r in by["ChestXray14"]:  # no header: list_index i is on 1-based line i + 1
            assert txt_lines[int(r["list_index"]) + 1 - 1].split()[0] == r["path"], r
        assert by["PadChest"][0]["path"] != by["ChestXray14"][0]["path"]  # the two lists really put different images at one index, so a rule applied to the wrong list shows
        # (b) the selection is unchanged: the same images in the same order as the pre-change logic, [paths[i] for i in random.Random(seed).sample(range(len(paths)), k)], and as its literal values
        for name, listing in (("PadChest", csv_list), ("ChestXray14", txt_list)):
            paths = image_paths(name, images, listing)  # the parser the trainer uses
            before = [os.path.relpath(paths[i], images) for i in random.Random(0).sample(range(len(paths)), 5)]
            assert [r["path"] for r in by[name]] == before, name
            assert [int(r["list_index"]) for r in by[name]] == DRAW_SEED0_N5, name
        # (c) a list shorter than --n (--n 20 over 12): all of it, in the order drawn, each image still at its true list position
        out = os.path.join(tmp, "capped")
        run_main(argv_for(out, config, images, "--seed", "3", datasets=("PadChest",), n=20))
        rows = read_stats(out)
        assert [r["tile"] for r in rows] == [str(i) for i in range(LONG)] and [int(r["list_index"]) for r in rows] == DRAW_SEED3_ALL
        assert DRAW_SEED3_ALL != sorted(DRAW_SEED3_ALL)  # not list order, or a tile number written as list_index would pass
        assert all(csv_lines[int(r["list_index"]) + 1].split(",")[0] == r["path"] for r in rows)
        # (d) a missing sampled image: the error gives its list index (7), not its tile (4, in this draw), and its path
        draw = random.Random(0).sample(range(LONG), LONG)
        assert draw.index(7) != 7
        os.remove(os.path.join(images, "p07.png"))
        message = str(exit_of(loader_audit.main, argv_for(os.path.join(tmp, "n12"), config, images, datasets=("PadChest",), n=LONG)))
        assert "cannot open" in message and "p07.png" in message and "(list index 7)" in message, message


TMP_NAME = re.compile(r"\.stats\.csv\.[A-Za-z0-9_]+\.tmp")  # the hidden temporary file the rows are written to before they are renamed onto stats.csv
SHEETS = ["PadChest_crops.png", "PadChest_whole.png"]  # what a PadChest-only run writes besides stats.csv


def test_audit_stats_csv_atomic_rename():
    """(a) The rows go to a hidden temporary file in --out, which is fsynced and then renamed onto stats.csv, the last thing the run does to files; afterwards there are the sheets and stats.csv, and no
    temporary file. (c) is test_audit_end_to_end's determinism check: two runs into two directories give identical bytes, which still holds."""
    with tempfile.TemporaryDirectory() as tmp:
        images, config = fake_dataset(tmp)
        out = os.path.join(tmp, "ok")
        events, real_fsync, real_replace = [], os.fsync, os.replace

        def spy_fsync(fd):
            events.append(("fsync", os.fstat(fd).st_size))  # how much the temporary file already holds when it is synced
            return real_fsync(fd)

        def spy_replace(src, dst, *args, **kwargs):
            events.append(("replace", os.path.dirname(src) == os.path.dirname(dst), os.path.basename(src), os.path.basename(dst)))
            return real_replace(src, dst, *args, **kwargs)

        os.fsync, os.replace = spy_fsync, spy_replace
        try:
            run_main(argv_for(out, config, images, datasets=("PadChest",)))
        finally:
            os.fsync, os.replace = real_fsync, real_replace
        assert [e[0] for e in events[-2:]] == ["fsync", "replace"] and events[-1][1] is True and events[-1][3] == "stats.csv" and TMP_NAME.fullmatch(events[-1][2]), events  # same directory, hidden .tmp name
        final_size = os.stat(os.path.join(out, "stats.csv")).st_size
        assert events[-2][1] > 0 and events[-2][1] == final_size, (events[-2], final_size)  # flushed before the fsync: the file already held all of stats.csv, not an empty file and a buffer
        assert sorted(os.listdir(out)) == SHEETS + ["stats.csv"], os.listdir(out)
        assert read_stats(out) and mode_of(out) == 0o700 and mode_of(os.path.join(out, "stats.csv")) == 0o600  # the renamed file keeps the temporary file's owner-only mode


def test_audit_stats_csv_write_failure():
    """(b) The writer fails partway, as a full disk does (an OSError) or an interrupt does (a BaseException): the temporary file was there, hidden, mode 600, beside both finished sheets, and no stats.csv
    existed; afterwards the temporary file is gone, nothing is named stats.csv, and the error got out of main (uncaught, it is a non-zero exit)."""
    with tempfile.TemporaryDirectory() as tmp:
        images, config = fake_dataset(tmp)
        for failure in (OSError(errno.ENOSPC, "No space left on device"), KeyboardInterrupt()):
            out, seen = os.path.join(tmp, "fail_" + type(failure).__name__), {}
            real_writerows = loader_audit.csv.DictWriter.writerows

            def failing_writerows(self, rows, failure=failure, out=out, seen=seen):
                self.writerow(rows[0])  # a half-written file
                seen["names"] = sorted(os.listdir(out))
                seen["modes"] = {name: mode_of(os.path.join(out, name)) for name in seen["names"] if name.endswith(".tmp")}
                raise failure

            loader_audit.csv.DictWriter.writerows = failing_writerows
            try:
                try:
                    run_main(argv_for(out, config, images, datasets=("PadChest",)))
                except type(failure) as error:
                    assert error is failure
                else:
                    raise AssertionError("the failure was swallowed: %r" % (failure,))
            finally:
                loader_audit.csv.DictWriter.writerows = real_writerows
            temporary = [name for name in seen["names"] if name.endswith(".tmp")]
            assert len(temporary) == 1 and TMP_NAME.fullmatch(temporary[0]), seen
            assert seen["names"] == sorted(SHEETS + temporary) and seen["modes"] == {temporary[0]: 0o600}, seen  # no stats.csv yet; the sheets already written; owner-only
            assert sorted(os.listdir(out)) == SHEETS, os.listdir(out)  # no stats.csv, no temporary file


@contextlib.contextmanager
def repo_at(path):
    """loader_audit's idea of the checkout for one block: the tool takes it from its own location (there is no flag for it), so in-process tests patch the module's REPO."""
    real = loader_audit.REPO
    loader_audit.REPO = Path(path)
    try:
        yield
    finally:
        loader_audit.REPO = real


def tree(root):
    return sorted(os.path.join(directory, name) for directory, directories, files in os.walk(root) for name in directories + files)


def test_audit_out_is_refused_by_identity():
    """--out is judged by what directory it is, not how it is spelled: when it, or an ancestor of it, is the repo root or a directory under <repo>/.container-runtime/ (nvidia.sh binds
    .container-runtime/nvidia at /runtime and sets TMPDIR=/runtime/tmp) it is refused under whatever name, and nothing is created; a legitimate directory is accepted."""
    with tempfile.TemporaryDirectory() as raw:
        tmp = os.path.realpath(raw)  # resolved: /var is a symlink on macOS, and the tool compares resolved paths
        repo = os.path.join(tmp, "repo")
        for sub in ("base", os.path.join(".container-runtime", "nvidia", "tmp"), os.path.join(".container-runtime", "gaudi", "habana_logs")):
            os.makedirs(os.path.join(repo, sub))
        images, config = fake_dataset(os.path.join(tmp, "data"))
        refused = lambda out: str(exit_of(loader_audit.check_out, out))
        before = tree(repo)
        with repo_at(repo):
            # a symlink to the repo as the parent of --out, through main(): refused before anything is made; a symlink to a subdirectory of the repo; and the bind simulated by a symlink, another name
            # for .container-runtime/nvidia, which is what /runtime is in the container
            os.symlink(repo, os.path.join(tmp, "to_repo"))
            message = str(exit_of(loader_audit.main, argv_for(os.path.join(tmp, "to_repo", "audit"), config, images, datasets=("PadChest",))))
            assert "inside the repo checkout" in message, message
            os.symlink(os.path.join(repo, "base"), os.path.join(tmp, "to_base"))
            assert "inside the repo checkout" in refused(os.path.join(tmp, "to_base", "audit", "deeper"))
            os.symlink(os.path.join(repo, ".container-runtime", "nvidia"), os.path.join(tmp, "runtime"))
            assert "inside the repo checkout" in refused(os.path.join(tmp, "runtime", "tmp", "audit"))
            # legitimate: a directory elsewhere, and one beside the repo under a name that starts like its own
            for fine in (os.path.join(tmp, "legit", "audit"), os.path.join(tmp, "repo_notes", "audit")):
                assert loader_audit.check_out(fine) == Path(fine), fine
        assert tree(repo) == before, "a refused run created something in the checkout"
        # the checkout under a spelling that resolved paths never use: the string check cannot match it, only identity can (the repo itself, a new child, a path whose first existing ancestor is inside)
        os.symlink(repo, os.path.join(tmp, "alias"))
        with repo_at(os.path.join(tmp, "alias")):
            for out in (repo, os.path.join(repo, "audit"), os.path.join(repo, "base", "new", "deeper")):
                assert "inside the repo checkout" in refused(out), out
        # the directories under .container-runtime, with no ancestor that is the repo: the container's /runtime/tmp and /var/log/habana_logs, here two directories elsewhere that the fake checkout's
        # .container-runtime holds by symlink (a bind is a directory under two names)
        repo2 = os.path.join(tmp, "repo2")
        os.makedirs(os.path.join(repo2, ".container-runtime", "gaudi"))
        for sub in (os.path.join("runtime", "tmp"), "habana_logs", "other"):
            os.makedirs(os.path.join(tmp, "elsewhere", sub))
        os.symlink(os.path.join(tmp, "elsewhere", "runtime"), os.path.join(repo2, ".container-runtime", "nvidia"))
        os.symlink(os.path.join(tmp, "elsewhere", "habana_logs"), os.path.join(repo2, ".container-runtime", "gaudi", "habana_logs"))
        with repo_at(repo2):
            for out in (("runtime", "tmp", "audit"), ("runtime", "audit"), ("habana_logs", "audit")):
                assert "inside the repo checkout" in refused(os.path.join(tmp, "elsewhere", *out)), out
            assert loader_audit.check_out(os.path.join(tmp, "elsewhere", "other", "audit")) == Path(tmp, "elsewhere", "other", "audit")  # not bound: fine
        # the resolved spelling still counts where identity cannot be asked: a checkout that is not there to stat (a new path under it is refused by its name)
        with repo_at(os.path.join(tmp, "ghost")):
            assert "inside the repo checkout" in refused(os.path.join(tmp, "ghost", "audit"))
        assert tree(os.path.join(tmp, "elsewhere")) == sorted(os.path.join(tmp, "elsewhere", sub) for sub in ("runtime", os.path.join("runtime", "tmp"), "habana_logs", "other")), "a refused run created something"


def test_audit_out_case_insensitive_spelling():
    """On a case-insensitive filesystem the checkout is one directory under many spellings (/Repo, /rEPO): refused by identity. Skipped where the filesystem tells letter cases apart."""
    with tempfile.TemporaryDirectory() as raw:
        tmp = os.path.realpath(raw)
        os.mkdir(os.path.join(tmp, "Checkout"))
        if not os.path.exists(os.path.join(tmp, "cHECKOUT")):
            raise unittest.SkipTest("this filesystem is case-sensitive")
        images, config = fake_dataset(os.path.join(tmp, "data"))
        with repo_at(os.path.join(tmp, "Checkout")):
            message = str(exit_of(loader_audit.main, argv_for(os.path.join(tmp, "cHECKOUT", "audit"), config, images, datasets=("PadChest",))))
            assert "inside the repo checkout" in message, message
            assert os.listdir(os.path.join(tmp, "Checkout")) == []


def test_audit_batch_script_restricts_its_log():
    """loader_audit.sh's first action makes the Slurm log owner-only: Slurm created it (the #SBATCH -o file) with the submitter's umask before the script ran, and an error for an unreadable image puts
    that image's path in it. A chmod that cannot be done is a warning, not a failed job. Run in a sandbox: the cluster paths substituted, a stand-in for nvidia.sh."""
    script = os.path.join(HERE, "loader_audit.sh")
    text = open(script).read()
    assert subprocess.run(["bash", "-n", script]).returncode == 0
    code = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    first = next(i for i, line in enumerate(code) if line.startswith("chmod 600"))
    assert all(line.startswith(("set ", "LOG=")) for line in code[:first]), code[:first]  # only the shell options and the log's path come before it
    assert all(not line.startswith(("umask", "mkdir", "cd ", "DATA_BIND")) for line in code[:first])
    with tempfile.TemporaryDirectory() as raw:
        tmp = os.path.realpath(raw)
        sandbox = os.path.join(tmp, "scratch", "jdefili4")
        root, new = os.path.join(sandbox, "2026Fall-CSE507-Ark-Container"), os.path.join(sandbox, "datasets-new")
        os.makedirs(root)
        os.makedirs(new)
        stub = os.path.join(root, "nvidia.sh")  # records what it is given, and the log's mode at the moment the container would start
        with open(stub, "w") as f:
            f.write('#!/usr/bin/env bash\nhere=$(dirname "$0")\nprintf "%s\\n" "$@" > "$here/argv.txt"\nprintf "%s" "${DATA_BIND:-}" > "$here/data_bind.txt"\n'
                    + '"%s" -c \'import os, stat, sys; print(oct(stat.S_IMODE(os.stat(sys.argv[1]).st_mode)))\' "$here/../datasets-new/loader-audit.$SLURM_JOB_ID.out" > "$here/log_mode.txt" 2>/dev/null\nexit 0\n' % sys.executable)
        os.chmod(stub, 0o755)
        sandboxed = os.path.join(tmp, "loader_audit_sandboxed.sh")
        with open(sandboxed, "w") as f:
            f.write(text.replace("/scratch/jdefili4", sandbox))
        header_log = re.search(r"^#SBATCH -o (\S+)$", text, re.M).group(1).replace("/scratch/jdefili4", sandbox)  # the -o path, %j still in it
        assert header_log == os.path.join(new, "loader-audit.%j.out"), header_log
        env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}

        def submit(job, *args):
            return subprocess.run(["bash", sandboxed, *args], env=dict(env, SLURM_JOB_ID=job), capture_output=True, text=True)

        # the log as Slurm leaves it (umask 022: 644), at the path the #SBATCH -o line gives for this job: owner-only by the time the container starts, and the rest of the script as before
        log = header_log.replace("%j", "4242")
        with open(log, "w") as f:
            f.write("written by slurm\n")
        os.chmod(log, 0o644)
        done = submit("4242", "--n", "5")
        assert done.returncode == 0 and done.stderr == "", (done.returncode, done.stderr)
        assert mode_of(log) == 0o600 and open(os.path.join(root, "log_mode.txt")).read().strip() == "0o600", (oct(mode_of(log)), open(os.path.join(root, "log_mode.txt")).read())
        assert open(os.path.join(root, "argv.txt")).read().splitlines() == ["python", "/workspace/base/tools/loader_audit.py", "--out", os.path.join(new, "loader_audit", "4242"), "--n", "5"]
        assert open(os.path.join(root, "data_bind.txt")).read() == "/data:/data:ro,%s:%s:rw" % (new, new)
        assert mode_of(os.path.join(new, "loader_audit")) == 0o700 and mode_of(os.path.join(new, "loader_audit", "4242")) == 0o700
        # no log at that path (an sbatch -o on the command line moved it): a warning naming the file on stderr, and the job goes on
        done = submit("4243")
        assert done.returncode == 0 and "WARNING" in done.stderr and "loader-audit.4243.out" in done.stderr, (done.returncode, done.stderr)
        assert os.path.isdir(os.path.join(new, "loader_audit", "4243")) and open(os.path.join(root, "argv.txt")).read().splitlines()[:2] == ["python", "/workspace/base/tools/loader_audit.py"]
        # outside Slurm there is no job id: stop at once, before touching anything
        done = subprocess.run(["bash", sandboxed], env=env, capture_output=True, text=True)
        assert done.returncode != 0 and "SLURM_JOB_ID" in done.stderr and not os.path.exists(os.path.join(new, "loader_audit", "4244")), (done.returncode, done.stderr)


def test_audit_second_dataset_failure():
    """stats.csv is written once, after every dataset: when the second dataset fails, the first one's finished sheets are there and there is no stats.csv and no temporary file; the exit is non-zero;
    and a rerun into that directory, which is no longer empty, is refused."""
    with tempfile.TemporaryDirectory() as tmp:
        images, config = long_dataset(tmp)
        assert 5 not in DRAW_SEED0_N5 and LONG - 1 - DRAW_SEED0_N5[0] == 5  # PadChest's draw (--seed 0 --n 5) misses p05; ChestXray14's list is reversed, so its first sampled image (list index 6) is p05
        os.remove(os.path.join(images, "p05.png"))
        out = os.path.join(tmp, "two")
        done, _ = cli(out, config, images, datasets=("PadChest", "ChestXray14"), n=5)
        assert done.returncode != 0 and "ChestXray14: cannot open" in done.stderr and "p05.png" in done.stderr and "(list index 6)" in done.stderr, (done.returncode, done.stderr)
        assert sorted(os.listdir(out)) == ["PadChest_crops.png", "PadChest_whole.png"], os.listdir(out)  # the first dataset's sheets; no stats.csv, no .stats.csv.*.tmp
        before = snapshot(out)
        again, _ = cli(out, config, images, datasets=("PadChest", "ChestXray14"), n=5)
        assert again.returncode != 0 and "is not empty" in again.stderr and snapshot(out) == before, (again.returncode, again.stderr)


def test_audit_jpeg_uses_the_runs_decode_side():
    """The loader decodes a JPEG through PIL's draft at the dataset class's decode side, which the run sets to --pre_resize (512: draft takes a 2200 x 2100 file to 550 x 525); left at the class default
    (1024) it would be 1100 x 1050, other pixels. PNG, the other tests' fixtures, ignores draft, so only a JPEG shows it: the tool's numbers, whole-image tile and crops must be the run's."""
    with tempfile.TemporaryDirectory() as tmp:
        images, cfg = os.path.join(tmp, "jpeg_images"), os.path.join(tmp, "jpeg_cfg")
        os.makedirs(images)
        os.makedirs(cfg)
        h, w = 2100, 2200
        yy, xx = np.mgrid[0:h, 0:w]
        pixels = np.clip(128 + 55 * np.random.RandomState(1).standard_normal((h, w)) + 45 * np.sin(xx / 5.0) * np.cos(yy / 7.0), 0, 255).astype(np.uint8)  # noise and a fine pattern: the detail a draft decode changes
        path = os.path.join(images, "big.jpg")
        Image.fromarray(pixels).save(path, quality=90)  # 8-bit grey, as the JPEG datasets are
        with open(os.path.join(cfg, "mimic.csv"), "w") as f:
            f.write("Path\nbig.jpg\n")
        config = os.path.join(cfg, "datasets_config.yaml")
        with open(config, "w") as f:
            yaml.safe_dump({"MIMIC": {"data_dir": "/nonexistent/mimic", "train_list": "mimic.csv"}}, f)
        sizes = []
        for side in (P, 1024):  # the premise: the two decode sides decode the file differently
            with Image.open(path) as jpeg:
                jpeg.draft("L", (side, side))
                sizes.append(jpeg.size)
        assert sizes == [(550, 525), (1100, 1050)], sizes
        out = os.path.join(tmp, "out")
        run_main(argv_for(out, config, images, datasets=("MIMIC",), n=1))
        row = read_stats(out)[0]
        identity = lambda img: img
        run = np.asarray(SSLImageDataset([path], identity, P, P)[0][0])  # the trainer's own dataset class at the run's decode side (ssl_pretrain.py:653: --pre_resize)
        default = np.asarray(SSLImageDataset([path], identity, pre_resize=P)[0][0])  # decode side left at the class default
        numbers = lambda a: dict(rescaled_mean=a.mean(), rescaled_std=a.std(), rescaled_p1=np.percentile(a, 1), rescaled_p50=np.percentile(a, 50), rescaled_p99=np.percentile(a, 99))
        for key, value in numbers(run).items():
            assert abs(float(row[key]) - value) < 2e-3, (key, row[key], value)
        assert abs(float(row["rescaled_std"]) - numbers(default)["rescaled_std"]) > 0.1, "the class default gives the same numbers: this fixture cannot tell the decode sides apart"
        whole = np.asarray(Image.open(os.path.join(out, "MIMIC_whole.png")))
        assert np.array_equal(whole[PAD + LABEL_H:PAD + LABEL_H + P, PAD:PAD + P], run) and not np.array_equal(run, default)  # tile 0 is the image as the run decodes it
        aug = DataAugmentationCXR(local_crops_number=8, global_crops_size=G, local_crops_size=L, gram_teacher_crops_size=G, emit_gram_teacher_crops=False, teacher_no_distortions=True)
        torch.manual_seed(0)
        made = aug(Image.fromarray(run))  # the crops are cut from that image
        mean, std = (torch.tensor(v).view(3, 1, 1) for v in (IMAGENET_MEAN, IMAGENET_STD))
        undo = lambda crop: np.rint(((crop * std + mean)[0].clamp(0, 1) * 255).numpy()).astype(int)
        crops = np.asarray(Image.open(os.path.join(out, "MIMIC_crops.png")))
        assert np.abs(crops[PAD + LABEL_H:PAD + LABEL_H + G, PAD:PAD + G].astype(int) - undo(made["global_crops"][0])).max() <= 1


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
            except unittest.SkipTest as skipped:  # pytest reports the same exception as a skip
                print("SKIP", name, "-", skipped)
            else:
                print("OK", name)
    print("ALL OK")
