"""What the SSL training loader actually feeds the model, per pretraining dataset (CPU only): whether the high-bit-depth images come out flat or dark, and how much real detail the crops hold.

Run from anywhere:  python base/tools/loader_audit.py --out DIR [--datasets NAME ...] [--n 20] [--seed 0] [--config base/datasets_config.yaml] [--pre_resize 512]
                    [--global_size 512] [--local_size 224] [--data_root_override NAME=PATH ...]          (on Sol: sbatch base/tools/loader_audit.sh)
Nothing of the loader is re-implemented: every image goes through ssl_dinov3.data.SSLImageDataset (decode, the per-image rescale of high-bit-depth images, --pre_resize) and DataAugmentationCXR,
set up as base/ssl_pretrain.py sets them up for chain/newdata512.sh (the constants below say where each value comes from). Per dataset it draws --n images from the SSL train list
of --config as random.Random(--seed).sample(range(len(list)), n): the images come out in the order drawn, and that order is the order of the tiles on the sheets. It writes to --out:
  stats.csv             one row per image, in tile order. The first columns name the image: dataset; tile, its position on the sheets (0 .. n-1: the label of its tile in <dataset>_whole.png and of its row in
                        <dataset>_crops.png); list_index, its 0-based index in the list of image paths that ssl_dinov3.data.image_paths returns for the dataset's train list, the list the sample is drawn from;
                        path, relative to the dataset's data_dir. list_index does NOT count the list file's header: image_paths skips the header row of the csv lists (CheXpert, MIMIC, PadChest, TAIXRay,
                        ReXGradient), so for them an image's 1-based line in the list file is list_index + 2, and for ChestXray14 and VinDrCXR, whose lists have no header and one image per line, it is
                        list_index + 1. A list shorter than --n is audited whole, still with each image's true list_index. Then the source (PIL mode, dtype, bits, size; raw min / p1 / p50 / p99 / max
                        over every sample of the file as stored) and the image as it enters the augmentation, i.e. after the loader's rescale and the --pre_resize squash, before any crop, jitter or
                        normalization, as 8-bit grey: min, p1, p50, p99, max, mean, std, p99_minus_p1, fraction_at_0, fraction_at_255. Written last and atomically, so it exists only if every dataset finished and is never
                        half-written: the rows go to a hidden temporary file in --out (.stats.csv.<random>.tmp, mode 600), which is flushed, fsynced and renamed onto stats.csv only after every dataset
                        and every sheet is done. If writing fails, the temporary file is removed and the run exits non-zero; a process killed outright can leave only that temporary file.
  <dataset>_whole.png   contact sheet of those images, each tile labelled with its tile number
  <dataset>_crops.png   for tiles 0-3 (the first 4 images), one row each, labelled with the tile number: the two global crops and the first 4 local crops of one DataAugmentationCXR call (normalization
                        undone), each at its own size
and prints one summary line per dataset. A missing list, image root or image is an error (exit 1), never a skip; an error that names a sampled image gives its list_index and path. --out must be outside the repo checkout and not under a results/ directory:
the sheets hold dataset images, one dataset's data-use agreement forbids sharing them, and results/ is group-readable on the cluster. The checkout is recognized by what directory it is, not by how a path
is spelled (a symlink, another letter case, or the container's /runtime, which is <repo>/.container-runtime/nvidia under another name, all count): --out is refused when it, or any ancestor of it, is the
repo root or a directory under <repo>/.container-runtime/ by identity (os.path.samefile), as well as by its resolved spelling. It must also be new or empty: a non-empty --out is refused
before any work is done, with nothing in it touched, so no stale stats.csv or sheets of an earlier run can mix with this run's ("stats.csv exists" keeps meaning "every dataset finished").
The directory is created if missing (parents too) and set to mode 700 either way with an explicit chmod, so the mode depends neither on the umask nor on an older directory's permissions;
the tool exits if the mode does not stick (a filesystem that ignores chmod).
A relative train_list in --config is read from the config file's directory (for base/datasets_config.yaml that is base/, the trainer's working directory).
--pre_resize 0 (the Stage A runs: crops cut from the full decode) is refused: the sheets and the crop arithmetic below assume the squash to a square.
"""
import argparse
import contextlib
import csv
import math
import os
import random
import stat
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # base/: ssl_dinov3 and dinov3 import from here (the trainer runs with cwd = base/)
REPO = BASE.parent
sys.path.insert(0, str(BASE))

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from ssl_dinov3.data import DATASET_NAMES, IMAGENET_MEAN, IMAGENET_STD, DataAugmentationCXR, SSLImageDataset, image_paths

# The 512 run's loader settings (chain/newdata512.sh:62-64, through chain/ssl_common.sh:36-39, into base/ssl_pretrain.py) and where each value comes from:
#  - DataAugmentationCXR is built as ssl_pretrain.py:650-652 builds it. global / local size: --global_size / --local_size (the run's --global_crops_size 512 and --local_crops_size 224,
#    newdata512.sh:62). gram_teacher_crops_size = the global size (newdata512.sh:62 --gram_teacher_size 512; --no_gram accepts that one value, ssl_pretrain.py:343). emit_gram_teacher_crops False
#    (--no_gram, newdata512.sh:64). teacher_no_distortions True (--cyclic_ark_teacher, newdata512.sh:35, implies --teacher_clean_view, ssl_pretrain.py:359).
#  - local_crops_number: --local_crops_number's default, ssl_pretrain.py:273; no chain script passes it.
#  - everything else is DataAugmentationCXR's own default, which ssl_pretrain.py leaves alone: crop scales (0.25, 1.0) global and (0.04, 0.25) local (ssl_dinov3/data.py:134-135), horizontal
#    flips p 0.5 (:140, :150), ImageNet mean / std (:141-142), bicubic RandomResizedCrop (:156, :162), ColorJitter(brightness 0.2, contrast 0.2) p 0.8 (:185), GaussianBlur(9, sigma 0.1-1.0) p 0.5 (:188).
#    The scales are read back from the built object below, not repeated here.
#  - the decode side SSLImageDataset gets is --pre_resize when that is set (ssl_pretrain.py:653: args.pre_resize or transform.decode_min_side).
LOCAL_CROPS_NUMBER = 8

FLAT_BELOW = 64  # p99_minus_p1 under this: the middle 98% of the pixels sit in under a quarter of the 0-255 range
PAD, LABEL_H, BACKGROUND = 4, 14, 48  # contact sheets: gap around the tiles, label strip above each tile (px), background grey
WHOLE_COLS, CROP_ROWS, LOCAL_SHOWN = 5, 4, 4  # whole-image sheet columns; images that get a row in the crops sheet; local crops drawn in it
STAT_NAMES = ("min", "p1", "p50", "p99", "max")
FONT = ImageFont.load_default()
MEAN, STD = (torch.tensor(v).view(3, 1, 1) for v in (IMAGENET_MEAN, IMAGENET_STD))


def fail(message):
    sys.exit(f"loader_audit: {message}")


def same_dir(a, b):
    """Whether two paths are one directory by identity (device and inode), not by spelling; False if either cannot be examined (it does not exist)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def checkout_dirs():
    """The directories that are the repo checkout, or sit inside it and are bound into a container under another name: the repo root, and what is under <repo>/.container-runtime/, one and two levels
    down: the deepest any script binds (nvidia.sh:18 binds .container-runtime/nvidia at /runtime, and nvidia.sh:22 sets TMPDIR=/runtime/tmp, so a path under /runtime has /runtime as an ancestor;
    gaudi.sh:43 and :45 bind .container-runtime/gaudi at /runtime and .container-runtime/gaudi/habana_logs at /var/log/habana_logs). A symlink there counts as the directory it points to, as a bind does."""
    runtime = REPO / ".container-runtime"
    below = sorted(d for pattern in ("*", "*/*") for d in runtime.glob(pattern) if d.is_dir()) if os.path.isdir(runtime) else []
    return [REPO, *below]


def check_out(out):
    """The resolved --out, or exit. Path policy first: the sheets hold dataset images, so not inside the repo checkout and not under a results/ directory (group-readable on the cluster). The checkout
    is decided by identity, not spelling: besides the resolved spelling (which still catches a path that does not exist yet), --out or any ancestor of it, from the first existing one up to the
    filesystem root, being the same directory as the repo root or one under <repo>/.container-runtime/ (see checkout_dirs) refuses it. The results/ rule stays a rule about names, over the
    resolved chain, as before. Then the directory rule: --out is new or empty, never a directory with files in it (a stale stats.csv, whose presence means every dataset finished, or stale sheets would
    mix with this run's). Checks only: nothing is created or changed here, so a refusal leaves everything as it was."""
    path = Path(out).expanduser().resolve()
    if any(part.lower() == "results" for part in path.parts):
        fail(f"--out {path} is under a results/ directory; the sheets hold dataset images and results/ is group-readable on the cluster")
    if path == REPO or REPO in path.parents:
        fail(f"--out {path} is inside the repo checkout {REPO}; the sheets hold dataset images and must stay out of git and out of any rsync of the repo")
    targets = checkout_dirs()
    for ancestor in (path, *path.parents):  # --out if it exists, else its first existing ancestor (where it would be made), and on up to the root
        for target in targets:
            if same_dir(ancestor, target):
                fail(f"--out {path} is inside the repo checkout: {ancestor} is {target}, the same directory under another name; the sheets hold dataset images and must stay out of git and out of any rsync of the repo")
    if path.exists() and not path.is_dir():
        fail(f"--out {path} exists and is not a directory; the tool writes only into a new or empty directory")
    if path.is_dir() and any(path.iterdir()):
        fail(f"--out {path} is not empty: the tool writes only into a new or empty directory, so that files of an earlier run cannot mix with this run's stats.csv and sheets; "
             "give a new --out, or move or delete that directory's contents first")
    return path


def make_out(out):
    """Creates --out if it is missing (parents too) and sets it to exactly mode 700 either way, with an explicit chmod: the sheets hold dataset images, and the mode must depend neither on the umask
    nor on the permissions an existing empty directory had. Exits if the mode does not stick (a filesystem that ignores chmod)."""
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(out, 0o700)
    except OSError as error:
        fail(f"--out {out}: cannot set its mode to 700: {error}")
    mode = stat.S_IMODE(os.stat(out).st_mode)
    if mode != 0o700:
        fail(f"--out {out} is mode {mode:o} after chmod 700 (a filesystem that ignores chmod?); refusing to write dataset images into it")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES), help="names as in datasets_config.yaml (default: the seven of the 512 run)")
    p.add_argument("--n", type=int, default=20, help="images per dataset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output directory: new or empty (a non-empty one is refused), set to mode 700; not inside the repo checkout, not under results/")
    p.add_argument("--config", default=str(BASE / "datasets_config.yaml"))
    p.add_argument("--pre_resize", type=int, default=512, help="the run's --pre_resize: every image is squashed to this square before the crops (at least 1)")
    p.add_argument("--global_size", type=int, default=512)
    p.add_argument("--local_size", type=int, default=224)
    p.add_argument("--data_root_override", action="append", default=[], metavar="NAME=PATH", help="read NAME's images from PATH instead of its data_dir (repeatable)")
    args = p.parse_args(argv)
    for name in ("n", "pre_resize", "global_size", "local_size"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be at least 1, got {getattr(args, name)}")
    return args


def dataset_files(name, config, config_path, overrides):
    """(data_dir, train list) of one dataset, both checked to exist."""
    entry = config.get(name)
    if not isinstance(entry, dict):
        fail(f"{name}: no such entry in {config_path}")
    data_dir, list_file = overrides.get(name, entry.get("data_dir")), entry.get("train_list")
    if not data_dir or not list_file:
        fail(f"{name}: {config_path} gives it no data_dir or train_list")
    list_file = Path(list_file)
    if not list_file.is_absolute():
        list_file = config_path.parent / list_file
    if not list_file.is_file():
        fail(f"{name}: train list {list_file} is missing")
    if not Path(data_dir).is_dir():
        fail(f"{name}: image root {data_dir} is missing")
    return str(data_dir), str(list_file)


def describe(a):
    """min, p1, p50, p99, max over every sample of the array (all channels of a colour image), as plain numbers (a numpy float would print as np.float64(...) in the csv)."""
    p1, p50, p99 = np.percentile(a, [1, 50, 99])
    return a.min().item(), round(float(p1), 3), round(float(p50), 3), round(float(p99), 3), a.max().item()


def stats(prefix, a):
    return {f"{prefix}_{key}": value for key, value in zip(STAT_NAMES, describe(a))}


def to_image(crop):
    """A normalized 3 x H x W crop back to the 8-bit grey it was made from (v2.RGB() copied one grey channel three times)."""
    return Image.fromarray(((crop * STD + MEAN).clamp(0, 1)[0] * 255).round().byte().numpy())


def put(canvas, draw, x, y, label, image):
    draw.text((x, y + 1), label, fill=255, font=FONT)
    canvas.paste(image, (x, y + LABEL_H))


def whole_sheet(images):
    """Every image at its own size (they are all pre_resize x pre_resize), WHOLE_COLS to a row, a label strip above each."""
    side = images[0].size[0]
    cols, rows = min(WHOLE_COLS, len(images)), math.ceil(len(images) / WHOLE_COLS)
    canvas = Image.new("L", (PAD + cols * (side + PAD), PAD + rows * (side + LABEL_H + PAD)), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    for i, image in enumerate(images):
        put(canvas, draw, PAD + i % WHOLE_COLS * (side + PAD), PAD + i // WHOLE_COLS * (side + LABEL_H + PAD), str(i), image)
    return canvas


def crops_sheet(rows):
    """rows: one list of (label, image) per sampled image; every tile at its own size, a row as tall as its tallest tile plus the label strip."""
    height = lambda row: max(image.size[1] for _, image in row) + LABEL_H + PAD
    canvas = Image.new("L", (PAD + max(sum(image.size[0] + PAD for _, image in row) for row in rows), PAD + sum(height(row) for row in rows)), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    y = PAD
    for row in rows:
        x = PAD
        for label, image in row:
            put(canvas, draw, x, y, label, image)
            x += image.size[0] + PAD
        y += height(row)
    return canvas


def audit_dataset(name, data_dir, list_file, args, aug, out):
    """Samples, measures and draws one dataset; returns its stats rows."""
    paths = image_paths(name, data_dir, list_file)  # the trainer's own parser for this dataset's list
    if not paths:
        fail(f"{name}: {list_file} lists no images")
    k = min(args.n, len(paths))
    if k < args.n:
        print(f"{name}: the list holds {len(paths)} images, fewer than --n {args.n}: auditing all of them", flush=True)
    picked = random.Random(args.seed).sample(range(len(paths)), k)  # the draw, unchanged: list indices in the order drawn, so tile t is picked[t]; only the indices are kept now, not dropped
    chosen = [paths[i] for i in picked]
    # the real decode + rescale + squash; the identity transform hands back the image DataAugmentationCXR would receive. decode_side is --pre_resize (ssl_pretrain.py:653)
    loader = SSLImageDataset(chosen, lambda img: img, args.pre_resize, args.pre_resize)
    wholes, rows = [], []
    for tile, (list_index, path) in enumerate(zip(picked, chosen)):
        try:
            with Image.open(path) as src:  # the file as stored, for the raw columns
                mode, (width, height), raw = src.mode, src.size, np.asarray(src)
            whole = loader[tile][0]
        except OSError as error:  # missing, unreadable, truncated or not an image (PIL's UnidentifiedImageError is one)
            fail(f"{name}: cannot open {path} (list index {list_index}): {error}")  # the list index, which finds the image in the list file; a tile number would not
        grey = np.asarray(whole)
        rescaled = stats("rescaled", grey)
        rows.append({
            "dataset": name, "tile": tile, "list_index": list_index, "path": os.path.relpath(path, data_dir),
            "src_mode": mode, "src_dtype": raw.dtype.name, "src_bits": raw.dtype.itemsize * 8, "src_width": width, "src_height": height,
            **stats("raw", raw), **rescaled,
            "rescaled_mean": round(float(grey.mean()), 3), "rescaled_std": round(float(grey.std()), 3),
            "p99_minus_p1": round(rescaled["rescaled_p99"] - rescaled["rescaled_p1"], 3),
            "fraction_at_0": round(float((grey == 0).mean()), 5), "fraction_at_255": round(float((grey == 255).mean()), 5),
        })
        wholes.append(whole)
    whole_sheet(wholes).save(out / f"{name}_whole.png")

    torch.manual_seed(args.seed)  # the augmentation draws from torch's global RNG: the same crops on every run
    cropper = SSLImageDataset(chosen[:CROP_ROWS], aug, args.pre_resize, args.pre_resize)  # the trainer's dataset: that decode + rescale + squash, then the real augmentation
    crop_rows = []
    for tile in range(len(cropper)):
        crops = cropper[tile][0]
        crop_rows.append([(f"{tile} global {j + 1}", to_image(c)) for j, c in enumerate(crops["global_crops"])]
                         + [(f"{tile} local {j + 1}", to_image(c)) for j, c in enumerate(crops["local_crops"][:LOCAL_SHOWN])])
    crops_sheet(crop_rows).save(out / f"{name}_crops.png")

    # Real pixels per global crop. RandomResizedCrop(scale=(lo, hi)) cuts a region of area fraction s of the image it is given, here the pre_resize x pre_resize squash, and s is drawn
    # uniformly in (lo, hi). The region's geometric-mean side sqrt(w x h) is therefore  side(s) = sqrt(s) x pre_resize  pixels of the squashed image (exact at aspect ratio 1), and the
    # crop is resized to global_size, i.e. upsampled by  global_size / side(s)  (below 1: shrunk). Shown at the scale range's minimum, median ((lo + hi) / 2) and maximum.
    lo, hi = aug.geometric_augmentation_global.transforms[0].scale
    scales = (lo, (lo + hi) / 2, hi)
    sides = [math.sqrt(s) * args.pre_resize for s in scales]
    p99_minus_p1 = [r["p99_minus_p1"] for r in rows]
    print(f"{name}: n={k} | p99-p1 median {np.median(p99_minus_p1):.1f}, {sum(v < FLAT_BELOW for v in p99_minus_p1)} of {k} below {FLAT_BELOW} | "
          f"source median {np.median([r['src_width'] for r in rows]):g}x{np.median([r['src_height'] for r in rows]):g} | "
          f"global crop at scale {'/'.join(f'{s:g}' for s in scales)} = {'/'.join(f'{x:.0f}' for x in sides)} px of the {args.pre_resize}x{args.pre_resize} image, "
          f"upsampled to {args.global_size} by {'/'.join(f'x{args.global_size / x:.2f}' for x in sides)}", flush=True)
    return rows


def write_stats(out, rows):
    """Writes out/stats.csv so that it never exists half-written: the rows go to a hidden temporary file in the same directory (so the same filesystem), created mode 600 and named
    .stats.csv.<random>.tmp so it is not taken for a result; it is flushed and fsynced, then renamed onto stats.csv, an atomic replace. On any failure the temporary file is removed (if it can be) and
    the error propagates, so the run exits non-zero with nothing named stats.csv; a process killed outright can leave only the temporary file."""
    fd, tmp = tempfile.mkstemp(prefix=".stats.csv.", suffix=".tmp", dir=out)  # O_EXCL, mode 600
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out / "stats.csv")
    except BaseException:  # BaseException: an interrupt in the middle of the write must not leave the temporary file either
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def main(argv=None):
    args = parse_args(argv)
    out = check_out(args.out)
    args.datasets = list(dict.fromkeys(args.datasets))
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        fail(f"--config {config_path} is missing")
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}  # what base/utils.get_config does, without its sklearn / scipy imports
    overrides = {}
    for item in args.data_root_override:
        name, _, path = item.partition("=")
        if name not in args.datasets or not path:
            fail(f"--data_root_override {item!r}: expected NAME=PATH with NAME one of --datasets ({' '.join(args.datasets)})")
        overrides[name] = path
    files = {name: dataset_files(name, config, config_path, overrides) for name in args.datasets}  # every list and root is checked before the first image is decoded
    aug = DataAugmentationCXR(local_crops_number=LOCAL_CROPS_NUMBER, global_crops_size=args.global_size, local_crops_size=args.local_size,
                              gram_teacher_crops_size=args.global_size, emit_gram_teacher_crops=False, teacher_no_distortions=True)
    make_out(out)  # after every check above, so a run that fails them creates and changes nothing
    print(f"loader_audit: {' '.join(args.datasets)} | n={args.n} seed={args.seed} | pre_resize {args.pre_resize} | global crops {args.global_size} scale "
          f"{aug.geometric_augmentation_global.transforms[0].scale}, local {args.local_size} scale {aug.geometric_augmentation_local.transforms[0].scale} x{LOCAL_CROPS_NUMBER} | out {out}", flush=True)
    rows = []
    for name in args.datasets:
        rows += audit_dataset(name, *files[name], args, aug, out)
    write_stats(out, rows)  # last: after every dataset and every sheet
    print(f"loader_audit: {len(rows)} rows in {out / 'stats.csv'}, a whole and a crops sheet per dataset beside it", flush=True)


if __name__ == "__main__":
    main()
