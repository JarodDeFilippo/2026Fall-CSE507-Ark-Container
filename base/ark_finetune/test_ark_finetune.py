"""CPU checks for base/ark_finetune and base/ark_linearprobe, B2: the lab's embedding linear probe (Ark_Plus/Linearprobing/main_linearprobing.py) as shipped, run on a backbone's cached
embeddings, and run_emb_gaudi.sh.
Run from base/:  python -m ark_finetune.test_ark_finetune   (or python ark_finetune/test_ark_finetune.py)
(a) vendoring guard: every hunk that differs from the upstream files carries a '# ported:' marker, and every line it replaces its own; the Linearprobing utils.py is byte-identical; the Finetuning loop is gone
(b) the real list files of the four datasets parse with the vendored official classes (counts printed)
(c) full official train lists and no validation split: VinDrCXR's 15,000, no 90/10 files, no val_list anywhere
(d) the four dataset modules against the official classes: labels, embeddings, the alignment guard, smoke truncation, CheXpert's 5 scored findings
(e) the probe as shipped: defaults, float64, sigmoid + BCELoss, Adam, batch 512, shuffled train loader, the lr that RISES (1e-4 -> 0.00626 at epoch 50 -> 0.02409 at epoch 99), CPU only
(f) best epoch on TEST per run: the peak epoch is the one reported, ties keep the earlier, a run without a defined mAUC is refused
(g) the runner: 10 runs, seeds, the JSON and its cross-checks, flags, refusals and the preflight order, the JSON written whole or not at all
(h) the extractor on tiny synthetic images and a tiny ViT (train and test only, one center crop each), and the normalization presets
(i) extractor -> runner on the four official list formats
(i3) the Ark+ Swin teacher through the extractor: a lab-style checkpoint's TEACHER (not its student) is embedded before the projector, 1024-d for swin_base; a model that does not fit the file (embed_dim, window, input size),
     a file with no teacher or with a missing block weight, a model name the loader does not build, a ViT's file under a swin name, and a crop that is not the Swin's size are refused; a file with no attn_mask buffers
     loads and says its input size is unverified
(j) run_emb_gaudi.sh and its A100 sibling run_emb_class.sh in a fake repo, preflight, the allow-list of extractor arguments and the ARK_EMB_NO_FIT switch included, and the drift guard on the block the two share;
    the two-job probe: run_emb_fit_class.sh (the CPU-only fit: refusals, the resume rule, --check) and submit_probe_class.sh (the extraction and the fit as two sbatch lines behind a stub sbatch)
"""
import ast
import contextlib
import csv
import difflib
import errno
import hashlib
import io
import json
import os
import pickle
import random
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import warnings
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
REPO = os.path.dirname(BASE)
LP_HERE = os.path.join(BASE, "ark_linearprobe")
# base/ first, and the vendored utils.py / dataloader.py (next to this file and in ark_linearprobe/) never as top-level modules: they would shadow the container's
sys.path[:] = [BASE] + [p for p in sys.path if os.path.abspath(p or ".") not in (HERE, LP_HERE, BASE)]
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
from PIL import Image

import models
from accelerator import Accelerator
from models import ArkViT, ArkViTDinoV2, DINOV2_VITL16_KWARGS, DINOV3_VITB16_KWARGS
from ssl_dinov3.model import export_backbone
from utils import get_config
from ark_finetune import dataloader, dataset_lists, embedding_mode, extract_embeddings
from ark_linearprobe import embedding_datasets, main_linearprobing, run_linearprobe
from ark_linearprobe import utils as lp_utils

DEFAULT_THREADS = torch.get_num_threads()  # before the runner (g) pins torch to one thread for the rest of the run: (i3)'s Swin models take this back

UPSTREAM = os.environ.get("ARK_UPSTREAM")  # the Ark_Plus folder of a local checkout of the upstream repo (github.com/jlianglab/Ark); unset, the vendoring guard is skipped
VENDORED = ((os.path.join(UPSTREAM, "Finetuning"), HERE, ("dataloader.py",)), (os.path.join(UPSTREAM, "Linearprobing"), LP_HERE, ("main_linearprobing.py", "utils.py"))) if UPSTREAM else ()
NAMES = ("ChestXray14", "CheXpert", "MIMIC", "VinDrCXR")
real_config = get_config(os.path.join(BASE, "datasets_config.yaml"))
# the same ViT as the container's with ONE block of the real width: 2 * 768 = 1536-d features, and 32x32 inputs make it cheap on the CPU
TINY = dict(DINOV3_VITB16_KWARGS, img_size=32, depth=1)


def rejects(exception, call, expected=None):
    """The exception message when call() raises `exception`; fails when it returns."""
    try:
        call()
    except exception as error:
        assert expected is None or expected in str(error), (expected, str(error))
        return str(error)
    raise AssertionError("expected {}".format(exception.__name__))


def quiet(call):
    """(result, stdout, stderr) of call() with both streams captured."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        result = call()
    return result, out.getvalue(), err.getvalue()


# (a) the vendoring guard ----------------------------------------------------------------------------------------------
def unmarked_hunks(upstream, vendored):
    """(number of hunks, the problem hunks) of the diff of two line lists, hunks being the groups of changed lines with no unchanged line between them. A hunk is a problem when it has no
    '# ported:' marker, or when it replaces or deletes upstream lines (it has removed lines) and one of its added lines lacks the marker itself: an unmarked edit next to a marked
    addition must not hide in the marked hunk. (autojunk is off: difflib otherwise treats blank lines as noise, and edits separated only by a blank line would run into one hunk.)"""
    problems = []
    groups = [g for g in difflib.SequenceMatcher(None, upstream, vendored, autojunk=False).get_grouped_opcodes(0) if any(tag != "equal" for tag, *_ in g)]  # (identical files give one 'equal' group)
    for group in groups:
        removed = [line for tag, i1, i2, j1, j2 in group if tag in ("replace", "delete") for line in upstream[i1:i2]]
        added = [line for tag, i1, i2, j1, j2 in group if tag in ("replace", "insert") for line in vendored[j1:j2]]
        marked = [line for line in added if "# ported:" in line]
        if not marked or (removed and len(marked) != len(added)):
            problems.append("\n".join(["-" + line for line in removed] + ["+" + line for line in added]))
    return len(groups), problems


# the guard itself: an unmarked change (an edit, a deletion, an addition) is found, a marked one passes
original = ["a = 1", "b = 2", "c = 3", "d = 4"]
assert unmarked_hunks(original, original) == (0, [])
assert unmarked_hunks(original, ["a = 1", "b = 2  # ported: why", "c = 3", "d = 4"])[1] == []
assert unmarked_hunks(original, ["a = 1", "# ported: why", "c = 3", "d = 4"])[1] == []  # a deletion left a marked comment behind
assert unmarked_hunks(original, ["a = 1", "b = 22", "c = 3", "d = 4"])[0] == 1 and len(unmarked_hunks(original, ["a = 1", "b = 22", "c = 3", "d = 4"])[1]) == 1
assert len(unmarked_hunks(original, ["a = 1", "c = 3", "d = 4"])[1]) == 1  # an unmarked deletion
assert len(unmarked_hunks(original, original + ["e = 5"])[1]) == 1  # an unmarked addition
two_hunks = unmarked_hunks(original, ["a = 11", "b = 2", "c = 3", "d = 4  # ported: why"])  # two separate hunks, only the second marked
assert two_hunks[0] == 2 and len(two_hunks[1]) == 1 and "a = 11" in two_hunks[1][0]
assert unmarked_hunks(original, original[:2] + ["# ported: why", "x = 1", "y = 2"] + original[2:])[1] == []  # new lines inserted between upstream lines: one marker for the block
assert len(unmarked_hunks(original, ["a = 1", "b = 2", "x = 1  # ported: why", "c = 33", "d = 4"])[1]) == 1  # an unmarked edit right next to a marked addition is still caught
assert unmarked_hunks(original, ["a = 1", "b = 2", "x = 1  # ported: why", "c = 33  # ported: why", "d = 4"])[1] == []  # ... and passes once it carries its own marker
blank = ["a = 1", "", "b = 2"]
assert len(unmarked_hunks(blank, ["a = 11  # ported: why", "", "b = 22"])[1]) == 1 and unmarked_hunks(blank, ["a = 11  # ported: why", "", "b = 22  # ported: why"]) == (2, [])  # a blank line keeps two edits apart

for gone in ("main_classification.py", "engine.py", "trainer.py", "utils.py", "run_probe.py"):  # the Finetuning loop and its runner are not vendored any more: nothing uses them
    assert not os.path.exists(os.path.join(HERE, gone)), gone
if not UPSTREAM or not os.path.isdir(UPSTREAM):
    print("SKIP (a) vendoring guard against the upstream files: {}; set ARK_UPSTREAM to the Ark_Plus folder of a local Ark checkout (github.com/jlianglab/Ark)"
          .format("ARK_UPSTREAM is not set" if not UPSTREAM else "{} is not a directory".format(UPSTREAM)))
else:
    summary = {}
    for upstream_dir, local_dir, files in VENDORED:
        for name in files:
            with open(os.path.join(upstream_dir, name)) as f:
                upstream = f.read().splitlines()
            with open(os.path.join(local_dir, name)) as f:
                vendored = f.read().splitlines()
            count, unmarked = unmarked_hunks(upstream, vendored)
            assert not unmarked, "{}: hunks without a '# ported:' marker (or replacing a line without a marker of its own):\n{}".format(name, "\n---\n".join(unmarked))
            summary[os.path.join(os.path.basename(local_dir), name)] = count
    assert summary["ark_linearprobe/utils.py"] == 0, summary  # nothing in the Linearprobing utils needed a port
    assert summary["ark_linearprobe/main_linearprobing.py"] > 0 and summary["ark_finetune/dataloader.py"] > 0, summary
    with open(os.path.join(UPSTREAM, "Linearprobing", "utils.py"), "rb") as a, open(os.path.join(LP_HERE, "utils.py"), "rb") as b:
        assert a.read() == b.read()
    print("OK (a) vendoring guard: every changed hunk against {} carries '# ported:', and every replaced line its own ({}); ark_linearprobe/utils.py is byte-identical; the guard finds unmarked edits, "
          "deletions and additions, also next to a marked addition; "
          "the vendored Finetuning loop (engine, trainer, main_classification, utils) is gone".format(UPSTREAM, ", ".join("{} {}".format(n, c) for n, c in summary.items())))

# (b) the real list files, parsed by the vendored official classes -------------------------------------------------------
EXPECTED = {"ChestXray14": (75312, 25596), "CheXpert": (223414, 668), "MIMIC": (368960, 5159), "VinDrCXR": (15000, 3000)}
counts = {}
for name in NAMES:
    entry = dataset_lists.dataset_entry(name)
    counts[name] = []
    for split in ("train", "test"):
        extra = dict(uncertain_label="Ones") if (name, split) == ("MIMIC", "test") else {}  # as the official main_classification builds the MIMIC test set
        dataset = getattr(dataloader, name)(images_path="/nonexistent", file_path=entry[split + "_list"], augment=None, **extra)  # parses the list; no image is read
        labels = np.array(dataset.img_label, dtype=np.float64)
        assert labels.shape == (len(dataset), len(entry["diseases"])), (name, split, labels.shape)
        assert all(os.path.isabs(p) for p in dataset.img_list)
        counts[name].append(len(dataset))
        if name in ("ChestXray14", "VinDrCXR") or (name, split) in (("CheXpert", "test"), ("MIMIC", "test")):
            assert np.isin(labels, (0, 1)).all(), (name, split)  # the hard-label splits (MIMIC's test set is built with uncertain_label="Ones")
        elif name == "MIMIC":  # drawn once, in the official constructor: LSR-Ones turns each -1 into a soft label in [0.55, 0.85]
            soft = labels[~np.isin(labels, (0, 1))]
            assert len(soft) > 0 and 0.55 <= soft.min() and soft.max() <= 0.85, (name, split)
        else:  # CheXpert train keeps -1 and draws at every access
            assert np.isin(labels, (-1, 0, 1)).all() and (labels == -1).any()
    assert tuple(counts[name]) == EXPECTED[name], (name, counts[name])
with open(dataset_lists.dataset_entry("MIMIC")["train_list"], newline="") as f:
    header = next(csv.reader(f))
assert header[5:19] == real_config["MIMIC"]["diseases"] == real_config["MIMIC"]["test_diseases_name"]  # the config names the csv columns, in the csv's order
print("OK (b) real lists parse with the vendored classes (images not read): " + "; ".join("{} {} / {}".format(n, *counts[n]) for n in NAMES) + " (train / test)")

# (c) the full official train lists, no validation split ------------------------------------------------------------------
vindr_dir = os.path.join(BASE, "dataset", "VinDrCXR")
assert sorted(f for f in os.listdir(vindr_dir) if f.endswith(".txt")) == ["VinDrCXR_test_pe_global_one.txt", "VinDrCXR_train_pe_global_one.txt"]  # no 90/10 lists
assert not hasattr(dataset_lists, "split_vindr") and not hasattr(dataset_lists, "VINDR_LISTS")
for name in NAMES:
    entry = dataset_lists.dataset_entry(name)
    assert "val_list" not in entry, name  # nothing can read a validation list by accident
    assert entry["train_list"] == os.path.normpath(os.path.join(BASE, real_config[name]["train_list"])) and entry["test_list"] == os.path.normpath(os.path.join(BASE, real_config[name]["test_list"]))
vindr = dataset_lists.dataset_entry("VinDrCXR")
assert vindr["train_list"] == os.path.join(vindr_dir, "VinDrCXR_train_pe_global_one.txt") and vindr["test_list"] == os.path.join(vindr_dir, "VinDrCXR_test_pe_global_one.txt")
train_ids = [l.split()[0] for l in open(vindr["train_list"], "rb").read().splitlines()]
test_ids = [l.split()[0] for l in open(vindr["test_list"], "rb").read().splitlines()]
assert (len(train_ids), len(set(train_ids)), len(test_ids)) == (15000, 15000, 3000)  # all 15,000 train images, each once
assert {i.replace(b"train_jpeg/", b"") for i in train_ids}.isdisjoint({i.replace(b"test_jpeg/", b"") for i in test_ids})
assert real_config["VinDrCXR"]["val_list"] == real_config["VinDrCXR"]["test_list"]  # datasets_config.yaml is untouched: the other arms read it as it is
print("OK (c) splits: every dataset trains on its full official train list (VinDrCXR all 15000, the 6-label global list) and is scored on its official test list; no val_list in dataset_entry, "
      "no 90/10 files, datasets_config.yaml untouched")


# synthetic datasets in the four official list formats -----------------------------------------------------------------------
def hard_labels(n, classes, shift=0):
    """0/1 labels with both values in every column: class c is positive for the rows i with (i + c + shift) % 3 == 0."""
    return [[1 if (i + c + shift) % 3 == 0 else 0 for c in range(classes)] for i in range(n)]


def make_dataset(root, name, sizes, images=False):
    """Lists in the official format (and, with images=True, tiny PNGs at the paths they name) under root/<name>; returns dataset_lists.dataset_entry's dict for it.
    Uncertain (-1) and blank cells are in the CheXpert train csv, and in every MIMIC csv (its test set reads -1 as 1: uncertain_label="Ones")."""
    diseases = real_config[name]["diseases"]
    folder = os.path.join(root, name)
    entry = dict(real_config[name], data_dir=os.path.join(folder, "images"))
    entry.pop("val_list")
    n_classes = len(diseases)

    def image(relative, seed):
        if images:
            os.makedirs(os.path.dirname(os.path.join(entry["data_dir"], relative)), exist_ok=True)
            pixels = np.random.RandomState(seed).randint(0, 256, (40, 40), dtype=np.uint8)
            Image.fromarray(pixels).save(os.path.join(entry["data_dir"], relative), format="PNG")  # also under a .jpeg name: PIL goes by content

    os.makedirs(folder, exist_ok=True)
    for k, (split, n) in enumerate(sizes.items()):
        labels = hard_labels(n, n_classes, shift=k)
        path = os.path.join(folder, split + (".csv" if name in ("CheXpert", "MIMIC") else ".txt"))
        rows = []
        for i in range(n):
            if name == "ChestXray14":
                relative = "{}_{:03d}.png".format(split, i)
                rows.append(relative + " " + " ".join(map(str, labels[i])))
            elif name == "VinDrCXR":
                relative = "{}_jpeg/{}{:03d}.jpeg".format("test" if split == "test" else "train", split, i)
                rows.append(relative[:-len(".jpeg")] + " " + " ".join(map(str, labels[i])) + " \r")  # CRLF and a trailing space, as the real lists
            else:
                cells = [str(float(v)) for v in labels[i]]
                if name == "MIMIC" and i == 1:
                    cells[0] = "-1.0"  # row 1, column 0 is negative otherwise: uncertain
                    cells[1] = ""  # a blank cell: unknown_label
                if name == "CheXpert" and split == "train" and i in (1, 4):
                    cells[0], cells[2] = "-1.0", "-1.0"
                if name == "CheXpert":
                    relative = "CheXpert-v1.0/{}/patient{:05d}/view1_frontal.png".format(split, i)
                    rows.append(",".join([relative] + ([] if split == "test" else ["Female", "68", "Frontal", "AP"]) + cells))
                else:
                    relative = "files/p10/{}_{:03d}.png".format(split, i)
                    rows.append(",".join([relative, "dicom{}".format(i), "5{}".format(i), "1000{}".format(i), split] + cells))
            image(relative, 1000 * k + i)
        with open(path, "w", newline="") as f:
            if name == "CheXpert":
                f.write(",".join((["Path"] if split == "test" else ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"]) + diseases) + "\n")
            if name == "MIMIC":
                f.write(",".join(["image_path", "dicom_id", "study_id", "subject_id", "split"] + diseases) + "\n")
            f.write("\n".join(rows) + "\n")
        entry[split + "_list"] = path
    return entry


def official_dataset(name, entry, split, **kwargs):
    if (name, split) == ("MIMIC", "test"):
        kwargs.setdefault("uncertain_label", "Ones")  # as the official branch builds it
    return getattr(dataloader, name)(images_path=entry["data_dir"], file_path=entry[split + "_list"], augment=None, **kwargs)


WEIGHTS_SHA256 = hashlib.sha256(b"the stock DINOv3 ViT-B/16 file").hexdigest()
WEIGHTS_META = {"path": "/workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", "size_bytes": 342852041, "sha256": WEIGHTS_SHA256}
TRANSFORM_META = {"resize": 256, "crop": 224, "pipelines": {"train": ["Resize", "CenterCrop", "ToTensor", "Normalize"], "test": ["Resize", "CenterCrop", "ToTensor", "Normalize"]}}


def write_cache(folder, name, entry, seed=0, max_images=None, signal=False, model="vit_base_dinov3", normalization="imagenet", tag="t", weights=WEIGHTS_META):
    """A cache in the layout extract_embeddings.py writes, from random float16 embeddings (with signal=True, class c shifts coordinate c by 3 where it is positive).
    Returns {split: float16 array}."""
    os.makedirs(folder)
    rng = np.random.RandomState(seed)
    arrays, counts = {}, {}
    for split in embedding_mode.SPLITS:
        dataset = official_dataset(name, entry, split)
        keep = len(dataset) if max_images is None else min(max_images, len(dataset))
        x = rng.randn(keep, 1536).astype(np.float32)
        if signal:
            y = (np.array(dataset.img_label[:keep], dtype=np.float64) != 0).astype(np.float32)
            x[:, :y.shape[1]] += 3 * y
        arrays[split], counts[split] = x.astype(np.float16), keep
        npy, paths = embedding_mode.split_files(folder, split)
        np.save(npy, arrays[split])
        embedding_mode.write_paths(paths, dataset.img_list[:keep])
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump({"tag": tag, "dataset": name, "model": model, "weights": weights, "normalization": normalization, "transform": TRANSFORM_META, "counts": counts, "max_images": max_images,
                   "feature_dim": 1536}, f)
    return arrays


def probe_argv(name, entry, cache, results_dir, *extra):
    """The options the runner hands the shipped script's parser for one dataset (CPU, no workers)."""
    return ["--dataset", name, "--num_class", str(len(entry["diseases"])), "--emb_dim", "1536", "--emb_path", cache, "--train_list", entry["train_list"], "--test_list", entry["test_list"],
            "--data_dir", entry["data_dir"], "--results_dir", results_dir, "--arch", "t", "--exp_name", "b2", "--num_workers", "0", "--device", "cpu"] + list(extra)


def run_script(argv):
    """The shipped main() on the options `argv`: (its returned records, stdout)."""
    records, out, _ = quiet(lambda: main_linearprobing.main(main_linearprobing.get_args_parser(argv)))
    return records, out


# (d) the four dataset modules against the official classes -------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    sizes = {"train": 9, "test": 6}
    data = {name: make_dataset(tmp, name, sizes, images=True) for name in NAMES}
    for name in NAMES:
        entry = data[name]
        cache = os.path.join(tmp, "cache", name)
        arrays = write_cache(cache, name, entry)
        Module = embedding_datasets.EMBEDDING_MODULES[name]
        assert Module.NAME == name and issubclass(Module, embedding_datasets.EmbeddingDataloaderModule)
        random.seed(11)
        module, out, _ = quiet(lambda: Module(cache, entry["train_list"], entry["test_list"], entry["data_dir"], batch_size=4, num_workers=0))
        assert out == "#train:  9\n#test:   6\n", out  # the shipped modules' prints
        random.seed(11)  # MIMIC draws its soft labels when the dataset is built: the same state gives the same draws
        theirs = {"train": official_dataset(name, entry, "train"), "test": official_dataset(name, entry, "test")}
        for split in ("train", "test"):
            ours = getattr(module, split + "_set")
            assert isinstance(ours, embedding_datasets.InMemoryEmbeddingDataset) and len(ours) == len(theirs[split]) == sizes[split]
            assert ours.parsed.img_list == theirs[split].img_list and ours.parsed.img_label == theirs[split].img_label
            for i in range(len(ours)):
                random.seed(100 + i)
                sample, label = ours[i]
                random.seed(100 + i)
                image, expected = theirs[split][i]  # the official __getitem__: opens the image, draws CheXpert's soft labels
                assert isinstance(image, Image.Image) and label.dtype == expected.dtype == torch.float32 and torch.equal(label, expected), (name, split, i, label, expected)
                assert sample.dtype == torch.float16 and sample.shape == (1536,) and torch.equal(sample, torch.from_numpy(arrays[split][i]))  # the cached row, as it is
        # the shipped DataLoader settings: batch from the option, train shuffled, test in list order; a batch is (float16 embeddings, float32 labels)
        train_loader, test_loader = module.train_dataloader(), module.test_dataloader()
        assert (train_loader.batch_size, test_loader.batch_size, train_loader.num_workers) == (4, 4, 0)
        assert type(train_loader.sampler).__name__ == "RandomSampler" and type(test_loader.sampler).__name__ == "SequentialSampler"
        x, y = next(iter(train_loader))
        assert x.dtype == torch.float16 and x.shape == (4, 1536) and y.dtype == torch.float32 and y.shape == (4, len(real_config[name]["diseases"]))
        assert torch.equal(torch.cat([x for x, _ in test_loader]), torch.from_numpy(arrays["test"]))
        # CheXpert is scored on its 5 competition findings (the Finetuning branch's test_diseases); the others on every column
        diseases = real_config["CheXpert"]["diseases"]
        assert module.test_columns == ([diseases.index(c) for c in ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]] if name == "CheXpert" else None), (name, module.test_columns)
        assert module.dropped_columns == []  # every scored column has a positive and a negative test example here
    assert embedding_datasets.scored_columns("CheXpert") == [8, 2, 6, 5, 10] and embedding_datasets.scored_columns("MIMIC") is None  # MIMIC names all 14, in column order
    # CheXpert: uncertain labels are drawn at every access, in the official ranges, whatever uncertain_label says: the copied label block equals the official __getitem__ in all four modes
    for mode, low, high in (("LSR-Ones", 0.55, 0.85), ("LSR-Zeros", 0.0, 0.3), ("Ones", 1, 1), ("Zeros", 0, 0)):
        theirs = official_dataset("CheXpert", data["CheXpert"], "train", uncertain_label=mode)
        seen = set()
        for i in range(len(theirs)):
            for seed in (1, 2):
                random.seed(seed)
                a = embedding_mode.chexpert_label(theirs, i)
                random.seed(seed)
                assert torch.equal(a, theirs[i][1]), (mode, i)
                if i in (1, 4):
                    assert all(low <= a[c] <= high for c in (0, 2)), (mode, a)
                    seen.add(float(a[0]))
        assert len(seen) == (2 if mode.startswith("LSR") else 1), (mode, seen)  # a fresh draw per access for LSR, a constant otherwise
    print("OK (d1) dataset modules: for ChestXray14, CheXpert, MIMIC and VinDrCXR, train / test, img_list, img_label and every label equal the official class's (CheXpert's soft labels under the same "
          "random seed, in all four uncertain_label modes; MIMIC's draws at construction, test with uncertain_label='Ones'); a sample is the cached float16 row; train loader shuffled, batch from the "
          "option; CheXpert scores columns [8, 2, 6, 5, 10] (its 5 findings), the others all of them")

    # the alignment guard and the other refusals
    entry, cache = data["ChestXray14"], os.path.join(tmp, "cache", "ChestXray14")
    Chest = embedding_datasets.ChestXray14_DataloaderModule
    build = lambda e=entry, c=cache, **kw: quiet(lambda: Chest(c, e["train_list"], e["test_list"], e["data_dir"], batch_size=4, num_workers=0, **kw))[0]
    assert len(build().train_set) == 9
    for split in ("train", "test"):
        paths_file = embedding_mode.split_files(cache, split)[1]
        original_paths = open(paths_file).read()
        lines = original_paths.splitlines()
        try:
            for label, broken in (("two entries swapped", [lines[1], lines[0]] + lines[2:]), ("one entry short", lines[:-1]), ("one entry too many", lines + [lines[0]]),
                                  ("another data_dir", [l.replace(entry["data_dir"], "/other") for l in lines]), ("empty", [])):
                with open(paths_file, "w") as f:
                    f.write("".join(l + "\n" for l in broken))
                message = rejects(ValueError, build, "the cached image paths are not the dataset's img_list")
                assert "ChestXray14 {}".format(split) in message and "first difference at index" in message, (split, label, message)
        finally:
            with open(paths_file, "w") as f:
                f.write(original_paths)
    assert len(build().train_set) == 9
    other_test = os.path.join(tmp, "other_test.txt")  # --test_list names another file, with the train list's content: the test cache was not extracted from it
    open(other_test, "w").write(open(entry["train_list"]).read())
    rejects(ValueError, lambda: build(e=dict(entry, test_list=other_test)), "ChestXray14 test: the cached image paths are not the dataset's img_list")
    rejects(ValueError, lambda: build(e=dict(entry, data_dir="/elsewhere")), "ChestXray14 train: the cached image paths are not the dataset's img_list")  # relocated images
    rejects(ValueError, lambda: build(few_shot=0.5), "--few_shot")
    rejects(FileNotFoundError, lambda: Chest(os.path.join(tmp, "nowhere"), entry["train_list"], entry["test_list"], entry["data_dir"]))
    npy = embedding_mode.split_files(cache, "train")[0]
    cached_rows = np.load(npy)
    try:
        for label, bad, expected in (("float32", cached_rows.astype(np.float32), "float32"), ("narrow", cached_rows[:, :1000], "[9, 1000]"), ("short", cached_rows[:-1], "[8, 1536]")):
            np.save(npy, bad)
            rejects(ValueError, build, expected)
    finally:
        np.save(npy, cached_rows)
    assert len(build().train_set) == 9
    # a smoke cache holds the first N entries of each list: the module keeps those, and the guard still applies when the meta does not say so
    smoke_cache = os.path.join(tmp, "smoke")
    write_cache(smoke_cache, "ChestXray14", entry, max_images=4)
    kept = build(c=smoke_cache)
    for split in ("train", "test"):
        full = official_dataset("ChestXray14", entry, split)
        part = getattr(kept, split + "_set")
        assert len(part) == 4 and part.parsed.img_list == full.img_list[:4] and part.parsed.img_label == full.img_label[:4] and len(part.embeddings) == 4
    meta_path = os.path.join(smoke_cache, "meta.json")
    meta = json.load(open(meta_path))
    json.dump(dict(meta, max_images=None), open(meta_path, "w"))  # the cache has 4 rows, the meta says it is whole
    rejects(ValueError, lambda: build(c=smoke_cache), "the cached image paths are not the dataset's img_list")
    print("OK (d2) alignment guard: swapped, missing, extra and relocated paths in either split, another split's list and an empty cache raise ValueError naming the dataset, split and first "
          "differing index; --few_shot, a missing cache and a cache that is not float16 [N, 1536] are refused; a max_images=4 cache serves the first 4 entries of each list")


# (d3) scoring needs both classes: single-valued test columns are dropped, recorded, and a dataset with nothing to score is refused ---------------------------
def force_columns(path, name, values):
    """Rewrite the test list at `path` so that label column c holds values[c] in every row (ChestXray14: 'image l1 ... lC'; CheXpert's test csv: Path, then the label cells)."""
    if name == "ChestXray14":
        rows = [line.split() for line in open(path).read().splitlines()]
        for row in rows:
            for c, value in values.items():
                row[1 + c] = str(value)
        open(path, "w").write("".join(" ".join(row) + "\n" for row in rows))
    else:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        for row in rows[1:]:
            for c, value in values.items():
                row[1 + c] = "{:.1f}".format(value)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(rows)


with tempfile.TemporaryDirectory() as tmp:
    diseases = real_config["ChestXray14"]["diseases"]
    entry = make_dataset(tmp, "ChestXray14", {"train": 9, "test": 8})
    force_columns(entry["test_list"], "ChestXray14", {3: 1, 5: 0})  # column 3 all positive, column 5 all negative in the evaluated test set
    cache = os.path.join(tmp, "cache_chest")
    write_cache(cache, "ChestXray14", entry)
    module = quiet(lambda: embedding_datasets.ChestXray14_DataloaderModule(cache, entry["train_list"], entry["test_list"], entry["data_dir"], batch_size=4, num_workers=0))[0]
    assert module.test_columns == [c for c in range(14) if c not in (3, 5)]
    assert module.dropped_columns == [{"column": 3, "reason": "all positive"}, {"column": 5, "reason": "all negative"}], module.dropped_columns
    # the shipped test() on all 14 columns: sklearn gives the all-positive column no AUC (a NaN with a warning, or a ValueError in older versions), so the run's mAUC is not a number and no
    # epoch ever beats the initial best; with the module's columns the shipped script scores the other 12
    labels = torch.tensor(np.array(module.test_set.parsed.img_label), dtype=torch.float32)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (all_mAUC, *_), (all_aucs, *_) = main_linearprobing.test(main_linearprobing.LinearClassifier(1536, 14).double(), [(torch.randn(8, 1536).half(), labels)], io.StringIO())
        assert np.isnan(all_mAUC) and np.isnan(all_aucs).sum() == 1
    except ValueError:
        pass
    records, _ = run_script(probe_argv("ChestXray14", entry, cache, os.path.join(tmp, "res_chest"), "--epochs", "3"))
    assert len(records[0]["scores"][0]) == 12 and records[0]["best_epoch"] >= 0
    # CheXpert: the official 5 findings first, then the single-valued ones among them go (Edema, column 5); a single-valued column that is not one of the 5 is not the probe's business
    entry = make_dataset(tmp, "CheXpert", {"train": 9, "test": 8})
    force_columns(entry["test_list"], "CheXpert", {5: 1, 0: 1})  # Edema all positive, No Finding (not scored) all positive
    cache = os.path.join(tmp, "cache_chexpert")
    write_cache(cache, "CheXpert", entry)
    module = quiet(lambda: embedding_datasets.CheXpert_DataloaderModule(cache, entry["train_list"], entry["test_list"], entry["data_dir"], batch_size=4, num_workers=0))[0]
    assert module.test_columns == [8, 2, 6, 10] and module.dropped_columns == [{"column": 5, "reason": "all positive"}], (module.test_columns, module.dropped_columns)
    # nothing left to score: refused, by the rule and by the module
    one_row = make_dataset(tmp, "ChestXray14", {"train": 9, "test": 1})  # one test image: every column is single-valued
    cache = os.path.join(tmp, "cache_one")
    write_cache(cache, "ChestXray14", one_row)
    rejects(ValueError, lambda: quiet(lambda: embedding_datasets.ChestXray14_DataloaderModule(cache, one_row["train_list"], one_row["test_list"], one_row["data_dir"], num_workers=0)),
            "ChestXray14: none of the 14 columns it is scored on has both a positive and a negative example among the 1 test images: there is nothing to score")
    rejects(ValueError, lambda: embedding_datasets.evaluable_columns("VinDrCXR", np.zeros((3, 6))), "none of the 6 columns")
    rejects(ValueError, lambda: embedding_datasets.evaluable_columns("MIMIC", np.ones((3, 14))), "column 0 all positive")
    rejects(ValueError, lambda: embedding_datasets.evaluable_columns("ChestXray14", np.zeros((0, 14))), "no test images")
    assert embedding_datasets.evaluable_columns("VinDrCXR", np.array([[0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0]])) == ([0, 1, 4, 5], [{"column": 2, "reason": "all negative"}, {"column": 3, "reason": "all positive"}])
    assert embedding_datasets.evaluable_columns("ChestXray14", np.array([[0] * 14, [1] * 14])) == (None, [])  # everything scoreable: the shipped path, columns=None
    print("OK (d3) scoring needs both classes: of the columns a dataset is scored on (CheXpert: its 5 findings first), those that are all positive or all negative among the evaluated test images are dropped "
          "and recorded (the shipped test() gives the all-positive one a NaN AUC and so a NaN mAUC), the rest are scored; a dataset with no column left, or no test image, is refused with a message before anything is fitted")


# (e) the probe as shipped ---------------------------------------------------------------------------------------------------
def random_dataset(tmp, n_train, n_test, shift, scale, seed, classes=14):
    """A ChestXray14-format dataset with random labels (35 % positive) and a cache whose first 40 coordinates are noise, class c shifted by `shift` on coordinate c where it is positive, all
    times `scale`, the other 1496 coordinates zero. Returns (entry, cache)."""
    rng = np.random.RandomState(seed)
    folder = os.path.join(tmp, "random_{}".format(len(os.listdir(tmp))))
    os.makedirs(folder)
    entry = {"data_dir": os.path.join(folder, "images"), "diseases": real_config["ChestXray14"]["diseases"][:classes]}
    cache = os.path.join(folder, "cache")
    os.makedirs(cache)
    for split, n in (("train", n_train), ("test", n_test)):
        y = (rng.rand(n, classes) < 0.35).astype(int)
        names = ["{}_{:04d}.png".format(split, i) for i in range(n)]
        with open(os.path.join(folder, split + ".txt"), "w") as f:
            f.write("".join(name + " " + " ".join(map(str, row)) + "\n" for name, row in zip(names, y)))
        entry[split + "_list"] = os.path.join(folder, split + ".txt")
        x = np.zeros((n, 1536), np.float32)
        x[:, :40] = rng.randn(n, 40)
        x[:, :classes] += shift * y
        npy, paths = embedding_mode.split_files(cache, split)
        np.save(npy, (x * scale).astype(np.float16))
        embedding_mode.write_paths(paths, [os.path.join(entry["data_dir"], name) for name in names])
    with open(os.path.join(cache, "meta.json"), "w") as f:
        json.dump({"tag": "t", "dataset": "ChestXray14", "model": "vit_base_dinov3", "normalization": "imagenet", "counts": {"train": n_train, "test": n_test}, "max_images": None}, f)
    return entry, cache


defaults = main_linearprobing.get_args_parser([])
assert (defaults.batch_size, defaults.lr, defaults.epochs, defaults.optimizer, defaults.runs, defaults.num_workers, defaults.patience, defaults.mode, defaults.multiclass, defaults.gpu) \
    == (512, 1e-4, 100, "adam", 1, 4, 10, "train", False, "0,1,2,3")  # the shipped defaults, untouched
assert (defaults.device, defaults.results_dir, defaults.seed, defaults.train_list, defaults.test_list, defaults.data_dir) == ("cpu", "/results", 0, None, None, None)  # the ported options
assert all(getattr(main_linearprobing, f) is getattr(lp_utils, f) for f in ("meanAUC", "meanMCC", "meanAP", "meanF1", "AverageMeter", "save_model"))  # the metrics are the script's own utils.py

with tempfile.TemporaryDirectory() as tmp:
    entry, cache = random_dataset(tmp, 200, 400, shift=1.5, scale=10.0, seed=0)  # a signal the shipped recipe can learn
    seen, epochs_mAUC = [], []
    stock_train, stock_test = main_linearprobing.train_one_epoch, main_linearprobing.test

    def spy_train(model, criterion, optimizer, scheduler, train_loader, epoch, multiclass, log_writter):
        seen.append(dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"], model=type(model).__name__, layers=[type(m).__name__ for m in model.model], dtype=next(model.parameters()).dtype,
                         optimizer=type(optimizer).__name__, defaults_lr=optimizer.defaults["lr"], criterion=type(criterion).__name__, scheduler=type(scheduler).__name__,
                         T_max=scheduler.T_max, eta_min=scheduler.eta_min, batch=train_loader.batch_size, sampler=type(train_loader.sampler).__name__, workers=train_loader.num_workers,
                         multiclass=multiclass, weight_shape=tuple(model.model[0].weight.shape)))
        return stock_train(model, criterion, optimizer, scheduler, train_loader, epoch, multiclass, log_writter)

    def spy_test(model, test_loader, log_writter, columns=None):
        result = stock_test(model, test_loader, log_writter, columns)
        epochs_mAUC.append(result[0][0])
        assert next(model.parameters()).dtype == torch.float64 and not model.training  # evaluated in eval mode, in float64
        return result

    main_linearprobing.train_one_epoch, main_linearprobing.test = spy_train, spy_test
    try:
        records, out = run_script(probe_argv("ChestXray14", entry, cache, os.path.join(tmp, "res")))
    finally:
        main_linearprobing.train_one_epoch, main_linearprobing.test = stock_train, stock_test
    assert [s["epoch"] for s in seen] == list(range(100)) and len(epochs_mAUC) == 100  # 100 epochs, the shipped default
    first = seen[0]
    assert (first["model"], first["layers"], first["dtype"], first["optimizer"], first["defaults_lr"], first["criterion"], first["batch"], first["sampler"], first["multiclass"],
            first["weight_shape"]) == ("LinearClassifier", ["Linear"], torch.float64, "Adam", 1e-4, "BCELoss", 512, "RandomSampler", False, (14, 1536))
    assert all(s["dtype"] == torch.float64 and s["scheduler"] == "CosineAnnealingLR" and (s["T_max"], s["eta_min"]) == (1000, 1) and s["workers"] == 0 for s in seen)
    # the schedule: CosineAnnealingLR(T_max=1000, eta_min=1) stepped once per epoch, from a base lr of 1e-4 BELOW eta_min: the lr RISES (the shipped script's behavior, pinned)
    lrs = [s["lr"] for s in seen]
    closed = lambda t: 1 + (1e-4 - 1) * (1 + np.cos(np.pi * t / 1000)) / 2
    assert lrs[0] == 1e-4 and abs(lrs[50] - 0.00625521) < 1e-8 and abs(lrs[99] - 0.02408629) < 1e-8, (lrs[0], lrs[50], lrs[99])  # 1e-4 -> 0.00626 -> 0.02409
    assert all(b > a for a, b in zip(lrs, lrs[1:])) and max(abs(lrs[t] - closed(t)) for t in range(100)) < 1e-12
    with open(os.path.join(tmp, "res", "log.txt")) as f:  # and it is what the script logged at the start of every epoch
        logged = re.findall(r"^learning_rate: (\S+),(\d+)$", f.read(), re.M)
    assert [(float(lr), int(e)) for lr, e in logged] == list(zip(lrs, range(100)))
    # the recipe learns: chance level at epoch 0, a real signal at the end; the best epoch is the one with the best test mAUC
    assert epochs_mAUC[0] < 0.55 and max(epochs_mAUC) > 0.7 and records[0]["performance"][0] == max(epochs_mAUC) == epochs_mAUC[records[0]["best_epoch"]]
    print("OK (e1) the probe as shipped: defaults (batch 512, lr 1e-4, 100 epochs, adam, 4 workers, patience 10), a float64 nn.Linear inside LinearClassifier, Adam, BCELoss, a shuffled train loader of 512, "
          "CosineAnnealingLR(T_max=1000, eta_min=1) stepped per epoch: lr {:.6g} at epoch 0, {:.8f} at 50, {:.8f} at 99, rising every epoch, matching its closed form to 1e-12 and the script's own log; "
          "test mAUC {:.3f} at epoch 0 -> {:.3f} at epoch {}".format(lrs[0], lrs[50], lrs[99], epochs_mAUC[0], max(epochs_mAUC), records[0]["best_epoch"]))

    # train_one_epoch and test on their own: float64 sigmoid + BCELoss on float16 embeddings and float32 labels, and the CheXpert columns
    class SpyLoss(torch.nn.BCELoss):
        calls = []

        def forward(self, outputs, target):
            SpyLoss.calls.append((outputs.detach().clone(), target.detach().clone()))
            return super().forward(outputs, target)

    torch.manual_seed(0)
    model = main_linearprobing.LinearClassifier(1536, 3).double()
    reference = main_linearprobing.LinearClassifier(1536, 3).double()
    reference.load_state_dict(model.state_dict())
    x16, labels = torch.randn(7, 1536).half(), torch.rand(7, 3)
    log = io.StringIO()
    loss_avg = main_linearprobing.train_one_epoch(model, SpyLoss(), torch.optim.Adam(model.parameters(), lr=1e-4), None, [(x16, labels)], 0, False, log)
    (outputs, target), = SpyLoss.calls
    assert outputs.dtype == target.dtype == torch.float64 and torch.equal(outputs, torch.sigmoid(reference(x16.double()))) and torch.equal(target, labels.double())  # sigmoid on the logits, then BCELoss
    assert loss_avg == float(torch.nn.functional.binary_cross_entropy(outputs, target)) and "Train: [0][1/1]" in log.getvalue() and "lr 0.0001" in log.getvalue()
    truth = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]], dtype=torch.float32)
    test_x = torch.randn(6, 1536).half()
    (all_mean, *_), (all_auc, *_) = main_linearprobing.test(model, [(test_x, truth)], io.StringIO())
    (two_mean, *_), (two_auc, *_) = main_linearprobing.test(model, [(test_x, truth)], io.StringIO(), columns=[0, 2])
    assert len(all_auc) == 3 and len(two_auc) == 2 and two_auc == [all_auc[0], all_auc[2]]  # the same model scored on two of its three classes
    print("OK (e2) train_one_epoch / test: float16 embeddings and float32 labels become float64, the loss sees torch.sigmoid(model(x)) and the labels, BCELoss; test(columns=[0, 2]) scores just those classes")

    # CPU only: --device accepts "cpu" and nothing else, with a clear message, and the shipped CUDA / DataParallel branch is gone
    for bad in ("cuda", "cuda:0", "hpu", "CPU", ""):
        _, _, usage = quiet(lambda: rejects(SystemExit, lambda: main_linearprobing.get_args_parser(["--device", bad])))
        assert "argument --device: {!r}: the probe runs on the CPU only".format(bad) in usage, (bad, usage)
    assert main_linearprobing.get_args_parser(["--device", "cpu"]).device == "cpu" and main_linearprobing.get_args_parser([]).device == "cpu"
    tree = ast.parse(open(os.path.join(LP_HERE, "main_linearprobing.py")).read())
    assert not [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in ("DataParallel", "cuda", "is_available", "device_count", "benchmark")]  # no CUDA call is left in the code
    entry_cpu, cache_cpu = random_dataset(tmp, 30, 20, shift=1.5, scale=1.0, seed=1)
    kinds = []
    before = os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def spy_kind(model, criterion, optimizer, scheduler, train_loader, epoch, multiclass, log_writter):
        kinds.append((type(model).__name__, next(model.parameters()).device.type))
        return stock_train(model, criterion, optimizer, scheduler, train_loader, epoch, multiclass, log_writter)

    main_linearprobing.train_one_epoch = spy_kind
    try:
        run_script(probe_argv("ChestXray14", entry_cpu, cache_cpu, os.path.join(tmp, "res_cpu"), "--epochs", "2", "--gpu", "0,1"))  # --gpu is the shipped option, and means nothing now
        assert kinds == [("LinearClassifier", "cpu")] * 2 and "CUDA_VISIBLE_DEVICES" not in os.environ
    finally:
        main_linearprobing.train_one_epoch = stock_train
        if before is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = before
    print("OK (e3) device: --device accepts only cpu (cuda, cuda:0, hpu, CPU and '' are refused with 'the probe runs on the CPU only'), the model is a plain LinearClassifier on the CPU, "
          "CUDA_VISIBLE_DEVICES is untouched even with --gpu, and no DataParallel / .cuda() / torch.cuda call is left in the vendored script")


# (f) best epoch on TEST per run, as shipped -----------------------------------------------------------------------------------
def scripted_test(values):
    """A stand-in for the script's test(): the k-th call returns mean metrics `values[k]` and per-class scores that identify the epoch (AUC of class j = value + 0.001 j)."""
    sequence = iter(values)

    def fake_test(model, test_loader, log_writter, columns=None):
        value = next(sequence)
        return [value, 0.1, 0.2, 0.3], [[value + 0.001 * j for j in range(14)], [0.0] * 14, [0.0] * 14, [0.0] * 14]
    return fake_test


with tempfile.TemporaryDirectory() as tmp:
    entry, cache = random_dataset(tmp, 30, 20, shift=1.5, scale=1.0, seed=2)
    stock_test = main_linearprobing.test

    def scripted_run(values, label, *flags):
        main_linearprobing.test = scripted_test(values)
        try:
            return run_script(probe_argv("ChestXray14", entry, cache, os.path.join(tmp, label), *flags)) + (os.path.join(tmp, label),)
        finally:
            main_linearprobing.test = stock_test

    def checkpoint_epoch(results_dir, run):
        return torch.load(os.path.join(results_dir, "linear_eval", "ChestXray14", "t", "b2", "run_{}".format(run), "ckpt.pth"), map_location="cpu")["epoch"]

    # a test mAUC that peaks mid-run (a triangle, 0.9 at epoch 37 of 80): that epoch is reported, its checkpoint is the saved one, and later epochs never replace it
    peak = [0.5 + 0.4 * (1 - abs(e - 37) / 37) for e in range(80)]
    records, out, results_dir = scripted_run(peak * 2, "peak", "--epochs", "80", "--runs", "2")
    assert [r["best_epoch"] for r in records] == [37, 37] and all(r["performance"] == [peak[37], 0.1, 0.2, 0.3] for r in records), records
    assert all(r["scores"][0] == [peak[37] + 0.001 * j for j in range(14)] for r in records)  # the per-class scores of THAT epoch, not the last
    assert [checkpoint_epoch(results_dir, run) for run in (0, 1)] == [37, 37]
    improved = re.findall(r"^Epoch (\d{4}): test performance improved", out, re.M)
    assert improved[-1] == "0037" and improved.count("0000") == 2 and "Run 0: Best performance is [0.9, 0.1, 0.2, 0.3] at epoch 37" in out, out
    with open(os.path.join(results_dir, "linear_eval", "ChestXray14", "t", "b2", "log.txt")) as f:
        assert "Average performance from 2 runs: ['0.9000(0.0000)', '0.1000(0.0000)', '0.2000(0.0000)', '0.3000(0.0000)']" in f.read()
    # equal mAUCs: only a strictly better epoch replaces the best, so the earlier one stays
    tied = [0.5] * 80
    tied[20] = tied[60] = 0.9
    records, _, results_dir = scripted_run(tied, "tie", "--epochs", "80", "--runs", "1")
    assert records[0]["best_epoch"] == 20 and checkpoint_epoch(results_dir, 0) == 20 and records[0]["scores"][0][0] == 0.9
    # a run whose test mAUC is never a number keeps the shipped initial state (best epoch -1): a failed measurement, which the runner refuses (g)
    records, _, _ = scripted_run([float("nan")] * 5, "nan", "--epochs", "5", "--runs", "1")
    assert records[0]["best_epoch"] == -1 and records[0]["scores"] is None and records[0]["performance"] == [-1, -1, -1, -1]
    # the real thing: the reported epoch is the first epoch of the largest test mAUC of the run, and not the last epoch
    entry, cache = random_dataset(tmp, 200, 400, shift=1.5, scale=300.0, seed=0)
    sequence = []

    def recording_test(model, test_loader, log_writter, columns=None):
        result = stock_test(model, test_loader, log_writter, columns)
        sequence.append(result[0][0])
        return result

    main_linearprobing.test = recording_test
    try:
        records, out = run_script(probe_argv("ChestXray14", entry, cache, os.path.join(tmp, "real")))
    finally:
        main_linearprobing.test = stock_test
    best = records[0]["best_epoch"]
    assert len(sequence) == 100 and best == int(np.argmax(sequence)) and records[0]["performance"][0] == sequence[best]
    assert 0 < best < 99 and sequence[best] > sequence[-1] + 0.005, (best, sequence[best], sequence[-1])  # a mid-run peak with a margin: selection on test is not "the last epoch"
    print("OK (f) best epoch on test per run, as shipped: a scripted mAUC peaking at epoch 37 -> epoch 37 reported with that epoch's per-class scores, checkpoint and log lines; a tie keeps the earlier "
          "epoch; no defined mAUC leaves best epoch -1; a real 100-epoch run reports epoch {} (test mAUC {:.4f}), above the last epoch's {:.4f}".format(best, sequence[best], sequence[-1]))


# (g) the runner ---------------------------------------------------------------------------------------------------------------
def run_fit(tag, emb_root, res_root, *flags):
    """run_linearprobe.main on the synthetic lists: (stdout, stderr) of the run."""
    _, out, err = quiet(lambda: run_linearprobe.main(["--tag", tag, "--embeddings_root", emb_root, "--results_root", res_root] + list(flags)))
    return out, err


@contextlib.contextmanager
def short_epochs(n):
    """The runner refuses to change the probe's epochs, so the tests shorten a run from inside: the script's main() gets options.epochs = n before it runs (and the JSON records what ran)."""
    stock = main_linearprobing.main

    def shortened(options):
        options.epochs = n
        return stock(options)

    main_linearprobing.main = shortened
    try:
        yield
    finally:
        main_linearprobing.main = stock


def load_result(res_root, tag, name):
    with open(os.path.join(res_root, tag, name + ".json")) as f:
        return json.load(f)


def script_dir(res_root, tag, name):
    """The shipped script's own files for a (tag, dataset): its results_dir and its run directory (linear_eval/<dataset>/<tag>/b2)."""
    results_dir = os.path.join(res_root, tag, "probe", name)
    return results_dir, os.path.join(results_dir, "linear_eval", name, tag, "b2")


def clone_cache(emb_root, src_tag, dst_tag, name, **meta_changes):
    """The `src_tag` cache of `name` under another tag (meta.json says which tag a cache was extracted under, and the runner checks it); meta_changes replace top-level meta keys."""
    shutil.copytree(os.path.join(emb_root, src_tag, name), os.path.join(emb_root, dst_tag, name))
    meta_file = os.path.join(emb_root, dst_tag, name, "meta.json")
    meta = json.load(open(meta_file))
    json.dump(dict(meta, tag=dst_tag, **meta_changes), open(meta_file, "w"))


def protocol_of(epochs, runs=10, seed=0):
    """The protocol text the runner must write for the shipped settings (a literal, not built from the runner's code)."""
    return ("the lab's embedding linear probe (Ark_Plus/Linearprobing/main_linearprobing.py) as shipped: float64 nn.Linear on the 1536-d embedding, sigmoid + BCELoss, adam lr 0.0001, batch 512, "
            "{} epochs, CosineAnnealingLR(T_max=1000, eta_min=1) stepped per epoch (the lr rises), {} runs seeded {}..{}, device cpu; full train list, official test list, best epoch on test".format(
                epochs, runs, seed, seed + runs - 1))


stock_entry = run_linearprobe.dataset_entry
with tempfile.TemporaryDirectory() as tmp:
    fit_sizes = {"train": 96, "test": 40}
    entries = {name: make_dataset(tmp, name, fit_sizes) for name in NAMES}  # lists only: the fit reads no image
    run_linearprobe.dataset_entry = lambda name: entries[name]
    try:
        emb_root, res_root = os.path.join(tmp, "emb"), os.path.join(tmp, "res")
        for name in NAMES:
            write_cache(os.path.join(emb_root, "sig", name), name, entries[name], signal=True, tag="sig")
        with short_epochs(3):
            out, err = run_fit("sig", emb_root, res_root)  # --runs is left at the runner's default
        classes = {"ChestXray14": real_config["ChestXray14"]["diseases"], "CheXpert": real_config["CheXpert"]["test_diseases_name"], "MIMIC": real_config["MIMIC"]["diseases"],
                   "VinDrCXR": real_config["VinDrCXR"]["diseases"]}
        keys = ["tag", "dataset", "model", "weights", "protocol", "embeddings_dir", "normalization", "max_images", "n", "runs", "seed", "epochs", "classes", "dropped_columns", "per_run",
                "best_epoch_per_run", "test_mAUC_mean", "test_mAUC_std", "test_AUC_mean", "test_AUC_std", "flags", "warnings", "script_log", "elapsed_s"]
        for name in NAMES:
            result = load_result(res_root, "sig", name)
            assert list(result) == keys, list(result)
            assert (result["tag"], result["dataset"], result["model"], result["normalization"], result["runs"], result["seed"], result["epochs"], result["max_images"]) \
                == ("sig", name, "vit_base_dinov3", "imagenet", 10, 0, 3, None) and result["n"] == fit_sizes and result["classes"] == classes[name], (name, result["classes"])
            assert result["weights"] == WEIGHTS_META and result["protocol"] == protocol_of(3) and result["embeddings_dir"] == os.path.join(emb_root, "sig", name)
            assert result["dropped_columns"] == [] and result["warnings"] == []
            runs = result["per_run"]
            assert [r["run"] for r in runs] == list(range(10)) and [r["seed"] for r in runs] == list(range(10))  # 10 runs, run i seeded with --seed + i
            assert all(0 <= r["best_epoch"] < 3 and list(r["test_AUC"]) == classes[name] for r in runs) and result["best_epoch_per_run"] == [r["best_epoch"] for r in runs]
            maucs = [r["test_mAUC"] for r in runs]
            assert all(abs(r["test_mAUC"] - np.mean(list(r["test_AUC"].values()))) < 1e-12 for r in runs)  # a run's mAUC is the mean of its per-class AUCs at its best epoch
            assert result["test_mAUC_mean"] == float(np.mean(maucs)) and result["test_mAUC_std"] == float(np.std(maucs))  # np.average / np.std (ddof 0), as the script prints them
            assert list(result["test_AUC_mean"]) == list(result["test_AUC_std"]) == classes[name]
            for j, class_name in enumerate(classes[name]):  # the per-class mean and s.d. over the runs
                column = [r["test_AUC"][class_name] for r in runs]
                assert abs(result["test_AUC_mean"][class_name] - np.mean(column)) < 1e-12 and abs(result["test_AUC_std"][class_name] - np.std(column)) < 1e-12
            assert result["flags"] == {"selection": "best epoch on test (as shipped)", "rad_dino_pretraining_overlap": None}
            results_dir, run_dir = script_dir(res_root, "sig", name)
            assert sorted(os.listdir(run_dir)) == sorted(["log.txt"] + ["run_{}".format(i) for i in range(10)]) and all(os.path.isfile(os.path.join(run_dir, "run_{}".format(i), "ckpt.pth")) for i in range(10))
            assert os.path.isfile(os.path.join(results_dir, "log.txt")) and os.path.join(res_root, "sig", result["script_log"]) == os.path.join(run_dir, "log.txt")
            assert "PROBE sig {}: test mAUC {:.4f} ({:.4f}) over 10 runs, best epoch on test (as shipped), best epochs {}".format(
                name, result["test_mAUC_mean"], result["test_mAUC_std"], result["best_epoch_per_run"]) in out
        print("OK (g1) the runner on synthetic embeddings, all four datasets: 10 runs by default, seeds 0..9, per run the best test epoch, its mAUC and per-class AUCs, mean / s.d. over the runs "
              "(np.average / np.std), scored classes (CheXpert: its 5 findings), the weights (path, size) and the protocol text built from the effective settings, flags.selection, the script's own "
              "files (log.txt, run_<i>/ckpt.pth) beside the JSON")

        # run i is seeded with --seed + i, before its data module is built: --seed 1 run 0 is --seed 0 run 1 (MIMIC's soft labels and CheXpert's draws included); the same seed repeats exactly
        for tag in ("a", "b", "c", "d"):
            for name in NAMES:
                clone_cache(emb_root, "sig", tag, name)
        with short_epochs(3):
            for tag, flags in (("a", ("--runs", "3")), ("b", ("--runs", "1", "--seed", "1")), ("c", ("--runs", "3")), ("d", ("--runs", "1", "--seed", "5"))):
                run_fit(tag, emb_root, res_root, *flags)
        key = lambda r: (r["best_epoch"], r["test_mAUC"], r["test_AUC"])
        for name in NAMES:
            a, b, c, d = (load_result(res_root, tag, name) for tag in "abcd")
            assert [key(r) for r in c["per_run"]] == [key(r) for r in a["per_run"]], name  # deterministic
            assert key(b["per_run"][0]) == key(a["per_run"][1]) and b["per_run"][0]["seed"] == 1, name  # run i = seed + i
            assert key(d["per_run"][0]) not in [key(r) for r in a["per_run"]] and key(a["per_run"][0]) != key(a["per_run"][1]), name  # other seeds, other results
        # python's random module is seeded with seed + run BEFORE the data module is built: MIMIC's one uncertain train cell is the first draw of random.Random(seed + run)
        recorded = []
        stock_mimic = embedding_datasets.EMBEDDING_MODULES["MIMIC"]

        class RecordingMimic(stock_mimic):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                recorded.append(self.train_set.parsed.img_label[1][0])  # row 1, column 0 of the synthetic MIMIC train list, drawn when the dataset is built

        embedding_datasets.EMBEDDING_MODULES["MIMIC"] = RecordingMimic  # the dict that main_linearprobing looks the modules up in
        try:
            clone_cache(emb_root, "sig", "mimic_seed", "MIMIC")
            with short_epochs(1):
                run_fit("mimic_seed", emb_root, res_root, "--datasets", "MIMIC", "--runs", "3", "--seed", "4")
        finally:
            embedding_datasets.EMBEDDING_MODULES["MIMIC"] = stock_mimic
        assert recorded == [random.Random(4 + run).uniform(0.55, 0.85) for run in range(3)] and len(set(recorded)) == 3, recorded
        print("OK (g2) seeds: the same --seed reproduces every run exactly; --seed 1 run 0 equals --seed 0 run 1 on all four datasets; another seed gives other results; random is seeded with "
              "seed + run before the data module is built (MIMIC's draws)")

        # the JSON is checked against the script's own files: a changed average line, a best epoch that its checkpoint does not carry, and a run without a defined mAUC are refused
        for name in ("ChestXray14", "MIMIC"):
            for tag in ("t1", "t2", "t3", "t4"):
                clone_cache(emb_root, "sig", tag, name)

        def tampered(kind, stock_main):
            def wrapper(options):
                records = stock_main(options)
                exp = os.path.join(options.results_dir, "linear_eval", options.dataset, options.arch, options.exp_name)
                if kind == "average":
                    text = open(os.path.join(exp, "log.txt")).read()
                    open(os.path.join(exp, "log.txt"), "w").write(re.sub(r"(Average performance from \d+ runs: \[')\d\.\d{4}", r"\g<1>0.1234", text))
                elif kind == "epoch":
                    records[0]["best_epoch"] += 1
                elif kind == "columns":
                    records[1]["columns"] = [0, 1, 2]
                return records
            return wrapper

        for kind, tag, expected in (("average", "t1", "recomputed"), ("epoch", "t2", "its checkpoint is epoch"), ("columns", "t4", "the script scored columns [0, 1, 2] and dropped [], the plan was None and []")):
            with short_epochs(3):
                shortened = main_linearprobing.main
                main_linearprobing.main = tampered(kind, shortened)
                try:
                    rejects(RuntimeError, lambda: run_fit(tag, emb_root, res_root, "--datasets", "ChestXray14", "--runs", "2"), expected)
                finally:
                    main_linearprobing.main = shortened
            assert not os.path.exists(os.path.join(res_root, tag, "ChestXray14.json"))  # nothing is written for a result that did not check out
        main_linearprobing.test = scripted_test([float("nan")] * 3)
        try:
            with short_epochs(3):
                rejects(RuntimeError, lambda: run_fit("t3", emb_root, res_root, "--datasets", "MIMIC", "--runs", "1"), "no epoch had a defined test mAUC")
        finally:
            main_linearprobing.test = stock_test
        assert not os.path.exists(os.path.join(res_root, "t3", "MIMIC.json"))
        print("OK (g3) cross-checks: a JSON whose mean (s.d.) differs from the script's own average line, a best epoch that its checkpoint does not carry, columns that the script scored but the runner "
              "did not plan, or a run whose test mAUC was never defined is refused, and no JSON is written")

        # the preflight: results that exist stop the run FIRST, before the cache is looked at (and run_emb_gaudi.sh checks the same two paths before it starts the extraction)
        assert run_linearprobe.results_of("R", "tag", "MIMIC") == [os.path.join("R", "tag", "MIMIC.json"), os.path.join("R", "tag", "probe", "MIMIC")]
        message = rejects(FileExistsError, lambda: run_fit("sig", emb_root, res_root, "--datasets", "ChestXray14"), "results already exist for tag sig")
        assert os.path.join(res_root, "sig", "ChestXray14.json") in message and os.path.join(res_root, "sig", "probe", "ChestXray14") in message and "refusing to overwrite" in message
        os.makedirs(os.path.join(res_root, "ghost", "probe", "MIMIC"))  # a half-done fit of a tag whose cache was deleted: the reason to stop is the results, not the missing cache
        rejects(FileExistsError, lambda: run_fit("ghost", emb_root, res_root, "--datasets", "MIMIC"), "results already exist for tag ghost")
        os.makedirs(os.path.join(res_root, "ghost2"))
        open(os.path.join(res_root, "ghost2", "VinDrCXR.json"), "w").write("{}")
        rejects(FileExistsError, lambda: run_fit("ghost2", emb_root, res_root), "results already exist for tag ghost2")  # any of the datasets to be fit
        rejects(FileNotFoundError, lambda: run_fit("nowhere", emb_root, res_root), "no embeddings for tag nowhere")
        assert not os.path.exists(os.path.join(res_root, "nowhere"))
        snapshot = lambda: sorted(os.path.join(d, f) for d, _, fs in os.walk(os.path.join(res_root, "sig")) for f in fs)
        before = snapshot()
        rejects(FileExistsError, lambda: run_fit("sig", emb_root, res_root))
        assert snapshot() == before  # a refused run touches nothing
        clone_cache(emb_root, "sig", "logonly", "VinDrCXR")
        os.makedirs(os.path.join(res_root, "logonly"))
        open(os.path.join(res_root, "logonly", "VinDrCXR.log"), "w").write("the wrapper's redirect\n")  # run_emb_gaudi.sh creates the log before the fit: it is not a result
        with short_epochs(2):
            run_fit("logonly", emb_root, res_root, "--datasets", "VinDrCXR", "--runs", "1")
        assert load_result(res_root, "logonly", "VinDrCXR")["runs"] == 1
        print("OK (g4) preflight: results that exist (the JSON or the script's directory, of any dataset to be fit) raise FileExistsError naming the tag and the paths before the cache is looked at; "
              "a missing cache is FileNotFoundError; a refused run touches nothing; the wrapper's .log alone is not a result")

        # flags: RAD-DINO's pretraining data holds all of ChestXray14 (train + val + test) and nothing the probe uses of the other datasets
        rad = lambda model, tag, dataset: run_linearprobe.rad_dino_flag({"model": model}, tag, dataset)
        assert [rad("vit_base_raddino", "x", d) for d in NAMES] == ["train+val+test", None, None, None]
        assert [rad("vit_base_dinov2", t, "ChestXray14") for t in ("rad_dino", "RAD-DINO", "raddino_s1", "stock", "dinov2")] == ["train+val+test", "train+val+test", "train+val+test", None, None]
        assert rad("vit_base_dinov3", "rad_dino", "CheXpert") is None and rad("vit_base_dinov3", "rad_dino", "MIMIC") is None
        for name in ("ChestXray14", "CheXpert"):
            clone_cache(emb_root, "sig", "rad", name, model="vit_base_raddino")
        for name in ("MIMIC", "VinDrCXR"):
            clone_cache(emb_root, "sig", "rad", name, model="vit_base_raddino")  # one backbone under the tag: all four caches say so
        with short_epochs(2):
            run_fit("rad", emb_root, res_root, "--datasets", "ChestXray14", "CheXpert", "--runs", "1")
        assert load_result(res_root, "rad", "ChestXray14")["flags"]["rad_dino_pretraining_overlap"] == "train+val+test" and load_result(res_root, "rad", "ChestXray14")["model"] == "vit_base_raddino"
        assert load_result(res_root, "rad", "CheXpert")["flags"] == {"selection": "best epoch on test (as shipped)", "rad_dino_pretraining_overlap": None}
        print("OK (g5) flags: rad_dino_pretraining_overlap is 'train+val+test' for ChestXray14 under a vit_base_raddino cache (or a rad-dino tag), null for CheXpert, MIMIC, VinDrCXR and other backbones")

        # single-valued test columns: dropped (and named in the JSON and on stderr) so that the shipped mAUC is never NaN; nothing to score is refused before anything is fitted
        assert classes["ChestXray14"][3] == "Infiltration" and classes["ChestXray14"][7] == "Pneumothorax" and real_config["CheXpert"]["diseases"][5] == "Edema"
        originals = dict(entries)
        ray_test = os.path.join(tmp, "ChestXray14", "test_single_valued.txt")
        shutil.copy(originals["ChestXray14"]["test_list"], ray_test)
        force_columns(ray_test, "ChestXray14", {3: 0, 7: 1})  # Infiltration never positive, Pneumothorax always
        expert_test = os.path.join(tmp, "CheXpert", "test_single_valued.csv")
        shutil.copy(originals["CheXpert"]["test_list"], expert_test)
        force_columns(expert_test, "CheXpert", {5: 1, 0: 1})  # Edema always positive (one of the 5 findings), No Finding always positive (not scored)
        entries["ChestXray14"], entries["CheXpert"] = dict(originals["ChestXray14"], test_list=ray_test), dict(originals["CheXpert"], test_list=expert_test)
        for name in NAMES:
            clone_cache(emb_root, "sig", "drop", name)
        with short_epochs(2):
            _, err = run_fit("drop", emb_root, res_root, "--datasets", "ChestXray14", "CheXpert", "--runs", "2")
        ray, expert = load_result(res_root, "drop", "ChestXray14"), load_result(res_root, "drop", "CheXpert")
        assert ray["classes"] == [c for i, c in enumerate(classes["ChestXray14"]) if i not in (3, 7)] and len(ray["classes"]) == 12
        assert ray["dropped_columns"] == [{"column": 3, "reason": "all negative", "name": "Infiltration"}, {"column": 7, "reason": "all positive", "name": "Pneumothorax"}]
        assert ray["warnings"] == ["Infiltration: all negative among the evaluated test images, so no AUC: not scored (left out of the means)",
                                   "Pneumothorax: all positive among the evaluated test images, so no AUC: not scored (left out of the means)"]
        assert all(len(r["test_AUC"]) == 12 and np.isfinite(r["test_mAUC"]) for r in ray["per_run"]) and "WARNING drop ChestXray14: Infiltration: all negative" in err
        assert expert["classes"] == ["Atelectasis", "Cardiomegaly", "Consolidation", "Pleural Effusion"] and expert["dropped_columns"] == [{"column": 5, "reason": "all positive", "name": "Edema"}]
        assert "WARNING drop CheXpert: Edema: all positive" in err and all(np.isfinite(r["test_mAUC"]) and len(r["test_AUC"]) == 4 for r in expert["per_run"])
        entries.update(originals)
        one_row = make_dataset(os.path.join(tmp, "one"), "VinDrCXR", {"train": 9, "test": 1})  # one test image: every column is single-valued, nothing can be scored
        clone_cache(emb_root, "sig", "empty", "ChestXray14")
        write_cache(os.path.join(emb_root, "empty", "VinDrCXR"), "VinDrCXR", one_row, tag="empty")
        entries["VinDrCXR"] = one_row
        calls = []
        stock_main = main_linearprobing.main
        main_linearprobing.main = lambda options: calls.append(options) or stock_main(options)
        try:
            nothing = "VinDrCXR: none of the 6 columns it is scored on has both a positive and a negative example among the 1 test images"
            rejects(ValueError, lambda: run_fit("empty", emb_root, res_root, "--datasets", "VinDrCXR"), nothing)
            # ... also when it is the second of two datasets: every dataset's scoring is planned before the first is fitted
            rejects(ValueError, lambda: run_fit("empty", emb_root, res_root, "--datasets", "ChestXray14", "VinDrCXR"), nothing)
        finally:
            main_linearprobing.main = stock_main
            entries.update(originals)
        assert calls == [] and not os.path.exists(os.path.join(res_root, "empty"))  # refused before the script ran, and before any results directory was made
        print("OK (g6) single-valued columns: an all-negative and an all-positive ChestXray14 column and an all-positive CheXpert finding are dropped, named in dropped_columns and warnings (and on stderr), "
              "and every run's mAUC is finite; a dataset with nothing to score (even the second of two) is refused before the script runs and before any results directory exists")

        # a smoke cache (meta max_images = 5): the probe trains and is scored on the first 5 entries of each list, and the protocol says so
        write_cache(os.path.join(emb_root, "smoke", "VinDrCXR"), "VinDrCXR", entries["VinDrCXR"], max_images=5, tag="smoke")
        with short_epochs(2):
            run_fit("smoke", emb_root, res_root, "--datasets", "VinDrCXR", "--runs", "2")
        smoke = load_result(res_root, "smoke", "VinDrCXR")
        assert smoke["n"] == {"train": 5, "test": 5} and smoke["max_images"] == 5 and smoke["runs"] == 2 and smoke["classes"] == classes["VinDrCXR"]
        assert smoke["protocol"] == protocol_of(2, runs=2) + "; SMOKE cache: the first 5 entries of each list" and "SMOKE" not in load_result(res_root, "sig", "VinDrCXR")["protocol"]
        print("OK (g7) smoke cache (max_images 5): n = 5 train / 5 test, max_images and 'SMOKE cache: the first 5 entries of each list' in the JSON's protocol; the whole pipeline runs on it")

        # the protocol is the shipped one: the runner refuses to change it, and the JSON describes the settings that ran
        for tag in ("allowed", "refused", "effective"):
            for name in NAMES:
                clone_cache(emb_root, "sig", tag, name)
        refused = [["--lr", "1e-3"], ["--epochs", "5"], ["--batch_size", "256"], ["--optimizer", "lars"], ["--multiclass", "true"], ["--mode", "test"], ["--few_shot", "0.5"], ["--emb_dim", "768"],
                   ["--num_class", "3"], ["--num_workers", "4"], ["--workers", "2"], ["--threads", "4"], ["--patience", "3"], ["--gpu", "0"], ["--arch", "x"], ["--exp_name", "x"], ["--dataset", "MIMIC"],
                   ["--max_images", "512"], ["--bogus"], ["--lr", "1e-4", "--epochs", "100"]]  # even the shipped values: nothing is passed through
        def refusal_of(*flags, tag="refused"):
            """What argparse printed for flags it refuses (run_fit captures stderr itself, and loses it when the run exits)."""
            return quiet(lambda: rejects(SystemExit, lambda: run_linearprobe.main(["--tag", tag, "--embeddings_root", emb_root, "--results_root", res_root] + list(flags))))[2]

        for extra in refused:
            usage = refusal_of(*extra)
            assert "{}: not accepted. The probe runs exactly as shipped".format(" ".join(extra)) in usage and "only --runs, --seed and the paths may vary" in usage, (extra, usage)
        assert "extract_embeddings --max_images" in refusal_of("--max_images", "512")  # where the smoke subset is chosen
        assert "--runs must be at least 1" in refusal_of("--runs", "0")
        for twice in (["MIMIC", "MIMIC"], ["MIMIC", "CheXpert", "MIMIC"]):  # as the extractor: at parse time, before the results or the caches are looked at
            assert "--datasets lists a dataset twice" in refusal_of("--datasets", *twice, tag="no_such_tag"), twice
        for device in ("cuda", "hpu", "cuda:0"):  # refused as the options are parsed: before the results or the caches are looked at (this tag has neither)
            assert "argument --device: {!r}: the probe runs on the CPU only".format(device) in refusal_of("--device", device, tag="no_such_tag"), device
        assert not os.path.exists(os.path.join(res_root, "refused"))  # nothing was created by any refusal
        with short_epochs(2):
            run_fit("allowed", emb_root, res_root, "--datasets", "VinDrCXR", "--runs", "2", "--seed", "7", "--device", "cpu")  # the knobs that may vary
        allowed = load_result(res_root, "allowed", "VinDrCXR")
        assert (allowed["runs"], allowed["seed"], [r["seed"] for r in allowed["per_run"]], allowed["protocol"]) == (2, 7, [7, 8], protocol_of(2, runs=2, seed=7))
        # the settings the script ran with are in the JSON, not constants: a run that the script itself shortens says so
        with short_epochs(4):
            run_fit("effective", emb_root, res_root, "--datasets", "VinDrCXR", "--runs", "2")
        effective = load_result(res_root, "effective", "VinDrCXR")
        assert effective["epochs"] == 4 and effective["protocol"] == protocol_of(4, runs=2) and all(0 <= r["best_epoch"] < 4 for r in effective["per_run"])
        # ... and the runner stops before fitting if the vendored defaults were changed (here: the lr)
        stock_parser = main_linearprobing.get_args_parser

        def edited_parser(argv=None):
            options = stock_parser(argv)
            options.lr = 5e-4
            return options

        main_linearprobing.get_args_parser = edited_parser
        try:
            rejects(RuntimeError, lambda: run_fit("effective", emb_root, res_root, "--datasets", "MIMIC", "--runs", "1"), "the effective settings are not the shipped probe's: lr = 0.0005 (shipped 0.0001)")
        finally:
            main_linearprobing.get_args_parser = stock_parser
        assert not os.path.exists(os.path.join(res_root, "effective", "probe", "MIMIC")) and not os.path.exists(os.path.join(res_root, "effective", "MIMIC.json"))
        # the shipped protocol, whole: no shortening, 100 epochs per run
        run_fit("allowed", emb_root, res_root, "--datasets", "ChestXray14", "--runs", "1")
        whole = load_result(res_root, "allowed", "ChestXray14")
        assert whole["epochs"] == 100 and whole["protocol"] == protocol_of(100, runs=1) and 0 <= whole["per_run"][0]["best_epoch"] < 100
        with open(os.path.join(script_dir(res_root, "allowed", "ChestXray14")[0], "log.txt")) as f:
            assert len(re.findall(r"^learning_rate: ", f.read(), re.M)) == 100
        print("OK (g8) protocol: any option but --runs, --seed, --device cpu and the paths is refused (also the shipped values, and --max_images, whose subset is chosen at extraction), --device "
              "cuda / hpu with the CPU-only message, --runs 0 and a repeated --datasets name, all at parse time, before anything is created; the JSON's protocol text and epochs are the settings the script ran with (a 4-epoch run says 4, a smoke cache says "
              "so, --seed 7 --runs 2 says 'seeded 7..8'), changed vendored defaults stop the fit, and the unshortened run is 100 epochs")

        # provenance: every cache under the tag (whichever datasets are fit: the wrapper fits one per process) must come from the same backbone, extracted the same way
        for name in NAMES:
            clone_cache(emb_root, "sig", "prov", name)
        with short_epochs(2):
            run_fit("prov", emb_root, res_root, "--datasets", "CheXpert", "--runs", "1")
        assert load_result(res_root, "prov", "CheXpert")["weights"] == WEIGHTS_META  # equal caches fit, and the result says which weights they came from
        mismatches = (("model", "CheXpert", dict(model="vit_base_dinov2"), "model differs"),
                      ("weights path", "MIMIC", dict(weights=dict(WEIGHTS_META, path="/workspace/weights/other.pth")), "weights path differs"),
                      ("weights size", "VinDrCXR", dict(weights=dict(WEIGHTS_META, size_bytes=1)), "weights size differs"),
                      ("weights sha256", "MIMIC", dict(weights=dict(WEIGHTS_META, sha256=hashlib.sha256(b"another file, same path and size").hexdigest())), "weights sha256 differs"),  # path and size equal
                      ("normalization", "MIMIC", dict(normalization="raddino"), "normalization differs"),
                      ("transform", "ChestXray14", dict(transform=dict(TRANSFORM_META, resize=512)), "transform differs"),
                      ("max_images", "VinDrCXR", dict(max_images=3), "max_images differs"))
        for field, odd, change, expected in mismatches:
            tag = "mix_" + field.replace(" ", "_")
            for name in NAMES:
                clone_cache(emb_root, "sig", tag, name, **(change if name == odd else {}))
            for fitted in (odd, "MIMIC" if odd == "ChestXray14" else "ChestXray14"):  # the dataset being fit may or may not be the odd one out
                message = rejects(RuntimeError, lambda: run_fit(tag, emb_root, res_root, "--datasets", fitted), expected)
                assert "the caches under tag {} were not extracted from the same backbone the same way".format(tag) in message and all("{} = ".format(n) in message for n in NAMES), message
            assert not os.path.exists(os.path.join(res_root, tag))  # nothing was fitted or created
        clone_cache(emb_root, "sig", "nokey", "ChestXray14")
        clone_cache(emb_root, "sig", "nokey", "MIMIC")
        meta_file = os.path.join(emb_root, "nokey", "MIMIC", "meta.json")
        meta = json.load(open(meta_file))
        del meta["weights"]
        json.dump(meta, open(meta_file, "w"))
        rejects(RuntimeError, lambda: run_fit("nokey", emb_root, res_root, "--datasets", "ChestXray14"), "MIMIC's meta.json under tag nokey lacks weights.path")
        # the digest must be there, in the dataset being fit too, and be a digest: null or "" in every cache of a tag would be equal, and prove nothing
        clone_cache(emb_root, "sig", "nodigest", "VinDrCXR", weights={k: v for k, v in WEIGHTS_META.items() if k != "sha256"})
        rejects(RuntimeError, lambda: run_fit("nodigest", emb_root, res_root, "--datasets", "VinDrCXR"), "VinDrCXR's meta.json under tag nodigest lacks weights.sha256")
        assert not os.path.exists(os.path.join(res_root, "nodigest"))
        for k, bad in enumerate((None, "", "abc", "A" * 64, "g" * 64, 12, WEIGHTS_SHA256[:-1])):
            tag = "baddigest{}".format(k)
            for name in ("ChestXray14", "VinDrCXR"):
                clone_cache(emb_root, "sig", tag, name, weights=dict(WEIGHTS_META, sha256=bad))  # the same in both caches, so they agree
            message = rejects(RuntimeError, lambda: run_fit(tag, emb_root, res_root, "--datasets", "VinDrCXR"), "is not a sha256 hex digest")
            assert "weights.sha256 {!r} is not".format(bad) in message and not os.path.exists(os.path.join(res_root, tag)), message
        clone_cache(emb_root, "sig", "alone", "VinDrCXR")  # a tag with a single cache has nothing to disagree with
        with short_epochs(2):
            run_fit("alone", emb_root, res_root, "--datasets", "VinDrCXR", "--runs", "1")
        clone_cache(emb_root, "sig", "retag", "VinDrCXR")
        meta_file = os.path.join(emb_root, "retag", "VinDrCXR", "meta.json")
        json.dump(dict(json.load(open(meta_file)), tag="sig"), open(meta_file, "w"))  # a cache that was extracted under another tag
        rejects(RuntimeError, lambda: run_fit("retag", emb_root, res_root, "--datasets", "VinDrCXR"), "holds the embeddings of sig / VinDrCXR, not retag / VinDrCXR")
        print("OK (g9) provenance: caches under one tag must agree on model, weights path, size and sha256 (same path and size, another digest: refused), normalization, transform and max_images, whichever "
              "dataset is fit and whichever is the odd one out (RuntimeError naming the field and the datasets, nothing created); a meta.json without weights, or without a sha256 digest, or with one that is not "
              "a digest, is refused; one cache alone, or equal caches, fit and the JSON records the weights path, size and sha256")

        # the JSON is written whole or not at all: run_emb_fit_class.sh's resume rule takes its existence for a finished fit, so a file that is empty or cut short must never be there
        whole_root = os.path.join(tmp, "res_whole")  # a fresh results root: nothing in it but what these fits leave
        part_files = lambda: [os.path.join(d, f) for d, _, fs in os.walk(whole_root) for f in fs if f.endswith(".part")]
        for name in ("VinDrCXR", "ChestXray14"):
            clone_cache(emb_root, "sig", "nolink", name)  # a second tag, for the fits that run with hard links refused

        @contextlib.contextmanager
        def after_collect(hook):
            """run_linearprobe.collect with hook(result) run once the real one has returned: the fit has run, and nothing of the JSON is written yet."""
            stock = run_linearprobe.collect

            def collect(*args):
                result = stock(*args)
                hook(result)
                return result

            run_linearprobe.collect = collect
            try:
                yield
            finally:
                run_linearprobe.collect = stock

        @contextlib.contextmanager
        def publishing(link_error=None):
            """Yields the list in which the calls that publish the JSON are recorded, in order, while this runs: ('fsync', the file's inode, its size on disk at that moment), ('link', src, dst),
            ('replace', src, dst), ('remove', path). With link_error, os.link raises it (a filesystem that refuses a hard link)."""
            steps, stock = [], (os.fsync, os.link, os.replace, os.remove)

            def fsync(fd):
                steps.append(("fsync", os.fstat(fd).st_ino, os.fstat(fd).st_size))
                return stock[0](fd)

            def link(src, dst):
                steps.append(("link", src, dst))
                if link_error:
                    raise link_error
                return stock[1](src, dst)

            def replace(src, dst):
                steps.append(("replace", src, dst))
                return stock[2](src, dst)

            def remove(path):
                steps.append(("remove", path))
                return stock[3](path)

            os.fsync, os.link, os.replace, os.remove = fsync, link, replace, remove
            try:
                yield steps
            finally:
                os.fsync, os.link, os.replace, os.remove = stock

        def fit_whole(name, tag="sig"):
            with short_epochs(2):
                run_fit(tag, emb_root, whole_root, "--datasets", name, "--runs", "1")

        canonical = lambda text: text == json.dumps(json.loads(text), indent=2, allow_nan=False)  # json.dump(indent=2)'s format, as before (no trailing newline)
        # a NaN: the dump fails before any file exists
        with after_collect(lambda result: result.update(test_mAUC_std=float("nan"))):  # json cannot write it with allow_nan=False
            rejects(ValueError, lambda: fit_whole("MIMIC"), "not JSON compliant")
        assert not os.path.exists(os.path.join(whole_root, "sig", "MIMIC.json")) and part_files() == []  # no MIMIC.json of any size, and no .part either
        # an ordinary fit: the whole text is written and synced to probe/<dataset>/<dataset>.json.part, hard-linked to the JSON's name (atomic, no rename) and the .part removed
        with publishing() as steps:
            fit_whole("VinDrCXR")
        final, part = os.path.join(whole_root, "sig", "VinDrCXR.json"), os.path.join(script_dir(whole_root, "sig", "VinDrCXR")[0], "VinDrCXR.json.part")
        assert steps == [("fsync", os.stat(final).st_ino, os.path.getsize(final)), ("link", part, final), ("remove", part)], steps  # the file that was synced is the one linked, all of it on disk by then
        assert canonical(open(final).read()) and part_files() == []  # no .part anywhere under the root
        # a filesystem that refuses the hard link (EPERM): the same JSON, published by a rename
        eperm = PermissionError(errno.EPERM, os.strerror(errno.EPERM))
        with publishing(eperm) as steps:
            fit_whole("VinDrCXR", "nolink")
        final, part = os.path.join(whole_root, "nolink", "VinDrCXR.json"), os.path.join(script_dir(whole_root, "nolink", "VinDrCXR")[0], "VinDrCXR.json.part")
        assert steps == [("fsync", os.stat(final).st_ino, os.path.getsize(final)), ("link", part, final), ("replace", part, final)], steps  # the link was tried and refused, then the rename
        assert canonical(open(final).read()) and part_files() == []
        # a JSON that appears while the fit runs (another fit of the same tag and dataset finished first) is refused by os.link itself (the kernel's EEXIST, not a check of ours) and left as it is
        raced = os.path.join(whole_root, "sig", "ChestXray14.json")
        with publishing() as steps, after_collect(lambda result: open(raced, "w").write("{}")):
            message = rejects(FileExistsError, lambda: fit_whole("ChestXray14"), os.strerror(errno.EEXIST))
        assert raced in message and open(raced).read() == "{}" and [s[0] for s in steps] == ["fsync", "link"], (message, steps)
        # ... and where hard links are refused, by the check before the rename
        raced = os.path.join(whole_root, "nolink", "ChestXray14.json")
        with publishing(eperm) as steps, after_collect(lambda result: open(raced, "w").write("{}")):
            message = rejects(FileExistsError, lambda: fit_whole("ChestXray14", "nolink"), "refusing to overwrite")
        assert raced in message and open(raced).read() == "{}" and [s[0] for s in steps] == ["fsync", "link"], (message, steps)
        # the two refused fits leave their .part in probe/<dataset>/, where the resume rule's rm -rf clears it with the half-done fit; nothing else
        assert sorted(part_files()) == sorted(os.path.join(script_dir(whole_root, tag, "ChestXray14")[0], "ChestXray14.json.part") for tag in ("sig", "nolink")), part_files()
        print("OK (g10) the JSON is written whole or not at all: a NaN in the result fails the dump (ValueError) before any file exists, so no MIMIC.json of any size and no .part; an ordinary fit writes the "
              "whole text to probe/<dataset>/<dataset>.json.part, fsyncs it (all of it on disk by then), hard-links it to the JSON's name (os.link, atomic) and removes the .part, in json.dump(indent=2)'s format, "
              "with no .part left under the results root; with hard links refused (EPERM) the same JSON is published by os.replace; a JSON that appeared while the fit ran is refused (os.link's own EEXIST, or "
              "the check before the rename) and left as it is, the refused fit's .part staying in its probe/<dataset>/")
    finally:
        run_linearprobe.dataset_entry = stock_entry

# (h) the extractor ------------------------------------------------------------------------------------------------------------
# the normalization presets of the vendored build_transform_classification: a constant gray image shows each preset's mean and std
gray = Image.new("RGB", (16, 16), (128, 128, 128))
level = 128 / 255
PRESETS = {"imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), "chestx-ray": ([0.5056] * 3, [0.252] * 3), "raddino": ([0.5307] * 3, [0.2583] * 3), "none": None}
for preset, stats in PRESETS.items():
    for variant in (preset, preset.upper()):  # the official function lowercases the name
        tensor = dataloader.build_transform_classification(normalize=variant, crop_size=8, resize=8, mode="valid")(gray)
        expected = torch.stack([torch.full((8, 8), level if stats is None else (level - stats[0][c]) / stats[1][c]) for c in range(3)])
        assert tensor.shape == (3, 8, 8) and torch.allclose(tensor, expected, atol=1e-6), (preset, tensor[:, 0, 0], expected[:, 0, 0])
assert extract_embeddings.NORMALIZATIONS == ("imagenet", "chestx-ray", "none", "raddino")
rejects(SystemExit, lambda: quiet(lambda: dataloader.build_transform_classification(normalize="raddino2", mode="valid")))  # an unknown preset exits, as the official function does
print("OK (h1) normalization presets: imagenet (0.485 0.456 0.406 / 0.229 0.224 0.225), chestx-ray (0.5056 / 0.252) and none unchanged; raddino = 0.5307 / 0.2583 on all three channels; unknown presets still exit")


class Loader:
    """What extract_features needs of a DataLoader: iteration over (samples, targets) batches and .dataset."""

    def __init__(self, batches):
        self.batches = batches
        self.dataset = [None] * sum(len(samples) for samples, _ in batches)

    def __iter__(self):
        return iter(self.batches)


class FakeModel:
    def __init__(self, embed):
        self.embed = embed

    def eval(self):
        return self

    def generate_embeddings(self, x, after_proj):
        assert after_proj is False
        return self.embed(x)


class NoMove(torch.nn.Module):
    """A model whose .to() silently moves nothing: its parameter stays on the CPU."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def to(self, *args, **kwargs):
        return self


META = SimpleNamespace(device=torch.device("meta"), name="meta", pin_memory=False, mark_step=lambda: None)  # a device the data can be sent to, with no data
doubled = lambda x: torch.cat([x.flatten(1)[:, :768], x.flatten(1)[:, :768]], 1)  # a 1536-d "embedding" that shows which input it came from
x4 = torch.randn(5, 3, 16, 16)
cpu = Accelerator("cpu")
marks = []
cpu.lazy_mode, cpu._htcore = True, SimpleNamespace(mark_step=lambda: marks.append(1))  # an HPU run in lazy mode, with a counting htcore
images = extract_embeddings.extract_features(FakeModel(doubled), Loader([(x4[:2], None), (x4[2:], None)]), cpu)
assert images.shape == (5, 1536) and images.dtype == np.float16 and len(marks) == 2  # one mark_step per batch, like trainer.evaluate
assert np.array_equal(images, doubled(x4).numpy().astype(np.float16))  # row i is image i's embedding
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)  # numpy warns about the overflow in the cast; the extractor then raises
    rejects(RuntimeError, lambda: extract_embeddings.extract_features(FakeModel(lambda x: torch.full((len(x), 1536), 1e6)), Loader([(x4, None)]), Accelerator("cpu"), split="ChestXray14 test"),
            "ChestXray14 test: 7680 of 7680 features are not finite once cast to float16")
rejects(RuntimeError, lambda: extract_embeddings.extract_features(FakeModel(lambda x: torch.zeros(len(x), 128)), Loader([(x4, None)]), Accelerator("cpu")), "returned 128-d features")
rejects(RuntimeError, lambda: extract_embeddings.extract_features(FakeModel(lambda x: torch.zeros(len(x), 1536)), Loader([(x4, None)]), META), "the features landed on 'cpu' but device 'meta' was requested")
rejects(RuntimeError, lambda: extract_embeddings.to_device(NoMove(), META), "the model's parameters are on 'cpu' after .to(meta), but device 'meta' was requested")
assert next(extract_embeddings.to_device(torch.nn.Linear(2, 2), META).parameters()).device.type == "meta"  # a real .to() reaches the device: no error
cadence = []
extract_embeddings.extract_features(FakeModel(lambda x: torch.zeros(len(x), 1536)), Loader([(torch.zeros(1, 3, 2, 2), None)] * 45), Accelerator("cpu"), cadence.append, "train")
assert [line.split(" imgs")[0] for line in cadence] == ["extract train 1/45", "extract train 2/45", "extract train 20/45", "extract train 40/45"]  # batches 1, 2, then every 20th
assert all(re.fullmatch(r"extract train \d+/45 imgs, \d+\.\d img/s, device=cpu", line) for line in cadence), cadence
print("OK (h2) extract_features: [N, 1536] float16, one embedding per image (no crop dimension), one mark_step per batch; features that overflow float16, a wrong width, features or parameters on "
      "another device than requested raise RuntimeError; progress lines at batches 1, 2 and every 20th")

# the width: by model name (embedding_dim), from a cache's meta.json (feature_dim), and as extract_features' parameter (the built backbone's; the default is the ViT-B 1536)
assert embedding_mode.EMBEDDING_DIM == 1536 and [embedding_mode.embedding_dim(m) for m in ("vit_base_dinov3", "vit_base_dinov2", "vit_base_raddino", "vit_large_raydino", "vit_large_dinov3", "vit_huge_plus_dinov3", "vit_7b_dinov3", "swin_base", "swin_large", "swin_large_384", "swin_large_768")] == \
    [1536, 1536, 1536, 2048, 2048, 2560, 8192, 1024, 1536, 1536, 1536]
assert embedding_mode.embedding_dim("some_other_model") == 1536 and embedding_mode.feature_dim({}) == 1536 and embedding_mode.feature_dim({"feature_dim": 2048}) == 2048
for bad_dim in (0, -1, 1536.0, "2048", None, True, [2048]):
    rejects(ValueError, lambda: embedding_mode.feature_dim({"feature_dim": bad_dim}), "meta.json feature_dim {!r} is not a positive integer".format(bad_dim))
wide = extract_embeddings.extract_features(FakeModel(lambda x: torch.ones(len(x), 2048)), Loader([(x4[:2], None), (x4[2:], None)]), Accelerator("cpu"), dim=2048)
assert wide.shape == (5, 2048) and wide.dtype == np.float16 and (wide == 1).all()
rejects(RuntimeError, lambda: extract_embeddings.extract_features(FakeModel(lambda x: torch.zeros(len(x), 1536)), Loader([(x4, None)]), Accelerator("cpu"), dim=2048), "returned 1536-d features, the probe's head takes 2048")
rejects(RuntimeError, lambda: extract_embeddings.extract_features(FakeModel(lambda x: torch.zeros(len(x), 2048)), Loader([(x4, None)]), Accelerator("cpu")), "returned 2048-d features, the probe's head takes 1536")
print("OK (h2b) width: embedding_dim is 1536 for the three ViT-B names and the three Swin-L names, 2048 for vit_large_raydino and vit_large_dinov3, 2560 for vit_huge_plus_dinov3, 8192 for vit_7b_dinov3 and 1024 for swin_base (EMBEDDING_DIM stays 1536); feature_dim reads meta.json (absent = 1536; 0, -1, 1536.0, '2048', null, true and a list "
      "are refused); extract_features takes the width as a parameter: 2048 gives (N, 2048) float16 and a 1536-wide model is refused at dim=2048, a 2048-wide one at the default")

stock_kwargs, stock_extractor_entry = models.DINOV3_VITB16_KWARGS, extract_embeddings.dataset_entry
models.DINOV3_VITB16_KWARGS = TINY  # build_omni_model reads the module global, so load_backbone builds the small ViT
try:
    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(0)
        reference = ArkViT([], None, False, **TINY).eval()
        weights = os.path.join(tmp, "teacher_backbone_epoch_0004.pth")
        torch.save(reference.state_dict(), weights)  # a flat backbone file, like the Stage A export and the stock DINOv3 file
        sizes = {"train": 7, "test": 6}
        entries = {name: make_dataset(tmp, name, sizes, images=True) for name in NAMES}
        extract_embeddings.dataset_entry = lambda name: entries[name]
        root, res_root = os.path.join(tmp, "emb"), os.path.join(tmp, "res")
        base_cli = ["--model", "vit_base_dinov3", "--weights", weights, "--embeddings_root", root, "--input_size", "32", "--img_resize", "36", "--batch_size", "4", "--workers", "0", "--device", "cpu"]
        CACHE_FILES = sorted(["meta.json", "train.npy", "train_paths.txt", "test.npy", "test_paths.txt"])  # and nothing for a validation split: the probe has none

        def extract(tag, *flags, normalization="imagenet"):
            return quiet(lambda: extract_embeddings.main(["--tag", tag, "--normalization", normalization] + base_cli + list(flags)))

        def reference_rows(name, split, normalization="imagenet", limit=None):
            """Each image through the official 'valid' transform (resize, center crop) and the reference ViT on its own: [N, 1536]."""
            transform = dataloader.build_transform_classification(normalize=normalization, crop_size=32, resize=36, mode="valid")
            with torch.no_grad():
                return np.concatenate([reference.generate_embeddings(transform(Image.open(path).convert("RGB"))[None], after_proj=False).numpy()
                                       for path in official_dataset(name, entries[name], split).img_list[:limit]])

        _, out, err = extract("tiny")
        for name in NAMES:
            folder = os.path.join(root, "tiny", name)
            assert sorted(os.listdir(folder)) == CACHE_FILES, os.listdir(folder)
            for split in ("train", "test"):
                npy, paths = embedding_mode.split_files(folder, split)
                cached = np.load(npy)
                assert cached.dtype == np.float16 and cached.shape == (sizes[split], 1536), (name, split, cached.shape)  # ONE embedding per image, test included
                assert embedding_mode.read_paths(paths) == official_dataset(name, entries[name], split).img_list  # row i is img_list[i]
                assert len({row.tobytes() for row in cached}) == sizes[split]  # every image has its own embedding
                expected = reference_rows(name, split)
                assert np.allclose(cached, expected.astype(np.float16), atol=2e-3, rtol=2e-3), (name, split, np.abs(cached - expected).max())
            meta = json.load(open(os.path.join(folder, "meta.json")))
            assert meta["tag"] == "tiny" and meta["dataset"] == name and meta["model"] == "vit_base_dinov3" and meta["normalization"] == "imagenet" and meta["max_images"] is None
            assert meta["weights"] == {"path": weights, "size_bytes": os.path.getsize(weights), "sha256": hashlib.sha256(open(weights, "rb").read()).hexdigest()}
            assert meta["feature_dim"] == 1536 and meta["dtype"] == "float16"
            assert meta["counts"] == sizes and meta["shapes"] == {"train": [7, 1536], "test": [6, 1536]} and meta["device"] == "cpu"
            assert meta["transform"] == {"resize": 36, "crop": 32, "pipelines": {"train": ["Resize", "CenterCrop", "ToTensor", "Normalize"], "test": ["Resize", "CenterCrop", "ToTensor", "Normalize"]}}
            assert meta["data_dir"] == entries[name]["data_dir"] and set(meta["seconds"]) == {"train", "test"} and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", meta["created_utc"])
            assert {s: os.path.normpath(os.path.join(BASE, rel)) for s, rel in meta["lists"].items()} == {s: entries[name][s + "_list"] for s in ("train", "test")}
        assert re.findall(r"^EMB tiny (\w+): train 7 / test 6, dim 1536, \d+s$", out, re.M) == list(NAMES) and out.count("\n") == 4  # one line per dataset, nothing else on stdout
        for expected in ("weights sha256: start", "weights sha256: done in", "model load: start", "model load: done in", "extract ChestXray14 train: start", "extract VinDrCXR test: done in", "extract ChestXray14 train 4/7 imgs", "extract ChestXray14 test 4/6 imgs"):
            assert expected in err, expected
        assert "device=cpu" in err and not re.search(r"^EMB", err, re.M) and "val" not in err
        print("OK (h3) extractor on tiny synthetic images in the four list formats: {train,test}.npy float16 (N, 1536) with one center-crop embedding per image, no val files, paths files equal the "
              "official img_list, every row equals the official 'valid' transform + the reference ViT on that image, meta.json complete (the weights' path, size and sha256 included), one EMB line per dataset "
              "on stdout, progress on stderr")

        # the sha256 of the weights is streamed: chunk by chunk, never the whole file at once, and equal to hashlib's of the whole file at every chunk boundary
        reads = []

        class SpyFile:
            def __init__(self, f):
                self.f = f

            def __enter__(self):
                self.f.__enter__()
                return self

            def __exit__(self, *exc):
                return self.f.__exit__(*exc)

            def read(self, n=-1):
                reads.append(n)
                return self.f.read(n)

        extract_embeddings.open = lambda path, mode: SpyFile(open(path, mode))  # the module's own global, found before the builtin
        try:
            blob_path = os.path.join(tmp, "blob.bin")
            for size, chunk in ((0, 7), (1, 7), (6, 7), (7, 7), (8, 7), (14, 7), (15, 7), (100, 7), (5 << 19, None)):  # the last: 2.5 MiB at the default chunk (1 MiB)
                blob = os.urandom(size)
                with open(blob_path, "wb") as f:
                    f.write(blob)
                reads.clear()
                digest = extract_embeddings.sha256_file(blob_path) if chunk is None else extract_embeddings.sha256_file(blob_path, chunk=chunk)
                chunk = chunk or 1 << 20
                assert digest == hashlib.sha256(blob).hexdigest() and re.fullmatch(r"[0-9a-f]{64}", digest), size
                assert reads == [chunk] * (-(-size // chunk) + 1), (size, reads)  # full-size reads until the empty one that ends the file
        finally:
            del extract_embeddings.open
        print("OK (h3b) sha256_file reads the weights chunk by chunk (never the whole file at once) and equals hashlib's digest of the whole file for sizes around the chunk boundary and for 2.5 MiB at the default chunk")

        # it never overwrites, before the model loads: not a whole cache, not a part of one
        snapshot = lambda: {os.path.join(d, f): open(os.path.join(d, f), "rb").read() for d, _, fs in os.walk(os.path.join(root, "tiny")) for f in fs}
        message = rejects(FileExistsError, lambda: extract("tiny"), "refusing to overwrite")
        assert os.path.join(root, "tiny", "ChestXray14", "meta.json") in message
        os.remove(os.path.join(root, "tiny", "VinDrCXR", "test.npy"))  # a dataset left half done by a dead job
        before = snapshot()
        rejects(FileExistsError, lambda: extract("tiny", "--datasets", "VinDrCXR"), os.path.join("tiny", "VinDrCXR", "train.npy"))
        rejects(FileExistsError, lambda: extract("tiny", "--datasets", "ChestXray14", "VinDrCXR"), "refusing to overwrite")
        assert snapshot() == before  # nothing was touched, not even the missing file recreated
        shutil.rmtree(os.path.join(root, "tiny", "VinDrCXR"))  # delete its directory to redo it, as the refusal says
        extract("tiny", "--datasets", "VinDrCXR")  # and --datasets finishes it
        assert sorted(os.listdir(os.path.join(root, "tiny", "VinDrCXR"))) == CACHE_FILES
        print("OK (h4) overwrite refusal: a second run and a half-written dataset raise FileExistsError naming the files before anything runs, the files stay byte-identical; --datasets redoes one dataset")

        # --max_images: the first N entries of each split, recorded in meta.json; the same embeddings as the full run
        _, out, _ = extract("smoke", "--datasets", "VinDrCXR", "ChestXray14", "--max_images", "3")
        for name in ("VinDrCXR", "ChestXray14"):
            smoke_dir, full_dir = os.path.join(root, "smoke", name), os.path.join(root, "tiny", name)
            meta = json.load(open(os.path.join(smoke_dir, "meta.json")))
            assert meta["max_images"] == 3 and meta["counts"] == {"train": 3, "test": 3} and meta["shapes"]["test"] == [3, 1536]
            for split in ("train", "test"):
                smoke_rows, full_rows = np.load(embedding_mode.split_files(smoke_dir, split)[0]), np.load(embedding_mode.split_files(full_dir, split)[0])
                assert np.allclose(smoke_rows, full_rows[:3], atol=2e-3, rtol=2e-3)
                assert embedding_mode.read_paths(embedding_mode.split_files(smoke_dir, split)[1]) == official_dataset(name, entries[name], split).img_list[:3]
        assert list(os.listdir(os.path.join(root, "smoke"))) in (["VinDrCXR", "ChestXray14"], ["ChestXray14", "VinDrCXR"]) and "EMB smoke VinDrCXR: train 3 / test 3" in out
        print("OK (h5) --max_images 3: the first 3 entries of each split, equal to the first 3 rows of the full run; meta.json records it; --datasets selects")

        # --model is passed through as it is (B1 adds vit_base_dinov2 and vit_base_raddino to base/models.py), with no projector and no heads; --normalization reaches the transform
        seen = []
        stock_builder = extract_embeddings.build_omni_model
        extract_embeddings.build_omni_model = lambda args, num_classes_list: (seen.append((args, num_classes_list)), stock_builder(SimpleNamespace(**dict(vars(args), model_name="vit_base_dinov3")), num_classes_list))[1]
        try:
            _, out, _ = quiet(lambda: extract_embeddings.main(["--tag", "raddino_smoke", "--normalization", "raddino", "--model", "vit_base_raddino"] + base_cli[2:] + ["--datasets", "ChestXray14", "--max_images", "2"]))
        finally:
            extract_embeddings.build_omni_model = stock_builder
        (builder_args, builder_heads), = seen
        assert (builder_args.model_name, builder_args.projector_features, builder_args.use_mlp, builder_args.pretrained_weights, builder_heads) == ("vit_base_raddino", None, False, weights, [])
        meta = json.load(open(os.path.join(root, "raddino_smoke", "ChestXray14", "meta.json")))
        assert meta["model"] == "vit_base_raddino" and meta["normalization"] == "raddino"
        raddino_rows = np.load(embedding_mode.split_files(os.path.join(root, "raddino_smoke", "ChestXray14"), "train")[0])
        assert np.allclose(raddino_rows, reference_rows("ChestXray14", "train", "raddino", 2).astype(np.float16), atol=2e-3, rtol=2e-3)
        assert not np.allclose(raddino_rows, np.load(embedding_mode.split_files(os.path.join(root, "tiny", "ChestXray14"), "train")[0])[:2], atol=2e-3, rtol=2e-3)  # another normalization, other embeddings
        print("OK (h6) --model reaches build_omni_model unchanged (vit_base_raddino, no projector, no heads, the given weights); --normalization raddino reaches the transform and the embeddings change with it")

        # (h7) the Stage A export as the Gaudi trainer writes it (export_backbone: a flat dict of CPU tensors) is a pickle of protocol >= 4 (Sol job 66243085): torch >= 2.6's default torch.load (weights_only=True) refuses its FRAME opcode ("Unsupported operand 149"),
        # and timm 0.5.4's loader took that default. build_omni_model reads it with the file's weights, also from a 'model' / 'state_dict' wrapper whose keys carry the `module.` prefix DataParallel adds (stripped, as timm did)
        omni_from = lambda path: models.build_omni_model(SimpleNamespace(model_name="vit_base_dinov3", projector_features=None, use_mlp=False, pretrained_weights=path), [])
        state = export_backbone(reference)
        for name, saved in (("flat", state), ("model", {"model": state}), ("module_prefixed", {"state_dict": {"module." + k: v for k, v in state.items()}})):
            protocol4 = os.path.join(tmp, name + "_protocol4.pth")
            torch.save(saved, protocol4, pickle_protocol=4)  # torch's own default is 2, which the restricted loader reads
            rejects(pickle.UnpicklingError, lambda: quiet(lambda: torch.load(protocol4, map_location="cpu")), "Unsupported operand 149")  # this torch's default refuses the file (and says on stderr that 4 is not its protocol) ...
            loaded = omni_from(protocol4).state_dict()  # ... and the container loads it
            assert loaded.keys() == state.keys() and all(torch.equal(loaded[k], v) for k, v in state.items()), name
        no_file = os.path.join(tmp, "no_such_checkpoint.pth")
        rejects(FileNotFoundError, lambda: omni_from(no_file), "No checkpoint found at '{}'".format(no_file))
        print("OK (h7) a Stage A export saved at pickle protocol 4 (flat tensors, as export_backbone writes it), which this torch's default torch.load refuses (Unsupported operand 149, the FRAME opcode), loads through build_omni_model with every tensor equal to the file's, flat or in a 'model' / "
              "'state_dict' wrapper with module.-prefixed keys (prefix stripped); a missing checkpoint raises FileNotFoundError naming the path")

        # CLI errors
        for bad, expected in ((["--tag", "../x"], "letters, digits"), (["--tag", "a b"], "letters, digits"), (["--tag", "ok", "--max_images", "0"], "--max_images must be at least 1"),
                              (["--tag", "ok", "--weights", os.path.join(tmp, "missing.pth")], "is not a file"), (["--tag", "ok", "--datasets", "Shenzhen"], "invalid choice"),
                              (["--tag", "ok", "--datasets", "MIMIC", "MIMIC"], "lists a dataset twice")):
            argv = ["--model", "m", "--weights", weights, "--normalization", "imagenet", "--embeddings_root", root] + bad
            _, _, usage = quiet(lambda: rejects(SystemExit, lambda: extract_embeddings.parse_args(argv)))
            assert expected in usage, (bad, usage)
        # argparse takes an abbreviation for the option it abbreviates: run_emb_gaudi.sh therefore allow-lists what may follow <normalization> instead of listing the options it owns (j)
        abbreviated = extract_embeddings.parse_args(["--tag", "ok", "--model", "m", "--weights", weights, "--normalization", "imagenet", "--emb", os.path.join(tmp, "elsewhere")])
        assert abbreviated.embeddings_root == os.path.join(tmp, "elsewhere")

        # (i) extractor -> runner: the cache the extractor wrote is what the probe reads (paths, meta, shapes) in all four list formats
        run_linearprobe.dataset_entry = lambda name: entries[name]
        try:
            with short_epochs(2):
                out, err = run_fit("tiny", root, res_root, "--runs", "2")
            for name in NAMES:
                result = load_result(res_root, "tiny", name)
                assert result["n"] == sizes and result["model"] == "vit_base_dinov3" and result["normalization"] == "imagenet" and result["runs"] == 2 and result["max_images"] is None
                assert result["classes"] == classes[name] and len(result["per_run"]) == 2 and result["flags"]["rad_dino_pretraining_overlap"] is None and result["warnings"] == []
                assert result["weights"] == {"path": weights, "size_bytes": os.path.getsize(weights), "sha256": hashlib.sha256(open(weights, "rb").read()).hexdigest()}  # the extractor's record of the weights, passed on
                assert result["dropped_columns"] == [] and "pretraining_overlap_train" not in result  # that note is RayDINO's alone
            with short_epochs(2):
                run_fit("smoke", root, res_root, "--datasets", "VinDrCXR", "ChestXray14", "--runs", "1")
            for name in ("VinDrCXR", "ChestXray14"):
                smoke = load_result(res_root, "smoke", name)
                assert smoke["n"] == {"train": 3, "test": 3} and smoke["max_images"] == 3  # the probe used the first 3 entries of each list, as the cache holds
            # RAD-DINO's pretraining data holds all of ChestXray14: the flag follows the backbone, not the dataset alone
            with short_epochs(2):
                run_fit("raddino_smoke", root, res_root, "--datasets", "ChestXray14", "--runs", "1")
            flagged = load_result(res_root, "raddino_smoke", "ChestXray14")
            assert flagged["flags"]["rad_dino_pretraining_overlap"] == "train+val+test" and flagged["model"] == "vit_base_raddino"
        finally:
            run_linearprobe.dataset_entry = stock_entry
        print("OK (i) extractor -> runner on the four list formats: the probe reads the extractor's cache (meta, paths, shapes) for 2 runs, the smoke cache (3 per split) runs the whole pipeline, "
              "and a vit_base_raddino cache is flagged rad_dino_pretraining_overlap=train+val+test on ChestXray14")

        # (i2) RayDINO, a ViT-L/16: 2048-d through the extractor and the runner. The stand-in has the real width (build_omni_model fixes encoder_features=2048 for vit_large_raydino), four blocks and 32 px, and
        # its file is in DINOv2's block_chunks layout, as Meta's is (2 chunks of 2 blocks here): the loader renames the block keys and says so on stderr. Everything after it is 2048 wide
        TINY_RAY = dict(DINOV2_VITL16_KWARGS, img_size=32, depth=4)
        stock_ray_kwargs = models.DINOV2_VITL16_KWARGS
        models.DINOV2_VITL16_KWARGS = TINY_RAY
        try:
            torch.manual_seed(5)
            ray_model = ArkViTDinoV2([], None, False, encoder_features=2048, **dict(TINY_RAY, block_chunks=2)).eval()
            assert "blocks.1.3.norm1.weight" in ray_model.state_dict() and "blocks.3.norm1.weight" not in ray_model.state_dict()  # chunked, <block> over the whole depth
            ray_weights = os.path.join(tmp, "xray_dino_tiny.pth")
            torch.save(ray_model.state_dict(), ray_weights)
            _, out, err = quiet(lambda: extract_embeddings.main(["--tag", "raytiny", "--normalization", "imagenet", "--model", "vit_large_raydino", "--weights", ray_weights] + base_cli[4:]))
            assert err.count("RayDINO checkpoint is in DINOv2's chunked block layout (2 chunks of 2 blocks") == 1, err
            assert err.count("RayDINO checkpoint: ") == 1 and "pos_embed [1, 5, 1024]" in err and "layout 2 chunks, 4 blocks" in err, err  # the file's one-line summary, once, in the job's log...
            assert err.index("RayDINO checkpoint: ") < err.index("chunked block layout")  # ...before the renaming it announces
            ray_transform = dataloader.build_transform_classification(normalize="imagenet", crop_size=32, resize=36, mode="valid")
            for name in NAMES:
                folder = os.path.join(root, "raytiny", name)
                meta = json.load(open(os.path.join(folder, "meta.json")))
                assert meta["model"] == "vit_large_raydino" and meta["feature_dim"] == 2048 and meta["shapes"] == {"train": [7, 2048], "test": [6, 2048]} and meta["weights"]["path"] == ray_weights
                for split in ("train", "test"):
                    cached = np.load(embedding_mode.split_files(folder, split)[0])
                    assert cached.dtype == np.float16 and cached.shape == (sizes[split], 2048), (name, split, cached.shape)
                    with torch.no_grad():  # the chunked model itself: the renamed file must give its function
                        expected = np.concatenate([ray_model.generate_embeddings(ray_transform(Image.open(path).convert("RGB"))[None], after_proj=False).numpy()
                                                   for path in official_dataset(name, entries[name], split).img_list])
                    assert np.allclose(cached, expected.astype(np.float16), atol=2e-3, rtol=2e-3), (name, split, np.abs(cached - expected).max())
            assert re.findall(r"^EMB raytiny (\w+): train 7 / test 6, dim 2048, \d+s$", out, re.M) == list(NAMES) and out.count("\n") == 4
            run_linearprobe.dataset_entry = lambda name: entries[name]
            try:
                with short_epochs(2):
                    run_fit("raytiny", root, res_root, "--runs", "2")
                # a cache whose arrays are not as wide as its meta says is refused by the dataset module, naming both widths
                clone_cache(root, "raytiny", "raywrong", "MIMIC", feature_dim=1536)
                rejects(ValueError, lambda: run_fit("raywrong", root, res_root, "--datasets", "MIMIC", "--runs", "1"), "float16 [7, 2048] (float16 [7, 1536] expected)")
            finally:
                run_linearprobe.dataset_entry = stock_entry
            # RayDINO's corpus holds the TRAIN splits of ChestXray14, CheXpert and MIMIC (its Table 2) and none of their test lists, and not VinDr-CXR: no test-set overlap, so the dagger flag stays null for
            # all four datasets; the train-split overlap is the JSON's separate, informational field
            ray_note = "CheXpert, MIMIC-CXR, ChestX-ray14 training splits (RayDINO Table 2); VinDr-CXR not in its corpus"
            for name in NAMES:
                result = load_result(res_root, "raytiny", name)
                assert result["model"] == "vit_large_raydino" and result["n"] == sizes and result["runs"] == 2 and "float64 nn.Linear on the 2048-d embedding" in result["protocol"], result["protocol"]
                assert result["flags"] == {"selection": "best epoch on test (as shipped)", "rad_dino_pretraining_overlap": None}
                assert result["pretraining_overlap_train"] == ray_note and list(result)[-2:] == ["pretraining_overlap_train", "elapsed_s"]
                _, run_dir = script_dir(res_root, "raytiny", name)
                ckpt = torch.load(os.path.join(run_dir, "run_0", "ckpt.pth"), map_location="cpu", weights_only=False)
                assert tuple(ckpt["model"]["model.0.weight"].shape) == (len(entries[name]["diseases"]), 2048), (name, ckpt["model"]["model.0.weight"].shape)  # the probe's head takes the cache's width
            assert run_linearprobe.raydino_train_overlap({"model": "vit_large_raydino"}) == ray_note
            assert all(run_linearprobe.raydino_train_overlap({"model": m}) is None for m in ("vit_base_dinov3", "vit_base_dinov2", "vit_base_raddino"))  # the model name decides, not a tag
            assert [run_linearprobe.rad_dino_flag({"model": "vit_large_raydino"}, "raydino", d) for d in NAMES] == [None] * 4  # RayDINO is not RAD-DINO
        finally:
            models.DINOV2_VITL16_KWARGS = stock_ray_kwargs
        print("OK (i2) RayDINO: a vit_large_raydino file in DINOv2's block_chunks layout is summarised (one line: key count, non-block keys with shapes, first and last block key, layout) and renamed on load (one notice), both on stderr, the extractor writes float16 (N, 2048) caches equal to the chunked model's embeddings with "
              "feature_dim 2048 in every meta.json and 'dim 2048' on every EMB line, the runner fits a 2048-wide probe head (protocol text, ckpt weight shape) for 2 runs on all four datasets; the JSON carries "
              "pretraining_overlap_train, its flags.rad_dino_pretraining_overlap stays null (no test-set overlap), and other models carry no such field; a meta that disagrees with its arrays is refused")

        # (i3) the Ark+ Swin teacher: a swin_* name loads checkpoint['teacher'], the lab's convention for the encoder of an Ark checkpoint, never checkpoint['state_dict'] (the student, which build_omni_model would read into a Swin with no
        # key guard), and embeds before the projector: 1024-d for swin_base. The stand-in is the lab's own model in its checkpoint layout (every key under 'module.', the teacher beside the student) with a projector and two heads, at
        # swin_base's width, window and input size (embed_dim 128, window 7, 224 px) but TWO blocks a stage instead of (2, 2, 18, 2): the checks read the width, the window, the input size and the key names, never the depth, and a
        # full Swin-B is ~90 M weights to initialise and (i3) builds a model many times over. The loader builds its probe with the depths as written, so the Swin class it builds is swapped for one that replaces them, for all of (i3): every model built
        # below (the probes and the shape-only builds too) is the shallow one. Delete ShallowSwin and the two lines that set and restore it for the real depths
        stock_swin, threads = models.ArkSwinTransformer, torch.get_num_threads()

        class ShallowSwin(stock_swin):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **dict(kwargs, depths=(2, 2, 2, 2)))

        models.ArkSwinTransformer = ShallowSwin
        torch.set_num_threads(DEFAULT_THREADS)  # the runner (g) left torch on one thread
        try:
            SWIN_BASE = dict(patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))  # swin_base as build_omni_model builds it
            torch.manual_seed(21)
            swin_teacher = models.ArkSwinTransformer([14, 3], 1376, False, **SWIN_BASE).eval()
            torch.manual_seed(22)
            swin_student = models.ArkSwinTransformer([14, 3], 1376, False, **SWIN_BASE).eval()

            def save_ark(path, teacher_state, **others):
                """The lab's checkpoint layout around `teacher_state` (a plain state dict): every key under 'module.' (DataParallel), plus `others` (e.g. state_dict=<the student's keys under 'module.'>)."""
                torch.save(dict({"epoch": 50, "teacher": {"module." + k: v for k, v in teacher_state.items()}, "optimizer": {}}, **others), path)

            ark_path = os.path.join(tmp, "ark_tiny.pth.tar")
            save_ark(ark_path, swin_teacher.state_dict(), state_dict={"module." + k: v for k, v in swin_student.state_dict().items()})
            swin_transform = dataloader.build_transform_classification(normalize="imagenet", crop_size=224, resize=256, mode="valid")

            def swin_rows(model, split, limit=None):
                """Each ChestXray14 image through the official 'valid' transform at 224 and `model` on its own, before the projector: [N, 1024]."""
                with torch.no_grad():
                    return np.concatenate([model.generate_embeddings(swin_transform(Image.open(path).convert("RGB"))[None], after_proj=False).numpy()
                                           for path in official_dataset("ChestXray14", entries["ChestXray14"], split).img_list[:limit]])

            def swin_main(tag, size="224", resize="256"):
                return quiet(lambda: extract_embeddings.main(["--tag", tag, "--normalization", "imagenet", "--model", "swin_base", "--weights", ark_path, "--embeddings_root", root, "--input_size", size, "--img_resize", resize,
                                                              "--batch_size", "4", "--workers", "0", "--device", "cpu", "--datasets", "ChestXray14"]))

            _, out, err = swin_main("arkswin")
            swin_folder = os.path.join(root, "arkswin", "ChestXray14")
            meta = json.load(open(os.path.join(swin_folder, "meta.json")))
            assert meta["model"] == "swin_base" and meta["feature_dim"] == 1024 and meta["shapes"] == {"train": [7, 1024], "test": [6, 1024]} and meta["weights"]["path"] == ark_path, meta
            assert meta["transform"]["crop"] == 224 and meta["transform"]["resize"] == 256
            for split in ("train", "test"):
                cached = np.load(embedding_mode.split_files(swin_folder, split)[0])
                assert cached.dtype == np.float16 and cached.shape == (sizes[split], 1024), (split, cached.shape)
                expected = swin_rows(swin_teacher, split)
                assert np.allclose(cached, expected.astype(np.float16), atol=2e-3, rtol=2e-3), (split, np.abs(cached - expected).max())  # the teacher's pooled vector, before its 1376-d projector
                assert not np.allclose(cached[:2], swin_rows(swin_student, split, 2).astype(np.float16), atol=2e-3, rtol=2e-3)  # and not the student's
            assert re.findall(r"^EMB arkswin (\w+): train 7 / test 6, dim 1024, \d+s$", out, re.M) == ["ChestXray14"] and out.count("\n") == 1, out
            assert err.count("loaded into --model swin_base") == 1 and "saved no attn_mask" not in err, err  # the summary line, once, in the job's log
            for text in ("key 'teacher'", "embed_dim 128, window 7, input 224 px", "not loaded (the probe builds no projector or heads): 6 keys"):  # projector.{weight, bias} and omni_heads.{0, 1}.{weight, bias}
                assert text in err, (text, err)

            # refused: the file's architecture is not the model's (the message gives the file's tensors and the model that fits), a Swin-B at 448 px that no model name builds, a block weight gone, no teacher
            load_swin = lambda model, path=ark_path: quiet(lambda: extract_embeddings.load_backbone(model, path))
            message = rejects(RuntimeError, lambda: load_swin("swin_large"), "embed_dim 128, window 7, input 224 px")
            assert "Use --model swin_base with a 224 px crop (run_emb_gaudi.sh: ARK_EMB_INPUT_SIZE=224)" in message and "--model swin_large builds embed_dim 192, window 7, input 224 px" in message, message
            assert "patch_embed.proj.weight [128, 3, 4, 4], relative_position_bias_table [169, 4], layers.0.blocks.1.attn_mask [64, 49, 49]" in message and message.startswith("Ark checkpoint {} ('teacher') does not fit --model swin_large".format(ark_path)), message
            other = os.path.join(tmp, "ark_other.pth.tar")  # rewritten for each case below: the temp dir holds one checkpoint beside ark_tiny's
            torch.manual_seed(23)
            save_ark(other, models.ArkSwinTransformer([14, 3], 1376, False, img_size=448, **SWIN_BASE).state_dict())
            message = rejects(RuntimeError, lambda: load_swin("swin_base", other), "input 448 px")
            assert "No model name here builds that architecture (swin_base 128/7/224, swin_large 192/7/224, swin_large_384 192/12/384, swin_large_768 192/12/768)" in message and "builds embed_dim 128, window 7, input 224 px" in message, message
            gone = "layers.2.blocks.1.mlp.fc1.weight"
            save_ark(other, {k: v for k, v in swin_teacher.state_dict().items() if k != gone})
            message = rejects(RuntimeError, lambda: load_swin("swin_base", other), "missing keys")
            assert "missing keys ['{}'] and unexpected keys []".format(gone) in message, message  # the projector and the heads the probe does not build are not unexpected keys
            torch.save(swin_teacher.state_dict(), other)  # a plain state dict: no 'teacher' (nor 'state_dict') to take
            message = rejects(KeyError, lambda: load_swin("swin_base", other), "has no 'teacher' entry")
            assert "(top-level keys: ['patch_embed.proj.weight', " in message and "): not an Ark checkpoint" in message, message
            torch.save({"epoch": 1, "state_dict": {"a": torch.zeros(1)}}, other)  # a student only
            assert "(top-level keys: ['epoch', 'state_dict'])" in rejects(KeyError, lambda: load_swin("swin_base", other), "has no 'teacher' entry")
            torch.save(torch.zeros(1), other)  # not even a dict
            rejects(KeyError, lambda: load_swin("swin_base", other), "has no 'teacher' entry")

            # two refusals that were a bare traceback from inside the loader: a model name it does not build (a typo for swin_large_768 here: UnboundLocalError) and a ViT's file under a swin name (a KeyError in swin_architecture)
            rejects(ValueError, lambda: load_swin("swin_large_1152"), "--model swin_large_1152 is not built by build_omni_model_from_checkpoint (it builds swin_base, swin_large, swin_large_384, swin_large_768, conv_base, vit_base_dinov3, vit_large_dinov3, vit_huge_plus_dinov3, vit_7b_dinov3, vit_base_dinov2, vit_base_raddino, vit_large_raydino)")
            save_ark(other, ArkViT([], None, False, **TINY).state_dict())  # a ViT-shaped teacher: patch_embed.proj.weight and blocks.*, no relative_position_bias_table
            message = rejects(RuntimeError, lambda: load_swin("swin_base", other), "is not a Swin state dict")
            assert message == "Ark checkpoint {} ('teacher') is not a Swin state dict: it has no layers.0.blocks.0.attn.relative_position_bias_table; --model swin_base is a Swin".format(other), message

            # a file with no attn_mask buffers (derived: the model keeps its own) loads, with its teacher's weights, but its input size cannot be checked: said on stderr
            save_ark(other, {k: v for k, v in swin_teacher.state_dict().items() if not k.endswith("attn_mask")})
            loaded, _, err = load_swin("swin_base", other)
            assert "saved no attn_mask buffer" in err and "--model swin_base builds 224 px, unverified" in err and "input not saved px" in err and err.count("loaded into --model swin_base") == 1, err
            assert torch.equal(loaded.patch_embed.proj.weight, swin_teacher.patch_embed.proj.weight) and not torch.equal(loaded.patch_embed.proj.weight, swin_student.patch_embed.proj.weight)

            # a Swin takes the one input size it is built for: a 32 px crop is refused once the model is loaded and before any output directory is made (timm's PatchEmbed would assert at the first batch)
            rejects(ValueError, lambda: swin_main("arkbad", "32", "36"), "--model swin_base takes 224x224 input, but the crop is 32 px: set ARK_EMB_INPUT_SIZE=224 (run_emb_gaudi.sh), i.e. --input_size 224")
            assert not os.path.exists(os.path.join(root, "arkbad"))

            # the shapes behind the checks, from the models themselves, no forward: width, input size and what swin_architecture reads off the state dict
            for model_name, features, size, triple in (("swin_large_768", 1536, 768, (192, 12, 768)), ("swin_large_384", 1536, 384, (192, 12, 384)), ("swin_base", 1024, 224, (128, 7, 224))):
                built = models.build_omni_model(SimpleNamespace(model_name=model_name, projector_features=None, use_mlp=False, pretrained_weights=None), [])
                assert built.num_features == features and tuple(built.patch_embed.img_size) == (size, size), model_name
                assert models.swin_architecture(built.state_dict()) == triple and models.SWIN_ARCHITECTURES[triple] == model_name and embedding_mode.embedding_dim(model_name) == features, model_name
            assert models.swin_architecture({k: v for k, v in built.state_dict().items() if k != "layers.0.blocks.1.attn_mask"}) == (128, 7, None)  # no mask saved: no input size
            fake = {"patch_embed.proj.weight": torch.zeros(128, 3, 4, 4), "layers.0.blocks.0.attn.relative_position_bias_table": torch.zeros(169, 4)}
            assert models.swin_architecture(dict(fake, **{"layers.0.blocks.1.attn_mask": torch.zeros(64, 49, 49)})) == (128, 7, 224)
            assert models.swin_architecture(dict(fake, **{"layers.0.blocks.1.attn_mask": torch.zeros(63, 49, 49)})) == (128, 7, None)  # 63 windows are not a square grid: no input size either
        finally:
            models.ArkSwinTransformer = stock_swin
            torch.set_num_threads(threads)
        print("OK (i3) Ark+ Swin teacher: swin_base loads checkpoint['teacher'] (not the student's state_dict) from a lab-layout .pth.tar and the extractor writes float16 (N, 1024) caches equal to the teacher's pooled vector before "
              "its projector, feature_dim 1024, 'dim 1024' on the EMB line, one summary line on stderr (key, tensors, embed_dim / window / input, the 6 projector and head keys not loaded); refused: the same file as swin_large "
              "(names the file's tensors and --model swin_base at a 224 px crop), a 448 px Swin-B (no model name), a missing block weight, a plain state dict, a student-only file and a non-dict (no 'teacher'), swin_large_1152 (a model name the loader does not build: the message lists the ones it does), a ViT's file under swin_base (no Swin tensors: the message names the missing key), a 32 px crop for "
              "swin_base (before any directory); a file with no attn_mask loads with an 'unverified' warning; swin_large_768 / swin_large_384 / swin_base build 1536 / 1536 / 1024-d at 768 / 384 / 224 px and swin_architecture reads "
              "(192, 12, 768) / (192, 12, 384) / (128, 7, 224) off their state dicts, with no input size when the mask is gone or its window count is not a square (all at two blocks a stage)")
finally:
    models.DINOV3_VITB16_KWARGS, extract_embeddings.dataset_entry = stock_kwargs, stock_extractor_entry

# (j) run_emb_gaudi.sh ------------------------------------------------------------------------------------------------------
WRAPPER = os.path.join(REPO, "run_emb_gaudi.sh")
subprocess.run(["bash", "-n", WRAPPER], check=True)
assert os.access(WRAPPER, os.X_OK)
wrapper_text = open(WRAPPER).read()
assert wrapper_text.startswith("#!/usr/bin/env bash\n")
assert re.findall(r"^#SBATCH (.+)$", wrapper_text, re.M) == ["--job-name=b2-emb", "-A grp_jliang12", "-p gaudi", "-q public", "-t 08:00:00", "-N 1", "--gres=gpu:hl225:1", "-c 36", "--mem=128G"]
assert "export PT_HPU_LAZY_MODE=1" in wrapper_text and "/scratch/$USER/2026Fall-CSE507-Ark-Container" in wrapper_text and "./gaudi_check_cards.sh || exit 1" in wrapper_text.splitlines()
STOCK = "/workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
STOCK_HOST = "weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
STUB = """#!/usr/bin/env bash
printf 'OMP=%s %s\\n' "${OMP_NUM_THREADS-unset}" "$*" >> "$STUB_LOG"
case "$*" in
  *extract_embeddings*) exit "${EXTRACT_EXIT:-0}" ;;
  *run_linearprobe*)
    dataset=$(printf '%s' "$*" | sed -n 's/.*--datasets \\([A-Za-z0-9]*\\).*/\\1/p')
    echo "PROBE tag $dataset: stub"
    [[ "$dataset" == "${FAIL_FIT:-}" ]] && exit 1
    exit 0 ;;
esac
exit 99
"""
NVIDIA_STUB = STUB.replace("printf 'OMP=", "printf 'nvidia.sh OMP=")  # the same stub with each call logged behind "nvidia.sh ", so a wrapper that reaches the wrong launcher shows in its calls
assert NVIDIA_STUB != STUB
USAGE = "Usage: sbatch run_emb_gaudi.sh|run_emb_class.sh <tag: letters, digits, . _ - only> <model> <in-container weights path under /workspace> <imagenet|chestx-ray|none|raddino> [--max_images N] [--datasets NAME...] [--workers N]"  # both print it: $0 is Slurm's spool copy under sbatch, so neither can name itself


def run_wrapper(args, files=(STOCK_HOST,), card_check_exit=0, content="x", wrapper=WRAPPER, snapshot=False, **env):
    """`wrapper` (run_emb_gaudi.sh unless given) from a different directory, ARK_ROOT = a fake repo whose gaudi.sh and nvidia.sh log each call (one line: OMP_NUM_THREADS, then the arguments; nvidia.sh's behind
    "nvidia.sh ") and whose gaudi_check_cards.sh records a call. A `files` entry is a path, or (path, content). Returns (completed process, the launchers' calls or None, card check ran, results/<tag> listing or None),
    and with snapshot=True a fifth item: the fake repo afterwards as {path: content}, the stubs and their log left out. The ARK_EMB_* / PROBE_* / SLURM_* variables of the caller's own environment do not reach the wrapper."""
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as elsewhere:
        for name, body in (("gaudi.sh", STUB), ("nvidia.sh", NVIDIA_STUB), ("gaudi_check_cards.sh", "echo CARD_CHECK; exit {}".format(card_check_exit))):
            with open(os.path.join(root, name), "w") as f:
                f.write(body if body.startswith("#!") else "#!/usr/bin/env bash\n" + body + "\n")
            os.chmod(os.path.join(root, name), 0o755)
        for entry in files:
            name, body = entry if isinstance(entry, tuple) else (entry, content)
            os.makedirs(os.path.join(root, os.path.dirname(name)), exist_ok=True)
            with open(os.path.join(root, name), "w") as f:
                f.write(body)
        log = os.path.join(root, "stub.log")
        environment = {k: v for k, v in os.environ.items() if not k.startswith(("ARK_EMB_", "PROBE_", "SLURM_"))}
        done = subprocess.run(["bash", wrapper] + list(args), cwd=elsewhere, env=dict(environment, ARK_ROOT=root, STUB_LOG=log, **env), capture_output=True, text=True)
        calls = open(log).read().splitlines() if os.path.exists(log) else None
        results = os.path.join(root, "results", args[0] if args else "")
        found = (done, calls, "CARD_CHECK" in done.stdout, sorted(os.listdir(results)) if os.path.isdir(results) else None)
        if not snapshot:
            return found
        tree = {os.path.relpath(os.path.join(folder, name), root): open(os.path.join(folder, name)).read() for folder, _, names in os.walk(root) for name in names}
        return found + ({path: body for path, body in tree.items() if path not in ("gaudi.sh", "nvidia.sh", "gaudi_check_cards.sh", "stub.log")},)


done, calls, card_check, results = run_wrapper(["stock", "vit_base_dinov3", STOCK, "imagenet"])
assert done.returncode == 0 and done.stderr == "" and card_check, done
extraction = "OMP=unset python -m ark_finetune.extract_embeddings --device hpu --tag stock --model vit_base_dinov3 --weights {} --normalization imagenet --workers 32".format(STOCK)
assert calls[0] == extraction and len(calls) == 5, calls  # the extraction first, alone; then the fits
assert sorted(calls[1:]) == sorted("OMP=1 python -m ark_linearprobe.run_linearprobe --tag stock --datasets {}".format(d) for d in NAMES), calls  # four processes, one thread each
assert results == sorted(d + ".log" for d in NAMES) and sorted(l for l in done.stdout.splitlines() if l.startswith(("fit ", "PROBE "))) == \
    sorted(["fit {}: done".format(d) for d in NAMES] + ["PROBE tag {}: stub".format(d) for d in NAMES]), (results, done.stdout)
expanded = "./gaudi.sh " + extraction[len("OMP=unset "):]
smoke_command = "sbatch run_emb_gaudi.sh stock_smoke vit_base_dinov3 {} imagenet --max_images 512".format(STOCK)
# the extras (Codex r3, P2): after <normalization> only --max_images, --datasets and --workers may follow; they reach the extractor as they are and last (argparse: the last occurrence wins), and only the extractor
prefix = "OMP=unset python -m ark_finetune.extract_embeddings --device hpu --tag stock_smoke --model vit_base_dinov3 --weights {} --normalization imagenet --workers 32".format(STOCK)
for extra, passed in ((["--max_images", "512", "--workers", "16"], "--max_images 512 --workers 16"),
                      (["--datasets", "MIMIC", "CheXpert", "--max_images", "5"], "--datasets MIMIC CheXpert --max_images 5"),
                      (["--max_images=5", "--workers=4", "--datasets=MIMIC"], "--max_images=5 --workers=4 --datasets=MIMIC")):
    done, calls, _, _ = run_wrapper(["stock_smoke", "vit_base_dinov3", STOCK, "imagenet"] + extra)
    assert done.returncode == 0 and calls[0] == prefix + " " + passed, calls
    assert all(c.endswith("--datasets " + d) for c, d in zip(sorted(calls[1:]), sorted(NAMES))) and all("--max_images" not in c for c in calls[1:]), calls
# a model name is syntax-checked and nothing more: the builders of B1 (vit_base_dinov2, vit_base_raddino) need no change here
done, calls, _, _ = run_wrapper(["rad", "vit_base_raddino", "/workspace/weights/rad_dino_vitb14_pretrain.pth", "raddino"], files=("weights/rad_dino_vitb14_pretrain.pth",))
assert done.returncode == 0 and "--model vit_base_raddino --weights /workspace/weights/rad_dino_vitb14_pretrain.pth --normalization raddino" in calls[0], calls
# the RayDINO submission line, as pasted at the terminal (Meta's file name kept): the wrapper's tag / model / path / normalization checks accept it, and the extractor gets exactly those arguments, then the four fits
RAY_SUBMISSION = "sbatch run_emb_gaudi.sh raydino vit_large_raydino /workspace/weights/xray_dino_vitl16_pretrained-ad31c2b0.pth imagenet"
assert RAY_SUBMISSION in wrapper_text  # and it is the example in the wrapper's header
done, calls, card_check, results = run_wrapper(RAY_SUBMISSION.split()[2:], files=("weights/xray_dino_vitl16_pretrained-ad31c2b0.pth",))
assert done.returncode == 0 and done.stderr == "" and card_check and results == sorted(d + ".log" for d in NAMES), done
assert calls[0] == "OMP=unset python -m ark_finetune.extract_embeddings --device hpu --tag raydino --model vit_large_raydino --weights /workspace/weights/xray_dino_vitl16_pretrained-ad31c2b0.pth --normalization imagenet --workers 32", calls
assert sorted(calls[1:]) == sorted("OMP=1 python -m ark_linearprobe.run_linearprobe --tag raydino --datasets {}".format(d) for d in NAMES), calls

# failures: weights missing or empty -> exit 1 before the card check; a failed card check -> exit 1, nothing runs; the extraction's status ends the job, with no fit
for kwargs in (dict(files=()), dict(files=("weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth.partial",)), dict(content="")):
    done, calls, card_check, results = run_wrapper(["stock", "vit_base_dinov3", STOCK, "imagenet"], **kwargs)
    assert done.returncode == 1 and calls is None and not card_check and results is None, (kwargs, done)
    assert "ERROR: weights not found at ./weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth (host path for " + STOCK + ")" in done.stderr, done.stderr
done, calls, card_check, results = run_wrapper(["stock", "vit_base_dinov3", STOCK, "imagenet"], card_check_exit=1)
assert done.returncode == 1 and card_check and calls is None and results is None, done  # fail closed
done, calls, _, results = run_wrapper(["stock", "vit_base_dinov3", STOCK, "imagenet"], EXTRACT_EXIT="7")
assert done.returncode == 7 and calls == [extraction] and results is None, (done, calls)  # the extractor's status is the job's; no fit on a failed extraction
done, calls, _, results = run_wrapper(["stock", "vit_base_dinov3", STOCK, "imagenet"], FAIL_FIT="MIMIC")
assert done.returncode == 1 and len(calls) == 5 and "ERROR: fit MIMIC failed (see results/stock/MIMIC.log)" in done.stderr and "fit ChestXray14: done" in done.stdout, (done, calls)  # the other three finish

# the preflight (Codex r1, P2): results that already exist for the tag stop the job BEFORE the card check and the extraction, which would otherwise spend the allocation and then be refused by the fit.
# It checks the very paths that run_linearprobe refuses (results_of), for every dataset the fit covers; the runner's own entry point refuses them too (g4)
args = ["stock", "vit_base_dinov3", STOCK, "imagenet"]
for name in NAMES:
    for blocking in run_linearprobe.results_of("results", "stock", name):
        done, calls, card_check, results = run_wrapper(args, files=(STOCK_HOST, os.path.join(blocking, "log.txt") if not blocking.endswith(".json") else blocking))
        assert done.returncode == 1 and calls is None and not card_check, (blocking, done)  # nothing started: no extraction, not even the card check
        assert "ERROR: results already exist for tag stock: {}; refusing to start the extraction".format(blocking) in done.stderr and done.stdout == "", (blocking, done.stderr)
done, calls, card_check, results = run_wrapper(args, files=(STOCK_HOST, "results/stock/CheXpert.json", "results/stock/probe/VinDrCXR/log.txt"))  # several: all are named
assert done.returncode == 1 and calls is None and "results/stock/CheXpert.json results/stock/probe/VinDrCXR" in done.stderr, done.stderr
for files in ((STOCK_HOST, "results/stock/MIMIC.log"),  # the wrapper's own redirect target from an earlier attempt is not a result
              (STOCK_HOST, "results/other/ChestXray14.json", "results/other/probe/MIMIC/log.txt"),  # nor is another tag's
              (STOCK_HOST, "results/stock2/CheXpert.json")):  # nor a tag that merely starts with this one
    done, calls, card_check, results = run_wrapper(args, files=files)
    assert done.returncode == 0 and card_check and len(calls) == 5, (files, done)
# everything else is refused with exit 2 before ANYTHING runs, the weights and results checks and the card check included: the wrapper's own options (--tag --model --weights --normalization --device), the
# ones that move the cache or the results (--embeddings_root --results_root), the ones that change the embeddings (--input_size --img_resize --batch_size), abbreviations that argparse would resolve to
# them (--emb, --ta, --max_image), -h / --help (which would exit 0 and start the fit on nothing), and anything unknown; the message names every refused argument, in order
base = ["stock", "vit_base_dinov3", STOCK, "imagenet"]
for extra, named in ((["--tag", "other"], "--tag"), (["--tag=other"], "--tag=other"), (["--model", "vit_base_dinov2"], "--model"), (["--weights", "/workspace/weights/x.pth"], "--weights"),
                     (["--normalization", "none"], "--normalization"), (["--device", "cpu"], "--device"), (["--device=cuda"], "--device=cuda"),
                     (["--embeddings_root", "/tmp/e"], "--embeddings_root"), (["--embeddings_root=/tmp/e"], "--embeddings_root=/tmp/e"), (["--results_root", "/tmp/r"], "--results_root"),
                     (["--input_size", "518"], "--input_size"), (["--img_resize", "512"], "--img_resize"), (["--batch_size", "128"], "--batch_size"),
                     (["--emb", "/tmp/e"], "--emb"), (["--ta", "x"], "--ta"), (["--max_image", "5"], "--max_image"), (["--workers"] + ["--help"], "--help"), (["-h"], "-h"), (["--"], "--"), (["-5"], "-5"),
                     (["--note", "two words"], "--note"), (["--max_images 5"], "--max_images 5"), (["--workers="], "--workers="),
                     (["--max_images", "5", "--tag", "x"], "--tag"), (["--datasets", "MIMIC", "--device", "cuda", "--workers", "4"], "--device"),
                     (["--tag", "x", "--workers", "4", "--batch_size", "8", "--embeddings_root", "/tmp/e"], "--tag --batch_size --embeddings_root")):
    for files in ((STOCK_HOST,), (), (STOCK_HOST, "results/stock/MIMIC.json")):  # the weights present, missing, or results that exist: refused first all the same
        done, calls, card_check, results = run_wrapper(base + extra, files=files)
        assert done.returncode == 2 and calls is None and not card_check and done.stdout == "", (extra, files, done)
        assert results == (["MIMIC.json"] if len(files) == 2 else None), (extra, files, results)  # nothing was added to results/<tag>
        assert done.stderr.startswith("ERROR: {}: not accepted after <normalization>.".format(named)) and "Only --max_images N, --datasets NAME... and --workers N may follow" in done.stderr, (extra, done.stderr)
# ARK_EMB_INPUT_SIZE: the extractor's --input_size and --img_resize = round(size x 8/7), after --workers 32 and before the extras; unset or empty, neither is
# passed (the extractor's 224 / 256, the argv above unchanged); anything but a positive integer of at most 4 digits exits 2 before anything runs
for size, resize in (("224", "256"), ("448", "512"), ("512", "585"), ("768", "878")):
    assert resize == str(round(int(size) * 8 / 7))
    done, calls, card_check, _ = run_wrapper(["stock", "vit_base_dinov3", STOCK, "imagenet", "--max_images", "5"], ARK_EMB_INPUT_SIZE=size)
    assert done.returncode == 0 and card_check and calls[0] == extraction + " --input_size {} --img_resize {} --max_images 5".format(size, resize), (size, calls)
    assert all("--input_size" not in c for c in calls[1:]), calls  # the fit reads the size from meta.json, not from an argument
    emb_size_command = "./gaudi.sh " + calls[0][len("OMP=unset "):]
done, calls, _, _ = run_wrapper(["stock", "vit_base_dinov3", STOCK, "imagenet"], ARK_EMB_INPUT_SIZE="")
assert done.returncode == 0 and calls[0] == extraction, calls
for bad in ("0", "-512", "512px", "5 12", "12345", "0512"):
    done, calls, card_check, results = run_wrapper(["stock", "vit_base_dinov3", STOCK, "imagenet"], ARK_EMB_INPUT_SIZE=bad)
    assert done.returncode == 2 and "ARK_EMB_INPUT_SIZE" in done.stderr and calls is None and not card_check and results is None, (bad, done)
print("OK (j) ARK_EMB_INPUT_SIZE: 224 / 448 / 512 / 768 -> --input_size N --img_resize round(N x 8/7) (256 / 512 / 585 / 878); unset or empty -> neither "
      "(224 / 256); 0, negative, non-numeric, 5 digits, a leading 0 -> exit 2 before anything runs; 512: " + emb_size_command)
# ARK_EMB_NO_FIT: exactly 1 = extraction only (the extraction is the job's one gaudi.sh call, a stdout line says so, exit 0, results/<tag> never made); unset or empty = the extraction and the four fits as before; any other value
# exits 2 before anything runs, the weights check, the results preflight and the card check included. The results preflight still runs for a job that does not fit, and a failed extraction still ends it with its own status
def no_fit_line(tag):
    """The line both extraction launchers print once an ARK_EMB_NO_FIT=1 extraction has succeeded: it names the fit launcher (CPU only) and keeps the one-job way of finishing a per-dataset split."""
    return ("ARK_EMB_NO_FIT=1: extraction only, no fit for tag {0}; fit it with: sbatch run_emb_fit_class.sh {0} [--datasets NAME...] (CPU only, needs the caches it fits), "
            "or the job that extracts the last dataset of this tag without ARK_EMB_NO_FIT fits all four").format(tag)


NO_FIT_LINE = no_fit_line("stock")
done, calls, card_check, results = run_wrapper(base, ARK_EMB_NO_FIT="1")
assert done.returncode == 0 and done.stderr == "" and card_check and calls == [extraction] and results is None, (done, calls, results)
assert done.stdout.splitlines() == ["CARD_CHECK", NO_FIT_LINE], done.stdout
done, calls, card_check, results = run_wrapper(base, ARK_EMB_NO_FIT="")
assert done.returncode == 0 and card_check and calls[0] == extraction and len(calls) == 5 and results == sorted(d + ".log" for d in NAMES) and "extraction only" not in done.stdout, (done, calls)
for bad in ("0", "yes", "1 ", "true", "11"):
    for files in ((STOCK_HOST,), (), (STOCK_HOST, "results/stock/MIMIC.json")):  # the weights present, missing, or results that exist: refused first all the same
        done, calls, card_check, results = run_wrapper(base, files=files, ARK_EMB_NO_FIT=bad)
        assert done.returncode == 2 and done.stderr == "ERROR: ARK_EMB_NO_FIT='{}' must be unset, empty or 1\n".format(bad) and done.stdout == "" and calls is None and not card_check, (bad, files, done)
        assert results == (["MIMIC.json"] if len(files) == 2 else None), (bad, files, results)
done, calls, card_check, results = run_wrapper(base, files=(STOCK_HOST, "results/stock/CheXpert.json"), ARK_EMB_NO_FIT="1")  # the preflight still runs
assert done.returncode == 1 and calls is None and not card_check and "ERROR: results already exist for tag stock: results/stock/CheXpert.json; refusing to start the extraction" in done.stderr, done
done, calls, _, results = run_wrapper(base, EXTRACT_EXIT="7", ARK_EMB_NO_FIT="1")  # a failed extraction ends the job with its status, not as "extraction only"
assert done.returncode == 7 and calls == [extraction] and "extraction only" not in done.stdout and results is None, (done, calls)
# the Ark+ Swin-L submission of the header, split by dataset: a job that only extracts MIMIC, then the last dataset's job without the switch, which extracts VinDrCXR and fits all four
ARK_WEIGHTS = "/workspace/weights/Ark6_swinLarge768_ep50.pth.tar"
ARK_SUBMISSION = "ARK_EMB_INPUT_SIZE=768 sbatch -t 1-00:00:00 run_emb_gaudi.sh arkplus6 swin_large_768 {} imagenet".format(ARK_WEIGHTS)
assert ARK_SUBMISSION in wrapper_text  # the Ark+ example in the wrapper's header
ark_extraction = "OMP=unset python -m ark_finetune.extract_embeddings --device hpu --tag arkplus6 --model swin_large_768 --weights {} --normalization imagenet --workers 32 --input_size 768 --img_resize 878".format(ARK_WEIGHTS)
ark_args, ark_files = ["arkplus6", "swin_large_768", ARK_WEIGHTS, "imagenet"], ("weights/Ark6_swinLarge768_ep50.pth.tar",)
done, calls, card_check, results = run_wrapper(ark_args + ["--datasets", "MIMIC"], files=ark_files, ARK_EMB_INPUT_SIZE="768", ARK_EMB_NO_FIT="1")
assert done.returncode == 0 and card_check and calls == [ark_extraction + " --datasets MIMIC"] and results is None, (done, calls)
done, calls, _, results = run_wrapper(ark_args + ["--datasets", "VinDrCXR"], files=ark_files, ARK_EMB_INPUT_SIZE="768")
assert done.returncode == 0 and calls[0] == ark_extraction + " --datasets VinDrCXR" and sorted(calls[1:]) == sorted("OMP=1 python -m ark_linearprobe.run_linearprobe --tag arkplus6 --datasets {}".format(d) for d in NAMES), calls
print("OK (j) ARK_EMB_NO_FIT: 1 = the extraction is the only gaudi.sh call, the 'extraction only' line names the tag, exit 0, no results/<tag>; unset or empty = the extraction and the four fits (5 calls); 0, yes, '1 ', true, 11 -> exit 2 naming "
      "the value before the weights check, the preflight and the card check; the preflight still stops a job that does not fit (exit 1), a failed extraction ends it with its status (7); the Ark+ line expands to: " + ark_extraction +
      " --datasets MIMIC (switch on), and the last dataset's job without it runs the extraction and the four fits")
# usage: <tag> <model> <path under /workspace> <normalization>, each checked, exit 2, extra arguments do not bypass the checks
for bad in ([], ["stock"], ["stock", "vit_base_dinov3", STOCK], ["../x", "vit_base_dinov3", STOCK, "imagenet"], ["a b", "vit_base_dinov3", STOCK, "imagenet"], ["-x", "vit_base_dinov3", STOCK, "imagenet"],
            ["", "vit_base_dinov3", STOCK, "imagenet"], ["stock", "vit base", STOCK, "imagenet"], ["stock", "", STOCK, "imagenet"], ["stock", "../m", STOCK, "imagenet"],
            ["stock", "vit_base_dinov3", "weights/x.pth", "imagenet"], ["stock", "vit_base_dinov3", "/data/x.pth", "imagenet"], ["stock", "vit_base_dinov3", STOCK, "caffe"],
            ["stock", "vit_base_dinov3", STOCK, ""], ["stock", "vit_base_dinov3", "weights/x.pth", "imagenet", "--max_images", "5"], ["--max_images", "5"]):
    done, calls, card_check, results = run_wrapper(bad)
    assert done.returncode == 2 and done.stderr == USAGE + "\n" and done.stdout == "" and calls is None and not card_check and results is None, (bad, done)
print("OK (j) run_emb_gaudi.sh: bash -n, Slurm header (b2-emb, 08:00:00, hl225:1, -c 36, 128G), expands to: {} then run_linearprobe --tag stock --datasets <each of the four> with OMP_NUM_THREADS=1, in "
      "parallel, one log each; after <normalization> only --max_images, --datasets and --workers (also --opt=value) go to the extractor, unchanged and last (smoke: {}), and anything else (its own "
      "options, --embeddings_root / --results_root, --input_size / --img_resize / --batch_size, abbreviations, --help) exits 2 before the weights check, the preflight, the card check and the extraction; PREFLIGHT: results.json or probe/<dataset> of any of the four datasets "
      "(the paths run_linearprobe refuses) stop the job with exit 1 before the card check and the extraction, a .log, another tag's results or a longer tag do not; missing / empty weights exit 1 before "
      "the card check, a failed card check exits 1, the extractor's status (7) ends the job with no fit, one failed fit exits 1 after the others finish, bad usage exits 2".format(expanded, smoke_command))

# run_emb_class.sh, the A100 sibling, and the block the two wrappers share ------------------------------------------------------------------------------------------------
CLASS_WRAPPER = os.path.join(REPO, "run_emb_class.sh")
subprocess.run(["bash", "-n", CLASS_WRAPPER], check=True)
assert os.access(CLASS_WRAPPER, os.X_OK)
class_text = open(CLASS_WRAPPER).read()
assert class_text.startswith("#!/usr/bin/env bash\n")
assert re.findall(r"^#SBATCH (.+)$", class_text, re.M) == ["--job-name=b2-emb-c", "-A class_cse49478170fall2026", "-p public,htc", "-q class", "-t 04:00:00", "-N 1", "--gres=gpu:1", "--constraint=a100_80", "-c 12", "--mem=64G"]
assert "PT_HPU_LAZY_MODE" not in class_text and "/scratch/$USER/2026Fall-CSE507-Ark-Container" in class_text
# the drift guard: the argument, environment, weights and results checks are ONE text, between one pair of marker lines in each script, so a fix made to one cannot miss the other
BEGIN = "# --- begin: shared with run_emb_gaudi.sh / run_emb_class.sh (test_ark_finetune.py checks the two copies are identical) ---"
END = "# --- end: shared ---"


def shared_block(text):
    """What lies between the marker lines of one script, as it is. The markers must be there exactly once each, begin first, and no other line may look like one."""
    lines = text.split("\n")
    assert lines.count(BEGIN) == 1 and lines.count(END) == 1 and lines.index(BEGIN) < lines.index(END), "need exactly one begin and one end marker line, begin first"
    assert sum(l.startswith("# --- begin") for l in lines) == 1 and sum(l.startswith("# --- end") for l in lines) == 1, "another line looks like a marker"
    return "\n".join(lines[lines.index(BEGIN) + 1:lines.index(END)]) + "\n"


gaudi_shared, class_shared = shared_block(wrapper_text), shared_block(class_text)
assert gaudi_shared == class_shared, "the shared blocks differ:\n" + "".join(difflib.unified_diff(gaudi_shared.splitlines(True), class_shared.splitlines(True), "run_emb_gaudi.sh", "run_emb_class.sh"))
assert gaudi_shared.startswith("TAG_RE=") and "taken=()" in gaudi_shared and gaudi_shared.endswith("another tag\" >&2\n  exit 1\nfi\n") and "PT_HPU_LAZY_MODE" not in gaudi_shared  # TAG_RE= through the results preflight's refusal
print("OK (j) drift guard: run_emb_gaudi.sh and run_emb_class.sh each hold one pair of marker lines, and the {} lines between them (TAG_RE= through the results preflight's refusal: usage and argument checks, ARK_EMB_INPUT_SIZE, "
      "ARK_EMB_NO_FIT, the weights check, the preflight) are the same text; run_emb_class.sh: bash -n, executable, Slurm header (b2-emb-c, class QOS, public,htc, 04:00:00, gpu:1 a100_80, -c 12, 64G), no PT_HPU_LAZY_MODE".format(gaudi_shared.count("\n")))

# the same checks through both wrappers, in the one fake repo that holds both launchers' stubs: every call must come through the wrapper's own launcher, and only run_emb_gaudi.sh runs the card check. run_emb_class.sh gives the fits
# OMP_NUM_THREADS=1 through `env` inside the container (nvidia.sh's --cleanenv forwards none), so its nvidia.sh stub sees no host value: "unset"
EXTRACTION = {WRAPPER: "OMP=unset python -m ark_finetune.extract_embeddings --device hpu --tag {} --model {} --weights {} --normalization {} --workers 32",
              CLASS_WRAPPER: "nvidia.sh OMP=unset python -m ark_finetune.extract_embeddings --device cuda --tag {} --model {} --weights {} --normalization {} --workers 12"}
FIT = {WRAPPER: "OMP=1 python -m ark_linearprobe.run_linearprobe --tag {} --datasets {}",
       CLASS_WRAPPER: "nvidia.sh OMP=unset env OMP_NUM_THREADS=1 python -m ark_linearprobe.run_linearprobe --tag {} --datasets {}"}


def run_checked(wrapper, args, **kwargs):
    """run_wrapper through `wrapper`, plus what every run must satisfy: each call came through the wrapper's own launcher (gaudi.sh's calls carry no prefix, nvidia.sh's start with "nvidia.sh "), and the card check ran only for run_emb_gaudi.sh."""
    done, calls, card_check, results = run_wrapper(args, wrapper=wrapper, **kwargs)
    assert all(c.startswith("nvidia.sh ") == (wrapper == CLASS_WRAPPER) for c in calls or []) and (wrapper == WRAPPER or not card_check), (wrapper, calls, done.stdout)
    return done, calls, card_check, results


for wrapper in (WRAPPER, CLASS_WRAPPER):
    gaudi = wrapper == WRAPPER  # only run_emb_gaudi.sh goes through gaudi.sh and the card check
    run = lambda args, **kwargs: run_checked(wrapper, args, **kwargs)
    expect = EXTRACTION[wrapper].format("stock", "vit_base_dinov3", STOCK, "imagenet")
    # the job: the extraction alone first, then the four fits, one log each
    done, calls, card_check, results = run(base)
    assert done.returncode == 0 and done.stderr == "" and card_check == gaudi and calls[0] == expect and sorted(calls[1:]) == sorted(FIT[wrapper].format("stock", d) for d in NAMES), (wrapper, done, calls)
    assert results == sorted(d + ".log" for d in NAMES) and sorted(l for l in done.stdout.splitlines() if l.startswith(("fit ", "PROBE "))) == \
        sorted(["fit {}: done".format(d) for d in NAMES] + ["PROBE tag {}: stub".format(d) for d in NAMES]), (wrapper, results, done.stdout)
    # the extras allow-list: anything else exits 2 before anything runs, the weights there or not; --max_images, --datasets and --workers reach the extractor as they are, last
    for extra in (["--tag", "other"], ["--device=cuda"], ["--embeddings_root", "/tmp/e"], ["--input_size", "518"], ["--emb", "/tmp/e"], ["-h"], ["--max_images", "5", "--tag", "x"]):
        for files in ((STOCK_HOST,), ()):
            done, calls, card_check, results = run(base + extra, files=files)
            assert done.returncode == 2 and calls is None and not card_check and results is None and done.stdout == "" and "not accepted after <normalization>." in done.stderr, (wrapper, extra, files, done)
    done, calls, _, _ = run(base + ["--max_images", "5", "--workers", "4"])
    assert done.returncode == 0 and calls[0] == expect + " --max_images 5 --workers 4", (wrapper, calls)
    # bad usage: exit 2 and the one usage line that names both scripts
    for bad in ([], ["stock", "vit_base_dinov3", STOCK], ["../x", "vit_base_dinov3", STOCK, "imagenet"], ["stock", "vit_base_dinov3", "weights/x.pth", "imagenet"], ["stock", "vit_base_dinov3", STOCK, "caffe"]):
        done, calls, card_check, results = run(bad)
        assert done.returncode == 2 and done.stderr == USAGE + "\n" and done.stdout == "" and calls is None and not card_check and results is None, (wrapper, bad, done)
    # missing or empty weights, and results that exist (the JSON, the shipped script's directory): exit 1 before the extraction
    for kwargs in (dict(files=()), dict(content="")):
        done, calls, card_check, results = run(base, **kwargs)
        assert done.returncode == 1 and calls is None and not card_check and results is None and "ERROR: weights not found at ./" + STOCK_HOST in done.stderr, (wrapper, kwargs, done)
    for blocking, named in (("results/stock/CheXpert.json", "results/stock/CheXpert.json"), ("results/stock/probe/MIMIC/log.txt", "results/stock/probe/MIMIC")):
        done, calls, card_check, results = run(base, files=(STOCK_HOST, blocking))
        assert done.returncode == 1 and calls is None and not card_check and done.stdout == "" and "ERROR: results already exist for tag stock: {}; refusing to start the extraction".format(named) in done.stderr, (wrapper, blocking, done)
    # the extractor's status ends the job with no fit; one failed fit exits 1 after the other three finish
    done, calls, _, results = run(base, EXTRACT_EXIT="7")
    assert done.returncode == 7 and calls == [expect] and results is None, (wrapper, done, calls)
    done, calls, _, results = run(base, FAIL_FIT="MIMIC")
    assert done.returncode == 1 and len(calls) == 5 and "ERROR: fit MIMIC failed (see results/stock/MIMIC.log)" in done.stderr and all("fit {}: done".format(d) in done.stdout for d in NAMES if d != "MIMIC"), (wrapper, done, calls)
    # ARK_EMB_INPUT_SIZE: --input_size 768 --img_resize 878 after the wrapper's own arguments and before the extras, none for the fits; an invalid value exits 2 before anything runs
    done, calls, _, _ = run(base + ["--max_images", "5"], ARK_EMB_INPUT_SIZE="768")
    assert done.returncode == 0 and calls[0] == expect + " --input_size 768 --img_resize 878 --max_images 5" and all("--input_size" not in c for c in calls[1:]), (wrapper, calls)
    done, calls, card_check, results = run(base, ARK_EMB_INPUT_SIZE="12345")
    assert done.returncode == 2 and "ARK_EMB_INPUT_SIZE" in done.stderr and calls is None and not card_check and results is None, (wrapper, done)
    # ARK_EMB_NO_FIT: 1 = the extraction only and its stdout line (exit 0, no results/<tag>); another value exits 2 before the weights check; the preflight still runs; a failed extraction ends the job with its status
    done, calls, card_check, results = run(base, ARK_EMB_NO_FIT="1")
    assert done.returncode == 0 and done.stderr == "" and card_check == gaudi and calls == [expect] and results is None and done.stdout.splitlines() == (["CARD_CHECK"] if gaudi else []) + [NO_FIT_LINE], (wrapper, done, calls)
    for bad in ("0", "yes"):
        done, calls, card_check, results = run(base, files=(), ARK_EMB_NO_FIT=bad)
        assert done.returncode == 2 and done.stderr == "ERROR: ARK_EMB_NO_FIT='{}' must be unset, empty or 1\n".format(bad) and done.stdout == "" and calls is None and not card_check and results is None, (wrapper, bad, done)
    done, calls, card_check, results = run(base, files=(STOCK_HOST, "results/stock/CheXpert.json"), ARK_EMB_NO_FIT="1")
    assert done.returncode == 1 and calls is None and not card_check and "ERROR: results already exist for tag stock: results/stock/CheXpert.json; refusing to start the extraction" in done.stderr, (wrapper, done)
    done, calls, _, results = run(base, EXTRACT_EXIT="7", ARK_EMB_NO_FIT="1")
    assert done.returncode == 7 and calls == [expect] and "extraction only" not in done.stdout and results is None, (wrapper, done, calls)
print("OK (j) both wrappers, each through its own launcher only (run_emb_gaudi.sh: gaudi.sh and the card check, hpu, 32 workers; run_emb_class.sh: nvidia.sh, no card check, cuda, 12 workers): the extraction alone first, then the four fits, one log each "
      "(class: {} then {}); extras other than --max_images / --datasets / --workers, a bad usage (one usage line naming both scripts), missing or empty weights, results that exist and an invalid ARK_EMB_INPUT_SIZE / ARK_EMB_NO_FIT stop the job "
      "before anything runs (exit 2 / 2 / 1 / 1 / 2 / 2); the extractor's status (7) ends it with no fit; one failed fit exits 1 after the others finish; ARK_EMB_INPUT_SIZE=768 -> --input_size 768 --img_resize 878; "
      "ARK_EMB_NO_FIT=1 -> the extraction only, exit 0, no results/<tag>".format(EXTRACTION[CLASS_WRAPPER].format("stock", "vit_base_dinov3", STOCK, "imagenet")[len("nvidia.sh OMP=unset "):], FIT[CLASS_WRAPPER].format("stock", "<each of the four>")[len("nvidia.sh OMP=unset "):]))

# the Ark+ Swin-L submission of run_emb_class.sh's header, one job per dataset as in run_emb_gaudi.sh: the job that only extracts MIMIC is the one call and the extraction-only line
CLASS_SUBMISSION = "ARK_EMB_INPUT_SIZE=768 sbatch -t 14:00:00 run_emb_class.sh arkplus6 swin_large_768 {} imagenet --datasets MIMIC".format(ARK_WEIGHTS)
assert CLASS_SUBMISSION in class_text  # the example in the class script's header
class_ark_extraction = "nvidia.sh OMP=unset python -m ark_finetune.extract_embeddings --device cuda --tag arkplus6 --model swin_large_768 --weights {} --normalization imagenet --workers 12 --input_size 768 --img_resize 878 --datasets MIMIC".format(ARK_WEIGHTS)
done, calls, card_check, results = run_checked(CLASS_WRAPPER, ark_args + ["--datasets", "MIMIC"], files=ark_files, ARK_EMB_INPUT_SIZE="768", ARK_EMB_NO_FIT="1")
assert done.returncode == 0 and done.stderr == "" and calls == [class_ark_extraction] and not card_check and results is None, (done, calls)
assert done.stdout == no_fit_line("arkplus6") + "\n", done.stdout
print("OK (j) Ark+ Swin-L on the class lane: ARK_EMB_INPUT_SIZE=768 ARK_EMB_NO_FIT=1 with --datasets MIMIC expands to the one call {} and the extraction-only line".format(class_ark_extraction[len("nvidia.sh OMP=unset "):]))


# the two-job probe: run_emb_fit_class.sh (the CPU-only fit) and submit_probe_class.sh (submits the extraction and, behind it, the fit) --------------------------------------------------------------
FIT_LAUNCHER = os.path.join(REPO, "run_emb_fit_class.sh")
SUBMIT = os.path.join(REPO, "submit_probe_class.sh")
for script in (FIT_LAUNCHER, SUBMIT):
    subprocess.run(["bash", "-n", script], check=True)
    assert stat.S_IMODE(os.stat(script).st_mode) == 0o755, script
fit_text, submit_text = open(FIT_LAUNCHER).read(), open(SUBMIT).read()
assert fit_text.startswith("#!/usr/bin/env bash\n") and submit_text.startswith("#!/usr/bin/env bash\n")
# the fit launcher is CPU only: a header with no --gres, no constraint and no GPU of any kind (the class QOS counts GPU minutes), and no launcher but nvidia.sh (the class image; run_linearprobe is cpu only by construction)
fit_header = re.findall(r"^#SBATCH (.+)$", fit_text, re.M)
assert fit_header == ["--job-name=b2-fit-c", "-A class_cse49478170fall2026", "-p public,htc", "-q class", "-t 03:00:00", "-N 1", "-c 12", "--mem=64G"], fit_header
assert not re.search(r"gres|constraint|gpu", " ".join(fit_header)) and "PT_HPU_LAZY_MODE" not in fit_text and "./gaudi.sh" not in fit_text and "extract_embeddings" not in fit_text
assert not re.search(r"^#SBATCH", submit_text, re.M)  # not an sbatch script: it runs on the login node, submits, prints one line and exits
# the lines the fit launcher takes verbatim from the block the two extraction launchers share (that block is the extraction's: it also wants <model> <weights> <normalization> and the weights file, which the fit has none of)
REUSED = ("TAG_RE='^[A-Za-z0-9][A-Za-z0-9_.-]*$'   # the tag is a directory name: no path separators, no spaces", "datasets=(ChestXray14 CheXpert MIMIC VinDrCXR)",
          'cd "${ARK_ROOT:-/scratch/$USER/2026Fall-CSE507-Ark-Container}" || exit 1',
          'taken=()\nfor dataset in "${datasets[@]}"; do\n  for path in "results/$TAG/$dataset.json" "results/$TAG/probe/$dataset"; do\n    [[ -e $path ]] && taken+=("$path")\n  done\ndone')
for text in REUSED:
    assert ("\n" + text + "\n") in ("\n" + gaudi_shared) and ("\n" + text + "\n") in ("\n" + class_shared) and ("\n" + text + "\n") in ("\n" + fit_text), text
print("OK (j) drift guard, three scripts: run_emb_fit_class.sh repeats {} lines of the shared block verbatim (TAG_RE=, datasets=(, the cd to ARK_ROOT, the loop that fills taken with the two paths run_linearprobe refuses); "
      "its header is {} (no --gres, no constraint, no GPU); submit_probe_class.sh: bash -n, mode 755, no #SBATCH line".format(sum(t.count("\n") + 1 for t in REUSED), " ".join(fit_header)))

FIT_USAGE = "Usage: sbatch run_emb_fit_class.sh <tag: letters, digits, . _ - only> [--datasets NAME...]   (NAME: ChestXray14 CheXpert MIMIC VinDrCXR, each at most once)"
CACHES = tuple("embeddings/stock/{}/meta.json".format(d) for d in NAMES)  # a dataset's cache as the extractor leaves it: meta.json is the file it writes last
FIT_CALL = FIT[CLASS_WRAPPER]  # the fit's one call, through nvidia.sh (the class image), OMP_NUM_THREADS=1 inside the container


def fits(tag, names):
    return sorted(FIT_CALL.format(tag, d) for d in names)


def seeded(files, content="x"):
    """{path: content} of the files run_wrapper puts in the fake repo (a `files` entry is a path, or (path, content))."""
    return {(f[0] if isinstance(f, tuple) else f): (f[1] if isinstance(f, tuple) else content) for f in files}


def run_fit(args, files=CACHES, **kwargs):
    """run_wrapper through run_emb_fit_class.sh, plus what every run must satisfy: each call came through nvidia.sh (never gaudi.sh), none is an extraction, and no card check ran. snapshot=True adds the fake repo afterwards, {path: content}."""
    found = run_wrapper(args, files=files, wrapper=FIT_LAUNCHER, **kwargs)
    assert not found[2] and all(c.startswith("nvidia.sh ") and "extract_embeddings" not in c for c in found[1] or []), (args, found[1], found[0].stdout)
    return found


# a fresh fit: the four fits and nothing else, one log each, the PROBE lines in the job log; --datasets: only those, and only their caches are needed
done, calls, _, results, tree = run_fit(["stock"], snapshot=True)
assert done.returncode == 0 and done.stderr == "" and sorted(calls) == fits("stock", NAMES) and results == sorted(d + ".log" for d in NAMES), (done, calls, results)
assert sorted(l for l in done.stdout.splitlines() if l.startswith(("fit ", "PROBE "))) == sorted(["fit {}: done".format(d) for d in NAMES] + ["PROBE tag {}: stub".format(d) for d in NAMES]), done.stdout
assert not [l for l in done.stdout.splitlines() if l.startswith(("skip ", "refit "))] and tree == dict(seeded(CACHES), **{"results/stock/{}.log".format(d): "PROBE tag {}: stub\n".format(d) for d in NAMES}), (done.stdout, sorted(tree))
done, calls, _, results = run_fit(["stock", "--datasets", "MIMIC", "CheXpert"], files=CACHES[1:3])  # CheXpert's and MIMIC's caches only
assert done.returncode == 0 and sorted(calls) == fits("stock", ("CheXpert", "MIMIC")) and results == ["CheXpert.log", "MIMIC.log"], (done, calls, results)
done, calls, _, results = run_fit(["stock", "--datasets", "VinDrCXR"], files=CACHES[3:] + ("embeddings/other/MIMIC/meta.json",))
assert done.returncode == 0 and calls == fits("stock", ("VinDrCXR",)) and results == ["VinDrCXR.log"], (done, calls, results)
# a failed fit: exit 1 after the other three have finished, the failure named with its log
done, calls, _, results = run_fit(["stock"], FAIL_FIT="MIMIC")
assert done.returncode == 1 and len(calls) == 4 and "ERROR: fit MIMIC failed (see results/stock/MIMIC.log)" in done.stderr and all("fit {}: done".format(d) in done.stdout for d in NAMES if d != "MIMIC"), (done, calls)
print("OK (j) run_emb_fit_class.sh: four fits, each a nvidia.sh call 'env OMP_NUM_THREADS=1 python -m ark_linearprobe.run_linearprobe --tag stock --datasets <one>' in parallel, one log each, PROBE lines at the end of the job log, no extraction, no card check; "
      "--datasets runs only the named ones (and needs only their caches); one failed fit exits 1 after the others finish")

# refusals before anything runs: exit 2 for the arguments, whatever else is in place (the caches there or not, results that exist); nothing is made or removed
for bad in ([], ["../x"], ["a b"], ["-x"], [""], ["--check"], ["--datasets", "MIMIC"], ["stock", "MIMIC"], ["stock", "--datasets"], ["stock", "--datasets=MIMIC"], ["stock", "--tag", "other"], ["stock", "--max_images", "5"], ["stock", "--workers", "4"]):
    for files in (CACHES, (), CACHES + ("results/stock/MIMIC.json",)):
        done, calls, _, _, tree = run_fit(bad, files=files, snapshot=True)
        assert done.returncode == 2 and done.stderr == FIT_USAGE + "\n" and done.stdout == "" and calls is None and tree == seeded(files), (bad, files, done)
for extra, message in ((["--datasets", "MIMC"], "ERROR: 'MIMC' is not a dataset of the probe (ChestXray14 CheXpert MIMIC VinDrCXR)\n"), (["--datasets", "mimic"], "ERROR: 'mimic' is not a dataset of the probe"),
                       (["--datasets", "ChestXray14 CheXpert"], "ERROR: 'ChestXray14 CheXpert' is not a dataset of the probe"), (["--datasets", "MIMIC", "--workers", "4"], "ERROR: '--workers' is not a dataset of the probe"),
                       (["--datasets", "MIMIC", "MIMIC"], "ERROR: --datasets lists MIMIC twice\n"), (["--datasets", "CheXpert", "MIMIC", "CheXpert"], "ERROR: --datasets lists CheXpert twice\n")):
    for files in (CACHES, ()):
        done, calls, _, _, tree = run_fit(["stock"] + extra, files=files, snapshot=True)
        assert done.returncode == 2 and done.stderr.startswith(message) and done.stdout == "" and calls is None and tree == seeded(files), (extra, files, done)
for bad in ("0", "yes", "1 ", "true", "11"):  # ARK_EMB_FIT_RESUME: unset, empty or exactly 1
    for files in (CACHES, (), CACHES + ("results/stock/MIMIC.json",)):
        done, calls, _, _, tree = run_fit(["stock"], files=files, snapshot=True, ARK_EMB_FIT_RESUME=bad)
        assert done.returncode == 2 and done.stderr == "ERROR: ARK_EMB_FIT_RESUME='{}' must be unset, empty or 1\n".format(bad) and done.stdout == "" and calls is None and tree == seeded(files), (bad, files, done)
# the cache of every dataset to be fit must be there (meta.json, written last), and the message names each one that is not
for present, missing in ((CACHES[:2] + CACHES[3:], ("MIMIC",)), (CACHES[:1], ("CheXpert", "MIMIC", "VinDrCXR")), ((), NAMES)):
    done, calls, _, _, tree = run_fit(["stock"], files=present, snapshot=True)
    expected = " ".join("embeddings/stock/{}/meta.json".format(d) for d in missing)
    assert done.returncode == 1 and done.stdout == "" and calls is None and tree == seeded(present), (missing, done)
    assert done.stderr.startswith("ERROR: no embeddings for tag stock: {} missing; extract them first (ARK_EMB_NO_FIT=1 sbatch run_emb_class.sh stock ".format(expected)), (missing, done.stderr)
partial = tuple("embeddings/stock/MIMIC/" + f for f in ("train.npy", "test.npy", "train_paths.txt", "test_paths.txt"))  # a half-written cache has no meta.json yet
done, calls, _, _, tree = run_fit(["stock"], files=CACHES[:2] + CACHES[3:] + partial, snapshot=True)
assert done.returncode == 1 and calls is None and "embeddings/stock/MIMIC/meta.json missing;" in done.stderr and "ChestXray14" not in done.stderr, done
done, calls, _, _ = run_fit(["stock"], files=tuple("embeddings/other/{}/meta.json".format(d) for d in NAMES))  # another tag's cache is no cache
assert done.returncode == 1 and calls is None and "embeddings/stock/ChestXray14/meta.json embeddings/stock/CheXpert/meta.json" in done.stderr, done
done, calls, _, _ = run_fit(["stock", "--datasets", "MIMIC"], files=CACHES[:2])  # --datasets: the requested dataset's cache is the one that counts
assert done.returncode == 1 and calls is None and "no embeddings for tag stock: embeddings/stock/MIMIC/meta.json missing;" in done.stderr, done
print("OK (j) run_emb_fit_class.sh refusals: the usage line (exit 2) for no tag, a bad tag, a second option or --datasets without names; an unknown, repeated or space-joined dataset name, and ARK_EMB_FIT_RESUME other than unset / empty / 1, exit 2 with "
      "the cause named; a missing cache (exit 1) names each dataset's embeddings/stock/<dataset>/meta.json that is not there (a half-written cache, another tag's cache and an unrequested dataset's absence count as they should); all before anything runs, nothing made or removed")

# results that exist stop a fresh fit (exit 1) BEFORE anything is removed or run, whichever of the two paths run_linearprobe refuses it is, and before the cache is looked at
for name in NAMES:
    for blocking in run_linearprobe.results_of("results", "stock", name):
        files = CACHES + (os.path.join(blocking, "log.txt") if not blocking.endswith(".json") else blocking,)
        done, calls, _, _, tree = run_fit(["stock"], files=files, snapshot=True)
        assert done.returncode == 1 and calls is None and done.stdout == "" and tree == seeded(files), (blocking, done)
        assert done.stderr.startswith("ERROR: results already exist for tag stock: {}; refusing to fit over them. To finish a fit that stopped half way set ARK_EMB_FIT_RESUME=1 ".format(blocking)), (blocking, done.stderr)
done, calls, _, _ = run_fit(["stock"], files=CACHES + ("results/stock/CheXpert.json", "results/stock/probe/VinDrCXR/log.txt"))  # several: all are named
assert done.returncode == 1 and calls is None and "results/stock/CheXpert.json results/stock/probe/VinDrCXR;" in done.stderr, done.stderr
done, calls, _, _ = run_fit(["stock"], files=("results/stock/CheXpert.json",))  # results first, as run_linearprobe: the missing caches are not the reason
assert done.returncode == 1 and calls is None and done.stderr.startswith("ERROR: results already exist for tag stock: results/stock/CheXpert.json;"), done.stderr
done, calls, _, _ = run_fit(["stock"], files=CACHES, ARK_EMB_FIT_RESUME="")  # empty is unset
assert done.returncode == 0 and len(calls) == 4, done
done, calls, _, _ = run_fit(["stock"], files=CACHES + ("results/stock/CheXpert.json",), ARK_EMB_FIT_RESUME="")
assert done.returncode == 1 and calls is None, done
for files in (("results/stock/MIMIC.log",),  # the launcher's own redirect target from an earlier attempt is not a result
              ("results/other/ChestXray14.json", "results/other/probe/MIMIC/log.txt"),  # nor is another tag's
              ("results/stock2/CheXpert.json",)):  # nor a tag that merely starts with this one
    done, calls, _, _ = run_fit(["stock"], files=CACHES + files)
    assert done.returncode == 0 and len(calls) == 4, (files, done)
done, calls, _, _ = run_fit(["stock", "--datasets", "MIMIC"], files=CACHES + ("results/stock/CheXpert.json", "results/stock/probe/VinDrCXR/log.txt"))  # a dataset that is not being fit does not matter
assert done.returncode == 0 and calls == fits("stock", ("MIMIC",)), (done, calls)
print("OK (j) run_emb_fit_class.sh, fresh fit: results.json or probe/<dataset> of a dataset to be fit (the two paths run_linearprobe refuses) stop the job with exit 1, the message names them and ARK_EMB_FIT_RESUME=1, nothing is removed or run; "
      "that check comes before the cache check; a .log, another tag's results, a longer tag or the results of a dataset not being fit do not block")

# the resume rule (ARK_EMB_FIT_RESUME=1) on a synthetic results dir: ChestXray14 finished (its JSON, the script's directory, its log), CheXpert half done (the script's directory and a stale log, no JSON), MIMIC not started, VinDrCXR
# finished with its JSON alone, and another tag's results beside them
RESUME_SEED = (("results/stock/ChestXray14.json", "finished json\n"), ("results/stock/probe/ChestXray14/log.txt", "finished script log\n"), ("results/stock/probe/ChestXray14/run_0/ckpt.pth", "finished ckpt\n"),
               ("results/stock/ChestXray14.log", "PROBE stock ChestXray14: seeded\n"), ("results/stock/probe/CheXpert/log.txt", "half done\n"), ("results/stock/probe/CheXpert/run_0/ckpt.pth", "half-done ckpt\n"),
               ("results/stock/CheXpert.log", "stale log of the dead fit\n"), ("results/stock/VinDrCXR.json", "json alone\n"), ("results/stock/VinDrCXR.log", "PROBE stock VinDrCXR: seeded\n"),
               ("results/other/probe/CheXpert/log.txt", "another tag\n"), ("results/other/CheXpert.json", "another tag json\n"))
done, calls, _, _, tree = run_fit(["stock"], files=CACHES + RESUME_SEED, snapshot=True, ARK_EMB_FIT_RESUME="1")
lines = done.stdout.splitlines()
assert done.returncode == 0 and done.stderr == "" and sorted(calls) == fits("stock", ("CheXpert", "MIMIC")), (done, calls)  # only the two that are not finished are fit
assert [l for l in lines if l.startswith(("skip ", "refit "))] == ["skip ChestXray14: results/stock/ChestXray14.json exists, the fit is finished", "refit CheXpert: removing results/stock/probe/CheXpert, left half done by an earlier fit",
                                                                    "skip VinDrCXR: results/stock/VinDrCXR.json exists, the fit is finished"], lines
assert sorted(l for l in lines if l.startswith(("fit ", "PROBE "))) == ["PROBE stock ChestXray14: seeded", "PROBE stock VinDrCXR: seeded", "PROBE tag CheXpert: stub", "PROBE tag MIMIC: stub", "fit CheXpert: done", "fit MIMIC: done"], lines
expected = seeded(CACHES + RESUME_SEED)  # everything stays byte for byte, but the half-done directory (gone), the stale log (rewritten by the new fit) and MIMIC's new log
for gone in ("results/stock/probe/CheXpert/log.txt", "results/stock/probe/CheXpert/run_0/ckpt.pth"):
    del expected[gone]
expected.update({"results/stock/CheXpert.log": "PROBE tag CheXpert: stub\n", "results/stock/MIMIC.log": "PROBE tag MIMIC: stub\n"})
assert tree == expected, sorted(set(tree) ^ set(expected)) or [(k, tree[k], expected[k]) for k in tree if tree[k] != expected[k]]
# every dataset finished: four skip lines, the PROBE lines of their logs, no fit, nothing touched
finished = tuple(("results/stock/{}.json".format(d), "json\n") for d in NAMES) + tuple(("results/stock/{}.log".format(d), "PROBE stock {}: seeded\n".format(d)) for d in NAMES)
done, calls, _, _, tree = run_fit(["stock"], files=CACHES + finished, snapshot=True, ARK_EMB_FIT_RESUME="1")
assert done.returncode == 0 and calls is None and tree == seeded(CACHES + finished), (done, calls)
assert sorted(done.stdout.splitlines()) == sorted(["skip {0}: results/stock/{0}.json exists, the fit is finished".format(d) for d in NAMES] + ["PROBE stock {}: seeded".format(d) for d in NAMES]), done.stdout
# nothing started yet: the same as a fresh fit; a half-done dataset whose cache is missing is NOT cleared (the cache check comes first); --datasets: the others are not looked at; a failed fit exits 1 after the others finish
done, calls, _, _ = run_fit(["stock"], ARK_EMB_FIT_RESUME="1")
assert done.returncode == 0 and sorted(calls) == fits("stock", NAMES) and not [l for l in done.stdout.splitlines() if l.startswith(("skip ", "refit "))], (done, calls)
half = (("results/stock/probe/CheXpert/log.txt", "half done\n"), ("results/stock/CheXpert.log", "stale\n"))
done, calls, _, _, tree = run_fit(["stock"], files=CACHES[:1] + CACHES[2:] + half, snapshot=True, ARK_EMB_FIT_RESUME="1")
assert done.returncode == 1 and calls is None and "no embeddings for tag stock: embeddings/stock/CheXpert/meta.json missing;" in done.stderr and tree == seeded(CACHES[:1] + CACHES[2:] + half), (done, sorted(tree))
done, calls, _, _, tree = run_fit(["stock", "--datasets", "MIMIC"], files=CACHES + RESUME_SEED, snapshot=True, ARK_EMB_FIT_RESUME="1")
assert done.returncode == 0 and calls == fits("stock", ("MIMIC",)) and "results/stock/probe/CheXpert/log.txt" in tree and not [l for l in done.stdout.splitlines() if l.startswith(("skip ", "refit "))], (done, calls)
done, calls, _, _ = run_fit(["stock"], files=CACHES + RESUME_SEED, ARK_EMB_FIT_RESUME="1", FAIL_FIT="CheXpert")
assert done.returncode == 1 and len(calls) == 2 and "ERROR: fit CheXpert failed (see results/stock/CheXpert.log)" in done.stderr and "fit MIMIC: done" in done.stdout, (done, calls)
print("OK (j) run_emb_fit_class.sh resume rule (ARK_EMB_FIT_RESUME=1): a dataset whose results/<tag>/<dataset>.json exists is skipped with a line and never touched (its probe directory, log and JSON stay byte for byte); one without it has its "
      "results/<tag>/probe/<dataset> removed, with a line, and is fit again; the other tag's results are never touched; the fits are the unfinished datasets only, the job log carries every dataset's PROBE line; all finished -> no fit; "
      "nothing started -> a fresh fit; a missing cache stops it before anything is removed; --datasets limits what is looked at; one failed fit exits 1 after the others")

# --check, the first argument: the arguments and results checks, no cache check, nothing run, written or removed, and silence on success (submit_probe_class.sh prints one line of its own)
for args in (["--check", "stock"], ["--check", "stock", "--datasets", "MIMIC", "CheXpert"]):
    done, calls, _, results, tree = run_fit(args, files=(), snapshot=True)  # no caches: the extraction has not run yet when the wrapper asks
    assert done.returncode == 0 and done.stdout == "" and done.stderr == "" and calls is None and results is None and tree == {}, (args, done)
done, calls, _, _, tree = run_fit(["--check", "stock"], files=CACHES + RESUME_SEED, snapshot=True, ARK_EMB_FIT_RESUME="1")  # resume mode: nothing is removed
assert done.returncode == 0 and done.stdout == "" and calls is None and tree == seeded(CACHES + RESUME_SEED), done
for args, files, code, message in ((["--check"], (), 2, FIT_USAGE + "\n"), (["--check", "../x"], (), 2, FIT_USAGE + "\n"), (["--check", "stock", "--datasets", "MIMC"], (), 2, "ERROR: 'MIMC' is not a dataset of the probe"),
                                    (["--check", "stock", "--datasets", "MIMIC", "MIMIC"], (), 2, "ERROR: --datasets lists MIMIC twice\n"), (["--check", "stock"], ("results/stock/CheXpert.json",), 1, "ERROR: results already exist for tag stock: results/stock/CheXpert.json;"),
                                    (["--check", "stock"], ("results/stock/probe/MIMIC/log.txt",), 1, "ERROR: results already exist for tag stock: results/stock/probe/MIMIC;")):
    done, calls, _, _, tree = run_fit(args, files=files, snapshot=True)
    assert done.returncode == code and done.stderr.startswith(message) and done.stdout == "" and calls is None and tree == seeded(files), (args, files, done)
done, calls, _, _, tree = run_fit(["--check", "stock"], files=(), snapshot=True, ARK_EMB_FIT_RESUME="2")
assert done.returncode == 2 and done.stderr == "ERROR: ARK_EMB_FIT_RESUME='2' must be unset, empty or 1\n" and calls is None, done
print("OK (j) run_emb_fit_class.sh --check: silent exit 0 without the caches, with the same refusals as the real run (usage, dataset names, ARK_EMB_FIT_RESUME, results that exist), no call, nothing written or removed, not even in resume mode")

# run_emb_class.sh --check, the first argument: the shared block's checks (usage, extras, ARK_EMB_*, weights, results), then exit 0 with no extraction, no fit and no output
done, calls, card_check, results, tree = run_wrapper(["--check"] + base, wrapper=CLASS_WRAPPER, snapshot=True)
assert done.returncode == 0 and done.stdout == "" and done.stderr == "" and calls is None and results is None and tree == {STOCK_HOST: "x"}, done
done, calls, _, results = run_wrapper(["--check"] + base + ["--datasets", "MIMIC", "CheXpert"], wrapper=CLASS_WRAPPER, ARK_EMB_INPUT_SIZE="512", ARK_EMB_NO_FIT="1")
assert done.returncode == 0 and done.stdout == "" and done.stderr == "" and calls is None and results is None, done
for args, kwargs, code, message in ((["--check"], {}, 2, USAGE + "\n"), (["--check"] + base[:3], {}, 2, USAGE + "\n"), (["--check", "../x"] + base[1:], {}, 2, USAGE + "\n"), (["--check"] + base + ["--tag", "other"], {}, 2, "ERROR: --tag: not accepted after <normalization>."),
                                    (["--check"] + base, dict(files=()), 1, "ERROR: weights not found at ./" + STOCK_HOST), (["--check"] + base, dict(files=(STOCK_HOST, "results/stock/CheXpert.json")), 1, "ERROR: results already exist for tag stock: results/stock/CheXpert.json;"),
                                    (["--check"] + base, dict(ARK_EMB_INPUT_SIZE="12345"), 2, "ERROR: ARK_EMB_INPUT_SIZE='12345'"), (["--check"] + base, dict(ARK_EMB_NO_FIT="yes"), 2, "ERROR: ARK_EMB_NO_FIT='yes' must be unset, empty or 1\n")):
    done, calls, card_check, results = run_wrapper(args, wrapper=CLASS_WRAPPER, **kwargs)
    assert done.returncode == code and done.stderr.startswith(message) and done.stdout == "" and calls is None and not card_check, (args, kwargs, done)
print("OK (j) run_emb_class.sh --check: exit 0, no output, no call (no extraction, no fit) when the usage, extras, ARK_EMB_*, weights and results checks pass; the same exits and messages as the real run when they do not (2 / 2 / 2 / 1 / 1 / 2 / 2)")

# the extraction-only exit message names the fit launcher (both extraction launchers), with the tag filled in, and says it is CPU only
for wrapper in (WRAPPER, CLASS_WRAPPER):
    done, calls, _, results = run_wrapper(base, wrapper=wrapper, ARK_EMB_NO_FIT="1")
    assert done.returncode == 0 and done.stdout.splitlines()[-1] == no_fit_line("stock") and "sbatch run_emb_fit_class.sh stock [--datasets NAME...] (CPU only" in done.stdout, (wrapper, done.stdout)
print("OK (j) ARK_EMB_NO_FIT=1 exit message, both extraction launchers: " + no_fit_line("<tag>"))

# submit_probe_class.sh: the two sbatch lines behind a stub sbatch (and scancel) first on PATH; the launchers' own --check runs in the fake repo
SBATCH_STUB = """#!/usr/bin/env bash
n=$(( $(cat "$STUB_DIR/count" 2>/dev/null || echo 0) + 1 ))
echo $n > "$STUB_DIR/count"
{ printf 'NO_FIT=%s INPUT_SIZE=%s RESUME=%s\\n' "${ARK_EMB_NO_FIT-unset}" "${ARK_EMB_INPUT_SIZE-unset}" "${ARK_EMB_FIT_RESUME-unset}"; printf '%s\\n' "$@"; } > "$STUB_DIR/sbatch.$n"
[[ $n == "${SBATCH_FAIL_ON:-}" ]] && { echo "sbatch: error: stub failure" >&2; exit 1; }
printed="SBATCH_PRINT_$n"
echo "${!printed:-$((1000 + n))}"
"""
SCANCEL_STUB = """#!/usr/bin/env bash
echo "$*" >> "$STUB_DIR/scancel.log"
[[ -n "${SCANCEL_FAIL:-}" ]] && exit 1
exit 0
"""
SUBMIT_USAGE = "Usage: ./submit_probe_class.sh <tag> <model> <in-container weights path under /workspace> <imagenet|chestx-ray|none|raddino> [--datasets NAME...]"


def run_submit(args, files=(STOCK_HOST,), **env):
    """submit_probe_class.sh run directly (its shebang and mode) from another directory, ARK_ROOT = a fake repo holding `files` and a stub nvidia.sh (a launcher that really ran would log a call there), with stub sbatch and scancel first on PATH.
    Returns (completed process, one (environment line, argv) per sbatch call, the scancel argument lines, the launchers' calls or None). Every run must also satisfy: at most two sbatch calls, the second with the dependency on the id the first printed."""
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as elsewhere, tempfile.TemporaryDirectory() as stubs:
        os.makedirs(os.path.join(stubs, "bin"))
        for folder, name, body in ((os.path.join(stubs, "bin"), "sbatch", SBATCH_STUB), (os.path.join(stubs, "bin"), "scancel", SCANCEL_STUB), (root, "nvidia.sh", NVIDIA_STUB)):
            with open(os.path.join(folder, name), "w") as f:
                f.write(body)
            os.chmod(os.path.join(folder, name), 0o755)
        for name in files:
            os.makedirs(os.path.join(root, os.path.dirname(name)), exist_ok=True)
            with open(os.path.join(root, name), "w") as f:
                f.write("x")
        environment = {k: v for k, v in os.environ.items() if not k.startswith(("ARK_EMB_", "PROBE_", "SLURM_", "SBATCH_", "SCANCEL_"))}
        environment.update(ARK_ROOT=root, STUB_LOG=os.path.join(root, "stub.log"), STUB_DIR=stubs, PATH=os.path.join(stubs, "bin") + os.pathsep + os.environ["PATH"])
        environment.update(env)
        done = subprocess.run([SUBMIT] + list(args), cwd=elsewhere, env=environment, capture_output=True, text=True)
        sbatch = []
        while os.path.exists(os.path.join(stubs, "sbatch.%d" % (len(sbatch) + 1))):
            lines = open(os.path.join(stubs, "sbatch.%d" % (len(sbatch) + 1))).read().split("\n")[:-1]
            sbatch.append((lines[0], lines[1:]))
        scancel = open(os.path.join(stubs, "scancel.log")).read().splitlines() if os.path.exists(os.path.join(stubs, "scancel.log")) else []
        calls = open(os.path.join(root, "stub.log")).read().splitlines() if os.path.exists(os.path.join(root, "stub.log")) else None
    first_id = env.get("SBATCH_PRINT_1", "1001").split(";")[0]
    assert len(sbatch) <= 2 and (len(sbatch) < 2 or "--dependency=afterok:" + first_id in sbatch[1][1]), (args, sbatch)  # never a second submission without the dependency
    return done, sbatch, scancel, calls


X_DEFAULT = ["--parsable", "-J", "probe-stock-x", "-p", "public,htc", "-q", "class", "--constraint=a100", "-t", "01:15:00", "run_emb_class.sh"] + base
FIT_DEFAULT = ["--parsable", "-J", "probe-stock-fit", "-t", "03:00:00", "--dependency=afterok:1001", "run_emb_fit_class.sh", "stock"]
done, sbatch, scancel, calls = run_submit(base)
assert done.returncode == 0 and done.stderr == "" and done.stdout == "probe stock: extraction 1001, fit 1002 (afterok)\n" and scancel == [] and calls is None, (done, scancel, calls)
assert sbatch == [("NO_FIT=1 INPUT_SIZE=unset RESUME=unset", X_DEFAULT), ("NO_FIT=unset INPUT_SIZE=unset RESUME=unset", FIT_DEFAULT)], sbatch  # the extraction alone has ARK_EMB_NO_FIT=1; the fit is GPU-free: no -p/-q/constraint/gres of the extraction
assert not [a for a in sbatch[1][1] if "gres" in a or "constraint" in a or "gpu" in a or a.startswith("-p")], sbatch[1]
# --datasets goes to both jobs; the limits and the constraint are overridable; ARK_EMB_INPUT_SIZE reaches the extraction; an inherited ARK_EMB_FIT_RESUME does not reach the fit (a pair is a fresh fit); an inherited ARK_EMB_NO_FIT is the wrapper's 1 on the extraction
done, sbatch, scancel, calls = run_submit(base + ["--datasets", "MIMIC", "CheXpert"], ARK_EMB_INPUT_SIZE="512", PROBE_X_TIME="2:00:00", PROBE_X_CONSTRAINT="a100_80", PROBE_FIT_TIME="1-00:00:00", ARK_EMB_FIT_RESUME="1", ARK_EMB_NO_FIT="yes")
assert done.returncode == 0 and done.stdout == "probe stock: extraction 1001, fit 1002 (afterok)\n" and scancel == [] and calls is None, (done, scancel, calls)
assert sbatch[0] == ("NO_FIT=1 INPUT_SIZE=512 RESUME=unset", ["--parsable", "-J", "probe-stock-x", "-p", "public,htc", "-q", "class", "--constraint=a100_80", "-t", "2:00:00", "run_emb_class.sh"] + base + ["--datasets", "MIMIC", "CheXpert"]), sbatch[0]
assert sbatch[1] == ("NO_FIT=yes INPUT_SIZE=512 RESUME=unset", ["--parsable", "-J", "probe-stock-fit", "-t", "1-00:00:00", "--dependency=afterok:1001", "run_emb_fit_class.sh", "stock", "--datasets", "MIMIC", "CheXpert"]), sbatch[1]
done, sbatch, scancel, calls = run_submit(base, PROBE_X_TIME="", PROBE_X_CONSTRAINT="", PROBE_FIT_TIME="")  # empty is unset: the defaults
assert done.returncode == 0 and [argv for _, argv in sbatch] == [X_DEFAULT, FIT_DEFAULT], sbatch
done, sbatch, scancel, calls = run_submit(["a.b-c_1", "vit_large_raydino", "/workspace/weights/xray_dino_vitl16_pretrained-ad31c2b0.pth", "imagenet"], files=("weights/xray_dino_vitl16_pretrained-ad31c2b0.pth",))  # another tag, model and weights: all in both names and arguments
assert done.returncode == 0 and done.stdout == "probe a.b-c_1: extraction 1001, fit 1002 (afterok)\n" and sbatch[0][1][2] == "probe-a.b-c_1-x" and sbatch[1][1][2] == "probe-a.b-c_1-fit" and sbatch[1][1][-1] == "a.b-c_1", sbatch
done, sbatch, scancel, calls = run_submit(base, SBATCH_PRINT_1="4242;sol", SBATCH_PRINT_2="4243;sol")  # --parsable may print "<id>;<cluster>"
assert done.returncode == 0 and done.stdout == "probe stock: extraction 4242, fit 4243 (afterok)\n" and sbatch[1][1][5] == "--dependency=afterok:4242", (done, sbatch)
print("OK (j) submit_probe_class.sh: the extraction line 'ARK_EMB_NO_FIT=1 sbatch " + " ".join(X_DEFAULT) + "' then the fit line 'sbatch " + " ".join(FIT_DEFAULT) + "' (afterok on the id the first printed, '<id>;<cluster>' accepted), "
      "one stdout line 'probe <tag>: extraction <id1>, fit <id2> (afterok)'; --datasets reaches both jobs; PROBE_X_TIME / PROBE_X_CONSTRAINT / PROBE_FIT_TIME override (empty = default); ARK_EMB_INPUT_SIZE reaches the extraction; "
      "an inherited ARK_EMB_FIT_RESUME does not reach the fit; the fit line has no -p, -q, constraint or gres")

# nothing is submitted when a launcher's check refuses (exit 2 for the arguments, 1 for the weights file and for results that exist), when the shape is wrong (the wrapper's own usage), or inside a Slurm job; the launcher's message is kept
CHECK_REFUSED = "submit_probe_class.sh: a launcher's check refused the arguments (exit {}); nothing was submitted\n"
for args, files, env, code, stderr in ((base, (), {}, 1, None), (base[:3] + ["caffe"], (STOCK_HOST,), {}, 2, USAGE + "\n"), (["../x"] + base[1:], (STOCK_HOST,), {}, 2, USAGE + "\n"), (["stock", "vit_base_dinov3", "weights/x.pth", "imagenet"], (STOCK_HOST,), {}, 2, USAGE + "\n"),
                                       (base, (STOCK_HOST,), dict(ARK_EMB_INPUT_SIZE="12345"), 2, None), (base, (STOCK_HOST,), dict(ARK_EMB_INPUT_SIZE="0512"), 2, None),
                                       (base + ["--datasets"], (STOCK_HOST,), {}, 2, FIT_USAGE + "\n"), (base + ["--datasets", "MIMC"], (STOCK_HOST,), {}, 2, "ERROR: 'MIMC' is not a dataset of the probe (ChestXray14 CheXpert MIMIC VinDrCXR)\n"),
                                       (base + ["--datasets", "MIMIC", "MIMIC"], (STOCK_HOST,), {}, 2, "ERROR: --datasets lists MIMIC twice\n"), (base + ["--datasets", "MIMIC", "--max_images", "5"], (STOCK_HOST,), {}, 2, "ERROR: '--max_images' is not a dataset of the probe (ChestXray14 CheXpert MIMIC VinDrCXR)\n")):
    done, sbatch, scancel, calls = run_submit(args, files=files, **env)
    assert done.returncode == code and sbatch == [] and scancel == [] and calls is None and done.stdout == "", (args, env, done)
    assert done.stderr.endswith(CHECK_REFUSED.format(code)) and (stderr is None or done.stderr == stderr + CHECK_REFUSED.format(code)), (args, env, done.stderr)
assert "ERROR: weights not found at ./" + STOCK_HOST in run_submit(base, files=())[0].stderr and "ERROR: ARK_EMB_INPUT_SIZE='12345'" in run_submit(base, ARK_EMB_INPUT_SIZE="12345")[0].stderr
for name in NAMES:  # results that exist for the tag: the same preflight as the extraction launcher's, so nothing is queued that the fit would refuse to overwrite
    for blocking in run_linearprobe.results_of("results", "stock", name):
        done, sbatch, scancel, calls = run_submit(base, files=(STOCK_HOST, os.path.join(blocking, "log.txt") if not blocking.endswith(".json") else blocking))
        assert done.returncode == 1 and sbatch == [] and scancel == [] and calls is None and done.stdout == "", (blocking, done)
        assert "ERROR: results already exist for tag stock: {}; refusing to start the extraction".format(blocking) in done.stderr and done.stderr.endswith(CHECK_REFUSED.format(1)), (blocking, done.stderr)
done, sbatch, scancel, calls = run_submit(base, files=(STOCK_HOST, "results/stock/CheXpert.json"), ARK_EMB_FIT_RESUME="1")  # an inherited resume switch does not make an existing result acceptable
assert done.returncode == 1 and sbatch == [] and "results already exist for tag stock" in done.stderr, done
for args in ([], base[:3], base + ["--max_images", "5"], base + ["--workers", "4"], base + ["--tag", "x"], base + ["--datasets=MIMIC"]):
    done, sbatch, scancel, calls = run_submit(args)
    assert (done.returncode, done.stdout, done.stderr, sbatch, scancel, calls) == (2, "", SUBMIT_USAGE + "\n", [], [], None), (args, done)
done, sbatch, scancel, calls = run_submit(base, SLURM_JOB_ID="7")
assert done.returncode == 1 and done.stdout == "" and sbatch == [] and scancel == [] and calls is None and done.stderr == "ERROR: submit_probe_class.sh submits the jobs itself: run it on the login node, not inside Slurm job 7\n", done
print("OK (j) submit_probe_class.sh refusals, nothing submitted and no launcher run: a bad tag, normalization or weights path, an invalid ARK_EMB_INPUT_SIZE (the extraction launcher's check, exit 2), a missing weights file (1), results that exist (1; the JSON or "
      "probe/<dataset> of any of the four), --datasets with no names, an unknown or repeated name or another option after it (the fit launcher's check, 2), and results that exist even with an inherited ARK_EMB_FIT_RESUME=1; every one keeps the "
      "launcher's own message and ends with 'nothing was submitted'; the wrapper's own usage (2) for fewer than four arguments or a fifth that is not --datasets; exit 1 inside a Slurm job")

# a failed submission: the extraction alone (nothing is queued, sbatch's own message); a fit that cannot be submitted after the extraction was -> the extraction is cancelled by id; a printed id that is no id
done, sbatch, scancel, calls = run_submit(base, SBATCH_FAIL_ON="1")
assert done.returncode == 1 and done.stdout == "" and len(sbatch) == 1 and scancel == [], (done, sbatch, scancel)
assert done.stderr == "sbatch: error: stub failure\nERROR: sbatch refused the extraction job; nothing was submitted\n", done.stderr
done, sbatch, scancel, calls = run_submit(base, SBATCH_FAIL_ON="2")
assert done.returncode == 1 and done.stdout == "" and len(sbatch) == 2 and scancel == ["1001"], (done, sbatch, scancel)
assert done.stderr == "sbatch: error: stub failure\nERROR: the fit job was not submitted (sbatch printed ''); extraction job 1001 cancelled. Fix the cause and run this again\n", done.stderr
done, sbatch, scancel, calls = run_submit(base, SBATCH_FAIL_ON="2", SCANCEL_FAIL="1")
assert done.returncode == 1 and scancel == ["1001"] and "WARNING: scancel 1001 failed: cancel the extraction by hand" in done.stderr, done
done, sbatch, scancel, calls = run_submit(base, SBATCH_PRINT_2="oops")
assert done.returncode == 1 and done.stdout == "" and scancel == ["1001"] and "the fit job was not submitted (sbatch printed 'oops'); extraction job 1001 cancelled" in done.stderr, done
done, sbatch, scancel, calls = run_submit(base, SBATCH_PRINT_1="not-an-id")
assert done.returncode == 1 and done.stdout == "" and len(sbatch) == 1 and scancel == [] and done.stderr == "ERROR: sbatch printed 'not-an-id' for the extraction, not a job id; the fit was not submitted\n", done
print("OK (j) submit_probe_class.sh failures: sbatch failing on the extraction submits nothing more; failing (or printing no id) on the fit cancels the extraction by id (scancel 1001), a failed scancel is a WARNING; "
      "a non-numeric extraction id stops before the fit; the fit is never submitted without --dependency=afterok:<the extraction's id>")

print("ALL OK")
