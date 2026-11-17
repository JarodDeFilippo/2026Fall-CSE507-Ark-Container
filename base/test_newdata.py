"""CPU checks for the 512 px new-data run (chain/newdata512.sh): the three datasets added to Stage A's four (PadChest, TAIXRay, ReXGradient), SSLImageDataset's 16-bit-safe read and --pre_resize
squash, --no_gram with --gram_teacher_size = --global_crops_size, --init_new_heads_from (parse_args, load_init_state, and a tiny end-to-end start from a finished run's checkpoint with its --resume),
--pre_resize (its locks, and in that end-to-end start), the launch script and its sbatch wrapper in a fake repo, ssl_common.sh's SSL_DATASETS and gaudi.sh's EXTRA_BINDS.
Also its two variants and the 224 px run on the same seven datasets: --decode_min_side, base/datasets_config_taix_original.yaml (TAIXRay from the native files), chain/newdata512_cropsrc.sh and
chain/newdata224.sh with their wrappers, each against the launch line it copies (newdata512.sh, the 224 px Ark-teacher trunk's constlr.sh) to show the only differences, and the LR the trainer
computes from cropsrc's line (a base of 1e-4, a constant effective 1.4142e-4 at batch 128).
And the cosine-schedule run on the same seven datasets (chain/newdata224_cos50.sh, its wrapper and its pre-run smoke run_d1_8_newdata224_cos50_smoke_gaudi.sh): the launch line word for word, its
one-line settings, the 8-card refusals, and the smoke in a fake repo with a recording gaudi.sh (its launch lines against the real launcher's, the truncated lists, the summary, the exit codes, the refusals).
Run from base/: python test_newdata.py (needs torch, timm, torchvision, PIL; no Slurm, no Gaudi). The test_ functions are also collected by pytest."""
import contextlib
import glob
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import ssl_pretrain
from models import DINOV3_VITB16_KWARGS
from ssl_dinov3.data import DATASET_NAMES, DataAugmentationCXR, SSLImageDataset, build_data_loader, image_paths
from ssl_dinov3.model import build_models
from ssl_dinov3.optim import build_schedules, lr_scaling_factor
from utils import get_config

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
NEW = ("PadChest", "TAIXRay", "ReXGradient")
OLD = ("VinDrCXR", "ChestXray14", "CheXpert", "MIMIC")
SEVEN = ["MIMIC", "ReXGradient", "CheXpert", "TAIXRay", "PadChest", "ChestXray14", "VinDrCXR"]  # chain/newdata512.sh's SSL_DATASETS
TINY_VIT = dict(DINOV3_VITB16_KWARGS, embed_dim=64, depth=2, num_heads=2, ffn_ratio=2)
TINY_HEADS = dict(out_dim=128, hidden_dim=64, bottleneck_dim=32, nlayers=3)
P = lambda *extra: ssl_pretrain.parse_args(["--data_set", "MIMIC", "--pretrained_weights", "w", "--output_dir", "o"] + list(extra))


def raises(exc, fn, *args, **kwargs):
    """The message of the exception fn raises; fails when it returns."""
    try:
        fn(*args, **kwargs)
    except exc as error:
        return str(error)
    raise AssertionError("expected %s" % exc.__name__)


def refused(*extra, words=()):
    """parse_args exits 2 on these options and its error holds every word."""
    with contextlib.redirect_stderr(io.StringIO()) as err:
        try:
            P(*extra)
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError("expected SystemExit for %s" % (extra,))
    complaint = err.getvalue().split("error: ", 1)[1]
    assert all(word in complaint for word in words), (extra, complaint)


def test_lists_and_config():
    assert DATASET_NAMES == OLD + NEW
    with tempfile.TemporaryDirectory() as tmp:
        listing = os.path.join(tmp, "list.csv")
        with open(listing, "w") as f:
            f.write("Path\n0/a_1.png\n49/b-2.png\n")
        for name in NEW:  # one format: a csv with a header row, the image path in column 0, joined to data_dir
            assert image_paths(name, "/data/x/", listing) == ["/data/x/0/a_1.png", "/data/x/49/b-2.png"], name
        assert "unknown dataset" in raises(ValueError, image_paths, "Shenzhen", "/data/x/", listing)
    config = get_config(os.path.join(BASE, "datasets_config.yaml"))
    assert all(name in config for name in DATASET_NAMES)
    for name in NEW:  # pretraining only: a train list and no val_list, test_list or diseases
        assert set(config[name]) == {"data_dir", "train_list", "task_type"}, name
    # the shipped PadChest list: a header and the 160,828 paths of the source list (<dir>/<name>.png, no duplicates)
    listing = os.path.join(BASE, config["PadChest"]["train_list"])
    paths = image_paths("PadChest", config["PadChest"]["data_dir"], listing)
    assert open(listing).readline() == "Path\n" and len(paths) == len(set(paths)) == 160828
    assert all(re.fullmatch(r"\d+/[^/,\s]+\.png", os.path.relpath(p, config["PadChest"]["data_dir"])) for p in paths)


def old_read(path, side=1024):
    """SSLImageDataset's read before the 16-bit branch existed: draft, then convert("L")."""
    with Image.open(path) as img:
        img.draft("L", (side, side))
        return img.convert("L")


def test_sslimagedataset_reads():
    rng = np.random.RandomState(0)
    grey = rng.randint(0, 256, (40, 50), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name, image, ext in (("l", Image.fromarray(grey), "png"), ("rgb", Image.fromarray(rng.randint(0, 256, (40, 50, 3), dtype=np.uint8)), "png"),
                                 ("rgba", Image.fromarray(rng.randint(0, 256, (40, 50, 4), dtype=np.uint8)), "png"), ("p", Image.fromarray(grey).convert("P"), "png"),
                                 ("jpeg", Image.fromarray(grey), "jpeg")):
            paths.append(os.path.join(tmp, name + "." + ext))
            image.save(paths[-1])
        dataset = SSLImageDataset(paths, lambda img: img)
        for i, path in enumerate(paths):  # 8-bit inputs: bit for bit the convert("L") read of before
            out, target = dataset[i]
            assert target == () and out.mode == "L" and out.tobytes() == old_read(path).tobytes(), path

        def save(array, ext):
            path = os.path.join(tmp, "x%d.%s" % (len(os.listdir(tmp)), ext))
            Image.fromarray(array).save(path)
            return path

        read = lambda array, ext="png": np.asarray(SSLImageDataset([save(array, ext)], lambda img: img)[0][0])
        a = rng.randint(0, 4001, (40, 50)).astype(np.uint16)  # a 12-bit radiograph in a 16-bit PNG
        a[0, :2] = 0, 4000
        assert Image.open(save(a, "png")).mode in ("I;16", "I")  # how Pillow reads it
        out = read(a)
        assert out.dtype == np.uint8 and (out.min(), out.max()) == (0, 255)  # scaled per image, min to 0 and max to 255
        assert np.abs(out.astype(int) - np.floor(a * 255.0 / 4000)).max() <= 1
        assert (np.diff(out.ravel()[np.argsort(a.ravel(), kind="stable")]) >= 0).all()  # monotone in the input
        assert (out == 255).mean() < 0.02 and (np.asarray(Image.open(save(a, "png")).convert("L")) == 255).mean() > 0.9  # no clipping, which convert("L") does
        assert np.array_equal(read(a + 100), out)  # min-max: an offset changes nothing
        assert not read(np.full((8, 8), 1234, np.uint16)).any()  # a flat image is all 0
        out = read(a.astype(np.float32), "tiff")  # float images too
        assert out.dtype == np.uint8 and (out.min(), out.max()) == (0, 255)


def test_pre_resize_squash():
    rng = np.random.RandomState(0)
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name, array in (("i16", rng.randint(0, 4001, (2000, 1500)).astype(np.uint16)), ("l", rng.randint(0, 256, (400, 300), dtype=np.uint8))):  # 1500 x 2000 (12 bit in a 16-bit PNG), 300 x 400
            paths.append(os.path.join(tmp, name + ".png"))
            Image.fromarray(array).save(paths[-1])
        assert Image.open(paths[0]).mode in ("I;16", "I") and Image.open(paths[1]).mode == "L"
        plain, squashed = SSLImageDataset(paths, lambda img: img), SSLImageDataset(paths, lambda img: img, pre_resize=512)
        assert (plain.pre_resize, squashed.pre_resize) == (0, 512)
        for i, size in enumerate(((1500, 2000), (300, 400))):
            full, out = plain[i][0], squashed[i][0]
            assert full.mode == "L" and full.size == size, (i, full.mode, full.size)  # off: the whole image, as before
            assert out.mode == "L" and out.size == (512, 512), (i, out.mode, out.size)  # on: a 512 x 512 square, up or down
            assert out.tobytes() == full.resize((512, 512), Image.BICUBIC).tobytes(), i  # the squash comes after the grey conversion, BICUBIC like the Ark loader's resize


def test_pre_resize_flag():
    assert P().pre_resize == 0 == ssl_pretrain.ARG_DEFAULTS["pre_resize"] and P("--pre_resize", "512").pre_resize == 512
    assert "pre_resize" in ssl_pretrain.RESUME_MATCH and "pre_resize" not in ssl_pretrain.BRANCH_FREE
    refused("--pre_resize", "-1", words=("--pre_resize", "-1"))
    # locked on --resume and by --branch_from (it changes the inputs); a checkpoint written before the flag reads as 0
    ckpt = dict(args={k: v for k, v in vars(P()).items() if k != "pre_resize"}, world_size=1, step=0, epoch=0, data_fingerprint="f")
    for skip, action in (((), "--resume"), (ssl_pretrain.BRANCH_FREE, "--branch_from")):
        ssl_pretrain.check_resume(ckpt, P(), 1, 1, "f", skip, action)
        message = raises(ValueError, ssl_pretrain.check_resume, ckpt, P("--pre_resize", "512"), 1, 1, "f", skip, action)
        assert "cannot " + action in message and "pre_resize: checkpoint 0, now 512" in message, message


def test_decode_min_side_flag():
    assert P().decode_min_side == 0 == ssl_pretrain.ARG_DEFAULTS["decode_min_side"] and P("--decode_min_side", "1024").decode_min_side == 1024
    assert "decode_min_side" in ssl_pretrain.RESUME_MATCH and "decode_min_side" not in ssl_pretrain.BRANCH_FREE
    refused("--decode_min_side", "-1", words=("--decode_min_side", "-1"))
    refused("--decode_min_side", "1024", "--pre_resize", "512", words=("--decode_min_side cannot be combined with --pre_resize",))  # the squash size is the decode side then
    # locked on --resume and by --branch_from (it changes the inputs); a checkpoint written before the flag reads as 0, so every earlier launch line resumes as it did
    ckpt = dict(args={k: v for k, v in vars(P()).items() if k != "decode_min_side"}, world_size=1, step=0, epoch=0, data_fingerprint="f")
    for skip, action in (((), "--resume"), (ssl_pretrain.BRANCH_FREE, "--branch_from")):
        ssl_pretrain.check_resume(ckpt, P(), 1, 1, "f", skip, action)
        message = raises(ValueError, ssl_pretrain.check_resume, ckpt, P("--decode_min_side", "1024"), 1, 1, "f", skip, action)
        assert "cannot " + action in message and "decode_min_side: checkpoint 0, now 1024" in message, message


def test_no_gram_gram_teacher_size_equal_to_global():
    hires = ["--global_crops_size", "512", "--local_crops_size", "224"]
    parsed = P("--no_gram", *hires, "--gram_teacher_size", "512")
    assert (parsed.no_gram, parsed.gram_teacher_size, parsed.gram_weight, parsed.gram_refresh_steps) == (True, 512, 0.0, []) and ssl_pretrain.check_args(parsed) == 40
    assert P("--no_gram").gram_teacher_size == 448 and P("--no_gram", "--global_crops_size", "512").gram_teacher_size == 1024  # the default is still 2 x global
    for extra in (["--gram_teacher_size", "448"], ["--global_crops_size", "512", "--gram_teacher_size", "1024"], ["--global_crops_size", "512", "--gram_teacher_size", "528"]):
        refused("--no_gram", *extra, words=("--gram_teacher_size cannot be combined with --no_gram", "cut at"))  # any other explicit value
    refused("--no_gram", "--gram_weight", "2", words=("--gram_weight",))  # the other Gram knobs stay refused
    refused("--no_gram", "--gram_refresh_steps", "10", words=("--gram_refresh_steps",))
    assert P(*hires, "--gram_teacher_size", "512").no_gram is False  # a Gram run at Gram = global stays valid
    # the crops are cut at the global size (not 2 x) and the decode side follows the cut
    aug = DataAugmentationCXR(global_crops_size=64, local_crops_size=32, gram_teacher_crops_size=64, emit_gram_teacher_crops=False)
    assert tuple(aug.geometric_augmentation_global.transforms[0].size) == (64, 64)
    out = aug(Image.fromarray(np.random.RandomState(0).randint(0, 256, (200, 300), dtype=np.uint8)))
    assert "gram_teacher_crops" not in out and out["global_crops"][0].shape == (3, 64, 64) and out["local_crops"][0].shape == (3, 32, 32)
    twice = DataAugmentationCXR(global_crops_size=64, local_crops_size=32, gram_teacher_crops_size=128, emit_gram_teacher_crops=False)
    assert tuple(twice.geometric_augmentation_global.transforms[0].size) == (128, 128)
    assert DataAugmentationCXR(global_crops_size=512, local_crops_size=224, gram_teacher_crops_size=512, emit_gram_teacher_crops=False).decode_min_side == 1344
    assert DataAugmentationCXR(global_crops_size=512, local_crops_size=224, gram_teacher_crops_size=1024).decode_min_side == 2368


def test_init_new_heads_from_parse():
    ark = ["--init_from_checkpoint", "c.pth", "--cyclic_ark_teacher", "--heads", "per_dataset"]
    assert P().init_new_heads_from is None and ssl_pretrain.ARG_DEFAULTS["init_new_heads_from"] is None  # an old checkpoint reads as None
    assert "init_new_heads_from" in ssl_pretrain.RESUME_MATCH and "init_new_heads_from" in ssl_pretrain.BRANCH_FREE
    for source in ("MIMIC", "random"):
        assert P(*ark, "--init_new_heads_from", source).init_new_heads_from == source
    refused("--init_new_heads_from", "MIMIC", words=("--init_new_heads_from needs --init_from_checkpoint",))
    refused("--init_from_checkpoint", "c.pth", "--init_new_heads_from", "MIMIC", words=("--init_new_heads_from needs --heads per_dataset",))
    refused(*ark, "--init_new_heads_from", "bogus", words=("--init_new_heads_from", "invalid choice"))


def tiny_models(heads, seed):
    torch.manual_seed(seed)
    student, teacher, _ = build_models(TINY_VIT, None, TINY_HEADS, gram=False, heads=heads)
    return student, teacher


def part(state, prefix):  # the entries of a state dict under a prefix, the prefix dropped
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_load_init_state():
    old_student, old_teacher = tiny_models(["A", "B"], 0)
    with torch.no_grad():  # the teacher's heads differ from the student's, as after a blend: a copy from the wrong side would show
        for head in ("dino_head", "ibot_head"):
            for p in old_teacher[head].parameters():
                p.add_(1.0)
    init = dict(student=old_student.state_dict(), teacher=old_teacher.state_dict(), args=dict(data_set=["A", "B"], heads="per_dataset"))
    args = SimpleNamespace(data_set=["A", "B", "C"], heads="per_dataset", init_new_heads_from="B")
    student, teacher = tiny_models(["A", "B", "C"], 1)
    assert ssl_pretrain.load_init_state(student, teacher, init, args) == ["C"]  # strict=False, no unexpected key, no missing one but C's
    for what, module in (("student", student), ("teacher", teacher)):
        state = module.state_dict()
        assert same(part(state, "backbone."), part(init[what], "backbone."))
        for head in ("dino_head", "ibot_head"):
            assert same(part(state, head + ".A."), part(init[what], head + ".A.")) and same(part(state, head + ".B."), part(init[what], head + ".B."))  # A, B: the checkpoint's
            assert same(part(state, head + ".C."), part(init[what], head + ".B."))  # C: B's, from the same side (student's from the student, teacher's from the teacher)
    assert not same(part(student.state_dict(), "dino_head.C."), part(teacher.state_dict(), "dino_head.C."))
    with torch.no_grad():  # a copy, not shared storage
        next(student["dino_head"]["C"].parameters()).add_(1.0)
    assert same(part(student.state_dict(), "dino_head.B."), part(init["student"], "dino_head.B."))
    # "random": C keeps its fresh init, the rest loads
    student, teacher = tiny_models(["A", "B", "C"], 1)
    fresh_c = {k: v.clone() for k, v in student.state_dict().items() if ".C." in k}
    assert ssl_pretrain.load_init_state(student, teacher, init, SimpleNamespace(**dict(vars(args), init_new_heads_from="random"))) == ["C"]
    assert same({k: v for k, v in student.state_dict().items() if ".C." in k}, fresh_c) and same(part(student.state_dict(), "dino_head.A."), part(init["student"], "dino_head.A."))
    # equal data sets without the flag: the strict load, every tensor the checkpoint's
    student, teacher = tiny_models(["A", "B"], 1)
    assert ssl_pretrain.load_init_state(student, teacher, init, SimpleNamespace(data_set=["B", "A"], heads="per_dataset", init_new_heads_from=None)) == []
    assert same(student.state_dict(), init["student"]) and same(teacher.state_dict(), init["teacher"])
    # refusals: the models stay as they were
    student, teacher = tiny_models(["A", "B", "C"], 1)
    before = {k: v.clone() for k, v in student.state_dict().items()}
    for changes, words in ((dict(init_new_heads_from="C"), ("--init_new_heads_from C is not a dataset of the checkpoint (A, B)",)),  # a new dataset is no source
                           (dict(init_new_heads_from="MIMIC"), ("is not a dataset of the checkpoint",)),
                           (dict(init_new_heads_from=None), ("adds C", "pass --init_new_heads_from", "random")),
                           (dict(data_set=["A", "B"]), ("no dataset is new",)),
                           (dict(data_set=["B", "C"]), ("--data_set lacks A",)),
                           (dict(data_set=["B", "C"], init_new_heads_from=None), ("--data_set lacks A",))):
        message = raises(ValueError, ssl_pretrain.load_init_state, student, teacher, init, SimpleNamespace(**dict(vars(args), **changes)))
        assert all(word in message for word in words), (changes, message)
        assert same(student.state_dict(), before)
    shared = dict(init, args=dict(init["args"], heads="shared"))
    assert "shared heads" in raises(ValueError, ssl_pretrain.load_init_state, student, teacher, shared, args)
    # a checkpoint that does not fit beyond the new heads is refused, naming the key: a missing one, an unexpected one
    missing = dict(init, student={k: v for k, v in init["student"].items() if k != "dino_head.A.last_layer.weight"})
    assert "dino_head.A.last_layer.weight" in raises(ValueError, ssl_pretrain.load_init_state, student, teacher, missing, args)
    extra = dict(init, teacher=dict(init["teacher"], surprise=torch.zeros(1)))
    assert "surprise" in raises(ValueError, ssl_pretrain.load_init_state, student, teacher, extra, args)


def synthetic(root, counts):
    """Tiny images (8-bit PNGs, VinDrCXR's JPEGs, PadChest's 16-bit PNGs) and list files in the formats image_paths reads; returns the datasets_config dict."""
    rng, config = np.random.RandomState(0), {}
    for name, count in counts.items():
        data_dir, listing, rows = os.path.join(root, name), os.path.join(root, name + ".list"), []
        os.makedirs(data_dir)
        for i in range(count):
            h, w = (60, 80) if i % 2 else (90, 70)
            ext = "jpeg" if name == "VinDrCXR" else "png"
            image = rng.randint(0, 4096, (h, w)).astype(np.uint16) if name == "PadChest" else rng.randint(0, 256, (h, w), dtype=np.uint8)
            Image.fromarray(image).save(os.path.join(data_dir, "i%d.%s" % (i, ext)))
            rows.append({"ChestXray14": "i%d.png 0 1" % i, "VinDrCXR": "i%d 0 1" % i}.get(name, "i%d.png,F" % i))
        with open(listing, "w") as f:
            f.write(("" if name in ("ChestXray14", "VinDrCXR") else "Path,Sex\n") + "\n".join(rows) + "\n")
        config[name] = dict(data_dir=data_dir, train_list=listing)
    return config


def test_new_datasets_end_to_end():
    def args_for(out, datasets, *extra):  # 64 / 32 crops, 2 per card, one rank: a cycle of every dataset's release pass
        argv = [x for name in datasets for x in ("--data_set", name)] + [
            "--output_dir", out, "--device", "cpu", "--workers", "0", "--batch_size_per_gpu", "2", "--epochs", "3", "--warmup_epochs", "1", "--teacher_temp_warmup_epochs", "1",
            "--print_freq", "2", "--global_crops_size", "64", "--local_crops_size", "32", "--pretrained_weights", "unused", "--no_gram", "--cyclic_ark_teacher", "--heads",
            "per_dataset", "--no_lock_half_pass", "--keep_all_checkpoints"] + list(extra)
        args = ssl_pretrain.parse_args(argv)
        args.pretrained_weights = None  # random init: the tiny ViT has no stock weights
        return args

    last = {}  # the latest go(): the sizes of the images the crops were cut from (--workers 0: in this process) and the datasets build_dataset made

    def go(config, out, datasets, *extra):
        buf, sizes, built = io.StringIO(), set(), []
        real_call, real_build = DataAugmentationCXR.__call__, ssl_pretrain.build_dataset
        DataAugmentationCXR.__call__ = lambda self, image: (sizes.add(image.size), real_call(self, image))[1]
        ssl_pretrain.build_dataset = lambda *a, **k: built.append(real_build(*a, **k)) or built[-1]
        try:
            with contextlib.redirect_stdout(buf):
                summary = ssl_pretrain.run(args_for(out, datasets, *extra), TINY_VIT, TINY_HEADS, config)
        finally:
            DataAugmentationCXR.__call__, ssl_pretrain.build_dataset = real_call, real_build
        last.update(sizes=sizes, built=built)
        return summary, buf.getvalue()

    with tempfile.TemporaryDirectory() as tmp:
        counts = dict(MIMIC=12, CheXpert=8, ChestXray14=6, VinDrCXR=4, PadChest=6, TAIXRay=4, ReXGradient=4)
        config = synthetic(os.path.join(tmp, "data"), counts)
        # the finished 224-style trunk: the four Stage A sets, crops cut at 2 x global, 15 steps a cycle (6 + 4 + 3 + 2)
        trunk = os.path.join(tmp, "trunk")
        summary, text = go(config, trunk, OLD, "--stop_after_epochs", "1")
        assert summary["epoch"] == 1 and summary["epoch_length"] == 15
        assert "no Gram teacher (global crops cut at 128 and resized to 64, as in the Gram runs); JPEG decode min side" in text
        assert last["sizes"] == {(80, 60), (70, 90)} and [d.pre_resize for d in last["built"]] == [0]  # off: the crops are cut from the images as decoded
        source = os.path.join(trunk, "ckpt", "epoch_0001.pth")
        trunk_ckpt = torch.load(source, map_location="cpu", weights_only=False)  # the real trunk predates --pre_resize: no such key in its args
        assert trunk_ckpt["args"]["pre_resize"] == 0
        del trunk_ckpt["args"]["pre_resize"]
        torch.save(trunk_ckpt, source)
        # the new run: seven sets, crops cut at the global size from images squashed to 64 x 64 first (--init_from_checkpoint compares no setting of its source), the new heads from MIMIC's
        new = os.path.join(tmp, "new")
        extra = ["--gram_teacher_size", "64", "--pre_resize", "64", "--init_from_checkpoint", source, "--init_new_heads_from", "MIMIC"]
        summary, text = go(config, new, SEVEN, "--stop_after_epochs", "1", *extra)
        assert summary["epoch"] == 1 and summary["epoch_length"] == 6 + 4 + 3 + 2 + 3 + 2 + 2
        assert last["sizes"] == {(64, 64)} and [(d.pre_resize, d.decode_side) for d in last["built"]] == [(64, 64)]  # every pass's images were squashed (the mixed set too, which is not iterated here)
        assert ("no Gram teacher (global crops cut at 64, no resize); pre-resize 64x64 (whole image squashed to a square before the random crops, Ark loader convention), "
                "so JPEG decode min side 64") in text
        assert "teacher (heads included) loaded strictly but for the new datasets' heads (below)" in text
        assert ("new heads: ReXGradient, TAIXRay, PadChest (not among the checkpoint's VinDrCXR, ChestXray14, CheXpert, MIMIC): their DINO and iBOT heads start as copies of "
                "MIMIC's (student's from the checkpoint's student, teacher's from its teacher)") in text
        refused_dir = os.path.join(tmp, "refused")  # seven sets from a four-set checkpoint need the flag; a refused run creates nothing
        message = raises(ValueError, go, config, refused_dir, SEVEN, "--stop_after_epochs", "1", *extra[:-2])
        assert "--data_set adds ReXGradient, TAIXRay, PadChest" in message and "pass --init_new_heads_from" in message and not os.path.exists(refused_dir), message
        info = json.load(open(os.path.join(new, "args.json")))
        assert (info["init_new_heads_from"], info["gram_teacher_size"], info["global_crops_size"], info["no_gram"], info["data_set"]) == ("MIMIC", 64, 64, True, SEVEN)
        assert (info["pre_resize"], info["decode_min_side"]) == (64, 64)  # the effective decode side: the squash size, not the crops' 1024
        ckpt = torch.load(os.path.join(new, "ckpt", "epoch_0001.pth"), map_location="cpu", weights_only=False)
        assert {k.split(".")[1] for k in ckpt["student"] if k.startswith("dino_head.")} == set(SEVEN) and ckpt["gram_teacher"] is None
        rows = open(os.path.join(new, "ssl_losses.csv")).read().splitlines()
        assert {r.split(",")[-1].split(":")[0] for r in rows[1:]} == set(SEVEN)  # every dataset trained, PadChest's 16-bit PNGs included
        # --resume: the flag is locked (a checkpoint without the key reads as None), and the same line resumes
        message = raises(ValueError, go, config, new, SEVEN, "--resume", "--stop_after_epochs", "2", *extra[:-1], "random")
        assert "cannot --resume" in message and "init_new_heads_from: checkpoint 'MIMIC', now 'random'" in message, message
        message = raises(ValueError, go, config, new, SEVEN, "--resume", "--stop_after_epochs", "2", *extra[:2], *extra[4:])  # without --pre_resize: locked too
        assert "cannot --resume" in message and "pre_resize: checkpoint 64, now 0" in message, message
        summary, text = go(config, new, SEVEN, "--resume", "--stop_after_epochs", "2", *extra)
        assert summary["epoch"] == 2 and "resumed from" in text and "initialized from" not in text
        old = torch.load(source, map_location="cpu", weights_only=False)  # an older checkpoint: no such key in its args
        assert "init_new_heads_from" in old["args"]
        del old["args"]["init_new_heads_from"]
        torch.save(old, source)
        message = raises(ValueError, go, config, trunk, OLD, "--resume", "--stop_after_epochs", "2", "--init_from_checkpoint", source, "--init_new_heads_from", "MIMIC")
        assert "init_new_heads_from: checkpoint None, now 'MIMIC'" in message, message
        summary, text = go(config, trunk, OLD, "--resume", "--stop_after_epochs", "2")  # without the flag it resumes, as it always did
        assert summary["epoch"] == 2 and "resumed from" in text


INIT = "/workspace/ssl_dinov3_cxr_constlr_arkteacher_seed_100/ckpt/epoch_0200.pth"
RUN = "ssl_dinov3_cxr_newdata512_arkteacher_seed_100"
WEIGHT_NAME = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
SETTINGS = {"main_lr.txt": "0.002\n", "newdata512_stop_after.txt": "5\n", "newdata512_batch_per_gpu.txt": "8\n"}
ENV = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "HABANA_VISIBLE_", "SSL_DATASETS", "EXTRA_BINDS"))}
ENV4 = dict(ENV, SLURM_JOB_GPUS="0,1,2,3", USER="tester")  # the 4-card allocation ssl_launch counts


def fake_repo(tmp, settings=SETTINGS, init=True):
    """A fake repo root: the real phase script, ssl_common.sh and Ark's common.sh, a gaudi.sh and card checks that record how they were called, the stock weights, the Ark trunk's
    epoch-200 checkpoint (init=False: absent) and the settings files (a name mapped to None: absent)."""
    root = os.path.join(tmp, "repo")
    for directory in ("chain", "experiment_scripts", "weights", os.path.dirname(INIT.replace("/workspace/", ""))):
        os.makedirs(os.path.join(root, directory))
    for name in ("newdata512.sh", "newdata512_cropsrc.sh", "newdata224.sh", "newdata224_cos50.sh", "constlr.sh", "ssl_common.sh"):  # constlr.sh: the 224 px trunk's launcher, which newdata224.sh is compared with
        shutil.copy(os.path.join(REPO, "chain", name), os.path.join(root, "chain", name))
    shutil.copy(os.path.join(REPO, "experiment_scripts", "common.sh"), os.path.join(root, "experiment_scripts"))
    for name, body in (("gaudi.sh", 'printf "%s\\n" "$@" > "$PWD/argv.txt"; printf "%s" "${EXTRA_BINDS:-}" > "$PWD/extra_binds.txt"'),
                       ("gaudi_check_cards.sh", 'touch "$PWD/cards_checked"'), ("gaudi_verify_busy.sh", 'printf "%s\\n" "$1" > "$PWD/busy_arg.txt"')):
        with open(os.path.join(root, name), "w") as f:
            f.write("#!/usr/bin/env bash\n" + body + "\n")
        os.chmod(os.path.join(root, name), 0o755)
    for path, present in (("weights/" + WEIGHT_NAME, True), (INIT.replace("/workspace/", ""), init)):
        if present:
            open(os.path.join(root, path), "w").write("x")
    for name, content in settings.items():
        if content is not None:
            open(os.path.join(root, "chain", name), "w").write(content)
    return root


def launch(root, script="chain/newdata512.sh", env=ENV4, cwd=None):
    path = os.path.join(root, "argv.txt")
    if os.path.exists(path):
        os.remove(path)  # what an earlier launch recorded must not pass for this one's
    result = subprocess.run(["bash", script], cwd=cwd or root, env=env, capture_output=True, text=True)
    return result, (open(path).read().splitlines() if os.path.exists(path) else None)


def flags(argv):
    """The trainer's part of a launch line as {flag: [the value list of each time it was given]} (--data_set once per dataset, --batch_size_per_gpu twice: ssl_launch's default, then the run's)."""
    out, current = {}, None
    for word in argv[argv.index("ssl_pretrain.py") + 1:]:
        if word.startswith("--"):
            current = out.setdefault(word, [])
            current.append([])
        else:
            current[-1].append(word)
    return out


def changed(a, b):
    """The flags two launch lines do not give alike (one given in only one of them counts)."""
    fa, fb = flags(a), flags(b)
    return {flag for flag in fa.keys() | fb.keys() if fa.get(flag) != fb.get(flag)}


parse = lambda argv: ssl_pretrain.parse_args(argv[argv.index("ssl_pretrain.py") + 1:])


def test_newdata512_script():
    script = os.path.join(REPO, "chain", "newdata512.sh")
    subprocess.run(["bash", "-n", script], check=True)
    assert os.access(script, os.X_OK) and open(script).read().startswith("#!/usr/bin/env bash\n")
    assert "newdata512" not in [line.split("#")[0].strip() for line in open(os.path.join(REPO, "chain", "phases.txt"))]  # not a chain phase
    expected = (["python", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "4", "ssl_pretrain.py", "--device", "hpu"]
                + [x for name in SEVEN for x in ("--data_set", name)]
                + ["--pretrained_weights", "/workspace/weights/" + WEIGHT_NAME, "--batch_size_per_gpu", "64", "--workers", "16", "--seed", "100", "--output_dir", "/workspace/" + RUN, "--resume",
                   "--global_crops_size", "512", "--local_crops_size", "224", "--gram_teacher_size", "512", "--pre_resize", "512", "--batch_size_per_gpu", "8", "--lr", "0.001", "--min_lr", "0.001",
                   "--epochs", "1000", "--stop_after_epochs", "5", "--warmup_epochs", "2", "--teacher_temp_warmup_epochs", "2",
                   "--drop_path_rate", "0.3", "--rope_rescale_coords", "2", "--untie_local_cls_norm", "--no_gram", "--cyclic_ark_teacher", "--heads", "per_dataset", "--no_lock_half_pass",
                   "--keep_all_checkpoints", "--init_from_checkpoint", INIT, "--init_new_heads_from", "MIMIC"])
    with tempfile.TemporaryDirectory() as tmp:
        root = fake_repo(tmp)
        result, argv = launch(root, env=dict(ENV4, SSL_DATASETS="VinDrCXR"))  # an inherited SSL_DATASETS does not change the seven
        assert result.returncode == 0 and result.stderr == "" and argv == expected, (result.stderr, argv)
        lines = result.stdout.splitlines()
        assert lines[0].startswith("run: " + RUN + " (") and "from chain/newdata512_stop_after.txt)" in lines[0] and "from " + INIT in lines[0], lines[0]
        assert lines[1] == "LR: base 0.001 -> effective 0.000707107 at global batch 32 (8 x 4 cards; the trainer's sqrt_wrt_1024 rule; constlr: 0.002 at 256)", lines  # 0.002 / 2, x sqrt(32 / 256)
        parsed = ssl_pretrain.parse_args(argv[argv.index("ssl_pretrain.py") + 1:])  # the trainer accepts the line, and reads it as meant
        assert ssl_pretrain.check_args(parsed) == 5 and parsed.data_set == SEVEN and parsed.batch_size_per_gpu == 8 and parsed.output_dir == "/workspace/" + RUN
        assert (parsed.global_crops_size, parsed.local_crops_size, parsed.gram_teacher_size, parsed.no_gram, parsed.cyclic_ark_teacher, parsed.heads, parsed.lock_half_pass,
                parsed.keep_all_checkpoints) == (512, 224, 512, True, True, "per_dataset", False, True)
        assert (parsed.init_from_checkpoint, parsed.init_new_heads_from, parsed.epochs, parsed.warmup_epochs, parsed.teacher_temp_warmup_epochs) == (INIT, "MIMIC", 1000, 2, 2)
        assert parsed.pre_resize == 512  # the Ark loader convention: the whole image squashed to 512 x 512 before the crops
        assert (parsed.drop_path_rate, parsed.rope_rescale_coords, parsed.untie_local_cls_norm, parsed.no_lr_scaling) == (0.3, 2.0, True, False)
        assert parsed.lr == parsed.min_lr == 0.002 / lr_scaling_factor(256)  # the trainer scales it by 4 x sqrt(G / 1024): main_lr.txt exactly at G = 256
        # 2 cards, another batch and stop epoch: the cards and the LR adapt
        result, argv = launch(root, env=dict(ENV, SLURM_JOB_GPUS="0,1"))
        assert result.returncode == 0 and argv[5] == "2" and "at global batch 16 (8 x 2 cards" in result.stdout
        # the checkpoint to start from: refused when absent, unless the run already has an epoch checkpoint of its own (a resume reads no INIT)
        root = fake_repo(os.path.join(tmp, "no_init"), init=False)
        result, argv = launch(root)
        assert result.returncode == 1 and argv is None and result.stdout == "" and "ERROR: ./ssl_dinov3_cxr_constlr_arkteacher_seed_100/ckpt/epoch_0200.pth" in result.stderr, result
        os.makedirs(os.path.join(root, RUN, "ckpt"))
        open(os.path.join(root, RUN, "ckpt", "epoch_backup.pth"), "w").write("x")  # not an epoch checkpoint (the trainer's name test)
        assert launch(root)[0].returncode == 1
        open(os.path.join(root, RUN, "ckpt", "epoch_0003.pth"), "w").write("x")
        result, argv = launch(root)
        assert result.returncode == 0 and argv == expected, result.stderr
        # every settings file is needed and checked, naming it, before anything is launched
        for name in SETTINGS:
            result, argv = launch(fake_repo(tempfile.mkdtemp(dir=tmp), dict(SETTINGS, **{name: None})))
            assert result.returncode == 1 and argv is None and "ERROR" in result.stderr and name in result.stderr, (name, result.stderr)
        for name, value in (("main_lr.txt", "0"), ("main_lr.txt", "abc"), ("newdata512_stop_after.txt", "0"), ("newdata512_stop_after.txt", "2.5"), ("newdata512_stop_after.txt", "1001"),
                            ("newdata512_stop_after.txt", "18446744073709551617"), ("newdata512_batch_per_gpu.txt", "0"), ("newdata512_batch_per_gpu.txt", "abc"),
                            ("newdata512_batch_per_gpu.txt", "08"), ("newdata512_batch_per_gpu.txt", "12345")):
            result, argv = launch(fake_repo(tempfile.mkdtemp(dir=tmp), dict(SETTINGS, **{name: value + "\n"})))
            assert result.returncode == 1 and argv is None and "ERROR" in result.stderr and name in result.stderr, (name, value, result.stderr)
        result, argv = launch(fake_repo(tempfile.mkdtemp(dir=tmp), dict(SETTINGS, **{"newdata512_stop_after.txt": "1000\n"})))
        assert result.returncode == 0 and argv[argv.index("--stop_after_epochs") + 1] == "1000"


def test_newdata512_wrapper():
    wrapper = os.path.join(REPO, "run_d1_8_newdata512_arkteacher_gaudi.sh")
    subprocess.run(["bash", "-n", wrapper], check=True)
    assert os.access(wrapper, os.X_OK) and stat.S_IMODE(os.stat(wrapper).st_mode) == 0o755
    text = open(wrapper).read()
    lines = text.splitlines()
    assert [line[len("#SBATCH "):] for line in lines if line.startswith("#SBATCH ")] == [
        "--job-name=d1-8-newdata512", "-A grp_jliang12", "-p gaudi", "-q public", "-t 7-00:00:00", "-N 1", "--gres=gpu:hl225:4", "-c 72", "--mem=256G"], lines
    code = [line for line in lines if line and not line.startswith("#")]
    assert code[-1] == "EXTRA_BINDS=/scratch/$USER/datasets-new:/scratch/$USER/datasets-new:ro exec ./chain/newdata512.sh", code
    for needle in ("export PT_HPU_LAZY_MODE=1", "./gaudi_check_cards.sh || exit 1", "[[ -x ./gaudi_verify_busy.sh ]]", "./gaudi_verify_busy.sh 900 &"):
        assert any(needle in line for line in code), needle
    assert all(line.startswith("#") for line in lines if "phases.txt" in line)  # not a chain phase: phases.txt is named in comments only
    for needle in ("sbatch --gres=gpu:hl225:8 -c 144 --mem=480G run_d1_8_newdata512_arkteacher_gaudi.sh", "chain/newdata512_stop_after.txt", "chain/newdata512_batch_per_gpu.txt",
                   "6.5 GB", "690 GB per 100 cycles", "TAIXRay and ReXGradient entries", "datasets_config.yaml", "re-submit"):
        assert needle in text, needle
    cd = "cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1"
    assert text.count(cd) == 1
    with tempfile.TemporaryDirectory() as tmp:  # run from elsewhere, as sbatch does, with its cd pointed at the fake repo: the datasets mount reaches gaudi.sh, the card checks run first
        root = fake_repo(tmp)
        open(os.path.join(root, "wrapper.sh"), "w").write(text.replace(cd, "cd '%s' || exit 1" % root))
        elsewhere = tempfile.mkdtemp(dir=tmp)
        result, argv = launch(root, script=os.path.join(root, "wrapper.sh"), cwd=elsewhere)
        assert result.returncode == 0 and argv[-1] == "MIMIC" and "--init_new_heads_from" in argv, result.stderr
        assert open(os.path.join(root, "extra_binds.txt")).read() == "/scratch/tester/datasets-new:/scratch/tester/datasets-new:ro"
        assert os.path.exists(os.path.join(root, "cards_checked"))


RUN224 = "ssl_dinov3_cxr_newdata224_arkteacher_seed_100"
RUN_CROPSRC = "ssl_dinov3_cxr_newdata512_cropsrc_arkteacher_seed_100"
TRUNK = "ssl_dinov3_cxr_constlr_arkteacher_seed_100"
NATIVE_YAML = "datasets_config_taix_original.yaml"
SETTINGS_NEW = {"main_lr.txt": "0.002\n", "newdata224_stop_after.txt": "5\n", "constlr_arkteacher_stop_after.txt": "5\n", "newdata512_stop_after.txt": "5\n", "newdata512_batch_per_gpu.txt": "16\n",
                "newdata512_cropsrc_stop_after.txt": "5\n", "newdata512_cropsrc_batch_per_gpu.txt": "16\n"}  # the settings files of the two new launchers and of the two they are compared with
ENV8 = dict(ENV, SLURM_JOB_GPUS="0,1,2,3,4,5,6,7", USER="tester")  # the 8-card allocation newdata512 ran on


def test_native_taix_config():
    main = get_config(os.path.join(BASE, "datasets_config.yaml"))
    native = get_config(os.path.join(BASE, NATIVE_YAML))
    assert list(native) == ["ChestXray14", "CheXpert", "VinDrCXR", "MIMIC", "PadChest", "TAIXRay", "ReXGradient"] and sorted(native) == sorted(SEVEN)
    for name in SEVEN:  # the trainer reads data_dir and train_list; the other six datasets are the shipped config's, line for line
        want = {key: main[name][key] for key in ("data_dir", "train_list")}
        if name == "TAIXRay":
            want = dict(data_dir="/scratch/jdefili4/datasets-new/TAIX-Ray/png_original/", train_list="/scratch/jdefili4/datasets-new/TAIX-Ray/taix_original_train.csv")
        assert native[name] == want, name
    assert main["TAIXRay"]["data_dir"].endswith("/TAIX-Ray/png512/") and main["TAIXRay"]["train_list"].endswith("/TAIX-Ray/taix_train.csv")  # the entry the 512 px run and the probe read: untouched
    with tempfile.TemporaryDirectory() as tmp:  # the list taix_unpack_original.sh writes: Path, then <split>/<UID>.png, under png_original/
        listing = os.path.join(tmp, "taix_original_train.csv")
        open(listing, "w").write("Path\ntrain/a.png\ntrain/b.png\n")
        assert image_paths("TAIXRay", native["TAIXRay"]["data_dir"], listing) == ["/scratch/jdefili4/datasets-new/TAIX-Ray/png_original/train/a.png", "/scratch/jdefili4/datasets-new/TAIX-Ray/png_original/train/b.png"]


def check_wrapper(name, sbatch, chain_script, needles, env):
    """A new sbatch wrapper: bash -n, mode, #SBATCH lines, last line, the guards, the header's commands, and (in a fake repo, run from elsewhere as sbatch does) the datasets mount and the card check."""
    wrapper = os.path.join(REPO, name)
    subprocess.run(["bash", "-n", wrapper], check=True)
    assert os.access(wrapper, os.X_OK) and stat.S_IMODE(os.stat(wrapper).st_mode) == 0o755
    text = open(wrapper).read()
    lines = text.splitlines()
    assert [line[len("#SBATCH "):] for line in lines if line.startswith("#SBATCH ")] == sbatch, lines
    code = [line for line in lines if line and not line.startswith("#")]
    assert code[-1] == "EXTRA_BINDS=/scratch/$USER/datasets-new:/scratch/$USER/datasets-new:ro exec ./chain/" + chain_script, code
    for needle in ("export PT_HPU_LAZY_MODE=1", "./gaudi_check_cards.sh || exit 1", "[[ -x ./gaudi_verify_busy.sh ]]", "./gaudi_verify_busy.sh 900 &"):
        assert any(needle in line for line in code), needle
    assert all(line.startswith("#") for line in lines if "phases.txt" in line)  # not a chain phase: phases.txt is named in comments only
    for needle in needles:
        assert needle in text, needle
    cd = "cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1"
    assert text.count(cd) == 1
    with tempfile.TemporaryDirectory() as tmp:
        root = fake_repo(tmp, SETTINGS_NEW)
        open(os.path.join(root, "wrapper.sh"), "w").write(text.replace(cd, "cd '%s' || exit 1" % root))
        result, argv = launch(root, script=os.path.join(root, "wrapper.sh"), cwd=tempfile.mkdtemp(dir=tmp), env=env)
        assert result.returncode == 0 and argv[-1] == NATIVE_YAML, result.stderr  # both launch lines end with the config file
        assert open(os.path.join(root, "extra_binds.txt")).read() == "/scratch/tester/datasets-new:/scratch/tester/datasets-new:ro"
        assert os.path.exists(os.path.join(root, "cards_checked"))


def test_newdata224_script():
    script = os.path.join(REPO, "chain", "newdata224.sh")
    subprocess.run(["bash", "-n", script], check=True)
    assert os.access(script, os.X_OK) and open(script).read().startswith("#!/usr/bin/env bash\n")
    assert "newdata224" not in [line.split("#")[0].strip() for line in open(os.path.join(REPO, "chain", "phases.txt"))]  # not a chain phase
    expected = (["python", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "4", "ssl_pretrain.py", "--device", "hpu"]
                + [x for name in SEVEN for x in ("--data_set", name)]
                + ["--pretrained_weights", "/workspace/weights/" + WEIGHT_NAME, "--batch_size_per_gpu", "64", "--workers", "16", "--seed", "100", "--output_dir", "/workspace/" + RUN224, "--resume",
                   "--lr", "0.002", "--min_lr", "0.002", "--no_lr_scaling", "--epochs", "1000", "--stop_after_epochs", "5",
                   "--drop_path_rate", "0.3", "--rope_rescale_coords", "2", "--untie_local_cls_norm", "--no_gram", "--cyclic_ark_teacher", "--heads", "per_dataset", "--no_lock_half_pass",
                   "--keep_all_checkpoints", "--init_from_checkpoint", INIT, "--init_new_heads_from", "MIMIC", "--datasets_config", NATIVE_YAML])
    with tempfile.TemporaryDirectory() as tmp:
        root = fake_repo(tmp, SETTINGS_NEW)
        result, argv = launch(root, script="chain/newdata224.sh", env=dict(ENV4, SSL_DATASETS="VinDrCXR"))  # an inherited SSL_DATASETS does not change the seven
        assert result.returncode == 0 and result.stderr == "" and argv == expected, (result.stderr, argv)
        lines = result.stdout.splitlines()
        assert lines[0].startswith("run: " + RUN224 + " (") and "from chain/newdata224_stop_after.txt)" in lines[0] and "from " + INIT in lines[0], lines[0]
        assert lines[1] == "LR: 0.002 absolute (--no_lr_scaling) at global batch 256 (64 x 4 cards), flat after a 4-cycle warmup", lines
        # against the 224 px Ark-teacher trunk's own launch line (chain/constlr.sh CONSTLR_NOGRAM=1 CONSTLR_ARK=1): the data sets, the init pair, the run dir and the config file, and nothing else
        trunk_result, trunk = launch(root, script="chain/constlr.sh", env=dict(ENV4, CONSTLR_NOGRAM="1", CONSTLR_ARK="1"))
        assert trunk_result.returncode == 0 and flags(trunk)["--output_dir"] == [["/workspace/" + TRUNK]], trunk_result.stderr
        assert flags(trunk)["--data_set"] == [[name] for name in ("VinDrCXR", "ChestXray14", "MIMIC", "CheXpert")]
        assert changed(argv, trunk) == {"--data_set", "--output_dir", "--init_from_checkpoint", "--init_new_heads_from", "--datasets_config"}
        parsed, trunk_parsed = parse(argv), parse(trunk)  # the trainer accepts the line, and reads it as meant: the trunk's crops, decode side, batch, LR and warmups
        assert ssl_pretrain.check_args(parsed) == 5 and parsed.data_set == SEVEN and parsed.output_dir == "/workspace/" + RUN224
        assert (parsed.global_crops_size, parsed.local_crops_size, parsed.gram_teacher_size, parsed.no_gram, parsed.pre_resize, parsed.decode_min_side) == (224, 96, 448, True, 0, 0)
        assert DataAugmentationCXR(global_crops_size=224, local_crops_size=96, gram_teacher_crops_size=448, emit_gram_teacher_crops=False).decode_min_side == 1024  # STAGE_A_CROPS: the trunk's decode side
        assert (parsed.batch_size_per_gpu, parsed.warmup_epochs, parsed.teacher_temp_warmup_epochs, parsed.freeze_backbone_epochs, parsed.no_lr_scaling) == (64, 4, 12, 1, True)
        assert parsed.lr == parsed.min_lr == 0.002 and (parsed.init_from_checkpoint, parsed.init_new_heads_from, parsed.datasets_config) == (INIT, "MIMIC", NATIVE_YAML)
        assert {k for k in ssl_pretrain.RESUME_MATCH if k != "world_size" and getattr(parsed, k) != getattr(trunk_parsed, k)} == {"data_set", "init_from_checkpoint", "init_new_heads_from"}
        # 4 cards or nothing: another count would change the global batch and, the LR being absolute, the recipe
        for gpus in ("0,1", "0,1,2,3,4,5,6,7"):
            result, none = launch(root, script="chain/newdata224.sh", env=dict(ENV, SLURM_JOB_GPUS=gpus))
            assert result.returncode == 1 and none is None and "ERROR: %d cards in the allocation" % (gpus.count(",") + 1) in result.stderr, result
        # the checkpoint to start from: refused when absent, unless the run already has an epoch checkpoint of its own (a resume reads no INIT)
        bare = fake_repo(os.path.join(tmp, "no_init"), SETTINGS_NEW, init=False)
        result, none = launch(bare, script="chain/newdata224.sh")
        assert result.returncode == 1 and none is None and result.stdout == "" and "ERROR: ./ssl_dinov3_cxr_constlr_arkteacher_seed_100/ckpt/epoch_0200.pth" in result.stderr, result
        os.makedirs(os.path.join(bare, RUN224, "ckpt"))
        open(os.path.join(bare, RUN224, "ckpt", "epoch_0003.pth"), "w").write("x")
        result, resumed = launch(bare, script="chain/newdata224.sh")
        assert result.returncode == 0 and resumed == expected, result.stderr
        # both settings files are needed and checked, naming them, before anything is launched
        for name in ("main_lr.txt", "newdata224_stop_after.txt"):
            result, none = launch(fake_repo(tempfile.mkdtemp(dir=tmp), dict(SETTINGS_NEW, **{name: None})), script="chain/newdata224.sh")
            assert result.returncode == 1 and none is None and "ERROR" in result.stderr and name in result.stderr, (name, result.stderr)
        for name, value in (("main_lr.txt", "0"), ("main_lr.txt", "abc"), ("newdata224_stop_after.txt", "0"), ("newdata224_stop_after.txt", "2.5"), ("newdata224_stop_after.txt", "1001"),
                            ("newdata224_stop_after.txt", "18446744073709551617")):
            result, none = launch(fake_repo(tempfile.mkdtemp(dir=tmp), dict(SETTINGS_NEW, **{name: value + "\n"})), script="chain/newdata224.sh")
            assert result.returncode == 1 and none is None and "ERROR" in result.stderr and name in result.stderr, (name, value, result.stderr)
        result, top = launch(fake_repo(tempfile.mkdtemp(dir=tmp), dict(SETTINGS_NEW, **{"newdata224_stop_after.txt": "1000\n"})), script="chain/newdata224.sh")
        assert result.returncode == 0 and top[top.index("--stop_after_epochs") + 1] == "1000"


def test_newdata224_wrapper():
    check_wrapper("run_d1_8_newdata224_arkteacher_gaudi.sh", ["--job-name=d1-8-newdata224", "-A grp_jliang12", "-p gaudi", "-q public", "-t 7-00:00:00", "-N 1", "--gres=gpu:hl225:4", "-c 72", "--mem=256G"],
                  "newdata224.sh", ("A=$(sbatch --parsable run_d1_8_newdata224_arkteacher_gaudi.sh)", "sbatch -d afterany:$A run_d1_8_newdata224_arkteacher_gaudi.sh", "chain/newdata224_stop_after.txt",
                                    "taix_unpack_original.sh train", "6.5 GB", "690 GB per 100 cycles", "4763 steps"), ENV4)


def test_newdata512_cropsrc_script():
    script = os.path.join(REPO, "chain", "newdata512_cropsrc.sh")
    subprocess.run(["bash", "-n", script], check=True)
    assert os.access(script, os.X_OK) and open(script).read().startswith("#!/usr/bin/env bash\n")
    assert "newdata512_cropsrc" not in [line.split("#")[0].strip() for line in open(os.path.join(REPO, "chain", "phases.txt"))]  # not a chain phase
    with tempfile.TemporaryDirectory() as tmp:
        root = fake_repo(tmp, SETTINGS_NEW)
        result, argv = launch(root, script="chain/newdata512_cropsrc.sh", env=dict(ENV8, SSL_DATASETS="VinDrCXR"))  # an inherited SSL_DATASETS does not change the seven
        exemplar_result, exemplar = launch(root, env=ENV8)  # chain/newdata512.sh with the same settings, the run it copies
        assert result.returncode == 0 and result.stderr == "" and exemplar_result.returncode == 0, (result.stderr, exemplar_result.stderr)
        assert argv[5] == "8" and flags(argv)["--batch_size_per_gpu"] == [["64"], ["16"]] and flags(exemplar)["--pre_resize"] == [["512"]]
        # the exemplar's launch line with four edits and no others: the run dir, the base LR in --lr and --min_lr (1e-4, a tenth of the exemplar's 0.001: 1.4142e-4 effective at batch 128 where the exemplar's is 1.4142e-3),
        # --pre_resize 512 -> --decode_min_side 1024 (the same place), and the config file on the end
        want = list(exemplar)
        want[want.index("--output_dir") + 1] = "/workspace/" + RUN_CROPSRC
        for flag in ("--lr", "--min_lr"):
            assert want[want.index(flag) + 1] == "0.001"
            want[want.index(flag) + 1] = "0.0001"
        i = want.index("--pre_resize")
        want[i:i + 2] = ["--decode_min_side", "1024"]
        assert argv == want + ["--datasets_config", NATIVE_YAML], (argv, want)
        assert changed(argv, exemplar) == {"--output_dir", "--lr", "--min_lr", "--pre_resize", "--decode_min_side", "--datasets_config"}
        lines = result.stdout.splitlines()
        assert lines[0].startswith("run: " + RUN_CROPSRC + " (") and "from chain/newdata512_cropsrc_stop_after.txt)" in lines[0] and "with the new heads copied from MIMIC's" in lines[0], lines[0]
        assert lines[1] == "LR: base 0.0001 -> effective 0.000141421 at global batch 128 (16 x 8 cards; the trainer's sqrt_wrt_1024 rule)", lines
        assert exemplar_result.stdout.splitlines()[1] == "LR: base 0.001 -> effective 0.00141421 at global batch 128 (16 x 8 cards; the trainer's sqrt_wrt_1024 rule; constlr: 0.002 at 256)"
        parsed, exemplar_parsed = parse(argv), parse(exemplar)  # the trainer accepts the line: crops cut from the source (no squash), the decode side held at 1024 where the crops alone give 1344
        assert ssl_pretrain.check_args(parsed) == 5 and (parsed.pre_resize, parsed.decode_min_side, parsed.global_crops_size, parsed.local_crops_size, parsed.gram_teacher_size) == (0, 1024, 512, 224, 512)
        assert DataAugmentationCXR(global_crops_size=512, local_crops_size=224, gram_teacher_crops_size=512, emit_gram_teacher_crops=False).decode_min_side == 1344
        assert parsed.data_set == SEVEN and (parsed.init_from_checkpoint, parsed.init_new_heads_from, parsed.datasets_config) == (INIT, "MIMIC", NATIVE_YAML)
        assert {k for k in ssl_pretrain.RESUME_MATCH if k != "world_size" and getattr(parsed, k) != getattr(exemplar_parsed, k)} == {"lr", "min_lr", "pre_resize", "decode_min_side"}
        # the LR the trainer computes from the line (run()'s build_schedules call; epoch_length 10 stands in): effective peak = effective end = 1e-4 x 4 x sqrt(128 / 1024) = 1.4142135623730951e-4 exactly, 0 -> that over the 2 warmup cycles, flat after
        schedule = lambda p, g: build_schedules(lr=p.lr, min_lr=p.min_lr, epochs=p.epochs, warmup_epochs=p.warmup_epochs, epoch_length=10, global_batch_size=g, scale_lr=not p.no_lr_scaling,
                                                teacher_temp_warmup_epochs=p.teacher_temp_warmup_epochs)
        sch = schedule(parsed, 128)
        assert sch["factor"] == lr_scaling_factor(128) and sch["peak_lr"] == sch["end_lr"] == 1.4142135623730951e-4 and parsed.lr == parsed.min_lr == 1e-4 and not parsed.no_lr_scaling
        assert sch["lr"][0] == 0 and (sch["lr"][:20] < 1.4142135623730951e-4).all() and (sch["lr"][20:] == 1.4142135623730951e-4).all()
        assert math.isclose(schedule(exemplar_parsed, 128)["peak_lr"], 1.4142135623730951e-3, rel_tol=1e-15) and (parsed.warmup_epochs, parsed.teacher_temp_warmup_epochs) == (2, 2)
        # 4 cards: G = 64, where the trainer's factor 4 x sqrt(64 / 1024) is exactly 1, so the effective LR is the base itself
        result, four = launch(root, script="chain/newdata512_cropsrc.sh", env=dict(ENV, SLURM_JOB_GPUS="0,1,2,3"))
        assert result.returncode == 0 and result.stdout.splitlines()[1] == "LR: base 0.0001 -> effective 0.0001 at global batch 64 (16 x 4 cards; the trainer's sqrt_wrt_1024 rule)"
        assert math.isclose(schedule(parse(four), 64)["peak_lr"], 1e-4, rel_tol=1e-14)
        # LR_BASE is the one place to change it: another value is passed as --lr = --min_lr as it stands, and the trainer's factor makes the effective LR value x 1.4142135623730951 at batch 128; zero or text is refused before anything is launched
        for value, ok in (("1e-3", True), ("0", False), ("fast", False)):
            changed_root = fake_repo(tempfile.mkdtemp(dir=tmp), SETTINGS_NEW)
            path = os.path.join(changed_root, "chain", "newdata512_cropsrc.sh")
            text = open(path).read()
            assert text.count("\nLR_BASE=0.0001  #") == 1
            open(path, "w").write(text.replace("\nLR_BASE=0.0001  #", "\nLR_BASE=%s  #" % value))
            result, other_lr = launch(changed_root, script="chain/newdata512_cropsrc.sh", env=ENV8)
            if ok:
                other = parse(other_lr)
                assert result.returncode == 0 and other.lr == other.min_lr == 1e-3 and math.isclose(schedule(other, 128)["peak_lr"], 1.4142135623730951e-3, rel_tol=1e-14), result.stderr
            else:
                assert result.returncode == 1 and other_lr is None and "ERROR: LR_BASE holds '%s'" % value in result.stderr, result
        # a start from a checkpoint of the 224 px seven-dataset run: all seven heads are in it, so there is no --init_new_heads_from (the trainer refuses it when no dataset is new)
        later = "/workspace/%s/ckpt/epoch_0003.pth" % RUN224
        os.makedirs(os.path.join(root, RUN224, "ckpt"))
        open(os.path.join(root, RUN224, "ckpt", "epoch_0003.pth"), "w").write("x")
        result, from_later = launch(root, script="chain/newdata512_cropsrc.sh", env=dict(ENV8, CROPSRC_INIT=later, CROPSRC_NEW_HEADS_FROM="none"))
        edited = list(argv)
        edited[edited.index("--init_from_checkpoint") + 1] = later
        i = edited.index("--init_new_heads_from")
        del edited[i:i + 2]
        assert result.returncode == 0 and from_later == edited and parse(from_later).init_new_heads_from is None, result.stderr
        assert "from %s with all its datasets' heads already in it (no --init_new_heads_from)" % later in result.stdout.splitlines()[0]
        result, other = launch(root, script="chain/newdata512_cropsrc.sh", env=dict(ENV8, CROPSRC_NEW_HEADS_FROM="random"))  # any other value goes to the trainer, which checks it
        assert result.returncode == 0 and other[other.index("--init_new_heads_from") + 1] == "random"
        result, none = launch(root, script="chain/newdata512_cropsrc.sh", env=dict(ENV8, CROPSRC_INIT="/workspace/nowhere/ckpt/epoch_0001.pth"))
        assert result.returncode == 1 and none is None and "ERROR: ./nowhere/ckpt/epoch_0001.pth" in result.stderr, result
        # the default start is refused when absent, unless the run has an epoch checkpoint of its own
        bare = fake_repo(os.path.join(tmp, "no_init"), SETTINGS_NEW, init=False)
        result, none = launch(bare, script="chain/newdata512_cropsrc.sh", env=ENV8)
        assert result.returncode == 1 and none is None and "ERROR: ./ssl_dinov3_cxr_constlr_arkteacher_seed_100/ckpt/epoch_0200.pth" in result.stderr, result
        os.makedirs(os.path.join(bare, RUN_CROPSRC, "ckpt"))
        open(os.path.join(bare, RUN_CROPSRC, "ckpt", "epoch_0003.pth"), "w").write("x")
        result, resumed = launch(bare, script="chain/newdata512_cropsrc.sh", env=ENV8)
        assert result.returncode == 0 and resumed == argv, result.stderr
        # both settings files are needed and checked, naming them, before anything is launched; chain/main_lr.txt is not read (the LR is LR_BASE)
        names = ("newdata512_cropsrc_stop_after.txt", "newdata512_cropsrc_batch_per_gpu.txt")
        for name in names:
            result, none = launch(fake_repo(tempfile.mkdtemp(dir=tmp), dict(SETTINGS_NEW, **{name: None})), script="chain/newdata512_cropsrc.sh", env=ENV8)
            assert result.returncode == 1 and none is None and "ERROR" in result.stderr and name in result.stderr, (name, result.stderr)
        for name, value in ((names[0], "0"), (names[0], "1001"), (names[1], "0"), (names[1], "08"), (names[1], "12345")):
            result, none = launch(fake_repo(tempfile.mkdtemp(dir=tmp), dict(SETTINGS_NEW, **{name: value + "\n"})), script="chain/newdata512_cropsrc.sh", env=ENV8)
            assert result.returncode == 1 and none is None and "ERROR" in result.stderr and name in result.stderr, (name, value, result.stderr)
        result, no_lr_file = launch(fake_repo(tempfile.mkdtemp(dir=tmp), dict(SETTINGS_NEW, **{"main_lr.txt": None})), script="chain/newdata512_cropsrc.sh", env=ENV8)
        assert result.returncode == 0 and no_lr_file == argv, result.stderr


def test_newdata512_cropsrc_wrapper():
    check_wrapper("run_d1_8_newdata512_cropsrc_arkteacher_gaudi.sh", ["--job-name=d1-8-newdata512-cropsrc", "-A grp_jliang12", "-p gaudi", "-q public", "-t 7-00:00:00", "-N 1", "--gres=gpu:hl225:8", "-c 144", "--mem=480G"],
                  "newdata512_cropsrc.sh", ("A=$(sbatch --parsable run_d1_8_newdata512_cropsrc_arkteacher_gaudi.sh)", "sbatch -d afterany:$A run_d1_8_newdata512_cropsrc_arkteacher_gaudi.sh",
                                            "sbatch --gres=gpu:hl225:4 -c 72 --mem=256G run_d1_8_newdata512_cropsrc_arkteacher_gaudi.sh", "chain/newdata512_cropsrc_stop_after.txt",
                                            "chain/newdata512_cropsrc_batch_per_gpu.txt", "CROPSRC_INIT", "CROPSRC_NEW_HEADS_FROM=none", "taix_unpack_original.sh train", "6.5 GB", "690 GB per 100 cycles"), ENV8)


def test_ssl_datasets_default_and_gaudi_extra_binds():
    with tempfile.TemporaryDirectory() as tmp:
        root = fake_repo(tmp)
        open(os.path.join(root, "chain", "probe.sh"), "w").write('#!/usr/bin/env bash\nsource "$(dirname "${BASH_SOURCE[0]}")/ssl_common.sh"\nssl_launch run_x --lr 1\n')
        for sets, flags in (({}, ["VinDrCXR", "ChestXray14", "MIMIC", "CheXpert"]), (dict(SSL_DATASETS=""), ["VinDrCXR", "ChestXray14", "MIMIC", "CheXpert"]),
                            (dict(SSL_DATASETS="MIMIC  PadChest *"), ["MIMIC", "PadChest", "*"])):  # unset or empty: the four, as ever; a list: in its order, no glob expansion
            result, argv = launch(root, script="chain/probe.sh", env=dict(ENV4, **sets))
            assert result.returncode == 0 and argv[argv.index("--device") + 2:argv.index("--pretrained_weights")] == [x for name in flags for x in ("--data_set", name)], (sets, argv)
        # gaudi.sh: EXTRA_BINDS is one more --bind after the data bind, and nothing changes without it
        fake_bin = os.path.join(tmp, "bin")
        os.makedirs(fake_bin)
        with open(os.path.join(fake_bin, "apptainer"), "w") as f:
            f.write('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
        os.chmod(os.path.join(fake_bin, "apptainer"), 0o755)
        env = dict(ENV, PATH=fake_bin + os.pathsep + os.environ["PATH"], HABANA_VISIBLE_MODULES="0", DATA_BIND="/d:/d:ro", RUNTIME_ROOT=os.path.join(tmp, "runtime"))

        def binds(**extra):
            out = subprocess.run([os.path.join(REPO, "gaudi.sh"), "true"], env=dict(env, **extra), capture_output=True, text=True, check=True).stdout.splitlines()
            return [out[i + 1] for i, word in enumerate(out) if word == "--bind"]

        plain, extra = binds(), binds(EXTRA_BINDS="/s/new:/s/new:ro,/s/other:/s/other")
        assert plain[-1] == "/d:/d:ro" and extra == plain + ["/s/new:/s/new:ro,/s/other:/s/other"], (plain, extra)
        assert binds(EXTRA_BINDS="") == plain


RUN_COS = "ssl_dinov3_cxr_newdata224_cos50_tiedhead_arkteacher_seed_100"
REGISTRY_COS = "ssl-newdata224_cos50_tiedhead_arkteacher"
COS_LINE = ("--no_gram --cyclic_ark_teacher --no_lock_half_pass --keep_all_checkpoints --no_ibot_separate_head --drop_path_rate 0.1 --rope_rescale_coords 2 --untie_local_cls_norm "
            "--lr 2e-3 --min_lr 1e-6 --no_lr_scaling --epochs 50 --warmup_epochs 5 --teacher_temp_warmup_epochs 15 --weight_decay 0.04 --weight_decay_end 0.4 --pre_resize 512 "
            "--global_crops_scale 0.15 1.0 --local_crops_scale 0.05 0.15 --intensity_aug ark --no_blur --no_horizontal_flips --rotate_limit 10").split()  # the confirmed flag line (handoff 2026-11-16 section 4), its schedule halved to 50 cycles by Jarod the same day (warm-ups 5 / 15: 10% / 30%)
COS_TAIL = ["--freeze_backbone_epochs", "0", "--teacher_whole_image_size", "448", "--whole_image_dino_weight", "1.0", "--export_pass_end_every", "1", "--gram_teacher_size", "224", "--datasets_config", NATIVE_YAML]  # after it: Jarod's no frozen first cycle, then the script's settings


def cos_expected(cards=8):
    """The cos50 run's launch line as chain/newdata224_cos50.sh and ssl_launch give it: the cards, the seven --data_set, the stock weights, ssl_launch's defaults, the confirmed flag line, then the script's own settings."""
    return (["python", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", str(cards), "ssl_pretrain.py", "--device", "hpu"] + [x for name in SEVEN for x in ("--data_set", name)]
            + ["--pretrained_weights", "/workspace/weights/" + WEIGHT_NAME, "--batch_size_per_gpu", "64", "--workers", "16", "--seed", "100", "--output_dir", "/workspace/" + RUN_COS, "--resume"]
            + COS_LINE + COS_TAIL)


def test_newdata224_cos50_script():
    script = os.path.join(REPO, "chain", "newdata224_cos50.sh")
    subprocess.run(["bash", "-n", script], check=True)
    assert os.access(script, os.X_OK) and open(script).read().startswith("#!/usr/bin/env bash\n")
    assert "newdata224_cos50" not in [line.split("#")[0].strip() for line in open(os.path.join(REPO, "chain", "phases.txt"))]  # not a chain phase
    expected = cos_expected()
    with tempfile.TemporaryDirectory() as tmp:
        root = fake_repo(tmp, SETTINGS_NEW)
        result, argv = launch(root, script="chain/newdata224_cos50.sh", env=dict(ENV8, SSL_DATASETS="VinDrCXR"))  # an inherited SSL_DATASETS does not change the seven
        assert result.returncode == 0 and result.stderr == "" and argv == expected, (result.stderr, argv)
        i = argv.index(COS_LINE[0])
        assert argv[i:i + len(COS_LINE)] == COS_LINE  # the confirmed line's words, as one contiguous run
        lines = result.stdout.splitlines()
        assert len(lines) == 1 and lines[0].startswith("run: %s (registry id %s; " % (RUN_COS, REGISTRY_COS)), lines
        for needle in ("data sets " + " ".join(SEVEN), "8 cards x 64 = global batch 512", "cut at 224", " ".join(COS_LINE + COS_TAIL[:6])):  # the datasets, the batch, the cut size, the flags as launched
            assert needle in lines[0], (needle, lines[0])
        parsed = parse(argv)  # the trainer accepts the line, and reads it as meant
        assert ssl_pretrain.check_args(parsed) == 50 and parsed.data_set == SEVEN and parsed.output_dir == "/workspace/" + RUN_COS  # no stop epoch: the run trains all 50 cycles
        i = argv.index("--freeze_backbone_epochs")
        assert parsed.freeze_backbone_epochs == 0 and parse(argv[:i] + argv[i + 2:]).freeze_backbone_epochs == 1  # the trainer's default is the frozen first cycle: this line turns it off
        assert (parsed.pre_resize, parsed.global_crops_size, parsed.local_crops_size, parsed.gram_teacher_size) == (512, 224, 96, 224)  # cut at 224 straight from the 512 squash: no resize
        assert (parsed.teacher_whole_image_size, parsed.whole_image_dino_weight, parsed.no_gram, parsed.cyclic_ark_teacher) == (448, 1.0, True, True)
        assert parsed.export_pass_end_every == 1  # pass-end exports in every cycle
        assert (parsed.heads, parsed.no_ibot_separate_head, parsed.lock_half_pass, parsed.keep_all_checkpoints) == ("shared", True, False, True)  # one tied head for all seven datasets
        assert (parsed.lr, parsed.min_lr, parsed.no_lr_scaling, parsed.epochs, parsed.warmup_epochs, parsed.teacher_temp_warmup_epochs) == (2e-3, 1e-6, True, 50, 5, 15)  # the 50-cycle schedule, both warm-ups 10% / 30% of it
        assert (parsed.weight_decay, parsed.weight_decay_end, parsed.drop_path_rate, parsed.rope_rescale_coords, parsed.untie_local_cls_norm) == (0.04, 0.4, 0.1, 2.0, True)
        assert (parsed.global_crops_scale, parsed.local_crops_scale, parsed.intensity_aug, parsed.no_blur, parsed.no_horizontal_flips, parsed.rotate_limit) == ((0.15, 1.0), (0.05, 0.15), "ark", True, True, 10.0)
        assert (parsed.init_from_checkpoint, parsed.datasets_config, parsed.bf16, parsed.stop_after_epochs) == (None, NATIVE_YAML, False, None)  # the stock weights, the native TAIXRay files, fp32
        # no settings file and no start checkpoint are read: with neither in the repo the line is the same
        result, bare = launch(fake_repo(tempfile.mkdtemp(dir=tmp), {}, init=False), script="chain/newdata224_cos50.sh", env=ENV8)
        assert result.returncode == 0 and bare == expected, result.stderr
        # 8 cards or nothing: another count would change the global batch, the absolute LR's recipe and the length of the cycle; refused before anything is launched
        for cards in (2, 4, 16):
            result, none = launch(root, script="chain/newdata224_cos50.sh", env=dict(ENV, SLURM_JOB_GPUS=",".join(str(card) for card in range(cards))))
            assert result.returncode == 1 and none is None and result.stdout == "" and "ERROR: %d cards in the allocation" % cards in result.stderr, (cards, result)
        # the three settings are one line each, once, ahead of the flag line, each a one-line change
        text = open(script).read()
        head = text[:text.index("\nFLAGS=(")]
        setting = {"RUN": "RUN=%s  #" % RUN_COS, "REGISTRY_ID": "REGISTRY_ID=%s  #" % REGISTRY_COS, "GRAM_TEACHER_SIZE": "GRAM_TEACHER_SIZE=224  #", "EXPORT_PASS_END_EVERY": "EXPORT_PASS_END_EVERY=1  #"}
        for name, line in setting.items():
            assert text.count("\n" + line) == 1 and head.count("\n" + line) == 1, name

        def edited(name, value):  # the script with one setting's value replaced, launched in a fresh fake repo: (result, argv)
            other_root = fake_repo(tempfile.mkdtemp(dir=tmp), SETTINGS_NEW)
            path = os.path.join(other_root, "chain", "newdata224_cos50.sh")
            copy = open(path).read()
            open(path, "w").write(copy.replace("\n" + setting[name], "\n%s=%s  #" % (name, value)))
            return launch(other_root, script="chain/newdata224_cos50.sh", env=ENV8)

        result, other = edited("RUN", "ssl_other_run_seed_7")  # the run dir, nothing else
        assert result.returncode == 0 and changed(other, argv) == {"--output_dir"} and flags(other)["--output_dir"] == [["/workspace/ssl_other_run_seed_7"]], result.stderr
        assert ssl_pretrain.check_args(parse(other)) == 50 and result.stdout.startswith("run: ssl_other_run_seed_7 (registry id " + REGISTRY_COS + "; "), result.stdout
        result, other = edited("REGISTRY_ID", "ssl-other")  # only printed: the trainer does not read it
        assert result.returncode == 0 and other == argv and result.stdout.startswith("run: %s (registry id ssl-other; " % RUN_COS), result.stdout
        result, other = edited("EXPORT_PASS_END_EVERY", "2")  # every second cycle (Jarod may choose it): --export_pass_end_every and nothing else
        assert result.returncode == 0 and changed(other, argv) == {"--export_pass_end_every"} and parse(other).export_pass_end_every == 2, result.stderr
        result, other = edited("GRAM_TEACHER_SIZE", "448")  # --gram_teacher_size and nothing else changes ...
        assert result.returncode == 0 and changed(other, argv) == {"--gram_teacher_size"} and flags(other)["--gram_teacher_size"] == [["448"]], result.stderr
        with contextlib.redirect_stderr(io.StringIO()) as err:  # ... and the trainer refuses it: with --no_gram its one accepted --gram_teacher_size is the global size (the 448 cut is the flag left out, not given)
            try:
                parse(other)
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError("expected SystemExit: --no_gram takes no --gram_teacher_size but the global crops' size")
        assert "--gram_teacher_size cannot be combined with --no_gram" in err.getvalue(), err.getvalue()


def test_newdata224_cos50_wrapper():
    check_wrapper("run_d1_8_newdata224_cos50_gaudi.sh", ["--job-name=d1-8-newdata224-cos50", "-A grp_jliang12", "-p gaudi", "-q public", "-t 7-00:00:00", "-N 1", "--gres=gpu:hl225:8", "-c 144", "--mem=480G"],
                  "newdata224_cos50.sh", ("A=$(sbatch --parsable run_d1_8_newdata224_cos50_gaudi.sh)", "sbatch -d afterany:$A run_d1_8_newdata224_cos50_gaudi.sh", "taix_unpack_original.sh train",
                                          "run_d1_8_newdata224_cos50_smoke_gaudi.sh", "re-submit"), ENV8)


SMOKE_COS = os.path.join(REPO, "run_d1_8_newdata224_cos50_smoke_gaudi.sh")
SMOKE_ROWS = 60 * 8 * 64  # the smoke's cut: STEPS_PER_DATASET x cards x BATCH
SMOKE_DIR = "smoke_newdata224_cos50_x8_4242"  # the smoke dir of job 4242 on 8 cards
COS_HEADERS = {"MIMIC": "image_path,dicom_id,study_id", "CheXpert": "Path,Sex,Age", "PadChest": "Path", "TAIXRay": "Path", "ReXGradient": "Path"}  # the lists with a header row (ChestXray14's and VinDrCXR's have none)
COS_PASS_END = ['MIMIC', 'ReXGradient', 'CheXpert', 'TAIXRay', 'PadChest', 'ChestXray14']  # the smoke cycle's passes but the last (the cut lists tie, so the --data_set order: VinDrCXR, the short one, last)
COS_SHORT = {"VinDrCXR": 15000}  # the real list's size, below SMOKE_ROWS: the smoke keeps it whole (29 steps on 8 cards)
COS_COLUMNS = "step,epoch,dino_global,dino_local,koleo,ibot,gram,total,lr,wd,momentum,teacher_temp,gram_weight,step_time,data_time,dataset,dino_whole"
# the rows the stub trainer writes: (step, dino_global, dino_local, total, step_time, data_time, dataset, dino_whole); the first, step 0, holds the compile. Values exact in binary, so Python's sums are awk's
COS_ROWS = [(0, 9.0, 9.5, 30.0, 41.0, 6.0, "MIMIC:release", 9.75), (4, 8.0, 8.5, 20.0, 2.0, 0.25, "MIMIC:release", 8.75), (9, 7.0, 7.5, 18.0, 2.5, 0.75, "MIMIC:release", 7.75),
            (14, 6.0, 6.5, 14.0, 3.0, 1.25, "ReXGradient:release", 6.75), (19, 5.0, 5.5, 12.0, 2.0, 0.5, "ReXGradient:release", 5.75), (24, 4.0, 4.5, 10.0, 1.5, 0.25, "CheXpert:release", 4.75)]
COS_GAUDI_STUB = r'''#!/usr/bin/env bash
out=; prev=; for a in "$@"; do [[ $prev == --output_dir ]] && out=${a#/workspace/}; prev=$a; done
mkdir -p "$PWD/$out"
printf "%s\n" "$@" > "$PWD/$out/argv.txt"; printf "%s" "${EXTRA_BINDS:-}" > "$PWD/$out/extra_binds.txt"
if [[ -n ${FAKE_TRAINER:-} ]]; then  # what the trainer leaves behind in its run dir and prints, then its exit
  base=80
  mkdir -p "$PWD/$out/ckpt" "$PWD/$out/export"; touch "$PWD/$out/ckpt/epoch_0001.pth" "$PWD/$out/export/teacher_backbone_epoch_0001.pth"; echo '{}' > "$PWD/$out/args.json"
  for a in "$@"; do  # --export_pass_end_every: the trainer's pass-end exports of the cycle, after each of its passes but the last (4 bytes each here)
    if [[ $a == --export_pass_end_every ]]; then mkdir -p "$PWD/$out/export/pass_end"; for d in MIMIC ReXGradient CheXpert TAIXRay PadChest ChestXray14; do printf abcd > "$PWD/$out/export/pass_end/teacher_backbone_epoch_0001_after_$d.pth"; echo "pass-end export after step 1: /workspace/$out/export/pass_end/teacher_backbone_epoch_0001_after_$d.pth (2.5 s to write)"; done; fi
  done
  cp "$FAKE_CSV" "$PWD/$out/ssl_losses.csv"
  for step in 1 5 389; do for r in 0 1 2 3 4 5 6 7; do echo "peak_memory rank=$r step=$step GiB=${FAKE_PEAK:-$((base + step / 100 + r)).50}"; done; done  # FAKE_PEAK=unavailable: the trainer found no counter
  sleep 1
  [[ -z ${FAKE_ERR:-} ]] || echo "$FAKE_ERR" >&2
  exit "${FAKE_RC:-0}"
fi
'''
COS_HLSMI_STUB = r'''#!/usr/bin/env bash
n=$(cat "$HLSMI_COUNT" 2>/dev/null || echo 0); echo $((n + 1)) > "$HLSMI_COUNT"
echo "index, module_id, memory.used [MiB]"
for i in 0 1 2 3 4 5 6 7; do echo "$i, $((i + 2)), $((768 + n * 1000 + i * 10)) MiB"; done
'''


def put(path, text, mode=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    if mode:
        os.chmod(path, mode)


def cos_list(path, name, rows):
    """A train list in the format image_paths reads for `name`: a header row (but for ChestXray14 and VinDrCXR), then `rows` images."""
    row = {"ChestXray14": "%s_%06d.png 0 1", "VinDrCXR": "%s_%06d 0 1"}.get(name, "%s_%06d.png,F")
    put(path, "".join(([COS_HEADERS[name] + "\n"] if name in COS_HEADERS else []) + [row % (name.lower(), i) + "\n" for i in range(rows)]))


def smoke_repo(tmp, trainer=True, rows=SMOKE_ROWS + 50):
    """A fake repo for the cos50 smoke: the real launcher, ssl_common.sh, Ark's common.sh and gaudi_hlsmi.sh, a recording gaudi.sh (FAKE_TRAINER: it also plays the trainer), a card check that records it ran, the stock weights, the
    trainer's text (trainer=False: without --teacher_whole_image_size), base/datasets_config_taix_original.yaml as the real file with its two absolute lists moved under tmp, and the seven lists it names: `rows` images each,
    VinDrCXR's 15,000. The CSV the stub trainer writes, an hl-smi stub on PATH and the datasets-new directory."""
    root = os.path.join(tmp, "repo")
    for name in ("chain/newdata224_cos50.sh", "chain/ssl_common.sh", "experiment_scripts/common.sh", "gaudi_hlsmi.sh"):
        put(os.path.join(root, name), open(os.path.join(REPO, name)).read(), stat.S_IMODE(os.stat(os.path.join(REPO, name)).st_mode))
    put(os.path.join(root, "gaudi.sh"), COS_GAUDI_STUB, 0o755)
    put(os.path.join(root, "gaudi_check_cards.sh"), '#!/usr/bin/env bash\ntouch "$PWD/cards_checked"\n', 0o755)
    put(os.path.join(root, "weights", WEIGHT_NAME), "x")
    text = open(os.path.join(BASE, "ssl_pretrain.py")).read()
    assert '"--teacher_whole_image_size"' in text  # the quoted flag name, which the smoke greps for
    put(os.path.join(root, "base", "ssl_pretrain.py"), text if trainer else text.replace('"--teacher_whole_image_size"', '"--gone"'))
    config, native = open(os.path.join(BASE, NATIVE_YAML)).read(), get_config(os.path.join(BASE, NATIVE_YAML))
    for name in SEVEN:
        listing, count = native[name]["train_list"], COS_SHORT.get(name, rows)
        if listing.startswith("/"):  # TAIXRay and ReXGradient: absolute paths on the cluster; here under tmp
            fake = os.path.join(tmp, "datasets-new", name + ".csv")
            assert config.count("train_list: " + listing + "\n") == 1
            config = config.replace("train_list: " + listing + "\n", "train_list: " + fake + "\n")
            cos_list(fake, name, count)
        else:  # the other five: relative to base/
            cos_list(os.path.join(root, "base", listing), name, count)
    put(os.path.join(root, "base", NATIVE_YAML), config)
    put(os.path.join(tmp, "run.csv"), COS_COLUMNS + "\n" + "".join("%d,0,%s,%s,0.5,3.0,0.0,%s,0.0,0.04,1.0,0.04,0.0,%s,%s,%s,%s\n" % row for row in COS_ROWS))
    put(os.path.join(tmp, "bin", "hl-smi"), COS_HLSMI_STUB, 0o755)
    os.makedirs(os.path.join(tmp, "datasets-new"), exist_ok=True)
    return root


def smoke_script(tmp, root, datasets="datasets-new"):
    """The smoke with its cd pointed at the fake repo, its datasets check at the fake directory and its 5 s sample interval at 0.2 s (the stub trainer runs 1 s a line)."""
    text = open(SMOKE_COS).read()
    cd, check, sleep = "cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1", "[[ -d /scratch/$USER/datasets-new ]]", "sleep 5; done"
    assert text.count(cd) == 1 and text.count(check) == 1 and text.count(sleep) == 1
    path = os.path.join(root, "smoke.sh")
    put(path, text.replace(cd, "cd '%s' || exit 1" % root).replace(check, "[[ -d '%s' ]]" % os.path.join(tmp, datasets)).replace(sleep, "sleep 0.2; done"))
    return path


def smoke_env(tmp, cards=8, **extra):
    env = {k: v for k, v in ENV.items() if not k.startswith(("FAKE_", "HLSMI_"))}
    return dict(env, SLURM_JOB_GPUS=",".join(str(card) for card in range(cards)), SLURM_JOB_ID="4242", USER="tester", PATH=os.path.join(tmp, "bin") + os.pathsep + env["PATH"],
                HLSMI_COUNT=os.path.join(tmp, "hlsmi_count"), FAKE_CSV=os.path.join(tmp, "run.csv"), **extra)


def smoke_run(script, root, env, cwd=None):
    return subprocess.run(["bash", script], cwd=cwd or root, env=env, capture_output=True, text=True)


def recorded(root, out):
    """The argv the stub gaudi.sh recorded for --output_dir /workspace/<out> (None: it was never launched)."""
    path = os.path.join(root, out, "argv.txt")
    return open(path).read().splitlines() if os.path.exists(path) else None


def no_launch(root, smoke_dir_exists=False):
    """Nothing was launched or made: no argv recorded anywhere, and no smoke dir (the one a test made itself stays empty)."""
    assert not glob.glob(os.path.join(root, "**", "argv.txt"), recursive=True)
    if smoke_dir_exists:
        assert os.listdir(os.path.join(root, SMOKE_DIR)) == []
    else:
        assert not os.path.exists(os.path.join(root, SMOKE_DIR))


def test_newdata224_cos50_smoke_header():
    subprocess.run(["bash", "-n", SMOKE_COS], check=True)
    assert os.access(SMOKE_COS, os.X_OK) and stat.S_IMODE(os.stat(SMOKE_COS).st_mode) == 0o755
    text = open(SMOKE_COS).read()
    lines = text.splitlines()
    sbatch = [line[len("#SBATCH "):] for line in lines if line.startswith("#SBATCH ")]
    assert sbatch == ["--job-name=d1-8-newdata224-cos50-smoke", "-A grp_jliang12", "-p gaudi", "-q public", "-t 00:30:00", "-N 1", "--gres=gpu:hl225:8", "-c 144", "--mem=480G"], lines
    run = [line[len("#SBATCH "):] for line in open(os.path.join(REPO, "run_d1_8_newdata224_cos50_gaudi.sh")).read().splitlines() if line.startswith("#SBATCH ")]
    assert [s for s in sbatch if not s.startswith(("--job-name", "-t "))] == [s for s in run if not s.startswith(("--job-name", "-t "))]  # the run's account, partition, QOS, node, cards, CPUs and memory
    code = [line for line in lines if line and not line.startswith("#")]
    for needle in ("export PT_HPU_LAZY_MODE=1", "cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1", "source ./chain/ssl_common.sh", "source ./gaudi_hlsmi.sh", "./gaudi_check_cards.sh || exit 1",
                   "export EXTRA_BINDS=/scratch/$USER/datasets-new:/scratch/$USER/datasets-new:ro", "--stop_after_epochs 1 --print_freq 5 --log_peak_memory", '--datasets_config "/workspace/$SMOKE/datasets_config_smoke.yaml"'):
        assert any(needle in line for line in code), needle
    assert not any("main_lr" in line or "stop_after.txt" in line for line in code)  # no settings file
    for line in ("STEPS_PER_DATASET=60  #", "BATCH=64  #"):  # the two settings, one line each, once, near the top
        assert text.count("\n" + line) == 1 and text.index("\n" + line) < text.index("\nFLAGS=("), line
    for needle in ("sbatch run_d1_8_newdata224_cos50_smoke_gaudi.sh", "grep -E '^(SMOKE|peak_memory)'", "29 steps", "30,720", "not measured", "dino_whole", "The 30 min limit", "--export_pass_end_every 1 --gram_teacher_size"):
        assert needle in text, needle
    assert "bf16" not in text.lower() and "smoke_line" not in text  # one line, fp32 as the run: no report-only bf16 line any more (Jarod, 2026-11-16)
    assert "newdata224_cos50_smoke" not in [line.split("#")[0].strip() for line in open(os.path.join(REPO, "chain", "phases.txt"))]  # not a chain phase


def test_newdata224_cos50_smoke_lines():
    with tempfile.TemporaryDirectory() as tmp:
        root = smoke_repo(tmp)
        env = smoke_env(tmp, SSL_DATASETS="VinDrCXR")  # an inherited SSL_DATASETS changes neither the run's line nor the smoke's
        real = smoke_run("chain/newdata224_cos50.sh", root, env)  # the run's own launch line, in the same fake repo
        assert real.returncode == 0 and real.stderr == "", real.stderr
        run_argv = recorded(root, RUN_COS)
        assert run_argv == cos_expected(), run_argv
        result = smoke_run(smoke_script(tmp, root), root, dict(env, FAKE_TRAINER="1"), cwd=tempfile.mkdtemp(dir=tmp))  # from elsewhere, as sbatch does
        assert result.returncode == 0 and result.stderr == "", (result.stdout, result.stderr)
        smoke = recorded(root, SMOKE_DIR)  # the one line, the smoke dir its run dir
        assert smoke is not None and len(glob.glob(os.path.join(root, "**", "argv.txt"), recursive=True)) == 2  # the run's and the smoke's: one launch each
        assert changed(smoke, run_argv) == {"--output_dir", "--datasets_config", "--stop_after_epochs", "--print_freq", "--log_peak_memory"}, changed(smoke, run_argv)  # the run's line, with exactly these
        assert flags(smoke)["--export_pass_end_every"] == flags(run_argv)["--export_pass_end_every"] == [["1"]] and "--bf16" not in flags(smoke)  # the pass-end exports as the run's; fp32 as the run
        config = "/workspace/%s/datasets_config_smoke.yaml" % SMOKE_DIR
        assert flags(smoke)["--output_dir"] == [["/workspace/" + SMOKE_DIR]] and flags(smoke)["--datasets_config"] == [[config]] and flags(run_argv)["--datasets_config"] == [[NATIVE_YAML]]
        assert flags(smoke)["--stop_after_epochs"] == [["1"]] and flags(smoke)["--print_freq"] == [["5"]] and flags(smoke)["--log_peak_memory"] == [[]]
        assert smoke[:smoke.index("--output_dir")] == run_argv[:run_argv.index("--output_dir")]  # the cards, the trainer, the seven datasets, the weights, the batch of 64, the workers and the seed: the run's
        assert flags(smoke)["--batch_size_per_gpu"] == [["64"]] and flags(smoke)["--data_set"] == [[name] for name in SEVEN] and flags(smoke)["--gram_teacher_size"] == [["224"]]
        run_parsed, parsed = parse(run_argv), parse(smoke)
        assert ssl_pretrain.check_args(parsed) == 1 and ssl_pretrain.check_args(run_parsed) == 50  # one cycle of the run's 50-cycle schedule
        assert parsed.freeze_backbone_epochs == 0 and parsed.export_pass_end_every == 1 and parsed.print_freq == 5 and parsed.log_peak_memory and not parsed.bf16
        assert {k for k in vars(run_parsed) if vars(run_parsed)[k] != vars(parsed)[k]} == {"output_dir", "datasets_config", "stop_after_epochs", "print_freq", "log_peak_memory"}  # every other trainer setting, defaults included
        # a throwaway dir that cannot be the run's, nor resume into it
        assert SMOKE_DIR != RUN_COS and not SMOKE_DIR.startswith(RUN_COS) and not RUN_COS.startswith(SMOKE_DIR)
        assert open(os.path.join(root, SMOKE_DIR, "extra_binds.txt")).read() == "/scratch/tester/datasets-new:/scratch/tester/datasets-new:ro"
        assert os.path.exists(os.path.join(root, "cards_checked"))
        assert "SMOKE plan: 8 cards x per-card batch 64 (global batch 512), one cycle over lists cut to 30720 rows (60 steps a dataset; a shorter list is kept whole), smoke dir " + SMOKE_DIR in result.stdout


def test_newdata224_cos50_smoke_lists():
    with tempfile.TemporaryDirectory() as tmp:
        root = smoke_repo(tmp)
        result = smoke_run(smoke_script(tmp, root), root, smoke_env(tmp, FAKE_TRAINER="1"))
        assert result.returncode == 0, result.stderr
        native = get_config(os.path.join(root, "base", NATIVE_YAML))
        smoke = get_config(os.path.join(root, SMOKE_DIR, "datasets_config_smoke.yaml"))  # the host path of the file the trainer reads as /workspace/<smoke>/datasets_config_smoke.yaml
        assert sorted(smoke) == sorted(native) == sorted(SEVEN) and all(set(smoke[name]) == {"data_dir", "train_list"} for name in SEVEN)
        for name in SEVEN:
            assert smoke[name]["data_dir"] == native[name]["data_dir"] and smoke[name]["train_list"] == "/workspace/%s/lists/%s.list" % (SMOKE_DIR, name), name
            original = native[name]["train_list"] if native[name]["train_list"].startswith("/") else os.path.join(root, "base", native[name]["train_list"])
            cut = os.path.join(root, SMOKE_DIR, "lists", name + ".list")  # the host path of /workspace/<smoke>/lists/<name>.list
            whole, kept = image_paths(name, native[name]["data_dir"], original), image_paths(name, native[name]["data_dir"], cut)
            short = name in COS_SHORT
            assert len(whole) == (COS_SHORT[name] if short else SMOKE_ROWS + 50) and kept == whole[:SMOKE_ROWS], name  # the first rows of the list, in its order
            assert len(kept) == (COS_SHORT[name] if short else SMOKE_ROWS) and short == (len(kept) < SMOKE_ROWS), name  # VinDrCXR, below the cut, keeps all its rows
            assert open(cut).readline() == open(original).readline() and sum(1 for _ in open(cut)) == len(kept) + (name in COS_HEADERS), name  # the header row kept where there is one, none added where not
            steps = len(build_data_loader(range(len(kept)), 64, 0, 0, 8, 100, False, None)[0])  # the pass the trainer cuts from the list: 8 ranks, batch 64, both drop_last
            assert steps == (29 if short else 60), (name, steps)
        assert sorted(os.listdir(os.path.join(root, SMOKE_DIR))) == ["args.json", "argv.txt", "datasets_config_smoke.yaml", "extra_binds.txt", "hlsmi.txt", "lists", "ssl_losses.csv", "train.log"]  # the cut lists beside the run's files
        assert len(os.listdir(os.path.join(root, SMOKE_DIR, "lists"))) == 7


def cos_summary():
    """The summary lines the smoke prints from the stub's rows: the speed and projection over the rows after step 0, then each dataset in the CSV's order."""
    after = COS_ROWS[1:]
    lines = ["SMOKE speed: %.3f s/step, data wait %.3f s/step (the mean of the %d rows after step 0, which holds the compile; each row is the mean over the steps since the row before)"
             % (sum(r[4] for r in after) / len(after), sum(r[5] for r in after) / len(after), len(after)),
             "SMOKE projection: %.1f h = %.3f s/step x 119000 steps / 3600: arithmetic on the mean of this smoke, not a measurement of the run"
             % (sum(r[4] for r in after) / len(after) * 119000 / 3600, sum(r[4] for r in after) / len(after))]
    for dataset in dict.fromkeys(r[6] for r in COS_ROWS):
        mine, later = [r for r in COS_ROWS if r[6] == dataset], [r for r in after if r[6] == dataset]
        last = mine[-1]
        lines.append("SMOKE dataset %s: %d rows, mean %.3f s/step, data wait %.3f s/step; last row: dino_global %.4f, dino_local %.4f, dino_whole %.4f, total %.4f"
                     % (dataset, len(later), sum(r[4] for r in later) / len(later), sum(r[5] for r in later) / len(later), last[1], last[2], last[7], last[3]))
    return lines


def test_newdata224_cos50_smoke_summary():
    with tempfile.TemporaryDirectory() as tmp:
        root = smoke_repo(tmp)
        script = smoke_script(tmp, root)
        result = smoke_run(script, root, smoke_env(tmp, FAKE_TRAINER="1"))
        lines = result.stdout.splitlines()
        assert result.returncode == 0 and result.stderr == "", (result.stdout, result.stderr)
        run_dir = os.path.join(root, SMOKE_DIR)
        for rank in range(8):  # the allocator peak per rank: the largest of its printed values (80 + step / 100 + rank, at steps 1 / 5 / 389)
            assert "SMOKE allocator peak, rank %d: %.2f GiB" % (rank, 83.5 + rank) in lines, (rank, lines)
        peaks = {int(l.split("card index ")[1].split(":")[0]): int(l.split("peak ")[1].split(" MiB")[0]) for l in lines if l.startswith("SMOKE hl-smi card index")}
        assert sorted(peaks) == list(range(8)), peaks  # the allocated cards
        assert all(peaks[i] >= 768 + i * 10 and (peaks[i] - 768 - i * 10) % 1000 == 0 for i in peaks), peaks  # the largest of several samples (the stub's reading grows by 1000 MiB a call)
        for expected in cos_summary():
            assert expected in lines, (expected, lines)
        assert [l for l in lines if "pass-end export time" in l] == ["SMOKE pass-end export time: 6 files, mean 2.5 s to write each, 0.2 min per exporting cycle (6 files): 0.21 h over 50 cycles at "
                                                                   "--export_pass_end_every 1, half that at 2 (arithmetic on this smoke's mean)"], lines  # the trainer's per-file seconds (2.5 each here)
        assert sorted(l for l in lines if l.startswith("SMOKE") and "pass-end export " in l and "time" not in l) == sorted("SMOKE pass-end export teacher_backbone_epoch_0001_after_%s.pth: 4 bytes" % d for d in COS_PASS_END), lines  # reported before the cleanup (glob order is the locale's)
        assert "SMOKE verdict: FITS: 8 cards x per-card batch 64 ran one cycle over the cut lists (exit 0)" in lines and lines[-1].startswith("SMOKE verdict: ")
        assert not any("bf16" in l or "fp32" in l for l in lines if l.startswith("SMOKE")), lines
        assert sorted(os.listdir(run_dir)) == ["args.json", "argv.txt", "datasets_config_smoke.yaml", "extra_binds.txt", "hlsmi.txt", "lists", "ssl_losses.csv", "train.log"], os.listdir(run_dir)  # ckpt/ and export/ are gone, the evidence stays
        assert "peak_memory rank=7 step=389 GiB=90.50" in open(os.path.join(run_dir, "train.log")).read()
        # the trainer found no allocator counter (its lines say GiB=unavailable): no peak is made up from them, the summary says so per rank, and the hl-smi peaks, the speed and the verdict are all still there
        shutil.rmtree(run_dir)
        result = smoke_run(script, root, smoke_env(tmp, FAKE_TRAINER="1", FAKE_PEAK="unavailable"))
        lines = result.stdout.splitlines()
        assert result.returncode == 0, result.stderr
        assert [l for l in lines if l.startswith("SMOKE allocator peak")] == [
            "SMOKE allocator peak, rank %d: unavailable (no allocator counter on this stack: the hl-smi peaks below are the measure)" % rank for rank in range(8)], lines
        assert len([l for l in lines if l.startswith("SMOKE hl-smi card index")]) == 8 and cos_summary()[0] in lines
        assert "peak_memory rank=7 step=389 GiB=unavailable" in open(os.path.join(run_dir, "train.log")).read()
        assert "SMOKE verdict: FITS: 8 cards x per-card batch 64 ran one cycle over the cut lists (exit 0)" in lines and not any("0.00 GiB" in l for l in lines)
        # the projections read the run's cycle count from the line's own --epochs, not a second copy: 40 there gives 40 x 2380 = 95,200 steps and 40 cycles of saves
        shutil.rmtree(run_dir)
        text = open(script).read()
        assert text.count(" --epochs 50 ") == 1
        open(script, "w").write(text.replace(" --epochs 50 ", " --epochs 40 "))
        lines = smoke_run(script, root, smoke_env(tmp, FAKE_TRAINER="1")).stdout.splitlines()
        assert any(l.startswith("SMOKE projection: ") and " x 95200 steps / 3600" in l for l in lines) and any(l.startswith("SMOKE pass-end export time: ") and " over 40 cycles " in l for l in lines), lines


def test_newdata224_cos50_smoke_exit_codes():
    oom, boom = "RuntimeError: Habana: no storage tensor attached to lazy tensor", "ValueError: boom"
    with tempfile.TemporaryDirectory() as tmp:
        root = smoke_repo(tmp)
        script = smoke_script(tmp, root)

        def attempt(**extra):  # one smoke on a fresh smoke dir (the job id is fixed): (the result, its stdout lines)
            if os.path.exists(os.path.join(root, SMOKE_DIR)):
                shutil.rmtree(os.path.join(root, SMOKE_DIR))
            result = smoke_run(script, root, smoke_env(tmp, FAKE_TRAINER="1", **extra))
            return result, result.stdout.splitlines()

        # the trainer's exit code is the script's, and the verdict says which kind of failure it was; the cleanup runs either way
        result, lines = attempt(FAKE_RC="1", FAKE_ERR=oom)
        assert result.returncode == 1 and "SMOKE verdict: OUT OF MEMORY: 8 cards x per-card batch 64 (exit 1); the log above has the error" in lines and not any("FITS" in l for l in lines)
        result, lines = attempt(FAKE_RC="3", FAKE_ERR=boom)
        assert result.returncode == 3 and "SMOKE verdict: FAILED (exit 3) and not obviously memory: read the log above" in lines
        assert not os.path.exists(os.path.join(root, SMOKE_DIR, "ckpt")) and not os.path.exists(os.path.join(root, SMOKE_DIR, "export"))


def test_newdata224_cos50_smoke_refusals():
    with tempfile.TemporaryDirectory() as tmp:
        root = smoke_repo(tmp)
        script = smoke_script(tmp, root)
        # 4 cards: the run needs 8, and the lists are cut for 8 ranks; nothing is made or launched
        result = smoke_run(script, root, smoke_env(tmp, 4, FAKE_TRAINER="1"))
        assert result.returncode == 1 and "ERROR: 4 cards in the allocation" in result.stderr, result.stderr
        no_launch(root)
        # the smoke dir exists: it is new for every job
        os.makedirs(os.path.join(root, SMOKE_DIR))
        result = smoke_run(script, root, smoke_env(tmp, FAKE_TRAINER="1"))
        assert result.returncode == 1 and "ERROR: %s exists" % SMOKE_DIR in result.stderr, result.stderr
        no_launch(root, smoke_dir_exists=True)
        os.rmdir(os.path.join(root, SMOKE_DIR))
        # no datasets mount
        result = smoke_run(smoke_script(tmp, root, datasets="nowhere"), root, smoke_env(tmp, FAKE_TRAINER="1"))
        assert result.returncode == 1 and "datasets-new not found" in result.stderr, result.stderr
        no_launch(root)
        # a trainer here that does not know --teacher_whole_image_size
        old = smoke_repo(tempfile.mkdtemp(dir=tmp), trainer=False)
        result = smoke_run(smoke_script(os.path.dirname(old), old), old, smoke_env(os.path.dirname(old), FAKE_TRAINER="1"))
        assert result.returncode == 1 and "base/ssl_pretrain.py here has no --teacher_whole_image_size" in result.stderr, result.stderr
        no_launch(old)
        # a train list missing or empty, one a relative path (under base/) and one an absolute one: the error names the file, before anything is made
        for name, path in (("MIMIC", "base/dataset/MIMIC/mimic-cxr-2.0.0-train.csv"), ("TAIXRay", "../datasets-new/TAIXRay.csv"), ("VinDrCXR", "base/dataset/VinDrCXR/VinDrCXR_train_pe_global_one.txt")):
            for empty in (False, True):
                bare = smoke_repo(tempfile.mkdtemp(dir=tmp))
                target = os.path.normpath(os.path.join(bare, path))
                if empty:
                    open(target, "w").close()
                else:
                    os.remove(target)
                result = smoke_run(smoke_script(os.path.dirname(bare), bare), bare, smoke_env(os.path.dirname(bare), FAKE_TRAINER="1"))
                assert result.returncode == 1 and "ERROR: " in result.stderr and os.path.basename(path) in result.stderr and "is missing or empty" in result.stderr, (name, empty, result.stderr)
                no_launch(bare)
        # a config without one of the seven
        bare = smoke_repo(tempfile.mkdtemp(dir=tmp))
        config = os.path.join(bare, "base", NATIVE_YAML)
        text = open(config).read()
        assert text.count("\nReXGradient:\n") == 1
        open(config, "w").write(text.replace("\nReXGradient:\n", "\nOtherDataset:\n"))
        result = smoke_run(smoke_script(os.path.dirname(bare), bare), bare, smoke_env(os.path.dirname(bare), FAKE_TRAINER="1"))
        assert result.returncode == 1 and "has no entry with a data_dir and a train_list for ReXGradient" in result.stderr, result.stderr
        no_launch(bare)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("OK", name)
    print("ALL OK")
