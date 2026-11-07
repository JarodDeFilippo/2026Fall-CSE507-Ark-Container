"""CPU checks for the 512 px new-data run (chain/newdata512.sh): the three datasets added to Stage A's four (PadChest, TAIXRay, ReXGradient), SSLImageDataset's 16-bit-safe read and --pre_resize
squash, --no_gram with --gram_teacher_size = --global_crops_size, --init_new_heads_from (parse_args, load_init_state, and a tiny end-to-end start from a finished run's checkpoint with its --resume),
--pre_resize (its locks, and in that end-to-end start), the launch script and its sbatch wrapper in a fake repo, ssl_common.sh's SSL_DATASETS and gaudi.sh's EXTRA_BINDS.
Run from base/: python test_newdata.py (needs torch, timm, torchvision, PIL; no Slurm, no Gaudi). The test_ functions are also collected by pytest."""
import contextlib
import io
import json
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
from ssl_dinov3.data import DATASET_NAMES, DataAugmentationCXR, SSLImageDataset, image_paths
from ssl_dinov3.model import build_models
from ssl_dinov3.optim import lr_scaling_factor
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
    for name in ("newdata512.sh", "ssl_common.sh"):
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
    result = subprocess.run(["bash", script], cwd=cwd or root, env=env, capture_output=True, text=True)
    path = os.path.join(root, "argv.txt")
    return result, (open(path).read().splitlines() if os.path.exists(path) else None)


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


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("OK", name)
    print("ALL OK")
