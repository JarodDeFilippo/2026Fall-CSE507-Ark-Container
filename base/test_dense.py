"""CPU checks for base/dense (dense heads on a frozen ViT backbone): the ViT feature pyramid and Foundation X's UPerNet head and Dice.
Random-init backbones, no files needed. Run from base/: python test_dense.py"""
import warnings

warnings.filterwarnings("ignore", message="xFormers is ")  # DINOv2's layer modules warn once each when xformers is absent

from types import SimpleNamespace

import numpy as np
import torch

import models
from dense.pyramid import ViTPyramid, layers_for
from dense.upernet import SegHead, dice_score, mean_dice_coef, step_decay, torch_dice_coef_loss


def rejects(exception, call, expected=None):
    """The exception message when call() raises `exception`; fails when it returns."""
    try:
        call()
    except exception as error:
        assert expected is None or expected in str(error), (expected, str(error))
        return str(error)
    raise AssertionError("expected {}".format(exception.__name__))


# (pyramid) the ViTDet-style feature pyramid over the frozen backbone's patch tokens ------------------------------------------
# name, backbone constructor, layers_for(backbone), the four maps at 224 px (batch 2): x4, x2, x1, /2 of the patch grid (14 for patch 16, 16 for patch 14)
B16 = [(2, 128, 56, 56), (2, 256, 28, 28), (2, 512, 14, 14), (2, 1024, 7, 7)]
B14 = [(2, 128, 64, 64), (2, 256, 32, 32), (2, 512, 16, 16), (2, 1024, 8, 8)]
CASES = (("ViT-B/16 (DINOv3)", lambda: models.ArkViT(num_classes_list=[], **models.DINOV3_VITB16_KWARGS), [2, 5, 8, 11], B16),
         ("ViT-B/14 (DINOv2)", lambda: models.ArkViTDinoV2(num_classes_list=[], **models.DINOV2_VITB14_KWARGS), [2, 5, 8, 11], B14),
         ("ViT-L/16 (DINOv2)", lambda: models.ArkViTDinoV2(num_classes_list=[], **models.DINOV2_VITL16_KWARGS), [5, 11, 17, 23], B16))
torch.manual_seed(0)
x = torch.randn(2, 3, 224, 224)
for name, make, layers, expected in CASES:
    bb = make()
    assert layers_for(bb) == layers, (name, layers_for(bb))
    pyr = ViTPyramid(bb, layers_for(bb))
    assert pyr.feature_channels == [128, 256, 512, 1024]
    with torch.no_grad():
        maps = pyr(x)
    assert isinstance(maps, list) and [tuple(m.shape) for m in maps] == expected, (name, [tuple(m.shape) for m in maps])
    with torch.no_grad():  # level i is its layers applied to the backbone's own normed, reshaped patch tokens of block layers[i]
        tokens = bb.get_intermediate_layers(x, n=layers, reshape=True, norm=True)
        assert all(torch.allclose(m, level(t), atol=1e-6) for m, level, t in zip(maps, pyr.levels, tokens)), name
    # the backbone is frozen and stays in eval, whatever the pyramid is told; the pyramid's own layers train
    assert all(not p.requires_grad for p in bb.parameters()) and not bb.training
    pyr.train()
    assert not bb.training and pyr.levels.training
    pyr.eval()
    assert not bb.training
    pyr.train()
    assert all(p.requires_grad for p in pyr.levels.parameters())
    sum(m.mean() for m in pyr(x)).backward()
    assert all(p.grad is not None for p in pyr.levels.parameters()) and all(p.grad is None for p in bb.parameters())
    # layers_for refuses a depth it has no layer choice for
    depth = bb.n_blocks
    bb.n_blocks = 6
    try:
        rejects(ValueError, lambda: layers_for(bb))
    finally:
        bb.n_blocks = depth
    del bb, pyr, maps, tokens  # the ViT-L is 1.2 GB
assert layers_for(SimpleNamespace(n_blocks=12)) == [2, 5, 8, 11] and layers_for(SimpleNamespace(n_blocks=24)) == [5, 11, 17, 23]
print("OK (pyramid): ViT-B/16, ViT-B/14 and ViT-L/16 give (128, 256, 512, 1024)-channel maps at x4, x2, x1, /2 of the patch grid, level i built from block layers[i]'s "
      "normed patch tokens; the backbone is frozen and stays in eval under train(); gradients reach the pyramid layers only; layers_for raises ValueError off 12 / 24 blocks")

# (upernet) FX's PSP + FPN head on the pyramid's maps, and its Dice and LR helpers ---------------------------------------------
head = SegHead([128, 256, 512, 1024]).eval()
for shapes in (B16, B14):  # patch 16 and patch 14 (RAD-DINO) at 224 px: the head does not care about the grid
    feats = [torch.randn(s) for s in shapes]
    last = feats[-1]
    with torch.no_grad():
        logits = head(feats, (224, 224))
    assert logits.shape == (2, 1, 224, 224) and feats[-1] is last  # the caller's list is not edited
head([torch.randn(s) for s in B16], (224, 224)).sum().backward()
assert all(p.grad is not None for p in head.parameters())  # PSP, FPN and the 3x3 conv are all in the path
for name, make, _, _ in CASES[:2]:  # the whole chain on a real backbone: RAD-DINO's patch 14 included
    bb = make()
    with torch.no_grad():
        assert head(ViTPyramid(bb, layers_for(bb))(x), (224, 224)).shape == (2, 1, 224, 224), name
mask = np.zeros((2, 1, 8, 8))
mask[:, :, 2:6, 2:6] = 1
assert dice_score(mask, mask) == 1.0 and dice_score(mask, 1 - mask) == 0.0 and dice_score(np.zeros((8, 8)), np.zeros((8, 8))) == 1.0
y_true, y_pred = np.zeros((2, 1, 4, 4)), np.zeros((2, 1, 4, 4))
y_true[0] = y_pred[0] = 1  # image 0 matches
y_true[1, :, :2] = 1  # image 1 is disjoint
y_pred[1, :, 2:] = 1
assert mean_dice_coef(y_true, y_pred) == 0.5
ones = torch.ones(2, 1, 4, 4)
assert torch_dice_coef_loss(ones, ones) < 1e-3 and torch_dice_coef_loss(ones, torch.zeros_like(ones)) > 0.9
assert step_decay(0, 1e-4, 5000, 250) == 0.0 and abs(step_decay(250, 1e-4, 5000, 250) - 1e-4) < 1e-12 and abs(step_decay(5000, 1e-4, 5000, 250)) < 1e-12
print("OK (upernet): SegHead gives (2, 1, 224, 224) logits from the ViT-B/16 and ViT-B/14 pyramid shapes and behind real ViT-B/16 and ViT-B/14 backbones, "
      "leaves the caller's list alone, and every parameter takes a gradient; Dice is 1 for equal masks, 0 for disjoint ones, 1 for two empty ones; the per-image mean is 0.5 "
      "for one match and one miss; the Dice loss is ~0 on a match; step_decay is 0 at step 0, the LR at step 250, 0 at the end")

# (data) CANDID-PTX (DICOM + RLE) and JSRT lung datasets, on synthetic files ---------------------------------------------------
import contextlib
import os
import pickle
import tempfile

import cv2
from PIL import Image

from ark_finetune.dataloader import build_transform_classification
from dense import data

warnings.filterwarnings("ignore", message="ShiftScaleRotate is a special case")  # albumentations 2.0.8's notice, once per build_transforms


def encode_rle(m):
    """The CANDID-PTX / SIIM RLE of a 0/1 mask: pixels numbered down the columns, "start length" pairs, each start counted from the end of the previous run."""
    flat = np.concatenate([[0], m.T.flatten(), [0]])
    edges = np.flatnonzero(flat[1:] != flat[:-1])  # where a run starts (even entries) and where it ends, exclusive (odd entries)
    starts, ends = edges[0::2], edges[1::2]
    return " ".join("{} {}".format(s, n) for s, n in zip(starts - np.concatenate([[0], ends[:-1]]), ends - starts))


@contextlib.contextmanager
def fake_dcmread(pixels):
    """pydicom.dcmread (data.dicom is the pydicom module) serves pixels[path] as the pixel_array of a stand-in DICOM."""
    real = data.dicom.dcmread
    data.dicom.dcmread = lambda path: SimpleNamespace(pixel_array=pixels[path])
    try:
        yield
    finally:
        data.dicom.dcmread = real


def write_png(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array).save(path)


def plain_pipeline(rgb, size, normalization):
    """The pipeline in plain cv2 / numpy for a val or test item: INTER_AREA resize, then (x / 255 - mean) / std, as float32 [3, size, size]."""
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    if data.NORMALIZATION[normalization] is not None:
        resized = (resized - np.array(data.NORMALIZATION[normalization][0], np.float32)) / np.array(data.NORMALIZATION[normalization][1], np.float32)
    return torch.from_numpy(resized).permute(2, 0, 1)


# the RLE decoder: pixels numbered down the columns, width = columns, height = rows
rle_mask = data.rle2mask("0 3", width=4, height=5)  # a run down the first column
assert rle_mask.shape == (5, 4) and rle_mask.dtype == np.uint8 and rle_mask.sum() == 3 and rle_mask[0:3, 0].tolist() == [1, 1, 1]
rle_mask = data.rle2mask("4 2 3 1", 4, 5)  # numbers 4, 5 (the run) and 9 (3 on from its end): wraps from the first column into the second
assert rle_mask.sum() == 3 and (rle_mask[4, 0], rle_mask[0, 1], rle_mask[4, 1]) == (1, 1, 1)
for blank in ("-1", "", "   "):
    rle_mask = data.rle2mask(blank, 4, 5)
    assert rle_mask.shape == (5, 4) and rle_mask.dtype == np.uint8 and not rle_mask.any(), blank
rng = np.random.default_rng(0)
rle_mask = (rng.random((6, 9)) < 0.4).astype(np.uint8)  # non-square round trip, with runs at both ends of the numbering
rle_mask[0, 0] = rle_mask[-1, -1] = 1
assert np.array_equal(data.rle2mask(encode_rle(rle_mask), width=9, height=6), rle_mask)
for rows, cols in ((1250, 1024), (1024, 1250)):  # full-size non-square images: a block on the right, near the top, must stay there
    rle_mask = np.zeros((rows, cols), np.uint8)
    rle_mask[rows // 10:rows // 3, cols * 9 // 10:cols * 19 // 20] = 1
    rle = encode_rle(rle_mask)
    decoded = data.rle2mask(rle, width=cols, height=rows)
    assert decoded.shape == (rows, cols) and decoded.sum() == sum(int(v) for v in rle.split()[1::2]) == rle_mask.sum()
    ys, xs = np.nonzero(decoded)
    assert xs.min() >= cols * 9 // 10 and ys.max() < rows // 3 and np.array_equal(decoded, rle_mask)

# DICOM -> uint8: min-max per image, nothing clipped, a constant image is zeros
with fake_dcmread({"fourteen": np.array([[0, 8191, 16383]], dtype=np.uint16), "twelve": np.array([[100, 2047, 4095]], dtype=np.uint16),
                   "flat": np.full((3, 3), 777, dtype=np.uint16)}):
    assert data.dicom_to_uint8("fourteen").tolist() == [[0, 127, 255]]
    # the copied formula floors a float product, so the maximum comes out 254 for some ranges (this one) and 255 for others; no value clips
    assert data.dicom_to_uint8("twelve").dtype == np.uint8 and data.dicom_to_uint8("twelve").tolist() == [[0, 124, 254]]
    flat = data.dicom_to_uint8("flat")
    assert flat.dtype == np.uint8 and flat.shape == (3, 3) and not flat.any()

# the normalization constants are the linear probe's, read off its own transform; the registry is the lab's paths, /data/jliang12 only
for norm_name in ("imagenet", "chestx-ray", "raddino"):
    probe_norm = build_transform_classification(norm_name, mode="valid").transforms[-1]
    assert (tuple(probe_norm.mean), tuple(probe_norm.std)) == data.NORMALIZATION[norm_name], norm_name
assert data.NORMALIZATION["none"] is None and not hasattr(build_transform_classification("none", mode="valid").transforms[-1], "mean")
lab_lists = "/data/jliang12/nuislam/data_files_splits"
assert sorted(data.DATASETS) == ["candidptx", "jsrt_lung"]
assert data.DATASETS["candidptx"]["root"] == "/data/jliang12/shared/dataset/CANDID-PTX/dataset"
assert data.DATASETS["candidptx"]["lists"] == {"train": lab_lists + "/candid_ptx/train.txt", "val": lab_lists + "/candid_ptx/valid.txt", "test": lab_lists + "/candid_ptx/test.txt"}
assert data.DATASETS["jsrt_lung"]["root"] == "/data/jliang12/jpang12/dataset/JSRT/All247images"
assert data.DATASETS["jsrt_lung"]["lists"] == {"train": lab_lists + "/jsrt/train.txt", "val": lab_lists + "/jsrt/val.txt", "test": lab_lists + "/jsrt/test.txt"}
rejects(ValueError, lambda: data.build_transforms("bogus", "imagenet", 64), "train, val or test")
rejects(KeyError, lambda: data.build_transforms("train", "bogus", 64))

# normalization comes last: on a gradient image the train pipeline keeps values below -1 and above 1 whatever fires (normalized first,
# brightness/contrast would clip it into [0, 1])
gradient = np.repeat(np.linspace(0, 255, 64 * 64).astype(np.uint8).reshape(64, 64, 1), 3, axis=2)
train_pipeline = data.build_transforms("train", "imagenet", 64)
train_pipeline.set_random_seed(0)
for _ in range(40):
    out = train_pipeline(image=gradient, mask=np.zeros((64, 64), np.uint8))["image"]
    assert out.dtype == torch.float32 and out.shape == (3, 64, 64) and out.min() < -1 and out.max() > 1

# JSRT: images_png/<id>.png, the union of the two lung masks; "b" has its masks at half the image's resolution and valued 1, not 255; "align" is one bright block; "p" is "a" with a palette image and left mask
with tempfile.TemporaryDirectory() as jsrt_root:
    left = np.zeros((64, 64), np.uint8)
    left[10:50, 8:28] = 255
    right = np.zeros((64, 64), np.uint8)
    right[10:50, 36:56] = 255
    lungs = np.full((64, 64), 30, np.uint8)
    lungs[(left > 0) | (right > 0)] = 200
    block = np.zeros((64, 64), np.uint8)
    block[8:28, 40:60] = 255
    align = np.full((64, 64), 20, np.uint8)
    align[block > 0] = 220
    for id_, image, left_mask, right_mask in (("a", lungs, left, right), ("b", lungs, left[::2, ::2] // 255, right[::2, ::2] // 255), ("align", align, block, 0 * block)):
        write_png(os.path.join(jsrt_root, "images_png", id_ + ".png"), image)
        write_png(os.path.join(jsrt_root, "masks", "left_lung_png", id_ + ".png"), left_mask)
        write_png(os.path.join(jsrt_root, "masks", "right_lung_png", id_ + ".png"), right_mask)
    # "p": "a" as palette (P) PNGs, the image and the left mask; the right mask stays 8-bit
    Image.fromarray(lungs).convert("P").save(os.path.join(jsrt_root, "images_png", "p.png"))
    Image.fromarray(left).convert("P").save(os.path.join(jsrt_root, "masks", "left_lung_png", "p.png"))
    write_png(os.path.join(jsrt_root, "masks", "right_lung_png", "p.png"), right)
    jsrt_list = os.path.join(jsrt_root, "list.txt")
    with open(jsrt_list, "w") as f:
        f.write("a\nb\n\nalign\np\n")
    jsrt_test = data.JSRTLung(jsrt_root, jsrt_list, size=224, mode="test", normalization="imagenet")
    assert len(jsrt_test) == 4  # the blank line is skipped
    img, msk = jsrt_test[0]
    assert img.shape == (3, 224, 224) and img.dtype == torch.float32 and msk.shape == (1, 224, 224) and msk.dtype == torch.float32
    assert set(msk.unique().tolist()) <= {0.0, 1.0} and msk.sum() > 0
    assert np.array_equal(msk[0].numpy(), cv2.resize(left | right, (224, 224), interpolation=cv2.INTER_NEAREST) // 255)  # both lungs, nothing else
    assert torch.allclose(img, plain_pipeline(np.repeat(lungs[..., None], 3, axis=-1), 224, "imagenet"), atol=1e-4)
    assert torch.equal(jsrt_test[1][1], msk)  # the half-resolution masks of "b" give the same mask
    p_img, p_msk = jsrt_test[3]  # the palette image and mask of "p" give the same shapes and values as "a"
    assert p_img.shape == (3, 224, 224) and p_msk.shape == (1, 224, 224) and torch.equal(p_img, img) and torch.equal(p_msk, msk)
    jsrt_train = data.JSRTLung(jsrt_root, jsrt_list, size=224, mode="train", normalization="imagenet")
    img, msk = jsrt_train[0]
    assert img.shape == (3, 224, 224) and img.dtype == torch.float32 and msk.shape == (1, 224, 224) and msk.dtype == torch.float32
    assert set(msk.unique().tolist()) <= {0.0, 1.0} and msk.sum() > 0
    for _ in range(40):  # the mask follows the image through the shift, scale and rotation: the bright block stays inside it
        img, msk = jsrt_train[2]
        assert dice_score((img[0] > 0).numpy(), msk[0].numpy()) > 0.9
    jsrt_none = data.make_dataset("jsrt_lung", "val", 64, "none", root=jsrt_root, list_path=jsrt_list)
    assert isinstance(jsrt_none, data.JSRTLung) and torch.allclose(jsrt_none[0][0], plain_pipeline(np.repeat(lungs[..., None], 3, axis=-1), 64, "none"), atol=1e-6)
    jsrt_copy = pickle.loads(pickle.dumps(jsrt_train))  # DataLoader workers get a copy of the dataset
    assert len(jsrt_copy) == 4 and jsrt_copy[0][0].shape == (3, 224, 224)

    def draws(seed):  # six train items of a fresh dataset, as a worker whose torch seed is `seed` would draw them
        dataset = data.JSRTLung(jsrt_root, jsrt_list, size=64, mode="train", normalization="imagenet")
        torch.manual_seed(seed)
        return [dataset[i % 3][0] for i in range(6)]
    same_a, same_b, other = draws(7), draws(7), draws(8)
    assert all(torch.equal(p, q) for p, q in zip(same_a, same_b)) and not all(torch.equal(p, q) for p, q in zip(same_a, other))
    # a 16-bit PNG would convert to 8 bit with every value >= 255 white: refused
    write_png(os.path.join(jsrt_root, "images_png", "deep.png"), np.full((8, 8), 3000, np.uint16))
    with open(jsrt_list + ".deep", "w") as f:
        f.write("deep\n")
    rejects(ValueError, lambda: data.JSRTLung(jsrt_root, jsrt_list + ".deep", size=64, mode="test")[0], "PNG, expected 8-bit")

# CANDID-PTX: "uid,rle" lines, the DICOM at root/uid; uid-a is non-square (80 rows, 64 columns), uid-c is the RLE "0 3" on a 64 x 64 image
with tempfile.TemporaryDirectory() as candid_root:
    rng = np.random.default_rng(1)
    ptx = np.zeros((80, 64), np.uint8)
    ptx[50:70, 40:60] = 1  # lower right
    pixels = {os.path.join(candid_root, "uid-a"): rng.integers(1000, 3000, (80, 64)).astype(np.uint16),
              os.path.join(candid_root, "uid-b"): rng.integers(1000, 3000, (64, 64)).astype(np.uint16),
              os.path.join(candid_root, "uid-c"): rng.integers(1000, 3000, (64, 64)).astype(np.uint16)}
    candid_list = os.path.join(candid_root, "list.txt")
    with open(candid_list, "w") as f:
        f.write("uid-a,{}\nuid-b,-1\n\nuid-c,0 3\n".format(encode_rle(ptx)))
    with fake_dcmread(pixels):
        candid_test = data.CandidPTX(candid_root, candid_list, size=224, mode="test", normalization="imagenet")
        assert len(candid_test) == 3
        img, msk = candid_test[0]
        assert img.shape == (3, 224, 224) and img.dtype == torch.float32 and msk.shape == (1, 224, 224) and msk.dtype == torch.float32
        assert np.array_equal(msk[0].numpy(), cv2.resize(ptx, (224, 224), interpolation=cv2.INTER_NEAREST))  # the drawn mask, lower right, not transposed
        uid_a = data.dicom_to_uint8(os.path.join(candid_root, "uid-a"))
        assert torch.allclose(img, plain_pipeline(np.repeat(uid_a[..., None], 3, axis=-1), 224, "imagenet"), atol=1e-4)
        assert not candid_test[1][1].any()  # "-1": no pneumothorax
        assert 0 < candid_test[2][1].sum() < 224 * 224 * 0.05  # 3 pixels of the first column, each about 3.5 x 3.5 at 224
        candid_train = data.CandidPTX(candid_root, candid_list, size=224, mode="train", normalization="imagenet")
        img, msk = candid_train[0]
        assert img.shape == (3, 224, 224) and img.dtype == torch.float32 and msk.shape == (1, 224, 224) and msk.dtype == torch.float32
        assert set(msk.unique().tolist()) <= {0.0, 1.0}
        candid_made = data.make_dataset("candidptx", "test", 64, "raddino", root=candid_root, list_path=candid_list)
        assert isinstance(candid_made, data.CandidPTX) and candid_made[0][0].shape == (3, 64, 64)
    with open(candid_list + ".bad", "w") as f:
        f.write("uid-a\n")
    rejects(ValueError, lambda: data.CandidPTX(candid_root, candid_list + ".bad"), "uid,rle")
print("OK (data): rle2mask puts the runs down the columns of a (height, width) mask and gives zeros for -1 / blank, through a round trip on 6 x 9 and on 1250 x 1024 and "
      "1024 x 1250 masks (sum = run total, block stays on the right); dicom_to_uint8 spreads 12- and 14-bit ranges over 0-255 without clipping and gives zeros for a constant image; "
      "NORMALIZATION equals the linear probe's constants; JSRTLung (union of both lungs, half-resolution masks, palette PNGs read like 8-bit ones, 16-bit PNG refused) and CandidPTX (non-square, -1, blank lines, "
      "no-comma line refused) return float32 (3, S, S) / (1, S, S) {0, 1} items in test and train mode, equal to a plain cv2 resize + normalize; normalization comes last; "
      "the train mask follows the geometric augmentation; --seed (torch's seed) fixes the augmentation draws")

# (seg_probe) the training / evaluation entry point: a random-init ViT-B/16 at 64 px on a synthetic JSRT set, four iterations ----------------------------
import copy
import errno
import glob
import hashlib
import io
import json
import re
from unittest import mock

from dense import seg_probe


def quietly(call):
    """call() with stdout and stderr captured: (its value, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        value = call()
    return value, out.getvalue(), err.getvalue()


# evaluate on stand-ins with a known answer: the "image" carries the map to predict (the mask itself, or its inverse), the stub head turns it into +-20 logits
class EchoHead(torch.nn.Module):
    def forward(self, feats, out_size):
        return 40 * (feats[0][:, :1] - 0.5)


m1 = torch.zeros(2, 1, 8, 8)
m1[:, :, 2:6, 2:6] = 1  # two images, 16 ones each, predicted right
m2 = torch.zeros(1, 1, 8, 8)
m2[:, :, 0:4, 0:4] = 1  # one image, 16 ones, predicted as its inverse
echo, cpu = EchoHead().train(), torch.device("cpu")
loss, pooled, mean = seg_probe.evaluate(lambda img: [img], echo, [(m1.repeat(1, 3, 1, 1), m1), ((1 - m2).repeat(1, 3, 1, 1), m2)], cpu)
assert echo.training  # evaluate hands the head back in the mode it got it
assert abs(loss - (0 * 2 + (1 - 1 / 65) * 1) / 3) < 1e-5  # batch losses ~0 and 1 - 1/(16 + 48 + 1), averaged by batch size (2 and 1), not by batch
assert abs(pooled - 50.0) < 1e-4 and abs(mean - 200 / 3) < 1e-4  # pooled: 2 x 32 / (80 + 48); per image: 1, 1, 0
seg_probe.evaluate(lambda img: [img], echo.eval(), [(m1.repeat(1, 3, 1, 1), m1)], cpu)
assert not echo.training
assert seg_probe.evaluate(lambda img: [img], echo, [(m1.repeat(1, 3, 1, 1), m1)], cpu)[1:] == (100.0, 100.0)
with seg_probe.autocast(cpu):  # bf16 autocast is for CUDA only
    assert not torch.is_autocast_enabled("cpu")

with tempfile.TemporaryDirectory() as seg_dir:
    rng = np.random.default_rng(2)
    seg_ids = ["s{}".format(i) for i in range(10)]  # train s0-s4 (five: batch 2 leaves a partial last batch), val s5-s6, test s7-s9
    for id_ in seg_ids:
        left = np.zeros((64, 64), np.uint8)
        left[10:50, 8:28] = 255
        right = np.zeros((64, 64), np.uint8)
        right[10:50, 36:56] = 255
        image = rng.integers(0, 60, (64, 64)).astype(np.uint8)
        image[(left > 0) | (right > 0)] += 150
        write_png(os.path.join(seg_dir, "images_png", id_ + ".png"), image)
        write_png(os.path.join(seg_dir, "masks", "left_lung_png", id_ + ".png"), left)
        write_png(os.path.join(seg_dir, "masks", "right_lung_png", id_ + ".png"), right)
    seg_lists = os.path.join(seg_dir, "lists")
    os.makedirs(seg_lists)
    for split, split_ids in (("train", seg_ids[:5]), ("val", seg_ids[5:7]), ("test", seg_ids[7:])):
        with open(os.path.join(seg_lists, split + ".txt"), "w") as f:
            f.write("\n".join(split_ids) + "\n")
    seg_out = os.path.join(seg_dir, "results")

    def seg_argv(tag, weights="random", *extra):
        """The command line of a four-iteration run (batch 2, validation at 2 and 4, lr 2e-4, weight decay 5e-5) on the synthetic set; `extra` flags come last and win."""
        return ["--model", "vit_base_dinov3", "--weights", weights, "--normalization", "imagenet", "--dataset", "jsrt_lung", "--tag", tag, "--iters", "4", "--batch", "2",
                "--val_every", "2", "--workers", "0", "--size", "64", "--seed", "0", "--lr", "2e-4", "--wd", "5e-5", "--device", "cpu", "--out", seg_out, "--data_root", seg_dir,
                "--list_dir", seg_lists] + list(extra)

    def refused(*extra):
        """parse_args' error text for the run's command line plus `extra` (argparse prints it to stderr and exits 2)."""
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rejects(SystemExit, lambda: seg_probe.parse_args(seg_argv("refused", "random", *extra)))
        return err.getvalue()

    def partials():
        """Every staging directory (seed<S>.partial-<pid>-<hex>) under the results, whatever its tag or dataset."""
        return sorted(glob.glob(os.path.join(seg_out, "*", "*", "*.partial-*")))

    assert "needs --device cpu" in refused("--device", "cuda") and "is not a file" in refused("--weights", os.path.join(seg_dir, "missing.pth"))
    assert "letters, digits" in refused("--tag", "bad tag") and "at least 2" in refused("--iters", "1") and "invalid choice" in refused("--model", "vit_large_dinov3")
    assert "at least 0" in refused("--seed", "-1")  # the seed names a directory, seed<S>
    assert "--batch must be at least 2" in refused("--batch", "1")  # PSP's 1 x 1 bin gives training-mode BatchNorm2d one value per channel from a batch of one: refused here, not by the first forward pass
    assert seg_probe.parse_args(seg_argv("batch2", "random", "--batch", "2")).batch == 2
    assert [seg_probe.step_inc_for(n) for n in (5000, 20000, 100, 20, 4, 2)] == [250, 250, 5, 1, 1, 1]  # 250 at RayDINO's 5,000 iterations, never above iters
    defaults = seg_probe.parse_args(["--model", "vit_base_dinov3", "--weights", "random", "--normalization", "imagenet", "--dataset", "jsrt_lung", "--tag", "d", "--device", "cpu"])
    assert [getattr(defaults, k) for k in ("iters", "batch", "val_every", "lr", "wd", "seed", "workers", "size", "out", "data_root", "list_dir")] == \
        [5000, 16, 250, 1e-4, 1e-4, 0, 12, 224, "results/dense", None, None]  # RayDINO's budget

    # a run with spies: the model it builds, the backbone loads it asks for, the loaders, the learning rate and weight decay each optimizer step runs at, what the Dice loss is given
    real_build, real_load, real_step, real_loader, real_loss = seg_probe.build_model, seg_probe.load_backbone, torch.optim.AdamW.step, seg_probe.make_loader, seg_probe.torch_dice_coef_loss
    built, loads, lrs, loaders, loss_inputs = {}, [], [], [], []

    def spy_build(*args, **kwargs):
        built["pyramid"], built["head"] = real_build(*args, **kwargs)
        built["backbone"], built["head0"] = copy.deepcopy(built["pyramid"].backbone.state_dict()), copy.deepcopy(built["head"].state_dict())
        built["levels0"] = copy.deepcopy(built["pyramid"].levels.state_dict())
        return built["pyramid"], built["head"]

    def spy_load(model_name, weights):
        loads.append((model_name, weights))
        return real_load(model_name, weights)

    def spy_step(self, *args, **kwargs):
        lrs.append((self.param_groups[0]["lr"], self.param_groups[0]["weight_decay"]))
        return real_step(self, *args, **kwargs)

    def spy_loader(*args, **kwargs):
        loaders.append(real_loader(*args, **kwargs))
        return loaders[-1]

    def spy_loss(y_true, y_pred):
        loss_inputs.append((y_true.unique().tolist(), float(y_pred.min()), float(y_pred.max())))
        return real_loss(y_true, y_pred)

    with mock.patch.object(seg_probe, "build_model", spy_build), mock.patch.object(seg_probe, "load_backbone", spy_load), mock.patch.object(torch.optim.AdamW, "step", spy_step), \
            mock.patch.object(seg_probe, "make_loader", spy_loader), mock.patch.object(seg_probe, "torch_dice_coef_loss", spy_loss):
        result, stdout, stderr = quietly(lambda: seg_probe.train(seg_probe.parse_args(seg_argv("plain"))))
    plain_dir = os.path.join(seg_out, "plain", "jsrt_lung")
    run_dir = os.path.join(plain_dir, "seed0")  # <out>/<tag>/<dataset>/seed<S>/
    assert sorted(os.listdir(run_dir)) == ["head_best.pt", "result.json"] and stdout == "" and stderr.count("seg_probe: iteration ") == 2
    assert os.listdir(plain_dir) == ["seed0"] and partials() == []  # published, the emptied staging directory removed: no seed0.partial-* beside it
    with open(os.path.join(run_dir, "result.json")) as f:
        assert json.load(f) == result
    assert set(result) == {"tag", "model", "weights", "weights_sha256", "normalization", "dataset", "seed", "iters", "batch", "lr", "wd", "size", "step_inc", "best_iter", "best_val_loss",
                           "val_losses", "test_dice_pooled", "test_dice_mean", "n_train", "n_val", "n_test", "git_sha", "started", "finished"}
    assert [result[k] for k in ("tag", "model", "weights", "weights_sha256", "normalization", "dataset")] == ["plain", "vit_base_dinov3", "random", None, "imagenet", "jsrt_lung"]
    assert [result[k] for k in ("seed", "iters", "batch", "lr", "wd", "size", "step_inc", "n_train", "n_val", "n_test")] == [0, 4, 2, 2e-4, 5e-5, 64, 1, 5, 2, 3]
    assert len(result["val_losses"]) == 2 and all(0 <= v <= 1 for v in result["val_losses"]) and result["best_iter"] in (2, 4)
    assert result["best_val_loss"] == min(result["val_losses"]) == result["val_losses"][result["best_iter"] // 2 - 1]
    assert 0 <= result["test_dice_mean"] <= 100 and 0 <= result["test_dice_pooled"] <= 100
    assert result["git_sha"] is None or re.fullmatch("[0-9a-f]{40}", result["git_sha"])
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", result["started"]) and result["started"] <= result["finished"]
    assert loads == [("vit_base_dinov3", None)]  # 'random' reaches load_backbone as no weights
    # the loaders: train is shuffled by the run's seed and drops a partial last batch (a batch of one would break BatchNorm); val and test keep every image, in order
    train_loader, val_loader, test_loader = loaders
    assert isinstance(train_loader.sampler, torch.utils.data.RandomSampler) and train_loader.drop_last and train_loader.generator.initial_seed() == 0
    assert not isinstance(val_loader.sampler, torch.utils.data.RandomSampler) and not val_loader.drop_last and not test_loader.drop_last
    assert [len(b[0]) for b in test_loader] == [2, 1] and [len(b[0]) for b in train_loader] == [2, 2]
    # step_decay at iterations 1-4 of 4 with step_inc 1 and lr 2e-4: the full rate, then 0.75, 0.25 and nothing of it (cosine to zero at the last iteration); weight decay 5e-5 throughout
    assert len(lrs) == 4 and all(abs(lr - want) < 1e-12 and wd == 5e-5 for (lr, wd), want in zip(lrs, [2e-4, 1.5e-4, 0.5e-4, 0.0])), lrs
    # the Dice loss gets the sigmoid of the logits and a {0, 1} mask: 4 training steps, 2 validation passes of one batch, 2 test batches
    assert len(loss_inputs) == 8 and all(set(m) <= {0.0, 1.0} and 0 <= lo <= hi <= 1 for m, lo, hi in loss_inputs), loss_inputs
    # the backbone is frozen: every tensor of it as it was built, no gradient, still in eval; the pyramid's level layers and the head did move
    backbone = built["pyramid"].backbone
    assert all(torch.equal(v, built["backbone"][k]) for k, v in backbone.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in backbone.parameters()) and not backbone.training
    assert any(not torch.equal(v, built["head0"][k]) for k, v in built["head"].state_dict().items() if v.is_floating_point())
    assert any(not torch.equal(v, built["levels0"][k]) for k, v in built["pyramid"].levels.state_dict().items())
    # head_best.pt: plain tensors of the trained part only, keyed as the modules are, and the state the test list was scored with
    saved = torch.load(os.path.join(run_dir, "head_best.pt"), weights_only=True)
    assert sorted(saved) == ["head", "pyramid"] and not any(k.startswith("backbone") for k in saved["pyramid"])
    assert list(saved["pyramid"]) == list(built["pyramid"].levels.state_dict()) and list(saved["head"]) == list(built["head"].state_dict())
    assert all(torch.equal(saved["head"][k], v) for k, v in built["head"].state_dict().items())
    assert all(torch.equal(saved["pyramid"][k], v) for k, v in built["pyramid"].levels.state_dict().items())
    del built, backbone, saved
    # the same seed gives the same run (init, shuffle, augmentation draws); a result.json that exists is never overwritten
    again, _, _ = quietly(lambda: seg_probe.train(seg_probe.parse_args(seg_argv("again"))))
    assert again["best_iter"] == result["best_iter"] and all(abs(a - b) < 1e-9 for a, b in zip(again["val_losses"], result["val_losses"]))
    with open(os.path.join(run_dir, "result.json")) as f:
        before = f.read()
    rejects(FileExistsError, lambda: seg_probe.train(seg_probe.parse_args(seg_argv("plain"))), "refusing to overwrite")
    with open(os.path.join(run_dir, "result.json")) as f:
        assert f.read() == before
    # a run directory that exists is refused up front, not after the run, whatever is in it: here a dangling symlink at result.json, which os.path.exists would call absent
    os.makedirs(os.path.join(seg_out, "dangling", "jsrt_lung", "seed0"))
    os.symlink(os.path.join(seg_dir, "nowhere.json"), os.path.join(seg_out, "dangling", "jsrt_lung", "seed0", "result.json"))
    rejects(FileExistsError, lambda: seg_probe.train(seg_probe.parse_args(seg_argv("dangling"))), "refusing to overwrite")
    # and so is a dangling symlink at the run directory's own name (lexists, not exists), with no staging directory made for the refused run
    os.makedirs(os.path.join(seg_out, "dangling_dir", "jsrt_lung"))
    os.symlink(os.path.join(seg_dir, "nowhere"), os.path.join(seg_out, "dangling_dir", "jsrt_lung", "seed0"))
    rejects(FileExistsError, lambda: seg_probe.train(seg_probe.parse_args(seg_argv("dangling_dir"))), "refusing to overwrite")
    assert os.listdir(os.path.join(seg_out, "dangling_dir", "jsrt_lung")) == ["seed0"]
    # --seed names the output directory, so the seeds of one (tag, dataset) coexist: seed 1 lands next to seed 0, which it leaves alone, and is a different run
    loaders.clear()
    with mock.patch.object(seg_probe, "make_loader", spy_loader):
        other, _, _ = quietly(lambda: seg_probe.train(seg_probe.parse_args(seg_argv("plain", "random", "--seed", "1"))))
    assert sorted(os.listdir(plain_dir)) == ["seed0", "seed1"] and other["seed"] == 1 and other["val_losses"] != result["val_losses"]
    assert loaders[0].generator.initial_seed() == 1  # the train loader's shuffle follows --seed
    with open(os.path.join(plain_dir, "seed1", "result.json")) as f:
        assert json.load(f) == other
    with open(os.path.join(run_dir, "result.json")) as f:
        assert f.read() == before
    # another job of the same (tag, dataset, seed) publishes before this one does: this one runs to its end, then its publish fails naming both paths, its finished run stays in its own
    # staging directory beside the run directory, and the other job's directory is untouched. Three iterations, validation every two: evaluate runs at iteration 2, at the last
    # iteration (3), and on the test list
    race_dir, race_calls, race_paths, synced, real_publish, real_fsync = os.path.join(seg_out, "race", "jsrt_lung", "seed0"), [], [], [], seg_probe.publish, os.fsync

    def counting_evaluate(pyramid, head, loader, device):
        race_calls.append(len(loader.dataset))
        return 0.5, 50.0, 50.0

    def spy_fsync(fd):  # what the run's staging directory held when each fsync ran
        synced.append(sorted(os.listdir(partials()[0])))
        return real_fsync(fd)

    def publish_after_other_job(staging, final_dir):  # the other job's directory lands between this run's last write and its publish
        race_paths.append((staging, final_dir))
        os.makedirs(final_dir)
        with open(os.path.join(final_dir, "result.json"), "w") as f:
            f.write("{}")
        return real_publish(staging, final_dir)

    with mock.patch.object(seg_probe, "evaluate", counting_evaluate), mock.patch.object(seg_probe, "publish", publish_after_other_job), mock.patch.object(os, "fsync", spy_fsync):
        message = rejects(RuntimeError, lambda: quietly(lambda: seg_probe.train(seg_probe.parse_args(seg_argv("race", "random", "--iters", "3")))), "could not publish")
    [(race_staging, race_final)] = race_paths
    assert race_calls == [2, 2, 3] and race_final == race_dir and race_staging in message and race_dir in message
    assert os.path.dirname(race_staging) == os.path.dirname(race_dir) and re.fullmatch(r"seed0\.partial-{}-[0-9a-f]+".format(os.getpid()), os.path.basename(race_staging))
    assert sorted(os.listdir(race_staging)) == ["head_best.pt", "result.json"]  # the finished run is kept whole, for inspection
    assert synced == [["head_best.pt"], ["head_best.pt", "result.json"]]  # both files fsynced, head_best.pt first
    with open(os.path.join(race_staging, "result.json")) as f:
        assert json.load(f)["tag"] == "race"
    with open(os.path.join(race_dir, "result.json")) as f:
        assert f.read() == "{}" and os.listdir(race_dir) == ["result.json"]  # the other job's directory is as it left it
    assert partials() == [race_staging]
    # an EMPTY seed0 at the name when the publish comes (made by another process after the up-front check) is refused as well, though the files would move into it: the publish
    # claims the name with an mkdir first, so the finished run stays in its staging directory and the empty directory is still there, still empty
    empty_dir, empty_paths = os.path.join(seg_out, "empty", "jsrt_lung", "seed0"), []

    def publish_after_empty_dir(staging, final_dir):  # the empty directory lands between this run's last write and its publish
        empty_paths.append((staging, final_dir))
        os.makedirs(final_dir)
        return real_publish(staging, final_dir)

    with mock.patch.object(seg_probe, "evaluate", counting_evaluate), mock.patch.object(seg_probe, "publish", publish_after_empty_dir):
        message = rejects(RuntimeError, lambda: quietly(lambda: seg_probe.train(seg_probe.parse_args(seg_argv("empty", "random", "--iters", "3")))), "could not publish")
    [(empty_staging, empty_final)] = empty_paths
    assert empty_final == empty_dir and empty_staging in message and empty_dir in message
    assert sorted(os.listdir(empty_staging)) == ["head_best.pt", "result.json"]  # the finished run is kept whole, for inspection
    assert os.path.isdir(empty_dir) and not os.path.islink(empty_dir) and os.listdir(empty_dir) == []  # the other directory is still there, still empty
    assert partials() == sorted([race_staging, empty_staging])
    # publish itself, on hand-made staging directories under <seg_dir>/publish (where partials() does not look): the files move into the directory the claim made, head_best.pt and then
    # result.json, and the emptied staging directory goes. A directory is never renamed: Lustre fails a rename onto an existing empty directory with EEXIST and macOS replaces it, so a run on macOS
    # cannot see that failure, and this checks instead, on any filesystem, that the source of every rename is a file at the moment of the call
    pub_root, real_rename, renames = os.path.join(seg_dir, "publish"), os.rename, []

    def stage(name, *files):
        """(staging, final_dir) for pub_root/<name>/seed0: the staging directory exists beside it, holding `files` (each written with its own name), and final_dir does not exist."""
        final_dir = os.path.join(pub_root, name, "seed0")
        staging = final_dir + ".partial-1-0"
        os.makedirs(staging)
        for file in files:
            with open(os.path.join(staging, file), "w") as f:
                f.write(file)
        return staging, final_dir

    def spy_rename(src, dst):  # the source as it is when the call is made, before it moves
        renames.append((os.path.basename(src), os.path.isfile(src)))
        return real_rename(src, dst)

    def fail_on_result(src, dst):  # a filesystem that fails the second move
        if os.path.basename(src) == "result.json":
            raise OSError(errno.EIO, "Input/output error")
        return real_rename(src, dst)

    staging, final_dir = stage("whole", "head_best.pt", "result.json")
    with mock.patch.object(os, "rename", spy_rename):
        seg_probe.publish(staging, final_dir)
    assert renames == [("head_best.pt", True), ("result.json", True)]  # two renames, result.json last, each of a file: none of a directory
    assert sorted(os.listdir(final_dir)) == ["head_best.pt", "result.json"] and os.listdir(os.path.dirname(final_dir)) == ["seed0"]  # whole, and the emptied staging directory is gone
    # no head_best.pt to move: the claim is given back, and result.json does not go into the directory ahead of it
    staging, final_dir = stage("first", "result.json")
    message = rejects(RuntimeError, lambda: seg_probe.publish(staging, final_dir), "could not publish")
    assert staging in message and final_dir in message and "nothing moved into " + final_dir in message
    assert not os.path.lexists(final_dir) and os.listdir(staging) == ["result.json"]
    # result.json fails to move: head_best.pt stays in the final directory and result.json in staging, as they are, for the operator to finish by hand
    staging, final_dir = stage("second", "head_best.pt", "result.json")
    with mock.patch.object(os, "rename", fail_on_result):
        message = rejects(RuntimeError, lambda: seg_probe.publish(staging, final_dir), "could not publish")
    assert staging in message and final_dir in message and "head_best.pt moved into " + final_dir in message and "Input/output error" in message
    assert os.listdir(final_dir) == ["head_best.pt"] and os.listdir(staging) == ["result.json"]
    # refusals that cost nothing: a train list smaller than the batch (the loader drops its partial batch and would never yield), a size the patch grid does not divide
    rejects(ValueError, lambda: seg_probe.train(seg_probe.parse_args(seg_argv("small", "random", "--batch", "6"))), "at least --batch (6)")
    rejects(ValueError, lambda: seg_probe.train(seg_probe.parse_args(seg_argv("size", "random", "--size", "60"))), "patch size 16")
    assert os.listdir(os.path.join(seg_out, "small", "jsrt_lung")) == [] and os.listdir(os.path.join(seg_out, "size", "jsrt_lung")) == []  # not the run directory, not its staging directory

    # an exception before the publish removes the staging directory with whatever the run had written into it, and is raised as it was
    def failing_train_into(args, out_dir):
        with open(os.path.join(out_dir, "head_best.pt"), "w") as f:
            f.write("half a run")
        raise RuntimeError("boom")

    with mock.patch.object(seg_probe, "train_into", failing_train_into):
        rejects(RuntimeError, lambda: seg_probe.train(seg_probe.parse_args(seg_argv("boom"))), "boom")
    assert os.listdir(os.path.join(seg_out, "boom", "jsrt_lung")) == []

    # best-validation selection, through main(): validation losses 0.3 then 0.9, so iteration 2 is the best and the last iteration is worse. Scripted evaluate calls record the
    # val/test list they got and a head weight; the saved file and the test call must both hold the weight of iteration 2
    seg_weights = os.path.join(seg_dir, "backbone.pth")
    with open(seg_weights, "wb") as f:
        f.write(b"hashed, never loaded")
    recorded, loads = [], []

    def scripted_evaluate(pyramid, head, loader, device):
        recorded.append((list(loader.dataset.items), head.conv.weight.detach().clone()))
        return [(0.3, 11.0, 12.0), (0.9, 21.0, 22.0), (0.4, 55.0, 66.0)][len(recorded) - 1]  # a 4th call (a second test pass) is an IndexError

    def fake_load(model_name, weights):
        loads.append((model_name, weights))
        return real_load(model_name, None)

    with mock.patch.object(seg_probe, "evaluate", scripted_evaluate), mock.patch.object(seg_probe, "load_backbone", fake_load):
        _, summary, _ = quietly(lambda: seg_probe.main(seg_argv("sel", seg_weights)))
    sel_dir = os.path.join(seg_out, "sel", "jsrt_lung", "seed0")
    with open(os.path.join(sel_dir, "result.json")) as f:
        sel = json.load(f)
    assert sel["best_iter"] == 2 and sel["val_losses"] == [0.3, 0.9] and sel["best_val_loss"] == 0.3
    assert (sel["test_dice_pooled"], sel["test_dice_mean"]) == (55.0, 66.0)  # the test call's numbers, not a validation call's
    assert sel["weights"] == seg_weights and sel["weights_sha256"] == hashlib.sha256(b"hashed, never loaded").hexdigest() and loads == [("vit_base_dinov3", seg_weights)]
    (val_a, at_2), (val_b, at_4), (test_ids, at_test) = recorded
    assert val_a == val_b == ["s5", "s6"] and test_ids == ["s7", "s8", "s9"]
    assert not torch.equal(at_2, at_4)  # training went on past the best state, so the equalities below mean something
    saved = torch.load(os.path.join(sel_dir, "head_best.pt"), weights_only=True)
    assert torch.equal(saved["head"]["conv.weight"], at_2) and torch.equal(at_test, at_2)
    assert summary == "SEG sel jsrt_lung seed0: best iteration 2 (val loss 0.3000), test Dice mean 66.00, pooled 55.00\n", summary
    assert partials() == sorted([race_staging, empty_staging])  # the successful runs, the refused ones and the failed one left no staging directory; only the two failed publishes keep their runs

    # --runs N through main(): the seeds --seed, --seed + 1, ... one after another in one call, each in its own seed<S> directory and with its own SEG line
    multi_dir = os.path.join(seg_out, "multi", "jsrt_lung")
    _, multi_stdout, _ = quietly(lambda: seg_probe.main(seg_argv("multi", "random", "--seed", "3", "--runs", "2")))
    assert sorted(os.listdir(multi_dir)) == ["seed3", "seed4"] and partials() == sorted([race_staging, empty_staging])
    for run_seed in (3, 4):
        with open(os.path.join(multi_dir, "seed{}".format(run_seed), "result.json")) as f:
            assert json.load(f)["seed"] == run_seed
    assert [line.split(":")[0] for line in multi_stdout.splitlines()] == ["SEG multi jsrt_lung seed3", "SEG multi jsrt_lung seed4"], multi_stdout
    # every run's directory is checked before the first run starts: with seed4 there (empty), the job of seeds 3 and 4 is refused with no seed3 run and no staging directory made
    upfront_dir = os.path.join(seg_out, "upfront", "jsrt_lung")
    os.makedirs(os.path.join(upfront_dir, "seed4"))
    rejects(FileExistsError, lambda: quietly(lambda: seg_probe.main(seg_argv("upfront", "random", "--seed", "3", "--runs", "2"))), "refusing to overwrite " + os.path.join(upfront_dir, "seed4"))
    assert os.listdir(upfront_dir) == ["seed4"] and partials() == sorted([race_staging, empty_staging])
print("OK (seg_probe): evaluate gives the loss averaged by batch size and the pooled and per-image Dice x100 on stand-ins with a known answer, and hands the head back in its mode; "
      "a four-iteration run on a synthetic JSRT set writes result.json (every key, counts, null weights_sha256 for random, git sha, UTC times) and head_best.pt (trained part only, "
      "plain tensors, the state the test list saw); the optimizer steps at step_decay's rate (step_inc 1 for 4 iterations); the backbone is unchanged and gradient-free after training "
      "while the pyramid's layers and the head moved; the loss is given probabilities; the defaults are RayDINO's budget; the same seed repeats the run and another seed is a different run in its own seed<S> directory beside the first; a run directory that exists (a dangling symlink at its name too) is refused up front; a run is written (head_best.pt, then result.json, each fsynced) into a private seed<S>.partial-<pid>-<hex> directory beside it and published by an mkdir of seed<S> and a move of the two files into it (head_best.pt, then result.json; no directory is ever renamed, which Lustre refuses onto the empty one the mkdir made), which leaves no staging directory behind after a success or an exception, and which, when another job published first or made an empty seed<S>, fails naming both paths with the finished run kept in its staging directory and the other directory untouched (a move that fails on head_best.pt gives the name back, one that fails on result.json leaves head_best.pt in seed<S> and result.json in staging); validation losses 0.3 then 0.9 select iteration 2, and the test list is scored "
      "with iteration 2's weights, once; the weights file's sha256 is recorded; bad flags (--batch 1 among them; 2 is accepted), a train list under the batch and a size off the patch grid are refused; the train loader is shuffled by the seed and drops a partial last batch, val and test keep every image; --runs N runs the seeds --seed, --seed + 1, ... in one main() call, "
      "one seed<S> directory and one SEG line each, and refuses up front, before the first run starts, when any of those directories exists (seed4 there: seed3 is not run, no staging directory is made)")

print("ALL OK")
