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

# JSRT: images_png/<id>.png, the union of the two lung masks; "b" has its masks at half the image's resolution and valued 1, not 255; "align" is one bright block
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
    jsrt_list = os.path.join(jsrt_root, "list.txt")
    with open(jsrt_list, "w") as f:
        f.write("a\nb\n\nalign\n")
    jsrt_test = data.JSRTLung(jsrt_root, jsrt_list, size=224, mode="test", normalization="imagenet")
    assert len(jsrt_test) == 3  # the blank line is skipped
    img, msk = jsrt_test[0]
    assert img.shape == (3, 224, 224) and img.dtype == torch.float32 and msk.shape == (1, 224, 224) and msk.dtype == torch.float32
    assert set(msk.unique().tolist()) <= {0.0, 1.0} and msk.sum() > 0
    assert np.array_equal(msk[0].numpy(), cv2.resize(left | right, (224, 224), interpolation=cv2.INTER_NEAREST) // 255)  # both lungs, nothing else
    assert torch.allclose(img, plain_pipeline(np.repeat(lungs[..., None], 3, axis=-1), 224, "imagenet"), atol=1e-4)
    assert torch.equal(jsrt_test[1][1], msk)  # the half-resolution masks of "b" give the same mask
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
    assert len(jsrt_copy) == 3 and jsrt_copy[0][0].shape == (3, 224, 224)

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
      "NORMALIZATION equals the linear probe's constants; JSRTLung (union of both lungs, half-resolution masks, 16-bit PNG refused) and CandidPTX (non-square, -1, blank lines, "
      "no-comma line refused) return float32 (3, S, S) / (1, S, S) {0, 1} items in test and train mode, equal to a plain cv2 resize + normalize; normalization comes last; "
      "the train mask follows the geometric augmentation; --seed (torch's seed) fixes the augmentation draws")

print("ALL OK")
