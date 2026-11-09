"""Segmentation datasets for the dense probe on a frozen ViT: CANDID-PTX pneumothorax (DICOM + RLE) and JSRT lungs (PNG), read in place
from the lab's lists. An item is (image float32 [3, size, size], mask float32 [1, size, size] with values in {0, 1})."""
import os
import random

import albumentations as A
import cv2
import numpy as np
import pydicom as dicom
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


# copied from Foundation_X+/datasets_medical.py:1241-1250
def dicom_to_uint8(path):
    # CANDID-PTX DICOMs are uint16 with 10-14 bits stored. PIL's uint16 -> RGB conversion clips every
    # value >= 255 to white, so rescale each image to 0-255 first (per-image min-max, the formula the
    # CANDID-PTX segmentation loader already used).
    # ponytail: MONOCHROME1 images are not inverted (none in a 20-image test sample); add if a full scan finds any.
    a = dicom.dcmread(path).pixel_array
    lo, hi = a.min(), a.max()
    if hi == lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return ((a - lo) * (1 / (hi - lo) * 255)).astype('uint8')


def has_mask(rle):
    """False for the RLE of an image with no mask, "-1" or blank (the CANDID-PTX lists write " -1", with a leading space), True otherwise."""
    return rle.strip() not in ("", "-1")


def rle2mask(rle, width, height):
    """uint8 mask [height, width] in {0, 1} from a SIIM-style RLE: "start length start length ...", each start counted from the end of the
    previous run, pixels numbered down the columns. width is the number of columns and height the number of rows; "-1" or blank is no mask.
    Foundation X+'s rle2mask (datasets_medical.py:1977-1991) decodes the same way, but its caller passes (rows, columns), which transposes
    a non-square image."""
    mask = np.zeros(width * height, dtype=np.uint8)
    if has_mask(rle):
        runs = [int(x) for x in rle.split()]
        position = 0
        for start, length in zip(runs[0::2], runs[1::2]):
            position += start
            mask[position:position + length] = 1
            position += length
    return np.ascontiguousarray(mask.reshape(width, height).T)


# mean and std per --normalization, the constants of the linear probe (ark_finetune/dataloader.py:22-65); "none" is x / 255
NORMALIZATION = {"imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 "chestx-ray": ((0.5056,) * 3, (0.252,) * 3),
                 "raddino": ((0.5307,) * 3, (0.2583,) * 3),
                 "none": None}


def build_transforms(mode, normalization, size):
    """Resize; for train also Demo 2's ShiftScaleRotate and RandomBrightnessContrast, on the uint8 image; then normalization; then tensors.
    Normalization comes last, which Foundation X's loaders do not do: on an already normalized float image RandomBrightnessContrast
    clips everything into [0, 1]. The mask rides along as mask= (nearest interpolation), so it follows every geometric step."""
    if mode not in ("train", "val", "test"):
        raise ValueError("mode must be train, val or test, got {!r}".format(mode))
    steps = [A.Resize(size, size, interpolation=cv2.INTER_AREA)]
    if mode == "train":
        steps += [A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5), A.RandomBrightnessContrast(p=0.5)]
    stats = NORMALIZATION[normalization]
    steps.append(A.ToFloat(max_value=255) if stats is None else A.Normalize(mean=stats[0], std=stats[1]))
    return A.Compose(steps + [ToTensorV2()])


class _SegDataset(Dataset):
    """One item per non-blank line of list_path. A subclass says how to parse a line (parse) and how to read an item (read: uint8 image
    [H, W, 3] and uint8 mask [H, W] in {0, 1}); the transforms do the rest. mode is train (augmented), val or test."""

    def __init__(self, root, list_path, size=224, mode="train", normalization="imagenet"):
        self.root = root
        self.transform = build_transforms(mode, normalization, size)
        self._rng_seed = None
        with open(list_path) as f:
            self.items = [self.parse(line.strip()) for line in f if line.strip()]

    @staticmethod
    def parse(line):
        return line

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        # albumentations 2 draws from a random state of its own, which every DataLoader worker would inherit unchanged. Follow torch's seed
        # instead (the loader's base seed plus the worker id in a worker, the manual seed otherwise), so --seed reaches the augmentation.
        seed = torch.initial_seed()
        if seed != self._rng_seed:
            self.transform.set_random_state(np.random.default_rng(seed), random.Random(seed))
            self._rng_seed = seed
        image, mask = self.read(self.items[i])
        out = self.transform(image=image, mask=mask)
        return out["image"], (out["mask"] > 0).float().unsqueeze(0)


class CandidPTX(_SegDataset):
    """CANDID-PTX pneumothorax. List lines are "uid,rle": the image is the DICOM root/uid, the mask is the RLE ("-1": no pneumothorax)."""

    @staticmethod
    def parse(line):
        uid, comma, rle = line.partition(",")
        if not comma:
            raise ValueError("expected 'uid,rle', got {!r}".format(line))
        return uid, rle

    def read(self, item):
        uid, rle = item
        # ponytail: every access decodes its DICOM again; cache the resized images if the loader ever starves the GPU
        a = dicom_to_uint8(os.path.join(self.root, uid))
        return np.repeat(a[..., None], 3, axis=-1), rle2mask(rle, width=a.shape[1], height=a.shape[0])


class CandidPTXImages(Dataset):
    """CANDID-PTX as one binary label per image, for the linear probe (ark_finetune, --datasets CANDIDPTX). The list is read by CandidPTX, so the images are the dense
    probe's; an image with several rows is one item, at its first row, labelled [1] (pneumothorax) when any of its rows has a mask, [0] when none has. The interface is
    the official Finetuning classes' (ark_finetune/dataloader.py): img_list (root/uid), img_label, augment (a torchvision transform of a PIL image, or None), items (image, label)."""

    def __init__(self, images_path, file_path, augment):
        positive = {}
        for uid, rle in CandidPTX(images_path, file_path, mode="test").items:
            positive[uid] = positive.get(uid, False) or has_mask(rle)
        self.img_list = [os.path.join(images_path, uid) for uid in positive]
        self.img_label = [[int(p)] for p in positive.values()]
        self.augment = augment

    def __getitem__(self, index):
        image = Image.fromarray(dicom_to_uint8(self.img_list[index])).convert("RGB")  # the dense probe's DICOM read, gray on three channels as the official classes open an image
        return (image if self.augment is None else self.augment(image)), torch.FloatTensor(self.img_label[index])

    def __len__(self):
        return len(self.img_list)


class JSRTLung(_SegDataset):
    """JSRT lungs. List lines are ids: the image is root/images_png/<id>.png, the mask is the union of root/masks/{left,right}_lung_png/<id>.png."""

    def read(self, id_):
        path = os.path.join(self.root, "images_png", id_ + ".png")
        with Image.open(path) as im:
            if im.mode in ("P", "PA"):  # a palette PNG: apply the palette, as cv2's color read does
                im = im.convert("RGB")
            if im.mode not in ("L", "RGB"):  # a 16-bit PNG would convert with every value >= 255 clipped to white
                raise ValueError("{}: {} PNG, expected 8-bit L, RGB or P (palette)".format(path, im.mode))
            image = np.asarray(im.convert("L"))
        mask = np.zeros(image.shape, dtype=np.uint8)
        for side in ("left", "right"):
            with Image.open(os.path.join(self.root, "masks", side + "_lung_png", id_ + ".png")) as im:
                part = (np.asarray(im.convert("L")) > 0).astype(np.uint8)
            if part.shape != image.shape:  # the lung masks need not share the image's resolution
                part = cv2.resize(part, image.shape[::-1], interpolation=cv2.INTER_NEAREST)
            mask |= part
        return np.repeat(image[..., None], 3, axis=-1), mask


LISTS = "/data/jliang12/nuislam/data_files_splits"
DATASETS = {
    "candidptx": dict(cls=CandidPTX, root="/data/jliang12/shared/dataset/CANDID-PTX/dataset",
                      lists={"train": LISTS + "/candid_ptx/train.txt", "val": LISTS + "/candid_ptx/valid.txt", "test": LISTS + "/candid_ptx/test.txt"}),
    "jsrt_lung": dict(cls=JSRTLung, root="/data/jliang12/jpang12/dataset/JSRT/All247images",
                      lists={"train": LISTS + "/jsrt/train.txt", "val": LISTS + "/jsrt/val.txt", "test": LISTS + "/jsrt/test.txt"}),
}


def make_dataset(name, split, size, normalization, root=None, list_path=None):
    """The dataset `name` at `split` (train, val or test). root and list_path replace the lab's paths (for tests)."""
    spec = DATASETS[name]
    return spec["cls"](spec["root"] if root is None else root, spec["lists"][split] if list_path is None else list_path,
                       size=size, mode=split, normalization=normalization)
