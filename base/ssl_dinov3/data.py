"""SSL data for the Stage A DINOv3 continued-pretraining run: image lists, CXR augmentation, collate, data loader.
Upstream: DataAugmentationCXR mirrors DataAugmentationDINO (dinov3/data/augmentations.py); make_collate_fn mirrors
build_data_loader_from_cfg (dinov3/train/train.py); image_paths mirrors the dataset classes in base/dataloader.py.
"""
import csv
import functools
import math
import os
import random

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import v2

from dinov3.data.collate import collate_data_and_cast
from dinov3.data.masking import MaskingGenerator

DATASET_NAMES = ("VinDrCXR", "ChestXray14", "CheXpert", "MIMIC")
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # Ark's `--normalization imagenet` values
IMAGENET_STD = (0.229, 0.224, 0.225)
# ponytail: caps JPEG decode cost (MIMIC/CheXpert are ~2-3k px) while keeping >= 1024 px, so 448 crops at scale 0.25
# are still downsampled; raise it if the pixels matter more than loader throughput. Since 2026-10-18 it is the floor of
# decode_min_side(), which raises it with the crop sizes (2368 for 512 / 224 / 1024).
DECODE_MIN_SIDE = 1024
# Stage A's crop sizes (pilots, main, constlr) keep DECODE_MIN_SIDE exactly: those runs resume across this change (constlr is extended
# indefinitely) and the decode side is not locked on --resume. decode_min_side() alone would give them 1088.
STAGE_A_CROPS = (224, 96, 448)


def decode_min_side(crop_side, min_scale, ratio):
    """The decode side at which the smallest crop region still holds crop_side source pixels on its short side: a RandomResizedCrop of area
    fraction min_scale and aspect ratio in `ratio` has a short side of at least sqrt(min_scale x min ratio) of the image side, so
    ceil(crop_side / (sqrt(min_scale) x sqrt(min ratio))), rounded up to a multiple of 64, never below DECODE_MIN_SIDE. PIL's draft only
    shrinks JPEGs by 1/2, 1/4 or 1/8, to the smallest size that is still at least this on both sides, so the decoded side is >= the target
    (or the file's own size, if that is smaller)."""
    min_ratio = min(min(r, 1 / r) for r in ratio)  # short side / sqrt(area) at the range's ends: 3/4 for (3/4, 4/3)
    return max(DECODE_MIN_SIDE, math.ceil(crop_side / (math.sqrt(min_scale) * math.sqrt(min_ratio)) / 64) * 64)


def image_paths(name, data_dir, list_file):
    """Image paths of one dataset's list file, parsed as its class in base/dataloader.py does (labels dropped)."""
    if name not in DATASET_NAMES:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASET_NAMES}")
    paths = []
    with open(list_file, "r") as f:
        if name == "ChestXray14":
            line = True
            while line:
                line = f.readline()
                if line:
                    paths.append(os.path.join(data_dir, line.split()[0]))
        elif name == "VinDrCXR":
            line = f.readline().strip()
            while line:
                paths.append(os.path.join(data_dir, line.split()[0] + ".jpeg"))
                line = f.readline()
        else:  # CheXpert, MIMIC: csv with a header row, image path in column 0
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                # CheXpert: data_dir and the csv paths both carry CheXpert-v1.0; the loader joins them as is
                paths.append(os.path.join(data_dir, row[0]))
    return paths


def build_dataset(names, datasets_config, transform, check_exists=True, decode_side=DECODE_MIN_SIDE):
    """Concatenated TRAIN-list images of `names`; datasets_config is the parsed datasets_config.yaml dict. decode_side: the JPEG draft
    target (the transform's decode_min_side)."""
    paths = []
    for name in names:
        cfg = datasets_config[name]
        found = image_paths(name, cfg["data_dir"], cfg["train_list"])
        if check_exists:
            assert found, f"[ssl data] {name}: {cfg['train_list']} lists no images"
            assert os.path.isfile(found[0]), (
                f"[ssl data] {name}: first image {found[0]} does not exist (data_dir {cfg['data_dir']})"
            )
            print(f"[ssl data] {name}: {len(found)} images, first {found[0]} exists")
        paths += found
    if check_exists:
        print(f"[ssl data] total: {len(paths)} images")
    return SSLImageDataset(paths, transform, decode_side)


class SSLImageDataset(Dataset):
    """Grayscale image -> transform(img). The () target is what Meta's datasets return (target_transform
    `lambda _: ()`) and what collate_data_and_cast expects (samples_list[i][0] is the crop dict)."""

    def __init__(self, paths, transform, decode_side=DECODE_MIN_SIDE):
        self.paths = paths
        self.transform = transform
        self.decode_side = decode_side

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            img.draft("L", (self.decode_side, self.decode_side))  # JPEG: decode at 1/2, 1/4 or 1/8 size, >= decode_side; no-op for PNG
            img = img.convert("L")
        return self.transform(img), ()


class DataAugmentationCXR:
    """DataAugmentationDINO with gram_teacher_no_distortions=True, share_color_jitter=False,
    local_crops_subset_of_global_crops=False, teacher_no_color_jitter=False and the CXR values below.
    Takes a grayscale PIL image; returns the crop dict collate_data_and_cast expects (crops 3-channel, normalized).
    gram_teacher_crops_size=None drops the gram crops; 224 makes them the same size as the global crops.
    emit_gram_teacher_crops=False (--no_gram) leaves "gram_teacher_crops" out of the output and changes nothing else: the global crop is
    still cut at max(global, Gram size) and resized to the global size, same decode side, same RNG draws, so the student / teacher crops are
    bitwise those of the Gram run with the same sizes."""

    def __init__(
        self,
        global_crops_scale=(0.25, 1.0),  # CXR (RAD-DINO-style) value; DINOv3 default (0.32, 1.0)
        local_crops_scale=(0.04, 0.25),  # CXR (RAD-DINO-style) value; DINOv3 default (0.05, 0.32)
        local_crops_number=8,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=448,  # 2x high-res Gram teacher (DINOv3 Sec. 4.3); ssl_default_config None (no gram crops)
        horizontal_flips=True,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        emit_gram_teacher_crops=True,
    ):
        self.local_crops_number = local_crops_number
        self.emit_gram_teacher_crops = emit_gram_teacher_crops
        bicubic = v2.InterpolationMode.BICUBIC
        flip = 0.5 if horizontal_flips else 0.0

        # one max-size crop serves both the global crop and the gram-teacher crop
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size or 0)
        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(global_crop_max_size, scale=global_crops_scale, interpolation=bicubic),
                v2.RandomHorizontalFlip(p=flip),
            ]
        )
        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=bicubic),
                v2.RandomHorizontalFlip(p=flip),
            ]
        )
        # the JPEG decode side that keeps the smallest global / Gram crop and the smallest local crop from being upsampled (SSLImageDataset);
        # scale and aspect-ratio ranges read from the two RandomResizedCrops themselves
        global_rrc, local_rrc = self.geometric_augmentation_global.transforms[0], self.geometric_augmentation_local.transforms[0]
        self.decode_min_side = max(decode_min_side(global_crop_max_size, global_rrc.scale[0], global_rrc.ratio),
                                   decode_min_side(local_crops_size, local_rrc.scale[0], local_rrc.ratio))
        if (global_crops_size, local_crops_size, gram_teacher_crops_size) == STAGE_A_CROPS:
            self.decode_min_side = DECODE_MIN_SIDE

        # gram_teacher_no_distortions=True as the Gram-anchor config (ssl_default_config False): the gram crop gets no
        # distortions, so the global crop is resized before them (as upstream); with no gram crops, nothing to resize
        resize_global = nn.Identity()
        self.resize_gram_teacher = None
        if gram_teacher_crops_size is not None:
            resize_global = v2.Resize(global_crops_size, interpolation=bicubic)
            self.resize_gram_teacher = v2.Resize(gram_teacher_crops_size, interpolation=bicubic)

        # CXR (RAD-DINO-style) value; DINOv3 default ColorJitter(0.4, 0.4, 0.2, 0.1) + RandomGrayscale(p=0.2).
        # No saturation/hue/grayscale here: the images are already grayscale.
        color_jittering = v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8)
        # CXR (RAD-DINO-style) value; DINOv3 default sigma (0.1, 2.0). Upstream's GaussianBlur passes `1 - p` to
        # RandomApply (applies with prob 1-p); at p=0.5 both readings agree.
        blur = v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 1.0))], p=0.5)

        # grayscale PIL -> 1-channel float -> 3 equal channels -> ImageNet mean/std
        self.normalize = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.RGB(), v2.Normalize(mean=mean, std=std)]
        )

        # CXR (RAD-DINO-style) value; DINOv3 default: blur p=1.0 on global 1, blur p=0.1 + RandomSolarize(128, p=0.2)
        # on global 2 (nominal p, see the GaussianBlur note above). Local crops: blur p=0.5, same as DINOv3.
        self.global_transfo1 = v2.Compose([resize_global, color_jittering, blur, self.normalize])
        self.global_transfo2 = v2.Compose([resize_global, color_jittering, self.normalize])
        self.local_transfo = v2.Compose([color_jittering, blur, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops (same RNG call order as DataAugmentationDINO.__call__)
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)
        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)
        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # gram teacher crops: same crop as the global ones, no distortions
        if self.resize_gram_teacher is not None and self.emit_gram_teacher_crops:
            output["gram_teacher_crops"] = [
                self.normalize(self.resize_gram_teacher(im1_base)),
                self.normalize(self.resize_gram_teacher(im2_base)),
            ]

        output["local_crops"] = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["offsets"] = ()
        return output


def _collate_checked(samples, collate):
    out = collate(samples)
    # The vendored collate draws exactly int(N*p) patches per masked sample (MaskingGenerator.complete_mask_randomly),
    # so the masked-patch count equals `upperbound` every step: the static shape HPU lazy mode needs. If it ever
    # differs, the iBOT tensors change shape and the graph recompiles. (Meta's DINOv3 dropped DINOv2's explicit
    # padding to upperbound; this assertion is our guard that the count is static anyway.)
    n_masked = out["mask_indices_list"].shape[0]
    assert n_masked == out["upperbound"], (
        f"masked-patch count {n_masked} != upperbound {out['upperbound']}: iBOT tensor shapes would change "
        "from step to step and HPU lazy mode would recompile the graph"
    )
    return out


def make_collate_fn(
    global_crops_size=224,
    patch_size=16,
    mask_ratio_min_max=(0.1, 0.5),
    mask_sample_probability=0.5,
    dtype=torch.float32,
):
    """collate_data_and_cast + MaskingGenerator built as build_data_loader_from_cfg does (no multidistillation).
    Picklable (a partial of top-level functions), so DataLoader workers may use spawn."""
    img_size = global_crops_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )
    collate = functools.partial(
        collate_data_and_cast,
        mask_ratio_tuple=mask_ratio_min_max,
        mask_probability=mask_sample_probability,
        dtype=dtype,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        random_circular_shift=False,
        local_batch_size=None,
    )
    return functools.partial(_collate_checked, collate=collate)


def build_data_loader(dataset, batch_size_per_gpu, num_workers, rank, world_size, seed, pin_memory, collate_fn):
    """Returns (loader, sampler). DistributedSampler is used at world size 1 too; call set_epoch before each epoch."""
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True)
    generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
        generator=generator,
        persistent_workers=False,
    )
    return loader, sampler


def set_epoch(loader, sampler, epoch, seed, rank):
    sampler.set_epoch(epoch)
    # re-seeds the worker base seed per (epoch, rank) so augmentation/mask randomness is reproducible across
    # --resume and distinct per rank
    epoch_seed = seed + 1000 * epoch + rank
    loader.generator.manual_seed(epoch_seed)
    if loader.num_workers == 0:  # transforms and masks then run in this process, on its global RNGs
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
