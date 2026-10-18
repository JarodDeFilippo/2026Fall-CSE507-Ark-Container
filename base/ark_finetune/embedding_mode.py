"""The embedding cache of B2 (CSE 494 IP, Step VIII), shared by the extractor that writes it and the dataset modules of the lab's embedding linear probe (ark_linearprobe/) that read it,
and the label rules of the official Finetuning dataset classes that the cached rows are served with.
(The file kept its name from the Finetuning-loop embedding mode it first served; the fit now runs the lab's Linearprobing script instead, see ark_linearprobe/NOTICE.)

The cache is one directory per backbone tag and dataset: {train,test}.npy (float16, [N, 1536 (ViT-B) or 2048 (ViT-L)]: ONE center-crop embedding per image, test included), {split}_paths.txt (the dataset's
img_list, one path per line) and meta.json. There is no validation split: the probe has none (it picks its best epoch on test). A dataset module refuses to serve embeddings whose
path list is not exactly its own img_list: row i of the cache is only meaningful for img_list[i].

The labels are the official Finetuning dataset classes' (ark_finetune/dataloader.py): the module parses a list with the official class, and a row's label is what the official
__getitem__ returns for it. plain_label is that for ChestXray14, MIMIC and VinDrCXR (MIMIC drew its soft labels once, in the official constructor); chexpert_label is the label block of
the official CheXpert.__getitem__ copied line by line, because the official method cannot run without opening the image: its uncertain labels are drawn at every access.
Both are pinned against the official classes by test_ark_finetune.py.
"""
import json
import os
import random

import torch

EMBEDDING_DIM = 1536  # CLS token ++ mean patch token of a ViT-B (768 + 768), the vector Ark's projector receives: the ViT-B default, for callers that do not know the model (see embedding_dim)
EMBEDDING_DIMS = {"vit_large_raydino": 2048}  # a ViT-L (1024 + 1024); every other --model name here is a ViT-B
SPLITS = ("train", "test")


def embedding_dim(model_name):
    """The width of the embedding a backbone gives, by --model name: 2048 for vit_large_raydino (a ViT-L), else the ViT-B width EMBEDDING_DIM (1536)."""
    return EMBEDDING_DIMS.get(model_name, EMBEDDING_DIM)


def feature_dim(meta):
    """The width of the embeddings a cache holds, from its meta.json (extract_embeddings writes it from the backbone it built): a positive int. A meta without it is a ViT-B cache (EMBEDDING_DIM)."""
    dim = meta.get("feature_dim", EMBEDDING_DIM)
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
        raise ValueError("meta.json feature_dim {!r} is not a positive integer".format(dim))
    return dim


def split_files(embeddings_dir, split):
    """(the split's .npy, its paths file) in a dataset's cache directory."""
    return os.path.join(embeddings_dir, split + ".npy"), os.path.join(embeddings_dir, split + "_paths.txt")


def write_paths(path, paths):
    with open(path, "x", newline="\n") as f:  # "x": never overwrites
        f.write("".join(p + "\n" for p in paths))


def read_paths(path):
    with open(path, newline="\n") as f:
        text = f.read()
    if text and not text.endswith("\n"):
        raise ValueError("{} does not end with a newline: a partly written paths file".format(path))
    return text.split("\n")[:-1]


def read_meta(embeddings_dir):
    with open(os.path.join(embeddings_dir, "meta.json")) as f:
        return json.load(f)


def check_aligned(cached, img_list, where):
    """The alignment guard: the cached path list is exactly the dataset's img_list, entry by entry. Row i of the cache is only meaningful for img_list[i]."""
    if cached != img_list:
        first = next((i for i, (a, b) in enumerate(zip(cached, img_list)) if a != b), min(len(cached), len(img_list)))
        raise ValueError("{}: the cached image paths are not the dataset's img_list (cache {} paths, list {}; first difference at index {}: {} vs {}); "
                         "the embeddings were extracted from other lists or another data_dir".format(
                             where, len(cached), len(img_list), first, cached[first:first + 1] or "<end>", img_list[first:first + 1] or "<end>"))


def plain_label(parsed, index):
    """ChestXray14, VinDrCXR and MIMIC: the official __getitem__ takes the list's label as it is. `parsed` is the official dataset instance."""
    return torch.FloatTensor(parsed.img_label[index])


def chexpert_label(parsed, index):
    """CheXpert: the label block of the official __getitem__, line by line: an uncertain label (-1) is drawn at every access, LSR-Ones from random.uniform(0.55, 0.85)."""
    label = []
    for l in parsed.img_label[index]:
      if l == -1:
        if parsed.uncertain_label == "Ones":
          label.append(1)
        elif parsed.uncertain_label == "Zeros":
          label.append(0)
        elif parsed.uncertain_label == "LSR-Ones":
          label.append(random.uniform(0.55, 0.85))
        elif parsed.uncertain_label == "LSR-Zeros":
          label.append(random.uniform(0, 0.3))
      else:
        label.append(l)
    return torch.FloatTensor(label)
