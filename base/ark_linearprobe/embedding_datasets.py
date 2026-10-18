"""Ported additions to the lab's embedding linear probe (main_linearprobing.py): dataset modules for ChestXray14, CheXpert, MIMIC and VinDrCXR over B2's cached embeddings.

The shipped script has dataset modules for ChestDR and VinDrCXRAll only, whose embeddings are one .npy file per image. These stand in for them, with the same interface
(train_set / test_set, train_dataloader() / test_dataloader(), the same DataLoader settings) and three named ports:
  * in-memory arrays: the shipped EmbeddingDataset np.loads a file per item; InMemoryEmbeddingDataset serves rows of one array that holds a whole split (float16, as cached; the probe's
    train and test loops convert every batch with .double(), which is exact for float16). The arrays are written by ark_finetune/extract_embeddings.py, one center-crop embedding per image.
  * labels from the official Finetuning dataset classes: a list is parsed by the official class (ark_finetune/dataloader.py, vendored), so img_list and img_label are exactly what the
    Finetuning loop trained on, and a row's label is what the official __getitem__ returns (ark_finetune/embedding_mode.py: plain_label, and for CheXpert the per-access draws of its
    uncertain labels). Train sets use the official Finetuning default uncertain_label LSR-Ones (CheXpert draws at every access; MIMIC once, in the official constructor, i.e. when the
    module is built); test sets are built as the official Finetuning branches build them (MIMIC with uncertain_label="Ones"). The cached paths must equal img_list (the alignment guard).
  * CheXpert is scored on its 5 competition findings, as the lab's Finetuning branch does (its test_diseases; the classes are named by datasets_config.yaml): the module's test_columns
    makes the ported test() in main_linearprobing.py score just those, while the model still predicts all 14.
Scoring needs a positive and a negative example: of the columns a dataset is scored on (CheXpert's 5 findings, else every column), the module keeps those that have both among the test images it
serves, and records the others in dropped_columns (the shipped meanAUC skips an all-negative column silently, which shifts the per-class scores against the class names, and gives an all-positive one no defined AUC: sklearn raises or returns NaN, by version; a smoke cache of the first N images can make any column single-valued). If no column
is left it refuses to be built. A smoke cache (meta.json max_images = N) holds the first N entries of each list: the modules then keep just those.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ark_finetune import dataloader as official
from ark_finetune.dataset_lists import dataset_entry
from ark_finetune.embedding_mode import check_aligned, chexpert_label, feature_dim, plain_label, read_meta, read_paths, split_files


def scored_columns(name):
    """The columns scored on test, or None for all of them: the classes that `test_diseases_name` names in datasets_config.yaml, located in `diseases`.
    CheXpert: its 5 competition findings (the Finetuning branch's test_diseases). MIMIC names all 14 in column order, so None, as its Finetuning branch."""
    entry = dataset_entry(name)
    if "test_diseases_name" not in entry:
        return None
    columns = [entry["diseases"].index(disease) for disease in entry["test_diseases_name"]]
    return None if columns == list(range(len(entry["diseases"]))) else columns


def test_labels(name, images_path, test_file, keep=None):
    """The test labels [N, C] that the probe serves for dataset `name`: the official class's parse of the test list (MIMIC: uncertain_label="Ones"), the first `keep` entries of a smoke cache."""
    parsed = getattr(official, name)(images_path=images_path, file_path=test_file, augment=None, **EMBEDDING_MODULES[name].TEST_KWARGS)
    return np.array(parsed.img_label[:keep], dtype=np.float64)


def evaluable_columns(name, labels):
    """(columns, dropped) for the test labels [N, C] of dataset `name`. The columns the probe scores start from the dataset's own selection (scored_columns: CheXpert's 5 findings, else every
    column) and keep those with both a positive and a negative example in `labels`; `columns` is None when that is every column of the dataset, as the shipped script scores it. `dropped` lists
    the others as {"column": c, "reason": "all positive" | "all negative" | "no test images"}. Raises ValueError if no column is left: there would be nothing to score."""
    if labels.ndim != 2 or len(labels) == 0:
        raise ValueError("{}: no test images to score".format(name))
    selected = scored_columns(name)
    keep, dropped = [], []
    for c in (range(labels.shape[1]) if selected is None else selected):
        positive, negative = bool((labels[:, c] != 0).any()), bool((labels[:, c] == 0).any())
        if positive and negative:
            keep.append(c)
        else:
            dropped.append({"column": c, "reason": "all positive" if positive else ("all negative" if negative else "no test images")})
    if not keep:
        raise ValueError("{}: none of the {} columns it is scored on has both a positive and a negative example among the {} test images: there is nothing to score ({})".format(
            name, len(keep) + len(dropped), len(labels), ", ".join("column {} {}".format(d["column"], d["reason"]) for d in dropped)))
    return (None if keep == list(range(labels.shape[1])) else keep), dropped


class InMemoryEmbeddingDataset(Dataset):
    """Port: the shipped EmbeddingDataset(data, labels) np.loads one .npy per image at every item; this holds a split's embeddings in one array. A sample is
    (embedding row, label), the label from the official dataset class instance `parsed` through `label_of`."""

    def __init__(self, embeddings, parsed, label_of):
        self.embeddings = embeddings
        self.parsed = parsed
        self.label_of = label_of

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, item):
        return torch.from_numpy(self.embeddings[item]), self.label_of(self.parsed, item)


class EmbeddingDataloaderModule(Dataset):
    """The shipped modules' interface for one of the four datasets: subclasses name the official class (NAME), how its test set is built (TEST_KWARGS) and how a label is read (label_of)."""
    NAME = None
    TEST_KWARGS = {}
    label_of = staticmethod(plain_label)

    def __init__(self, emb_path, train_file, test_file, images_path, few_shot=-1, batch_size=512, num_workers=4):
        super().__init__()
        if few_shot > 0:
            raise ValueError("--few_shot subsamples the lists with a shuffle that the cached embeddings do not follow: not supported")
        self.batch_size = batch_size
        self.num_workers = num_workers
        meta = read_meta(emb_path)
        keep, dim = meta.get("max_images"), feature_dim(meta)  # the width the extractor wrote: 1536 for a ViT-B, 2048 for a ViT-L
        self.train_set = self._split(emb_path, "train", images_path, train_file, keep, dim)
        self.test_set = self._split(emb_path, "test", images_path, test_file, keep, dim, **self.TEST_KWARGS)
        self.test_columns, self.dropped_columns = evaluable_columns(self.NAME, np.array(self.test_set.parsed.img_label, dtype=np.float64))  # raises if no column can be scored
        print('#train: ', len(self.train_set))
        print('#test:  ', len(self.test_set))

    def _split(self, emb_path, split, images_path, file_path, keep, dim, **kwargs):
        parsed = getattr(official, self.NAME)(images_path=images_path, file_path=file_path, augment=None, **kwargs)  # the official parse; no transform: no image is read
        if keep is not None:
            parsed.img_list, parsed.img_label = parsed.img_list[:keep], parsed.img_label[:keep]
        npy, paths = split_files(emb_path, split)
        check_aligned(read_paths(paths), parsed.img_list, "{} {}".format(self.NAME, split))
        embeddings = np.load(npy)
        if embeddings.dtype != np.float16 or embeddings.shape != (len(parsed.img_list), dim):
            raise ValueError("{}: {} {} (float16 [{}, {}] expected)".format(npy, embeddings.dtype, list(embeddings.shape), len(parsed.img_list), dim))
        return InMemoryEmbeddingDataset(embeddings, parsed, self.label_of)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)


class ChestXray14_DataloaderModule(EmbeddingDataloaderModule):
    NAME = "ChestXray14"


class CheXpert_DataloaderModule(EmbeddingDataloaderModule):
    NAME = "CheXpert"
    label_of = staticmethod(chexpert_label)


class MIMIC_DataloaderModule(EmbeddingDataloaderModule):
    NAME = "MIMIC"
    TEST_KWARGS = {"uncertain_label": "Ones"}  # as the official Finetuning branch builds the MIMIC test set


class VinDrCXR_DataloaderModule(EmbeddingDataloaderModule):
    NAME = "VinDrCXR"  # the 6-label global lists the arms use (VinDrCXR_{train,test}_pe_global_one.txt)


EMBEDDING_MODULES = {module.NAME: module for module in (ChestXray14_DataloaderModule, CheXpert_DataloaderModule, MIMIC_DataloaderModule, VinDrCXR_DataloaderModule)}
