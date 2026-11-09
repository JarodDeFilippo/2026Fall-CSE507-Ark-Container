"""Where each dataset's images and lists are, for B2's extractor and fit: base/datasets_config.yaml, the splits every arm uses.

B2's probe is the lab's embedding linear probe, which has no validation split (it picks its best epoch on the test set, as shipped): it trains on the FULL official train list of each
dataset (VinDrCXR: all 15,000 images of VinDrCXR_train_pe_global_one.txt, the 6-label global list the arms use) and is scored on the official test list. dataset_entry therefore
returns no val_list, so nothing can read one by accident (datasets_config.yaml gives VinDrCXR a val_list that is its test_list).
CANDIDPTX (CANDID-PTX pneumothorax, one label) is opt-in: probed only when --datasets names it, so the default stays the four and so does every four-dataset mean. Its images
and lists are the dense probe's (dense/data.py), its train and test lists; its val list is not used, as the probe has no validation split.
"""
import os

from utils import get_config  # the container's base/utils.py

from . import dataloader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(BASE)
DATASETS = ("ChestXray14", "CheXpert", "MIMIC", "VinDrCXR")  # the four Ark names (the first four of ssl_dinov3.data.DATASET_NAMES): the default of --datasets
NAMES = DATASETS + ("CANDIDPTX",)  # what --datasets accepts


def dataset_class(name):
    """The class that parses `name`'s lists and reads its images: the official Finetuning class for the four, dense/data.py's CandidPTXImages for CANDIDPTX."""
    if name == "CANDIDPTX":
        from dense.data import CandidPTXImages  # imported here: the four datasets never load the dense probe's modules
        return CandidPTXImages
    return getattr(dataloader, name)


def dataset_entry(name, config=None):
    """datasets_config.yaml's entry for `name`, with train_list and test_list absolute (the config's are relative to base/) and without the val_list.
    CANDIDPTX's is the dense probe's registry entry (dense/data.py): its image root, train and test lists, and one class."""
    if name == "CANDIDPTX":
        from dense.data import DATASETS as DENSE
        spec = DENSE["candidptx"]
        return {"data_dir": spec["root"], "train_list": spec["lists"]["train"], "test_list": spec["lists"]["test"], "diseases": ["Pneumothorax"]}
    config = get_config(os.path.join(BASE, "datasets_config.yaml")) if config is None else config
    entry = dict(config[name])
    entry.pop("val_list", None)
    for key in ("train_list", "test_list"):
        entry[key] = os.path.normpath(os.path.join(BASE, entry[key]))
    return entry
