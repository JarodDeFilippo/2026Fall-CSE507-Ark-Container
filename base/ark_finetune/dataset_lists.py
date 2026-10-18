"""Where each dataset's images and lists are, for B2's extractor and fit: base/datasets_config.yaml, the splits every arm uses.

B2's probe is the lab's embedding linear probe, which has no validation split (it picks its best epoch on the test set, as shipped): it trains on the FULL official train list of each
dataset (VinDrCXR: all 15,000 images of VinDrCXR_train_pe_global_one.txt, the 6-label global list the arms use) and is scored on the official test list. dataset_entry therefore
returns no val_list, so nothing can read one by accident (datasets_config.yaml gives VinDrCXR a val_list that is its test_list).
"""
import os

from utils import get_config  # the container's base/utils.py

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(BASE)
DATASETS = ("ChestXray14", "CheXpert", "MIMIC", "VinDrCXR")  # the four Ark names (ssl_dinov3.data.DATASET_NAMES)


def dataset_entry(name, config=None):
    """datasets_config.yaml's entry for `name`, with train_list and test_list absolute (the config's are relative to base/) and without the val_list."""
    config = get_config(os.path.join(BASE, "datasets_config.yaml")) if config is None else config
    entry = dict(config[name])
    entry.pop("val_list", None)
    for key in ("train_list", "test_list"):
        entry[key] = os.path.normpath(os.path.join(BASE, entry[key]))
    return entry
