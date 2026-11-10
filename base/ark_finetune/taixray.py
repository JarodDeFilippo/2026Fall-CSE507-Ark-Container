"""TAIX-Ray (Truhn et al., the TAIX-Ray Data Descriptor) as eight binary findings per image, for the linear probe (--datasets TAIXRAY, opt-in like CANDIDPTX).

The lists are the label files that base/dataset/TAIXRay/taix_unpack_eval.sh writes beside png512/ (on Sol /scratch/jdefili4/datasets-new/TAIX-Ray): taix_train_labels.csv and taix_test_labels.csv, a header row
Path, UID, Split, Fold, PatientID, PhysicianID, StudyDate, Age, Sex, HeartSize, PulmonaryCongestion, PleuralEffusion_Right, PleuralEffusion_Left, PulmonaryOpacities_Right, PulmonaryOpacities_Left, Atelectasis_Right, Atelectasis_Left
and one row per image (a null is an empty field; Path is <UID>.png, relative to the images directory). taix_val_labels.csv is never read: the probe has no validation split. The columns are found by name; the eight outputs follow
FINDINGS, whose Left comes before Right (the file lists Right first).
A label file is read for the split the caller asks for (`split`, "train" or "test", given where the splits are built and never read off the file name) and refused if its Split column says another: the train and test files
swapped or misnamed would otherwise train the probe on the test images and score it on the train ones, with every other check passing.
HeartSize is graded 0 normal, 1 borderline enlarged, 2 enlarged, 3 massively enlarged; the other seven 0 none, 1 questionable, 2 mild, 3 moderate, 4 severe. The task is the paper's binary one: a finding is present (label 1) when
its grade is above 0, so a questionable finding and a borderline heart count as present. The raw grades are kept in `grades` (int8 [N, 8], FINDINGS' order) and the extractor caches them beside the embeddings, for an ordinal
probe later; the fit does not read them.
The images are 16-bit grey PNGs holding 12-bit content, read with the SSL trainer's own rule (grey_read), so the evaluation sees what the pretraining saw.
The interface is the official Finetuning classes' (ark_finetune/dataloader.py): img_list (images directory / Path), img_label, augment (a torchvision transform of a PIL image, or None), items (image, label).
"""
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

PATH, SPLIT = "Path", "Split"
GRADE_MAX = {"HeartSize": 3, "PulmonaryCongestion": 4, "PleuralEffusion_Left": 4, "PleuralEffusion_Right": 4, "PulmonaryOpacities_Left": 4, "PulmonaryOpacities_Right": 4, "Atelectasis_Left": 4, "Atelectasis_Right": 4}
FINDINGS = tuple(GRADE_MAX)  # the eight columns, in the order of the probe's outputs and of `grades`; each one's highest grade is GRADE_MAX's
SPLIT_ROWS = {"train": 137593, "val": 34860, "test": 42928}  # images per official split (Table 2 of the data descriptor, as taix_unpack_eval.sh's PAPER); patients do not cross splits
PROBE_SPLITS = ("train", "test")  # the splits the probe reads: val is never asked for
GRADE_TEXT = {text: grade for grade in range(5) for text in (str(grade), "{}.0".format(grade))}  # a grade as the label file holds it: 2, or 2.0 if the source column was a float one


def mount_hint(images_path):
    """What to add to an error about a missing TAIX-Ray label file, image or images directory: what writes the files, and the bind that mounts them. `images_path` is the images directory the caller was given (in a real run
    the data_dir of base/datasets_config.yaml's TAIXRay entry); the folder to mount is the one that holds its parent, the datasets-new folder where taix_unpack.sh and taix_unpack_eval.sh put the TAIX-Ray folder. It is
    worked out from the path in hand and not written out, so it follows the yaml (one user's scratch there is not another's)."""
    mount = os.path.dirname(os.path.dirname(os.path.normpath(images_path)))
    return 'taix_unpack.sh and taix_unpack_eval.sh (base/dataset/TAIXRay/) write them, and the container must mount {0}: DATA_BIND="/data:/data:ro,{0}:{0}:ro" (see the README)'.format(mount)


def grey_read(path):
    """One image as the SSL trainer reads it: ssl_dinov3/data.py SSLImageDataset.__getitem__ (lines 103-114), called as it is and not copied, so the two cannot drift apart. A 16-bit (or int32 / float) grey image is
    stretched per image, its minimum to 0 and its maximum to 255 (a flat one is all 0), where PIL's convert("L") would clip every value of 255 or more to white; any other mode goes through convert("L"). A PIL "L" image."""
    from ssl_dinov3.data import SSLImageDataset  # imported here: the fit, which reads no image, and the other datasets never load the trainer's modules
    return SSLImageDataset([path], lambda image: image)[0][0]


class TAIXRayImages(Dataset):
    """One item per row of a label file: (image on three grey channels, the 8 binary findings as a float32 tensor). A transform means the images will be read, so they are checked first, all of them, with one listing of the
    images directory; augment=None (the fit's parse, which reads no image) does not touch that directory. `split` ("train" or "test"; keyword only, no default, so no caller can leave it out) is the split being built.
    Raises FileNotFoundError for a missing label file, a missing images directory or a listed image that is not there, and ValueError for a split that is not "train" or "test", a header without the expected columns, a row
    of the wrong width, a Path that is not a bare file name, a grade that is not a whole number from 0 to the finding's highest, rows of more than one split, no rows, or a file whose Split is not `split` (the message
    names the file, both splits and the line of the first such row): all before any image is opened. A row count other than the split's official one (SPLIT_ROWS) is a WARNING on stderr, naming both numbers."""

    def __init__(self, images_path, file_path, augment, *, split):
        if split not in PROBE_SPLITS:
            raise ValueError("TAIXRAY split {!r}: the probe reads {} only, never the val split".format(split, " and ".join(PROBE_SPLITS)))
        if not os.path.isfile(file_path):
            raise FileNotFoundError("TAIXRAY label file {} not found: {}".format(file_path, mount_hint(images_path)))
        names, splits, grades, other = [], set(), [], None  # other: (line, Split) of the first row whose Split is not `split`
        with open(file_path, newline="") as f:
            rows = csv.reader(f)
            header = next(rows, [])
            missing = [c for c in (PATH, SPLIT) + FINDINGS if c not in header]
            if missing:
                raise ValueError("{}: the header lacks the columns {} (it has {})".format(file_path, missing, header))
            at = {c: header.index(c) for c in (PATH, SPLIT) + FINDINGS}
            for row in rows:
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError("{} line {}: {} fields, the header has {}".format(file_path, rows.line_num, len(row), len(header)))
                name = row[at[PATH]]
                if not name or "/" in name:
                    raise ValueError("{} line {}: Path {!r} is not a bare file name".format(file_path, rows.line_num, name))
                cells = []
                for finding in FINDINGS:
                    grade = GRADE_TEXT.get(row[at[finding]])
                    if grade is None or grade > GRADE_MAX[finding]:
                        raise ValueError("{} line {}: {} grade {!r} is not a whole number from 0 to {}".format(file_path, rows.line_num, finding, row[at[finding]], GRADE_MAX[finding]))
                    cells.append(grade)
                names.append(name)
                splits.add(row[at[SPLIT]])
                if other is None and row[at[SPLIT]] != split:
                    other = (rows.line_num, row[at[SPLIT]])
                grades.append(cells)
        if not names:
            raise ValueError("{}: no rows".format(file_path))
        if len(splits) != 1 or next(iter(splits)) not in SPLIT_ROWS:
            raise ValueError("{}: its Split column holds {}, but a label file holds one of {} alone".format(file_path, sorted(splits), "/".join(SPLIT_ROWS)))
        if other:  # one split, and not the one asked for: another split's file in this one's place (train and test swapped or misnamed, or the val file)
            raise ValueError("{} line {}: its Split is {!r}, but the {!r} split was asked for: this is another split's label file (are the train and test files swapped or misnamed?)".format(file_path, other[0], other[1], split))
        if len(names) != SPLIT_ROWS[split]:
            print("TAIXRAY: WARNING {}: expected {} rows for the {} split (the data descriptor's Table 2), found {}".format(file_path, SPLIT_ROWS[split], split, len(names)), file=sys.stderr)
        self.grades = np.array(grades, dtype=np.int8)
        self.img_list = [os.path.join(images_path, name) for name in names]
        self.img_label = (self.grades > 0).astype(int).tolist()
        self.augment = augment
        if augment is not None:
            self._check_images(images_path, names)

    @staticmethod
    def _check_images(images_path, names):
        try:
            present = set(os.listdir(images_path))
        except OSError as error:
            raise FileNotFoundError("TAIXRAY images directory {} cannot be listed ({}): {}".format(images_path, error, mount_hint(images_path))) from error
        absent = [name for name in names if name not in present]
        if absent:
            raise FileNotFoundError("TAIXRAY: {} of the {} listed images are not in {} (first {}): {}".format(len(absent), len(names), images_path, absent[:3], mount_hint(images_path)))

    def __getitem__(self, index):
        image = grey_read(self.img_list[index]).convert("RGB")  # the SSL trainer's grey read, then gray on three channels as the official classes open an image
        return (image if self.augment is None else self.augment(image)), torch.FloatTensor(self.img_label[index])

    def __len__(self):
        return len(self.img_list)
