"""CPU checks for tbx_localization/compare_detectors.py: no data files, no weights, no GPU.
Run from base/: python -m tbx_localization.test_compare_detectors
(a) patch 16: RoIPool(16) is bitwise torchvision's default MultiScaleRoIAlign on a 32x32 map at 512 px (the default's power-of-two scale is the stride there)
(b) patch 14: RoIPool(14) pools RAD-DINO's 37x37 map at 518 px at 1/14 (RoIAlign(7, 1/14, 2)); the default gets another scale on it, which is the bug
(c) a FasterRCNN on a frozen stride-14 stand-in backbone takes RoIPool(14) and runs one train-mode forward to a finite loss
(d) drop_extra removes the extra/ images and nothing else (every annotation of the others kept), and tb_split on the result leaves every other image on the side it had (each folder group is ranked on its own)
(e) the committed TBX11K manifest: 5,610 train and 990 selection names, none under extra/, disjoint, sorted, its algorithm string the one main() writes, and equal to 425a110's minus its extra/ names (that last part needs git)
"""
import inspect
import json
import os
import subprocess

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, RoIAlign

from .compare_detectors import RoIPool, drop_extra, main, tb_split

torch.manual_seed(0)


def boxes_in(size):  # four boxes inside a size x size image, one batch item
    return [torch.tensor([[10., 20., 200., 300.], [0., 0., size - 1., size - 1.], [100., 150., 260., 400.], [33.3, 47.1, 129.9, 222.2]])]


features, boxes = {'0': torch.randn(1, 256, 32, 32)}, boxes_in(512)
ours, default = RoIPool(16)(features, boxes, [(512, 512)]), MultiScaleRoIAlign(['0'], 7, 2)(features, boxes, [(512, 512)])
assert RoIPool(16).output_size == (7, 7) and ours.shape == (4, 256, 7, 7) and torch.equal(ours, default)
print("OK (a) patch 16: RoIPool(16) is bitwise torchvision's default MultiScaleRoIAlign on a 1x256x32x32 map at 512 px")

features, boxes = {'0': torch.randn(1, 256, 37, 37)}, boxes_in(518)
ours, default = RoIPool(14)(features, boxes, [(518, 518)]), MultiScaleRoIAlign(['0'], 7, 2)(features, boxes, [(518, 518)])
assert RoIPool(14).output_size == (7, 7) and torch.equal(ours, RoIAlign(7, 1 / 14, 2)(features['0'], boxes)) and not torch.equal(ours, default)
print("OK (b) patch 14: RoIPool(14) is RoIAlign(7, 1/14, 2) on a 1x256x37x37 map at 518 px and is not the default's output (the default pools it at another scale)")


class Frozen(nn.Module):  # stands in for the frozen ViT and its projection: a 256-channel map at stride 14, 37x37 for 518 px
    out_channels = 256

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, 14, stride=14)
        self.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


anchors = AnchorGenerator(sizes=((16, 32, 64, 128, 256),), aspect_ratios=((.5, 1., 2.),))
model = FasterRCNN(Frozen(), num_classes=2, min_size=518, max_size=518, rpn_anchor_generator=anchors, box_roi_pool=RoIPool(14))
model.transform.size_divisible = 14
model.train()
losses = model([torch.rand(3, 518, 518)], [{'boxes': torch.tensor([[100., 120., 300., 360.]]), 'labels': torch.ones(1, dtype=torch.int64)}])
loss = sum(losses.values())
assert torch.isfinite(loss), losses
print("OK (c) FasterRCNN with RoIPool(14) on a frozen stride-14 backbone: one train-mode forward on a 518 px image, finite loss", loss.item())


def coco(folders):  # {folder: (images, how many of them carry a box)} as a COCO dict
    images, annotations = [], []
    for folder, (n, boxed) in folders.items():
        for i in range(n):
            images.append({'id': len(images) + 1, 'file_name': '%s/%03d.png' % (folder, i), 'width': 512, 'height': 512})
            if i < boxed:
                annotations.append({'id': len(annotations) + 1, 'image_id': images[-1]['id'], 'category_id': 1, 'bbox': [10, 20, 100, 80], 'iscrowd': 0})
    return {'images': images, 'annotations': annotations, 'categories': [{'id': 1, 'name': 'tb'}]}


synthetic = coco({'health': (40, 0), 'sick': (40, 0), 'tb': (40, 30), 'extra/x/train': (20, 0)})
kept = drop_extra(synthetic)
extra_ids = {x['id'] for x in synthetic['images'] if x['file_name'].startswith('extra/')}
assert len(extra_ids) == 20 and len(kept['images']) == 120 and not any(x['file_name'].startswith('extra/') for x in kept['images'])
assert kept['annotations'] == synthetic['annotations'] and len(synthetic['images']) == 140  # every annotation kept, the input untouched
train_all, val_all = tb_split(synthetic)
train_kept, val_kept = tb_split(kept)
assert extra_ids & set(val_all) and set(train_kept) == set(train_all) - extra_ids and set(val_kept) == set(val_all) - extra_ids
assert len(drop_extra(coco({'tb': (10, 10), 'extra/y/val': (5, 5)}))['annotations']) == 10  # an extra/ image's boxes go with it
print("OK (d) drop_extra: the %d extra/ images are gone, all %d annotations of the other %d stay, and tb_split puts each of those on the same side as before (%d train, %d selection)"
      % (len(extra_ids), len(kept['annotations']), len(kept['images']), len(train_kept), len(val_kept)))

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = 'splits/tbx11k_train_selection_seed100.json'
with open(os.path.join(HERE, MANIFEST)) as f:
    manifest = json.load(f)
train, selection = manifest['train_filenames'], manifest['selection_filenames']
assert (len(train), len(selection)) == (5610, 990)
assert not [n for n in train + selection if n.startswith('extra/')] and not set(train) & set(selection)
assert train == sorted(train) and selection == sorted(selection)
assert manifest['algorithm'] in inspect.getsource(main)  # main() raises on a committed manifest that differs from what it would write
try:
    shown = subprocess.run(['git', 'show', '425a110:base/tbx_localization/' + MANIFEST], cwd=HERE, capture_output=True, text=True)
except OSError:
    shown = None
if shown is None or shown.returncode:
    compared = 'NOT compared with 425a110 (git or that commit is not available here%s)' % (': ' + shown.stderr.strip() if shown else '')
else:
    earlier = json.loads(shown.stdout)
    assert list(manifest) == list(earlier) and manifest['source'] == earlier['source'] and manifest['source_sha256'] == earlier['source_sha256']
    assert manifest['algorithm'] == earlier['algorithm'] + '; extra/ images excluded'
    assert train == [n for n in earlier['train_filenames'] if not n.startswith('extra/')]
    assert selection == [n for n in earlier['selection_filenames'] if not n.startswith('extra/')]
    compared = "equal to 425a110's with its %d + %d extra/ names removed" % (len(earlier['train_filenames']) - len(train), len(earlier['selection_filenames']) - len(selection))
print("OK (e) manifest: 5610 train and 990 selection names, none under extra/, disjoint, sorted, its algorithm string the one in main(); " + compared)

print("ALL OK")
