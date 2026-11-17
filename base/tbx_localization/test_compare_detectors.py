"""CPU checks for tbx_localization/compare_detectors.py: no data files, no weights, no GPU.
Run from base/: python -m tbx_localization.test_compare_detectors
(a) patch 16: RoIPool(16) is bitwise torchvision's default MultiScaleRoIAlign on a 32x32 map at 512 px (the default's power-of-two scale is the stride there)
(b) patch 14: RoIPool(14) pools RAD-DINO's 37x37 map at 518 px at 1/14 (RoIAlign(7, 1/14, 2)); the default gets another scale on it, which is the bug
(c) a FasterRCNN on a frozen stride-14 stand-in backbone takes RoIPool(14) and runs one train-mode forward to a finite loss
"""
import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, RoIAlign

from .compare_detectors import RoIPool

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
print("OK (c) FasterRCNN with RoIPool(14) on a frozen stride-14 backbone: one train-mode forward on a 518 px image, finite loss", float(loss))

print("ALL OK")
