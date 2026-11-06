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
    blocks = bb.blocks
    bb.blocks = blocks[:6]
    try:
        rejects(ValueError, lambda: layers_for(bb))
    finally:
        bb.blocks = blocks
    del bb, pyr, maps, tokens  # the ViT-L is 1.2 GB
assert layers_for(SimpleNamespace(blocks=[None] * 12)) == [2, 5, 8, 11] and layers_for(SimpleNamespace(blocks=[None] * 24)) == [5, 11, 17, 23]
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

print("ALL OK")
