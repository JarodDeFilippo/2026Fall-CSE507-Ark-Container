"""Feature pyramid over a frozen ViT backbone's patch tokens, for dense heads."""
import torch
import torch.nn as nn


def layers_for(backbone):
    """The four blocks whose patch tokens feed the pyramid: every 3rd block of a 12-block ViT, every 6th of a 24-block one."""
    depth = len(backbone.blocks)
    if depth == 12:
        return [2, 5, 8, 11]
    if depth == 24:
        return [5, 11, 17, 23]
    raise ValueError("no layer choice for a backbone with {} blocks (12 or 24)".format(depth))


class ViTPyramid(nn.Module):
    """Four maps from a frozen ViT: the patch tokens of block layers[i] go through a 1x1 conv to out_channels[i], then are resampled
    to x4, x2, x1 and /2 of the patch grid. That is the shape of ViTDet's simple feature pyramid (transposed convs, identity,
    max-pool), without ViTDet's normalization layers. The backbone stays frozen and in eval; the convs here are what trains."""

    def __init__(self, backbone, layers, out_channels=(128, 256, 512, 1024)):
        super().__init__()
        assert len(layers) == len(out_channels) == 4
        self.backbone = backbone.eval().requires_grad_(False)
        self.layers = list(layers)
        self.feature_channels = list(out_channels)
        levels = []
        for i, c in enumerate(out_channels):
            level = [nn.Conv2d(backbone.embed_dim, c, 1)]
            if i == 0:
                level += [nn.ConvTranspose2d(c, c, 2, 2), nn.ConvTranspose2d(c, c, 2, 2)]
            elif i == 1:
                level.append(nn.ConvTranspose2d(c, c, 2, 2))
            elif i == 3:
                level.append(nn.MaxPool2d(2))
            levels.append(nn.Sequential(*level))
        self.levels = nn.ModuleList(levels)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            maps = self.backbone.get_intermediate_layers(x, n=self.layers, reshape=True, norm=True)
        return [level(m) for level, m in zip(self.levels, maps)]
