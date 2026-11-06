import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# copied from Foundation_X+/models/dino/swin_transformer_CyclicSegmentation.py:218-278
## UperNet from Jiaxuan
class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                      for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)]
                                         * (len(feature_channels) - 1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]  ##
        P = [up_and_add(features[i], features[i - 1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  # P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x

def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


# copied from Foundation_X+/utils_segmentation.py:143-169
def dice_score(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1 > 0.5).astype(bool)
    im2 = np.asarray(im2 > 0.5).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def mean_dice_coef(y_true,y_pred):
    sum=0
    for i in range (y_true.shape[0]):
        sum += dice_score(y_true[i,:,:,:],y_pred[i,:,:,:])
    return sum/y_true.shape[0]

def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))


# copied from Foundation_X+/utils_segmentation.py:190-199
def step_decay(step, start_lr, total_epoch, step_inc=20):

    lr = start_lr
    progress = (step - step_inc) / float(total_epoch - step_inc)
    progress = np.clip(progress, 0.0, 1.0)
    lr = lr * 0.5 * (1. + np.cos(np.pi * progress))

    lr = lr * np.minimum(1., step / step_inc)

    return lr


class SegHead(nn.Module):
    """Foundation X's segmentation head as built at swin_transformer_CyclicSegmentation.py:875-876 and 890 (PSP on the last map, FPN
    fusion, a 3x3 conv to num_classes) and run as in extra_features_seg (:1046-1053). Returns logits at out_size."""

    def __init__(self, feature_channels, num_classes=1):
        super().__init__()
        self.ppn = PSPModule(feature_channels[-1])
        self.fpn = FPN_fuse(feature_channels, fpn_out=feature_channels[0])
        self.conv = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)

    def forward(self, feats, out_size):
        features = list(feats)  # FPN_fuse edits its list in place
        features[-1] = self.ppn(features[-1])
        x = self.conv(self.fpn(features))
        return F.interpolate(x, size=out_size, mode='bilinear')
