"""AdamW param groups and schedules for Stage A, following dinov3/train/{ssl_meta_arch.get_params_groups, train.build_optimizer,
train.build_schedulers_v2, train.apply_optim_scheduler}. Uses the vendored param_groups.py / cosine_lr_scheduler.py as is."""
import math

import torch

from dinov3.train.cosine_lr_scheduler import linear_warmup_cosine_decay
from dinov3.train.param_groups import fuse_params_groups, get_params_groups_with_decay_fsdp

LAYERWISE_DECAY = 0.9  # ssl_default_config optim.layerwise_decay
PATCH_EMBED_LR_MULT = 0.2  # optim.patch_embed_lr_mult
# Departure (ours, as base/models.py vit_param_groups does): no weight decay on the learned tokens. Meta's name rules
# (bias / norm / gamma / fourier_w) miss them; every other 1-D param is already caught by those rules.
NO_WD_TOKENS = ("cls_token", "storage_tokens", "mask_token")


def get_params_groups(student):
    """SSLMetaArch.get_params_groups: per-submodule groups, then fuse_params_groups (multi_tensor_optim). Extra key
    is_backbone (ours) so the backbone can be frozen for the first epoch(s)."""
    all_groups = []
    for name, module in student.items():
        module = getattr(module, "module", module)  # DDP wrapper
        groups = get_params_groups_with_decay_fsdp(
            model=module, lr_decay_rate=LAYERWISE_DECAY, patch_embed_lr_mult=PATCH_EMBED_LR_MULT, dino_head_wd_multiplier=1.0
        )
        for group in groups:
            group["is_backbone"] = name == "backbone"
            if group["name"] in NO_WD_TOKENS:
                group["wd_multiplier"] = 0.0
        all_groups += groups
    return list(fuse_params_groups(all_groups, keys=("lr_multiplier", "wd_multiplier", "is_last_layer", "is_backbone")))


def build_optimizer(param_groups):
    # train.build_optimizer with adamw_beta1/2 = 0.9/0.999; Meta's fused/foreach group flags are CUDA-only, so torch defaults
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def lr_scaling_factor(global_batch_size):
    """dinov3 train.py build_schedulers_v2, scaling_rule sqrt_wrt_1024: lr *= 4 * sqrt(global_batch / 1024)."""
    return 4 * math.sqrt(global_batch_size / 1024.0)


def build_schedules(*, lr, min_lr, epochs, warmup_epochs, epoch_length, global_batch_size, scale_lr=True,
                    teacher_temp_warmup_epochs, warmup_teacher_temp=0.04, teacher_temp=0.07, freeze_last_layer_epochs=1):
    """build_schedulers_v2: per-iteration numpy arrays over epochs * epoch_length steps. LR peak and end are both scaled,
    as Meta does; scale_lr=False keeps --lr / --min_lr absolute. Weight decay and teacher momentum are constants here
    (the Gram-anchor config's start == peak == end), so they are not arrays."""
    total = epochs * epoch_length
    factor = lr_scaling_factor(global_batch_size) if scale_lr else 1.0
    peak, end = lr * factor, min_lr * factor
    lr_schedule = linear_warmup_cosine_decay(
        start=0.0, peak=peak, end=end, warmup_iterations=warmup_epochs * epoch_length, total_iterations=total
    )
    last_layer_lr_schedule = lr_schedule.copy()
    last_layer_lr_schedule[: freeze_last_layer_epochs * epoch_length] = 0
    teacher_temp_schedule = linear_warmup_cosine_decay(
        start=warmup_teacher_temp, peak=teacher_temp, end=teacher_temp,
        warmup_iterations=teacher_temp_warmup_epochs * epoch_length, total_iterations=total,
    )
    return dict(lr=lr_schedule, last_layer_lr=last_layer_lr_schedule, teacher_temp=teacher_temp_schedule,
                peak_lr=peak, end_lr=end, factor=factor)


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr, freeze_backbone=False):
    """train.apply_optim_scheduler, plus our backbone freeze: LR 0 on backbone groups. AdamW with lr 0 leaves a param
    bitwise unchanged (decoupled decay multiplies by 1 - 0 * wd), the same mechanism Meta uses to freeze the head's last layer."""
    for group in optimizer.param_groups:
        group["weight_decay"] = wd * group["wd_multiplier"]
        if freeze_backbone and group["is_backbone"]:
            group["lr"] = 0.0
        elif group["is_last_layer"]:
            group["lr"] = last_layer_lr * group["lr_multiplier"]
        else:
            group["lr"] = lr * group["lr_multiplier"]
