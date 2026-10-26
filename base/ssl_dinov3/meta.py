"""One Stage A training step: dinov3/train/ssl_meta_arch.py SSLMetaArch (forward_backward, get_teacher_output,
get_student_output, get_gram_teacher_output, compute_losses, update_ema, update_gram) with the options this run uses fixed:
sinkhorn_knopp centering, separate iBOT head, global_ignore_diagonal, non-distributed KoLeo (weight 0.1), no local-loss
reweighting, Gram from a separate teacher (ema_teacher false) on all tokens, normalized, img_level, remove_neg false,
bicubic resize without antialias, constant Gram weight. Method bodies follow Meta's line for line; differences are marked.
gram_teacher=None is the no-Gram mode (--no_gram): no Gram teacher forward, no Gram loss term, no gram_loss key."""
import contextlib

import torch
from torch import nn

from dinov3.loss import DINOLoss, GramLoss, KoLeoLoss, iBOTPatchLoss


class SSLMetaArch(nn.Module):
    def __init__(self, student, teacher, gram_teacher, *, n_local_crops=8, patch_size=16, gram_loss_weight=2.0,
                 dino_loss_weight=1.0, koleo_loss_weight=0.1, ibot_loss_weight=1.0, autocast_dtype=None):
        super().__init__()
        assert n_local_crops > 0
        # The vendored iBOT Sinkhorn-Knopp all_reduces unconditionally, so even one process needs a process group;
        # ssl_pretrain.py creates a world-size-1 group when Accelerator does not (WORLD_SIZE=1).
        assert torch.distributed.is_initialized(), "SSLMetaArch needs torch.distributed initialized (world size 1 included)"
        self.student = student  # ModuleDict(backbone, dino_head, ibot_head); entries may be DDP-wrapped
        self.teacher = teacher
        self.model_ema = self.teacher
        self.gram_teacher = gram_teacher  # ModuleDict(backbone); None: no-Gram mode
        self.teacher.requires_grad_(False)
        if self.gram_teacher is not None:
            self.gram_teacher.requires_grad_(False)
        self.ema_params_lists = None
        self.gram_params_lists = None

        n_prototypes = teacher.dino_head.last_layer.out_features
        self.dino_loss = DINOLoss(n_prototypes)
        self.ibot_patch_loss = iBOTPatchLoss(teacher.ibot_head.last_layer.out_features)
        self.koleo_loss = KoLeoLoss()
        self.gram_loss = (GramLoss(apply_norm=True, remove_only_teacher_neg=False, remove_neg=False)
                          if self.gram_teacher is not None else None)
        self.dino_loss.init_weights()  # centers start at nan upstream; sinkhorn_knopp never reads them
        self.ibot_patch_loss.init_weights()

        self.n_local_crops = n_local_crops
        self.patch_size = patch_size
        self.dino_global_ignore_diagonal = True
        self.dino_loss_weight = dino_loss_weight
        self.dino_koleo_loss_weight = koleo_loss_weight
        self.ibot_loss_weight = ibot_loss_weight
        self.gram_loss_weight = gram_loss_weight
        self.gram_img_level = True
        self.gram_global_teacher_resize_method = "bicubic"
        self.gram_global_teacher_resize_antialias = False
        # ours: optional bf16 autocast around the forward passes only (Meta uses FSDP mixed precision instead)
        self.autocast_dtype = autocast_dtype

    def _autocast(self, device):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=self.autocast_dtype)

    def forward_backward(self, data, *, teacher_temp):
        # Shapes
        n_global_crops = 2
        n_local_crops = self.n_local_crops
        B = data["collated_local_crops"].shape[0] // n_local_crops
        assert data["collated_global_crops"].shape[0] == n_global_crops * B

        device = next(self.teacher.parameters()).device  # Meta: .cuda(non_blocking=True)
        global_crops = data["collated_global_crops"].to(device, non_blocking=True)
        local_crops = data["collated_local_crops"].to(device, non_blocking=True)
        masks = data["collated_masks"].to(device, non_blocking=True)
        mask_indices_list = data["mask_indices_list"].to(device, non_blocking=True)
        masks_weight = data["masks_weight"].to(device, non_blocking=True)
        # ours: float, not int64, for the all_reduce inside iBOT's Sinkhorn-Knopp (exact below 2**24; HCCL-safe dtype)
        n_masked_patches_tensor = data["n_masked_patches"].to(device=device, dtype=torch.float32, non_blocking=True)
        if self.gram_teacher is not None:  # no-Gram mode: the batch has no Gram crops
            gram_teacher_crops = data["collated_gram_teacher_crops"].to(device, non_blocking=True)

        with self._autocast(device):
            teacher_global = self.get_teacher_output(
                global_crops.unflatten(0, (n_global_crops, B)),
                teacher_temp=teacher_temp,
                n_masked_patches_tensor=n_masked_patches_tensor,
                mask_indices_list=mask_indices_list,
                upperbound=data["upperbound"],
            )
            student_global, student_local = self.get_student_output(
                global_crops=global_crops.unflatten(0, (n_global_crops, B)),
                local_crops=local_crops.unflatten(0, (n_local_crops, B)),
                upperbound=data["upperbound"],
                masks=masks,
                mask_indices_list=mask_indices_list,
            )
            if self.gram_teacher is None:  # no-Gram mode: no Gram forward, compute_losses skips the Gram loss
                gram_global = None
            else:
                gram_global = self.get_gram_teacher_output(
                    gram_teacher_crops.unflatten(0, (n_global_crops, B)),
                    masks=masks,
                    teacher_global=teacher_global,
                    student_global=student_global,
                    student_global_crops_size=global_crops.shape[-1],
                )
            loss_accumulator, loss_dict = self.compute_losses(
                teacher_global=teacher_global,
                student_global=student_global,
                student_local=student_local,
                gram_global=gram_global,
                masks=masks,
                mask_indices_list=mask_indices_list,
                masks_weight=masks_weight,
            )

        self.backprop_loss(loss_accumulator)
        return loss_accumulator, loss_dict

    @torch.no_grad()
    def get_teacher_output(self, images, *, upperbound, mask_indices_list, teacher_temp, n_masked_patches_tensor):
        n_crops, B, rgb, H, W = images.shape
        images = images.flatten(0, 1)

        backbone_out = self.teacher.backbone(images, is_training=True)
        cls = backbone_out["x_norm_clstoken"]  # [n_crops * B, D]
        reg = backbone_out["x_storage_tokens"]  # [n_crops * B, R, D]
        ibot_patch = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P, D]

        # IBOT head only on patches that are masked for the student
        buffer = torch.index_select(ibot_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        masked_patch_after_head = self.teacher.ibot_head(buffer)

        # DINO head on CLS tokens
        cls_after_head = self.teacher.dino_head(cls)  # [n_crops * B, K]

        # Center with sinkhorn-knopp
        cls_centered = self.dino_loss.sinkhorn_knopp_teacher(cls_after_head, teacher_temp=teacher_temp)
        cls_centered = cls_centered.unflatten(0, (n_crops, B))  # [n_crops, B, K]
        # ours: .forward, not __call__, to bypass the torch.compile wrapper iBOTPatchLoss.__init__ installs
        masked_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher.forward(
            masked_patch_after_head,
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
        )  # [n_masked_patches, K]

        return {
            "cls_pre_head": cls.unflatten(0, [n_crops, B]),
            "reg_pre_head": reg.unflatten(0, [n_crops, B]),
            "patch_pre_head": ibot_patch.unflatten(0, [n_crops, B]),
            "cls_after_head": cls_after_head.unflatten(0, [n_crops, B]),
            "cls_centered": cls_centered,
            "masked_patch_centered": masked_patch_centered,
        }

    def get_gram_teacher_output(self, images, *, masks, teacher_global, student_global, student_global_crops_size):
        # Get student patch features
        student_patches = student_global["patch_pre_head"].flatten(0, 1)  # [n_crops * B, P, D]

        n_crops, B, rgb, H, W = images.shape
        images = images.flatten(0, 1)  # [n_crops * B, rgb, H, W]
        with torch.no_grad():
            backbone_out = self.gram_teacher.backbone(images, is_training=True)
        teacher_patches = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P_T, D]

        # Downsample Gram teacher features if needed
        if teacher_patches.shape[1] != student_patches.shape[1]:
            N = H // self.patch_size
            assert teacher_patches.shape[1] == N**2
            N_student = student_global_crops_size // self.patch_size
            assert student_patches.shape[1] == N_student**2
            patches_hw = teacher_patches.transpose(-2, -1).unflatten(-1, (N, N))  # [n_crops * B, D, N, N]
            patches_hw = torch.nn.functional.interpolate(
                patches_hw,
                size=(N_student, N_student),
                mode=self.gram_global_teacher_resize_method,
                align_corners=False,
                antialias=self.gram_global_teacher_resize_antialias,
            )
            teacher_patches = patches_hw.flatten(-2, -1).transpose(-2, -1)  # [n_crops * B, N_student**2, D]
            assert teacher_patches.shape == student_patches.shape

        # gram.tokens_used == "all": every patch, masked or not
        return {"student_patches": student_patches, "teacher_patches": teacher_patches}

    def get_student_output(self, *, global_crops, local_crops, upperbound, masks, mask_indices_list):
        n_global_crops, B, rgb, H, W = global_crops.shape
        n_local_crops, B, rgb, H, W = local_crops.shape

        global_crops = global_crops.flatten(0, 1)

        # Forward global and local crops through the student backbone jointly. Entry 1 is the local crops: with
        # untie_global_and_local_cls_norm, Meta's forward_features_list applies local_cls_norm to idx == 1 (train mode only)
        global_out, local_out = self.student.backbone(
            [global_crops, local_crops.flatten(0, 1)],
            masks=[masks, None],
            is_training=True,
        )
        g_cls, g_reg, g_patch = global_out["x_norm_clstoken"], global_out["x_storage_tokens"], global_out["x_norm_patchtokens"]
        l_cls, l_reg, l_patch = local_out["x_norm_clstoken"], local_out["x_storage_tokens"], local_out["x_norm_patchtokens"]

        # IBOT head only on masked patches
        masked_patches_pre_head = torch.index_select(g_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        global_masked_patch_after_head = self.student.ibot_head(masked_patches_pre_head)

        # DINO head on CLS tokens (all in one pass)
        buffer = [g_cls, l_cls]
        sizes = [x.shape[0] for x in buffer]
        buffer = torch.cat(buffer, dim=0)
        buffer = self.student.dino_head(buffer)
        buffer = torch.split_with_sizes(buffer, sizes, dim=0)

        global_out = {
            "cls_pre_head": g_cls.unflatten(0, [n_global_crops, B]),
            "reg_pre_head": g_reg.unflatten(0, [n_global_crops, B]),
            "patch_pre_head": g_patch.unflatten(0, [n_global_crops, B]),
            "cls_after_head": buffer[0].unflatten(0, [n_global_crops, B]),
            "masked_patch_after_head": global_masked_patch_after_head,
            "masked_patch_pre_head": masked_patches_pre_head,
        }
        local_out = {
            "cls_pre_head": l_cls.unflatten(0, [n_local_crops, B]),
            "reg_pre_head": l_reg.unflatten(0, [n_local_crops, B]),
            "patch_pre_head": l_patch.unflatten(0, [n_local_crops, B]),
            "cls_after_head": buffer[1].unflatten(0, [n_local_crops, B]),
        }
        return global_out, local_out

    def compute_losses(self, *, teacher_global, student_global, student_local, gram_global, masks, mask_indices_list,
                       masks_weight):
        n_global_crops = student_global["cls_after_head"].shape[0]
        n_local_crops = student_local["cls_after_head"].shape[0]
        loss_dict = {}
        loss_accumulator = 0.0

        # Loss scales like in DINOv2, these are multiplied with the loss weights from the config
        dino_global_terms = (
            n_global_crops * (n_global_crops - 1) if self.dino_global_ignore_diagonal else n_global_crops**2
        )
        dino_local_terms = n_global_crops * n_local_crops
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
        koleo_scale = n_global_crops

        # DINO local loss: compare post-head CLS tokens: student(local crops) vs. teacher(global crops)
        dino_local_crops_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss
        local_weight = 1.0  # dino.reweight_dino_local_loss false
        loss_accumulator += self.dino_loss_weight * dino_local_scale * local_weight * dino_local_crops_loss

        # DINO global loss: compare post-head CLS tokens: student(global crops) vs. teacher(global crops)
        dino_global_crops_loss = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=self.dino_global_ignore_diagonal,
        )
        loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
        loss_accumulator += self.dino_loss_weight * dino_global_scale * dino_global_crops_loss

        # Koleo: regularize pre-head CLS tokens of student(global crops)
        koleo_loss = sum(self.koleo_loss(x) for x in student_global["cls_pre_head"]) / n_global_crops
        loss_dict["koleo_loss"] = koleo_loss
        loss_accumulator += self.dino_koleo_loss_weight * koleo_scale * koleo_loss

        # IBOT loss
        ibot_patch_loss = self.ibot_patch_loss.forward_masked(
            student_global["masked_patch_after_head"],
            teacher_global["masked_patch_centered"],
            student_masks_flat=masks,
            n_masked_patches=mask_indices_list.shape[0],
            masks_weight=masks_weight,
        )
        loss_dict["ibot_loss"] = ibot_patch_loss
        loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        # Gram loss (constant weight: no loss_weight_schedule); none in no-Gram mode (gram_global is None)
        if gram_global is not None:
            gram_loss = self.gram_loss(gram_global["student_patches"], gram_global["teacher_patches"], img_level=self.gram_img_level)
            loss_accumulator += gram_loss * self.gram_loss_weight
            loss_dict["gram_loss"] = gram_loss

        return loss_accumulator, loss_dict

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        if self.gram_teacher is not None:
            self.gram_teacher.eval()
        return self

    def backprop_loss(self, loss):
        loss.backward()

    def update_ema(self, m):
        if self.ema_params_lists is None:
            student_param_list = []
            teacher_param_list = []
            for k in self.student.keys():
                for ms, mt in zip(self.student[k].parameters(), self.model_ema[k].parameters()):
                    student_param_list += [ms]
                    teacher_param_list += [mt]
            self.ema_params_lists = (student_param_list, teacher_param_list)
        else:
            student_param_list, teacher_param_list = self.ema_params_lists
        with torch.no_grad():
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def update_gram(self, m=0):
        if self.gram_params_lists is None:
            teacher_param_list = []
            gramteacher_param_list = []
            for k in self.gram_teacher.keys():
                for mgt, mt in zip(self.gram_teacher[k].parameters(), self.teacher[k].parameters()):
                    gramteacher_param_list += [mgt]
                    teacher_param_list += [mt]
            self.gram_params_lists = (gramteacher_param_list, teacher_param_list)
        else:
            gramteacher_param_list, teacher_param_list = self.gram_params_lists

        with torch.no_grad():
            torch._foreach_mul_(gramteacher_param_list, m)
            torch._foreach_add_(gramteacher_param_list, teacher_param_list, alpha=1 - m)
