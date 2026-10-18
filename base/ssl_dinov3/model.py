"""Student / EMA teacher / Gram teacher for Stage A, as dinov3/train/ssl_meta_arch.py builds them (SSLMetaArch.__init__ +
init_weights), minus FSDP. Backbone = Meta's DinoVisionTransformer with Ark's DINOV3_VITB16_KWARGS (base/models.py), so the
exported teacher backbone loads into Ark's vit_base_dinov3 unchanged."""
import copy

import torch
from torch import nn

from dinov3.layers.dino_head import DINOHead
from dinov3.models.vision_transformer import DinoVisionTransformer

# dinov3 configs/ssl_default_config.yaml dino.* / ibot.* (separate iBOT head, same sizes)
HEAD_KWARGS = dict(out_dim=65536, hidden_dim=2048, bottleneck_dim=256, nlayers=3)


def build_backbone(vit_kwargs, weights_path=None):
    backbone = DinoVisionTransformer(**vit_kwargs)
    backbone.init_weights()
    if weights_path:
        # Meta's hub loads this flat backbone file with strict=True; so do we
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state_dict, strict=True)
    return backbone


def build_models(vit_kwargs, weights_path=None, head_kwargs=HEAD_KWARGS):
    """Returns (student, teacher, gram_teacher) as nn.ModuleDicts keyed like SSLMetaArch: student/teacher have
    backbone, dino_head, ibot_head; gram_teacher has backbone only. Teacher = copy of the student (init_weights:
    model_ema.load_state_dict(student.state_dict())); Gram teacher = the same stock backbone, frozen, eval."""
    backbone = build_backbone(vit_kwargs, weights_path)
    student = nn.ModuleDict(dict(
        backbone=backbone,
        dino_head=DINOHead(in_dim=backbone.embed_dim, **head_kwargs),
        ibot_head=DINOHead(in_dim=backbone.embed_dim, **head_kwargs),
    ))
    student.dino_head.init_weights()  # heads are random: the released weights ship none
    student.ibot_head.init_weights()
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    gram_teacher = nn.ModuleDict(dict(backbone=copy.deepcopy(backbone)))
    gram_teacher.requires_grad_(False)
    gram_teacher.eval()
    return student, teacher, gram_teacher


def export_backbone(backbone):
    """Flat CPU state dict of a backbone: the same keys as the stock file (188 for ViT-B/16), for Ark's strict guard."""
    return {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
