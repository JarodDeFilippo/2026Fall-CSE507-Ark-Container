# Modified from the course Ark+ container (see NOTICE)
import logging
import re

import torch
import torch.nn as nn
from functools import partial
from torch.hub import load_state_dict_from_url

import timm.models.vision_transformer as vit
import timm.models.swin_transformer as swin
from convnext import ConvNeXt
from dinov3.models.vision_transformer import DinoVisionTransformer
 
from timm.models.helpers import load_state_dict

from utils import remap_pretrained_keys_swin


logger = logging.getLogger("ark")


class ArkSwinTransformer(swin.SwinTransformer):
    def __init__(self, num_classes_list, projector_features = None, use_mlp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None
        
        self.projector = None 
        if projector_features:
            encoder_features = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = []
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, x, head_n=None, return_all=False, return_features=False):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        if return_features:
            return x
        if return_all:
            return x, [head(x) for head in self.omni_heads]
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]
    
    def generate_embeddings(self, x, after_proj = True):
        x = self.forward_features(x)
        if after_proj:
            x = self.projector(x)
        return x

class ArkConvNeXt(ConvNeXt):
    def __init__(self, num_classes_list, projector_features = None, use_mlp=False, encoder_features=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None
        
        self.projector = None 
        if projector_features:
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = []
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, x, head_n=None, return_all=False, return_features=False):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        if return_features:
            return x
        if return_all:
            return x, [head(x) for head in self.omni_heads]
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]
    
    def generate_embeddings(self, x, after_proj = True):
        x = self.forward_features(x)
        if after_proj:
            x = self.projector(x)
        return x

# dinov3_vitb16 config from Meta's hub (dinov3/hub/backbones.py). The rope min/max period, shift and jitter args are None
# there and in DinoVisionTransformer's defaults, so they are omitted.
DINOV3_VITB16_KWARGS = dict(
    img_size=224,
    patch_size=16,
    in_chans=3,
    pos_embed_rope_base=100,
    pos_embed_rope_normalize_coords="separate",
    # Meta's hub uses 2 (train-mode random coordinate rescale); disabled here so the Ark+ student and teacher see identical geometry (2026-09-30)
    pos_embed_rope_rescale_coords=None,
    pos_embed_rope_dtype="fp32",
    embed_dim=768,
    depth=12,
    num_heads=12,
    ffn_ratio=4,
    qkv_bias=True,
    drop_path_rate=0.0,
    layerscale_init=1.0e-05,
    norm_layer="layernormbf16",
    ffn_layer="mlp",
    ffn_bias=True,
    proj_bias=True,
    n_storage_tokens=4,
    mask_k_bias=True,
)

class ArkViT(DinoVisionTransformer):
    def __init__(self, num_classes_list, projector_features = None, use_mlp=False, encoder_features=1536, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None
        # DINOv3's hub calls this when not loading pretrained; the checkpoint overwrites the backbone weights anyway.
        self.init_weights()
        self.num_features = encoder_features

        self.projector = None
        if projector_features:
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = []
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward_features(self, x):
        # DINOv3 returns a dict; the Ark heads want one (N, 1536) vector: CLS token + mean of the patch tokens.
        out = super().forward_features(x)
        return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(dim=1)], dim=-1)

    def forward(self, x, head_n=None, return_all=False, return_features=False):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        if return_features:
            return x
        if return_all:
            return x, [head(x) for head in self.omni_heads]
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]

    def generate_embeddings(self, x, after_proj = True):
        x = self.forward_features(x)
        if after_proj:
            x = self.projector(x)
        return x

# Learned tokens: layer 0, and no weight decay (Meta's param_groups.py comment says so, but its name rules miss them).
VIT_TOKEN_PARAMS = ("cls_token", "storage_tokens", "mask_token")

def vit_param_groups(model, weight_decay, layer_decay, patch_embed_lr_mult):
    """Optimizer param groups for ArkViT, mirroring dinov3/train/param_groups.py.

    Layer id: patch_embed.*, rope_embed.* and the learned tokens -> 0, blocks.{i} -> i+1, everything else
    (norm, projector, omni_heads) -> num_blocks+1. lr_scale = layer_decay ** (num_blocks + 1 - layer_id), times
    patch_embed_lr_mult for patch_embed.*; layer_decay=None means no decay. No weight decay on ndim <= 1 params,
    the tokens, or Meta's name rules (bias / norm / gamma / fourier_w), weight_decay otherwise.
    One group per distinct (lr_scale, weight_decay), ordered by descending lr_scale so that param_groups[0] (the
    LR that is logged) is the head at scale 1.0. WarmupCosineScheduler multiplies the scheduled LR by `lr_scale`.
    """
    model = getattr(model, "module", model)  # DDP wrapper
    num_blocks = len(model.blocks)
    layer_decay = 1.0 if layer_decay is None else layer_decay
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        block = re.match(r"blocks\.(\d+)\.", name)
        if name.startswith(("patch_embed.", "rope_embed.")) or name in VIT_TOKEN_PARAMS:
            layer_id = 0
        elif block:
            layer_id = int(block.group(1)) + 1
        else:
            layer_id = num_blocks + 1
        lr_scale = layer_decay ** (num_blocks + 1 - layer_id)
        if name.startswith("patch_embed."):
            lr_scale *= patch_embed_lr_mult
        no_decay = (param.ndim <= 1 or name in VIT_TOKEN_PARAMS
                    or name.endswith("bias") or "norm" in name or "gamma" in name or "fourier_w" in name)
        groups.setdefault((lr_scale, 0.0 if no_decay else weight_decay), []).append(param)
    return [{"params": groups[key], "weight_decay": key[1], "lr_scale": key[0]}
            for key in sorted(groups, key=lambda key: (-key[0], -key[1]))]

def build_omni_model_from_checkpoint(args, num_classes_list, key):
    if args.model_name == "swin_base": #swin_base_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    elif args.model_name == "swin_large": #swin_large_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_384": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =384, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_768": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =768, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "conv_base":
        model = ArkConvNeXt(num_classes_list, args.projector_features, args.use_mlp, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    elif args.model_name == "vit_base_dinov3":
        model = ArkViT(num_classes_list, args.projector_features, args.use_mlp, **DINOV3_VITB16_KWARGS)

    if args.pretrained_weights is not None:
        checkpoint = torch.load(args.pretrained_weights, map_location='cpu', weights_only=False)
        state_dict = checkpoint[key]
        if any([True if 'module.' in k else False for k in state_dict.keys()]):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}

        msg = model.load_state_dict(state_dict, strict=False)
        logger.info('Loaded with msg: {}'.format(msg))
           
    return model

def build_omni_model(args, num_classes_list):
    if args.model_name == "swin_base": #swin_base_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    elif args.model_name == "swin_large": #swin_large_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_384": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =384, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_768": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =768, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_1152": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =1152, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "conv_base":
        model = ArkConvNeXt(num_classes_list, args.projector_features, args.use_mlp, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        # url='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth'
    elif args.model_name == "vit_base_dinov3":
        model = ArkViT(num_classes_list, args.projector_features, args.use_mlp, **DINOV3_VITB16_KWARGS)
    if args.pretrained_weights is not None:
        if args.pretrained_weights.startswith('https'):
            state_dict = load_state_dict_from_url(url=args.pretrained_weights, map_location='cpu')
        else:
            state_dict = load_state_dict(args.pretrained_weights)
        
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        k_del = []
        for k in state_dict.keys():
            if "attn_mask" in k:
                k_del.append(k)
        logger.info("Removing key {} from pretrained checkpoint".format(k_del))
        for k in k_del:
            del state_dict[k]
            
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info('Loaded with msg: {}'.format(msg))
        if args.model_name == "vit_base_dinov3":
            # Meta loads this checkpoint into DinoVisionTransformer with strict=True, so anything but the new
            # projector/heads missing means the backbone silently stayed at random init.
            bad_missing = [k for k in msg.missing_keys if not k.startswith(("projector.", "omni_heads."))]
            if bad_missing or msg.unexpected_keys:
                raise RuntimeError("DINOv3 checkpoint does not match the backbone: missing keys {} and unexpected keys {}"
                                   .format(sorted(bad_missing), sorted(msg.unexpected_keys)))

    return model

def save_checkpoint(state,filename='model'):

    torch.save(state, filename + '.pth.tar')
