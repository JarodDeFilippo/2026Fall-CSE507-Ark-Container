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
from dinov2.layers import MemEffAttention, NestedTensorBlock
from dinov2.models.vision_transformer import DinoVisionTransformer as DinoV2VisionTransformer
 
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

# dinov2_vitb14 from Meta's hub: _make_dinov2_model's defaults (dinov2/hub/backbones.py), then the arguments vit_base
# (dinov2/models/vision_transformer.py) fixes. img_size 518 only sizes the learned pos_embed (37x37 + CLS); a 224 px input
# gets it resampled to 16x16 by DINOv2's own interpolate_pos_encoding. Used for both DINOv2 (LVD-142M) and RAD-DINO weights.
DINOV2_VITB14_KWARGS = dict(
    img_size=518,
    patch_size=14,
    init_values=1.0,
    ffn_layer="mlp",
    block_chunks=0,
    num_register_tokens=0,
    interpolate_antialias=False,
    interpolate_offset=0.1,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
    in_chans=3,
    channel_adaptive=False,
)

# RayDINO (Moutakanni et al., Nature Communications 2026) is DINOv2's code as a ViT-L/16: DINOV2_VITB14_KWARGS with vit_large's sizes (1024-d tokens, 24 blocks, 16 heads;
# dinov2/models/vision_transformer.py) and patch 16. img_size 512 is its pretraining global-crop size and only sizes the learned pos_embed (32x32 + CLS); a 224 px input gets it
# resampled to 14x14 by DINOv2's own interpolate_pos_encoding. No register tokens (num_register_tokens=0, as in Meta's xray_dino_vitl16 hub entry in dinov2-main). That hub entry builds
# block_chunks=4 and Meta's released file is in that layout (blocks.<chunk>.<block>.*); the model here is built with block_chunks=0, like the ViT-B names, and build_omni_model renames the
# file's block keys first (flatten_chunked_blocks).
DINOV2_VITL16_KWARGS = dict(DINOV2_VITB14_KWARGS, img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16)

# --model name -> the label of its pretrained file in build_omni_model's strict key guard. All four are DINO backbones in
# the Ark wrapper below (three ViT-B, one ViT-L), so they share vit_param_groups (--layer-decay / --patch-embed-lr-mult).
DINO_VIT_MODELS = {"vit_base_dinov3": "DINOv3", "vit_base_dinov2": "DINOv2", "vit_base_raddino": "RAD-DINO", "vit_large_raydino": "RayDINO"}

class ArkDinoHeads:
    """The Ark wrapper API of the DINO ViTs: projector, omni_heads, forward, generate_embeddings. Mixed in ahead of the
    backbone class, whose forward_features the subclass turns into one (N, 2 * embed_dim) vector."""

    def _add_ark_heads(self, num_classes_list, projector_features, use_mlp, encoder_features):
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

class ArkViT(ArkDinoHeads, DinoVisionTransformer):
    def __init__(self, num_classes_list, projector_features = None, use_mlp=False, encoder_features=1536, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None
        # DINOv3's hub calls this when not loading pretrained; the checkpoint overwrites the backbone weights anyway.
        self.init_weights()
        self._add_ark_heads(num_classes_list, projector_features, use_mlp, encoder_features)

    def forward_features(self, x):
        # DINOv3 returns a dict; the Ark heads want one (N, 1536) vector: CLS token + mean of the patch tokens.
        out = super().forward_features(x)
        return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(dim=1)], dim=-1)

class ArkViTDinoV2(ArkDinoHeads, DinoV2VisionTransformer):
    """DINOv2's ViT (vendored in dinov2/) in the Ark wrapper: vit_base_dinov2 and vit_base_raddino, built with DINOV2_VITB14_KWARGS (encoder_features 1536),
    and vit_large_raydino, built with DINOV2_VITL16_KWARGS (encoder_features 2048)."""

    def __init__(self, num_classes_list, projector_features = None, use_mlp=False, encoder_features=1536, *args, **kwargs):
        super().__init__(*args, **kwargs)  # DINOv2's __init__ already ends with init_weights(), as in the hub
        assert num_classes_list is not None
        self._add_ark_heads(num_classes_list, projector_features, use_mlp, encoder_features)

    def forward_features(self, x):
        # Same feature as ArkViT: DINOv2's dict -> CLS token + mean of the patch tokens, both after the final norm (no register tokens).
        out = super().forward_features(x)
        return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(dim=1)], dim=-1)

# Learned tokens: layer 0, and no weight decay (Meta's param_groups.py comment says so, but its name rules miss them).
VIT_TOKEN_PARAMS = ("cls_token", "storage_tokens", "mask_token")
# dinov2/utils/param_groups.py, get_vit_lr_decay_rate with force_is_backbone (our names have no "backbone." prefix):
# a name containing any of these is layer 0.
DINOV2_LAYER0_NAMES = ("pos_embed", "patch_embed", "mask_token", "cls_token", "register_tokens")

def vit_param_groups(model, weight_decay, layer_decay, patch_embed_lr_mult):
    """Optimizer param groups for ArkViT, mirroring dinov3/train/param_groups.py, and for ArkViTDinoV2, mirroring
    dinov2/utils/param_groups.py.

    Layer id: patch_embed.*, rope_embed.* and the learned tokens -> 0, blocks.{i} -> i+1, everything else
    (norm, projector, omni_heads) -> num_blocks+1. lr_scale = layer_decay ** (num_blocks + 1 - layer_id), times
    patch_embed_lr_mult for patch_embed.*; layer_decay=None means no decay. No weight decay on ndim <= 1 params,
    the tokens, or Meta's name rules (bias / norm / gamma / fourier_w), weight_decay otherwise.
    ArkViTDinoV2 takes DINOv2's rules as written: names containing pos_embed / patch_embed / mask_token / cls_token /
    register_tokens -> layer 0; no weight decay only for names ending in ".bias" or containing "norm" or "gamma"
    (LayerScale). So pos_embed, cls_token and mask_token are decayed there: DINOv2's code does that and, unlike
    DINOv3's, has no comment saying they should not be.
    One group per distinct (lr_scale, weight_decay), ordered by descending lr_scale so that param_groups[0] (the
    LR that is logged) is the head at scale 1.0. WarmupCosineScheduler multiplies the scheduled LR by `lr_scale`.
    """
    model = getattr(model, "module", model)  # DDP wrapper
    num_blocks = len(model.blocks)
    layer_decay = 1.0 if layer_decay is None else layer_decay
    dinov2 = isinstance(model, ArkViTDinoV2)
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        block = re.match(r"blocks\.(\d+)\.", name)
        if dinov2:
            layer0 = any(key in name for key in DINOV2_LAYER0_NAMES)
            no_decay = name.endswith(".bias") or "norm" in name or "gamma" in name
        else:
            layer0 = name.startswith(("patch_embed.", "rope_embed.")) or name in VIT_TOKEN_PARAMS
            no_decay = (param.ndim <= 1 or name in VIT_TOKEN_PARAMS
                        or name.endswith("bias") or "norm" in name or "gamma" in name or "fourier_w" in name)
        if layer0:
            layer_id = 0
        elif block:
            layer_id = int(block.group(1)) + 1
        else:
            layer_id = num_blocks + 1
        lr_scale = layer_decay ** (num_blocks + 1 - layer_id)
        if name.startswith("patch_embed."):  # Meta's code tests "patch_embed" in name: the same tensors here
            lr_scale *= patch_embed_lr_mult
        groups.setdefault((lr_scale, 0.0 if no_decay else weight_decay), []).append(param)
    return [{"params": groups[key], "weight_decay": key[1], "lr_scale": key[0]}
            for key in sorted(groups, key=lambda key: (-key[0], -key[1]))]

# DINOv2's block_chunks=N layout puts the blocks in N BlockChunk lists, named blocks.<chunk>.<block>.<rest>. Each chunk's list is padded with Identity() up to its first block, so <block> counts
# over the whole depth (blocks.1.6.* ... blocks.1.11.* for 4 chunks of 6). A flat key, blocks.<block>.<rest>, has a module name (attn, norm1, ls1 ...) where <block> is.
CHUNKED_BLOCK_KEY = re.compile(r"blocks\.(\d+)\.(\d+)\.(.+)")

def flatten_chunked_blocks(state_dict, depth):
    """(state_dict, 0) when no block key is chunked; otherwise (a copy with blocks.<chunk>.<block>.<rest> renamed blocks.<block>.<rest>, the number of chunks). Meta's RayDINO file is in this
    layout (4 chunks of 6 blocks, as its xray_dino_vitl16 hub entry builds it) and the model is built flat (block_chunks=0). <block> as DINOv2 writes it (over the whole depth) is kept; an index
    counted within its chunk (blocks.1.0.* ... blocks.1.5.*) becomes chunk * size + index, so either reading places every block. Raises RuntimeError for a layout it cannot place: chunked and
    flat block keys together, a chunk count that is not below the depth or does not divide it, a block outside its chunk, or two keys that land on one name."""
    found = {k: CHUNKED_BLOCK_KEY.fullmatch(k) for k in state_dict if k.startswith("blocks.")}
    if not any(found.values()):
        return state_dict, 0
    chunks = len({int(m.group(1)) for m in found.values() if m})
    flat = [k for k, m in found.items() if not m]
    if flat or chunks >= depth or depth % chunks:
        raise RuntimeError("block keys that fit neither the flat layout (blocks.<block>.*) nor a chunked one (blocks.<chunk>.<block>.*, fewer chunks than the {} blocks, and a count that divides them): "
                           "{} chunk(s), {} flat key(s) {}".format(depth, chunks, len(flat), flat[:3]))
    size = depth // chunks
    renamed = {}
    for key, value in state_dict.items():
        m = found.get(key)
        if not m:
            renamed[key] = value
            continue
        chunk, block = int(m.group(1)), int(m.group(2))
        index = chunk * size + block if block < size else block  # counted within the chunk, or over the whole depth (what DINOv2 writes)
        if not chunk * size <= index < (chunk + 1) * size or index >= depth:
            raise RuntimeError("{}: block {} is not one of chunk {}'s blocks {}..{}".format(key, block, chunk, chunk * size, (chunk + 1) * size - 1))
        name = "blocks.{}.{}".format(index, m.group(3))
        if name in renamed:
            raise RuntimeError("{} and another key both become {}".format(key, name))
        renamed[name] = value
    return renamed, chunks

def checkpoint_summary(state_dict, listed=32):
    """One log line on what a RayDINO file holds, so that a layout or shape surprise can be read off the log of the job that met it: the key count; every key that is not a block (the first
    `listed`) with its shape; the first and last block key, in block order, with theirs (blocks.3.23.* against blocks.3.5.* shows whether <block> counts over the depth or within the chunk);
    and the block layout the names show: chunks and blocks, flat, or mixed."""
    shape = lambda key: list(state_dict[key].shape) if hasattr(state_dict[key], "shape") else type(state_dict[key]).__name__
    order = lambda key: [(0, int(part), "") if part.isdecimal() else (1, 0, part) for part in key.split(".")]  # blocks.3.9.x before blocks.3.10.x
    blocks = [k for k in state_dict if k.startswith("blocks.")]
    others = [k for k in state_dict if not k.startswith("blocks.")]
    chunked = [m for m in map(CHUNKED_BLOCK_KEY.fullmatch, blocks) if m]
    if not blocks:
        layout = "no block keys"
    elif len(chunked) == len(blocks):
        layout = "{} chunks, {} blocks".format(len({m.group(1) for m in chunked}), len({m.group(1, 2) for m in chunked}))
    elif not chunked:
        layout = "flat, {} blocks".format(len({k.split(".")[1] for k in blocks}))
    else:
        layout = "mixed, {} chunked and {} flat block keys".format(len(chunked), len(blocks) - len(chunked))
    text = "RayDINO checkpoint: {} keys. Not blocks ({}): {}".format(len(state_dict), len(others), ", ".join("{} {}".format(k, shape(k)) for k in others[:listed]) or "none")
    if len(others) > listed:
        text += ", ... +{} more".format(len(others) - listed)
    if blocks:
        first, last = min(blocks, key=order), max(blocks, key=order)
        text += ". Blocks ({}): first {} {}, last {} {}".format(len(blocks), first, shape(first), last, shape(last))
    return text + "; layout " + layout

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
    elif args.model_name in ("vit_base_dinov2", "vit_base_raddino"):  # one architecture; --pretrained_weights picks DINOv2 or RAD-DINO
        model = ArkViTDinoV2(num_classes_list, args.projector_features, args.use_mlp, **DINOV2_VITB14_KWARGS)
    elif args.model_name == "vit_large_raydino":  # RayDINO: DINOv2's ViT-L/16, CLS ++ mean patch = 2 * 1024 = 2048-d
        model = ArkViTDinoV2(num_classes_list, args.projector_features, args.use_mlp, encoder_features=2048, **DINOV2_VITL16_KWARGS)

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
    elif args.model_name in ("vit_base_dinov2", "vit_base_raddino"):  # one architecture; --pretrained_weights picks DINOv2 or RAD-DINO
        model = ArkViTDinoV2(num_classes_list, args.projector_features, args.use_mlp, **DINOV2_VITB14_KWARGS)
    elif args.model_name == "vit_large_raydino":  # RayDINO: DINOv2's ViT-L/16, CLS ++ mean patch = 2 * 1024 = 2048-d
        model = ArkViTDinoV2(num_classes_list, args.projector_features, args.use_mlp, encoder_features=2048, **DINOV2_VITL16_KWARGS)
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
        if args.model_name == "vit_large_raydino":  # Meta's file is in DINOv2's block_chunks=4 layout and the model is built flat: rename the block keys before the key guard
            logger.warning(checkpoint_summary(state_dict))  # what the file holds, before anything is renamed or loaded; a warning, like the notice below
            state_dict, chunks = flatten_chunked_blocks(state_dict, len(model.blocks))
            if chunks:  # a warning: the extractor sets no logging up, and an info line would not reach its log
                logger.warning("RayDINO checkpoint is in DINOv2's chunked block layout ({} chunks of {} blocks, blocks.<chunk>.<block>.*): renamed its block keys to blocks.<block>.*".format(chunks, len(model.blocks) // chunks))

        try:
            msg = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as error:  # torch raises on a shape mismatch even with strict=False: name the checkpoint, and keep torch's lines (the offending key and both shapes)
            if args.model_name not in DINO_VIT_MODELS:
                raise
            raise RuntimeError("{} checkpoint does not match the backbone: {}".format(DINO_VIT_MODELS[args.model_name], error)) from error
        logger.info('Loaded with msg: {}'.format(msg))
        if args.model_name in DINO_VIT_MODELS:
            # Meta loads these checkpoints into DinoVisionTransformer with strict=True, so anything but the new
            # projector/heads missing means the backbone silently stayed at random init.
            bad_missing = [k for k in msg.missing_keys if not k.startswith(("projector.", "omni_heads."))]
            if bad_missing or msg.unexpected_keys:
                raise RuntimeError("{} checkpoint does not match the backbone: missing keys {} and unexpected keys {}"
                                   .format(DINO_VIT_MODELS[args.model_name], sorted(bad_missing), sorted(msg.unexpected_keys)))

    return model

def save_checkpoint(state,filename='model'):

    torch.save(state, filename + '.pth.tar')
