"""CPU checks for the larger stock DINOv3 backbones in the Ark wrapper: vit_large_dinov3, vit_huge_plus_dinov3 and vit_7b_dinov3
(ArkViT, encoder_features = 2 * embed_dim). Run from base/: python test_dinov3_larger_vits.py
Only vit_large_dinov3 is built with real tensors and run forward. The other two are built on torch's meta device (the real builder, shapes only, nothing allocated: the 7B is about 27 GB in fp32)."""
import os
import tempfile
from types import SimpleNamespace

import torch

from ark_finetune.embedding_mode import embedding_dim
from dinov3.layers import Mlp, SwiGLUFFN
from models import (ArkViT, DINO_VIT_MODELS, DINOV3_VITB16_KWARGS, DINOV3_VITH16PLUS_KWARGS, DINOV3_VITL16_KWARGS, DINOV3_VIT7B16_KWARGS,
                    build_omni_model)

# name -> (config, the keys that differ from the ViT-B's, blocks, embed_dim, FFN class, parameters), written out here rather than read off the config under test.
# The parameters are counted by hand from Meta's hub configs (dinov3/hub/backbones.py); they match the README's 300M (L), 840M (H+) and 6,716M (7B).
SIZES = {
    "vit_large_dinov3": (DINOV3_VITL16_KWARGS, {"embed_dim", "depth", "num_heads"}, 24, 1024, Mlp, 303154176),
    "vit_huge_plus_dinov3": (DINOV3_VITH16PLUS_KWARGS, {"embed_dim", "depth", "num_heads", "ffn_ratio", "ffn_layer"}, 32, 1280, SwiGLUFFN, 840633600),
    "vit_7b_dinov3": (DINOV3_VIT7B16_KWARGS, {"embed_dim", "depth", "num_heads", "ffn_ratio", "qkv_bias", "ffn_layer", "untie_global_and_local_cls_norm"}, 40, 4096, SwiGLUFFN, 6716043264),
}


def ark_args(model_name, projector_features=None, pretrained_weights=None):
    return SimpleNamespace(model_name=model_name, projector_features=projector_features, use_mlp=False, pretrained_weights=pretrained_weights)


def rejects(exception, call, expected=None):
    """The exception message when call() raises `exception`; fails when it returns."""
    try:
        call()
    except exception as error:
        assert expected is None or expected in str(error), (expected, str(error))
        return str(error)
    raise AssertionError("expected {}".format(exception.__name__))


def check(name, model):
    """The structure the name stands for, whichever device the model is on."""
    config, _, blocks, embed, ffn, params = SIZES[name]
    assert type(model) is ArkViT and len(model.blocks) == model.n_blocks == blocks and model.embed_dim == config["embed_dim"] == embed
    assert model.projector is None and model.num_features == embedding_dim(name) == 2 * embed  # no projector built: the encoder's width, CLS ++ mean patch
    assert type(model.blocks[0].mlp) is ffn and (model.blocks[0].attn.qkv.bias is not None) == config["qkv_bias"]
    assert (model.local_cls_norm is not None) == (name == "vit_7b_dinov3")  # only the 7B's hub entry builds the untied local CLS norm
    assert sum(p.numel() for p in model.parameters()) == params


# (a) the configs are the ViT-B's with Meta's sizes and nothing else changed (the 7B adds one key); all three are registered under the DINOv3 label, so the strict key guard covers them
for name, (config, differing, *_) in SIZES.items():
    assert set(DINOV3_VITB16_KWARGS) <= set(config) and {k for k, v in config.items() if DINOV3_VITB16_KWARGS.get(k) != v} == differing, name
    assert DINO_VIT_MODELS[name] == "DINOv3", name
print("OK (a) configs: vit_large_dinov3 differs from the ViT-B's in embed_dim, depth and num_heads only; the H+ adds ffn_ratio 6.0 and the swiglu FFN; the 7B adds ffn_ratio 3, no qkv bias, swiglu64 "
      "and untie_global_and_local_cls_norm; all three are in DINO_VIT_MODELS as DINOv3")

# (b) vit_large_dinov3 with real tensors: 24 blocks of width 1024, 2048-d features = CLS ++ mean patch at 224 px
large = build_omni_model(ark_args("vit_large_dinov3"), []).eval()
check("vit_large_dinov3", large)
x = torch.zeros(1, 3, 224, 224)
with torch.no_grad():
    embeddings = large.generate_embeddings(x, after_proj=False)
    tokens = super(ArkViT, large).forward_features(x)  # Meta's dict
assert embeddings.shape == (1, 2048) and torch.isfinite(embeddings).all()  # NaN would mean the k-bias masks were never initialised
assert torch.equal(embeddings[:, :1024], tokens["x_norm_clstoken"]) and torch.equal(embeddings[:, 1024:], tokens["x_norm_patchtokens"].mean(dim=1))
assert tokens["x_norm_patchtokens"].shape == (1, 196, 1024) and tokens["x_storage_tokens"].shape == (1, 4, 1024)  # 14 x 14 patches at 224 px, 4 storage tokens
del large, embeddings, tokens
print("OK (b) vit_large_dinov3 built without weights: 24 blocks of width 1024, 303,154,176 parameters, an MLP FFN; features (1, 2048) = CLS ++ mean patch, finite, at 224 px (196 patches, 4 storage tokens)")

# (c) the other two on the meta device: the same structure checks, the parameter counts and the feature widths 2560 / 8192; no forward
for name in ("vit_huge_plus_dinov3", "vit_7b_dinov3"):
    with torch.device("meta"):
        model = build_omni_model(ark_args(name), [])
    assert all(t.is_meta for t in (*model.parameters(), *model.buffers()))  # nothing was allocated
    check(name, model)
print("OK (c) vit_huge_plus_dinov3 / vit_7b_dinov3 built on the meta device (no tensors, no forward): 32 / 40 blocks of width 1280 / 4096, a SwiGLU FFN, "
      "840,633,600 / 6,716,043,264 parameters, no qkv bias and a local_cls_norm for the 7B only; embedding widths 2560 / 8192 = embedding_dim()")

# (d) a ViT-B file (the likeliest wrong one: the stock ViT-B/16's) under --model vit_large_dinov3 is refused under the DINOv3 label, with torch's size-mismatch lines (the key and both shapes)
vit_b = build_omni_model(ark_args("vit_base_dinov3"), [])
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "vit_b.pth")
    torch.save(vit_b.state_dict(), path)  # no projector, no heads were built: a backbone-only file
    del vit_b
    message = rejects(RuntimeError, lambda: build_omni_model(ark_args("vit_large_dinov3", pretrained_weights=path), []))
assert message.startswith("DINOv3 checkpoint does not match the backbone"), message
assert "size mismatch for cls_token" in message and "torch.Size([1, 1, 768])" in message and "torch.Size([1, 1, 1024])" in message, message
print("OK (d) a ViT-B/16 state dict loaded as --pretrained_weights for vit_large_dinov3 raises RuntimeError 'DINOv3 checkpoint does not match the backbone: ...' with torch's size-mismatch line for cls_token "
      "(torch.Size([1, 1, 768]) against torch.Size([1, 1, 1024]))")

print("ALL OK")
