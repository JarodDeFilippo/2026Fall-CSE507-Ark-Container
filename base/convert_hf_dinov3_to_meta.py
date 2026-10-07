"""Convert a Hugging Face DINOv3 ViT-B/16 checkpoint (transformers DINOv3ViTModel, model.safetensors) into Meta's
state-dict layout, so it loads through build_omni_model's strict guard for vit_base_dinov3.

Run from base/:  python convert_hf_dinov3_to_meta.py <hf_dir_or_model.safetensors> <out.pth>
Needs `safetensors` in the Python that runs it.
"""
import os
import re
import sys

import torch
from safetensors.torch import load_file

from dinov3.models.vision_transformer import DinoVisionTransformer
from models import DINOV3_VITB16_KWARGS

# Meta key -> HF key: the inverse of the renames in transformers' convert_dinov3_vit_to_hf.py. The fused qkv is built
# separately below, and the final norm.{weight,bias} keeps its name.
RENAMES = [
    (r"^cls_token$", "embeddings.cls_token"),
    (r"^mask_token$", "embeddings.mask_token"),
    (r"^storage_tokens$", "embeddings.register_tokens"),
    (r"^patch_embed\.proj\.", "embeddings.patch_embeddings."),
    (r"^blocks\.(\d+)\.attn\.proj\.", r"layer.\1.attention.o_proj."),
    (r"^blocks\.(\d+)\.ls(\d)\.gamma$", r"layer.\1.layer_scale\2.lambda1"),
    (r"^blocks\.(\d+)\.mlp\.fc1\.", r"layer.\1.mlp.up_proj."),
    (r"^blocks\.(\d+)\.mlp\.fc2\.", r"layer.\1.mlp.down_proj."),
    (r"^blocks\.(\d+)\.(norm[12])\.", r"layer.\1.\2."),
]
# The only buffers allowed to come from the template: HF drops both (non-persistent inv_freq, bias_mask).
KEPT = r"rope_embed\.periods|blocks\.\d+\.attn\.qkv\.bias_mask"

if len(sys.argv) != 3:
    sys.exit(__doc__)
src, dst = sys.argv[1:]
path = os.path.join(src, "model.safetensors") if os.path.isdir(src) else src
# The hub file and save_pretrained name layers "layer.N"; transformers 5.x state_dict() says "model.layer.N". Take both.
hf = {re.sub(r"^model\.(?=layer\.)", "", k): t for k, t in load_file(path).items()}
n_hf = len(hf)
assert "embeddings.cls_token" in hf, "not a DINOv3 HF checkpoint layout, keys start with: {}".format(sorted(hf)[:4])

model = DinoVisionTransformer(**DINOV3_VITB16_KWARGS)
model.init_weights()  # fills rope_embed.periods and the k-bias masks (bias_mask), neither of which is in the HF file
template = model.state_dict()
params = {k for k, _ in model.named_parameters()}
kept = [k for k in template if k not in params]
assert all(re.fullmatch(KEPT, k) for k in kept), kept

out = {}
for k, ref in template.items():
    if k in kept:
        out[k] = ref
        continue
    m = re.fullmatch(r"blocks\.(\d+)\.attn\.qkv\.(weight|bias)", k)
    if m:
        n, w = m.groups()
        q, v = (hf.pop(f"layer.{n}.attention.{p}_proj.{w}") for p in "qv")
        # HF has no k bias (Meta masks it to zero via bias_mask), so a zero k slot is exact.
        kk = hf.pop(f"layer.{n}.attention.k_proj.weight") if w == "weight" else torch.zeros_like(q)
        t = torch.cat([q, kk, v])
    else:
        name = k
        for pat, rep in RENAMES:
            name = re.sub(pat, rep, name)
        t = hf.pop(name)
        if k == "mask_token":
            t = t.squeeze(1)  # HF stores (1, 1, C), Meta (1, C)
    assert t.shape == ref.shape, (k, tuple(t.shape), tuple(ref.shape))
    out[k] = t.to(ref.dtype)
assert not hf, "HF tensors not consumed: {}".format(sorted(hf))
assert all(torch.isfinite(t).all() for t in out.values()), "non-finite values in the converted state dict"

os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
torch.save(out, dst)  # plain dict of tensors, no wrapper key
print("HF tensors consumed:        {}".format(n_hf))
print("Meta params overwritten:    {}".format(len(out) - len(kept)))
print("Buffers kept from template: {} (rope_embed.periods, blocks.*.attn.qkv.bias_mask)".format(len(kept)))
print("Wrote {} ({} keys)".format(dst, len(out)))
