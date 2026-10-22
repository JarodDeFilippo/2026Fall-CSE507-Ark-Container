"""Convert a Hugging Face DINOv3 ViT checkpoint (transformers DINOv3ViTModel: one model.safetensors, or the shards of a
model.safetensors.index.json) into Meta's state-dict layout, so it loads through build_omni_model's strict guard for the same
--model: vit_base_dinov3, vit_large_dinov3, vit_huge_plus_dinov3 or vit_7b_dinov3.

Run from base/:  python convert_hf_dinov3_to_meta.py [--model vit_base_dinov3] <hf_dir_or_file> <out.pth>
<hf_dir_or_file> is the HF directory (its model.safetensors, else its model.safetensors.index.json and the shards that lists, loaded
one at a time) or one .safetensors / .index.json file. Reads them with the `safetensors` package, or with a built-in stdlib reader when the Python has none
(the Sol container's has none and cannot install one); the first line printed says which.
Tensors are written in the HF file's dtype (no upcast). The buffers HF drops (rope periods, k-bias masks) come from the Meta model, fp32.
Output names (Meta's release names; the hash suffix is Meta's, not this file's): dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth,
dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth, dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth, dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth.
The 7B (about 27 GB in fp32) converts on a CPU node with RAM for one checkpoint plus the fused qkv weights built in memory (about 8 GB);
the Meta model itself is only shapes (torch's meta device).
"""
import argparse
import json
import math
import mmap
import os
import re
import struct
import sys

import torch

from dinov3.models.vision_transformer import DinoVisionTransformer
from models import DINOV3_VITB16_KWARGS, DINOV3_VITH16PLUS_KWARGS, DINOV3_VITL16_KWARGS, DINOV3_VIT7B16_KWARGS

MODELS = {"vit_base_dinov3": DINOV3_VITB16_KWARGS, "vit_large_dinov3": DINOV3_VITL16_KWARGS,
          "vit_huge_plus_dinov3": DINOV3_VITH16PLUS_KWARGS, "vit_7b_dinov3": DINOV3_VIT7B16_KWARGS}

# Meta key -> HF key: the inverse of the renames in transformers' convert_dinov3_vit_to_hf.py. The fused qkv is built
# separately below, and the final norm.{weight,bias} keeps its name. The FFN is one of two: an MLP (fc1 -> up_proj, fc2 ->
# down_proj; ViT-B, ViT-L) or a SwiGLU (w1 -> gate_proj, the SiLU branch; w2 -> up_proj; w3 -> down_proj; ViT-H+, 7B).
RENAMES = [
    (r"^cls_token$", "embeddings.cls_token"),
    (r"^mask_token$", "embeddings.mask_token"),
    (r"^storage_tokens$", "embeddings.register_tokens"),
    (r"^patch_embed\.proj\.", "embeddings.patch_embeddings."),
    (r"^blocks\.(\d+)\.attn\.proj\.", r"layer.\1.attention.o_proj."),
    (r"^blocks\.(\d+)\.ls(\d)\.gamma$", r"layer.\1.layer_scale\2.lambda1"),
    (r"^blocks\.(\d+)\.mlp\.fc1\.", r"layer.\1.mlp.up_proj."),
    (r"^blocks\.(\d+)\.mlp\.fc2\.", r"layer.\1.mlp.down_proj."),
    (r"^blocks\.(\d+)\.mlp\.w1\.", r"layer.\1.mlp.gate_proj."),
    (r"^blocks\.(\d+)\.mlp\.w2\.", r"layer.\1.mlp.up_proj."),
    (r"^blocks\.(\d+)\.mlp\.w3\.", r"layer.\1.mlp.down_proj."),
    (r"^blocks\.(\d+)\.(norm[12])\.", r"layer.\1.\2."),
]
# The only buffers allowed to come from the template: HF drops both (non-persistent inv_freq, bias_mask). The 7B has no qkv bias, so no bias_mask.
KEPT = r"rope_embed\.periods|blocks\.\d+\.attn\.qkv\.bias_mask"

# safetensors dtype names -> torch dtypes: the ten this reader handles. The format has others (float8, unsigned 16/32/64 bit): refused.
SAFETENSORS_DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64,
                      "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


def load_file_stdlib(path):
    """safetensors.torch.load_file with the standard library alone: {name: tensor} in the header's order. The format: an 8-byte little-endian header length N,
    N bytes of JSON {name: {"dtype", "shape", "data_offsets": [begin, end]}, optional "__metadata__"}, then the raw bytes that the offsets count from.
    The tensors are views of one copy-on-write mmap of the file, not copies: pages are read as they are used and the file is never written. Never close() the mmap:
    torch.frombuffer holds the mmap object, not a buffer export, so a close would unmap the memory under live tensors. Dropping the last tensor releases it."""
    if sys.byteorder != "little":
        raise RuntimeError("safetensors data is little-endian; torch.frombuffer reads native order")
    with open(path, "rb") as f:
        if os.fstat(f.fileno()).st_size < 8:
            raise ValueError("{}: shorter than the 8-byte header length".format(path))
        buffer = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)  # copy-on-write is writable, so torch.frombuffer does not warn about a read-only buffer
    (header_len,) = struct.unpack_from("<Q", buffer)
    start = 8 + header_len  # the first byte of tensor data
    if start > len(buffer):
        raise ValueError("{}: header length {} runs past the end of the file".format(path, header_len))
    header = json.loads(buffer[8:start])
    header.pop("__metadata__", None)
    tensors = {}
    for name, info in header.items():
        if info["dtype"] not in SAFETENSORS_DTYPES:
            raise ValueError("{}: {} has dtype {}, but only {} are read".format(path, name, info["dtype"], ", ".join(SAFETENSORS_DTYPES)))
        dtype, shape = SAFETENSORS_DTYPES[info["dtype"]], info["shape"]
        begin, end = info["data_offsets"]
        numel = math.prod(shape)
        # ponytail: holes and overlaps between tensors are not refused (the package does); this bounds check keeps every read inside the file
        if not 0 <= begin <= end <= len(buffer) - start or end - begin != numel * torch.empty((), dtype=dtype).element_size():
            raise ValueError("{}: {} has data_offsets {}, which do not hold {} {} values of shape {}".format(path, name, [begin, end], numel, info["dtype"], shape))
        tensors[name] = torch.frombuffer(buffer, dtype=dtype, count=numel, offset=start + begin).reshape(shape) if numel else torch.empty(shape, dtype=dtype)
    return tensors


try:
    from safetensors.torch import load_file
    READER = "safetensors package"
except ImportError:  # the Sol container's Python has none and cannot install one
    load_file, READER = load_file_stdlib, "built-in stdlib reader (safetensors cannot be imported here)"


def read_hf(src):
    """The HF checkpoint's tensors by name, from one file or from the shards of an index, one shard at a time."""
    if os.path.isdir(src):
        single = os.path.join(src, "model.safetensors")
        src = single if os.path.exists(single) else os.path.join(src, "model.safetensors.index.json")
    if src.endswith(".json"):
        with open(src) as f:
            weight_map = json.load(f)["weight_map"]
        tensors = {}
        for shard in sorted(set(weight_map.values())):
            part = load_file(os.path.join(os.path.dirname(src), shard))
            if part.keys() != {k for k, s in weight_map.items() if s == shard}:
                raise ValueError("shard {} does not hold exactly the tensors that {} lists for it".format(shard, src))
            tensors.update(part)
    else:
        tensors = load_file(src)
    # The hub file and save_pretrained name layers "layer.N"; transformers 5.x state_dict() says "model.layer.N". Take both.
    return {re.sub(r"^model\.(?=layer\.)", "", k): t for k, t in tensors.items()}


def meta_template(kwargs):
    """The Meta model's state dict: params as shapes only (torch's meta device, no memory), the buffers HF drops filled in. -> (state dict, buffer keys)"""
    with torch.device("meta"):
        model = DinoVisionTransformer(**kwargs)
    params = {k for k, _ in model.named_parameters()}
    kept = [k for k in model.state_dict() if k not in params]
    assert all(re.fullmatch(KEPT, k) for k in kept), kept
    for m in model.modules():  # materialize only the buffers (a few dozen floats, 3 * embed_dim per k-bias mask), empty
        for name, buf in list(m.named_buffers(recurse=False)):
            setattr(m, name, torch.empty_like(buf, device="cpu"))
    model.init_weights()  # fills rope_embed.periods and the k-bias masks (bias_mask), neither of which is in the HF file; the params stay on meta
    return model.state_dict(), kept


def convert(src, name):
    """The HF checkpoint at src as the Meta state dict of --model `name`. -> (state dict, the keys that came from the template, HF tensors consumed)"""
    hf = read_hf(src)
    n_hf = len(hf)
    if "embeddings.cls_token" not in hf:
        raise ValueError("{} is not a DINOv3 HF checkpoint layout, keys start with: {}".format(src, sorted(hf)[:4]))
    template, kept = meta_template(MODELS[name])

    def take(hf_key, meta_key):
        if hf_key not in hf:
            raise ValueError("the HF checkpoint has no {} (the source of Meta's {} for --model {})".format(hf_key, meta_key, name))
        return hf.pop(hf_key)

    out = {}
    for k, ref in template.items():
        if k in kept:
            out[k] = ref
            continue
        m = re.fullmatch(r"blocks\.(\d+)\.attn\.qkv\.(weight|bias)", k)
        if m:
            n, w = m.groups()
            source = "layer.{}.attention.{{q,k,v}}_proj.{}".format(n, w)
            q, v = (take(f"layer.{n}.attention.{p}_proj.{w}", k) for p in "qv")
            # HF has no k bias (Meta masks it to zero via bias_mask), so a zero k slot is exact.
            kk = take(f"layer.{n}.attention.k_proj.weight", k) if w == "weight" else torch.zeros_like(q)
            t = torch.cat([q, kk, v])
        else:
            source = k
            for pat, rep in RENAMES:
                source = re.sub(pat, rep, source)
            t = take(source, k)
            if k == "mask_token":
                t = t.squeeze(1)  # HF stores (1, 1, C), Meta (1, C)
        if t.shape != ref.shape:
            raise ValueError("the HF checkpoint's {} has shape {}, but --model {} expects {} for Meta's {}".format(
                source, tuple(t.shape), name, tuple(ref.shape), k))
        out[k] = t  # the HF file's dtype, not upcast
    if hf:
        raise ValueError("{} HF tensors not consumed by --model {}, first: {}".format(len(hf), name, sorted(hf)[:8]))
    if not all(torch.isfinite(t).all() for t in out.values()):
        raise ValueError("non-finite values in the converted state dict")
    return out, kept, n_hf


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="vit_base_dinov3", choices=list(MODELS), help="the build_omni_model name the file is for (default: %(default)s)")
    parser.add_argument("src", help="the HF directory, or one .safetensors / .index.json file")
    parser.add_argument("dst", help="the output .pth")
    args = parser.parse_args(argv)
    print("safetensors reader: {}".format(READER))
    try:
        out, kept, n_hf = convert(args.src, args.model)
    except ValueError as error:
        sys.exit("error: {}".format(error))

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(out, args.dst)  # plain dict of tensors, no wrapper key
    params = [t for k, t in out.items() if k not in kept]
    buffers = [out[k] for k in kept]
    dtypes = lambda tensors: ", ".join(sorted({str(t.dtype) for t in tensors}))
    print("Model:                      {}".format(args.model))
    print("HF tensors consumed:        {}".format(n_hf))
    print("Meta params overwritten:    {} ({:,} elements)".format(len(params), sum(t.numel() for t in params)))
    print("Buffers kept from template: {} ({})".format(len(buffers), ", ".join(dict.fromkeys(re.sub(r"\.\d+\.", ".*.", k) for k in kept))))
    print("dtype written:              params {}, buffers {}".format(dtypes(params), dtypes(buffers)))
    print("Bytes written:              {:,} of tensors, {:,} on disk".format(sum(t.numel() * t.element_size() for t in out.values()), os.path.getsize(args.dst)))
    print("Wrote {} ({} keys)".format(args.dst, len(out)))


if __name__ == "__main__":
    main()
