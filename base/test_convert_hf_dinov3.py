"""CPU checks for convert_hf_dinov3_to_meta.py: Hugging Face DINOv3 checkpoints convert to Meta's layout and load through build_omni_model's strict key guard,
for vit_base_dinov3, vit_large_dinov3, vit_huge_plus_dinov3 and vit_7b_dinov3. Run from base/: python test_convert_hf_dinov3.py
No real HF file is needed. Each model is built in Meta's layout with a distinct random value in every parameter, exported to HF names with transformers' own forward mapping
(HF_RENAMES below, from its convert_dinov3_vit_to_hf.py), converted back by the script and compared with torch.equal, then loaded through build_omni_model.
The ViT-B and ViT-L are full size; the ViT-H+ and the 7B keep their real per-layer shapes but have 2 blocks (the 7B's 40 are about 27 GB in fp32).
The script reads safetensors with the package when the Python has it, else with its own stdlib reader (the Sol container has no package): (g)-(i) check that reader, and (i) runs this whole file
again in a subprocess with the package hidden. Real file for (h): $HF_DINOV3_VITB_DIR/model.safetensors (default ../../dinov3-vitb16-hf, the ViT-B HF download); $ARK_WEIGHTS_DIR (default ../weights)
may hold the stock dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth, which the converted ViT-B is then compared with. A check whose file is missing prints a SKIP line and the rest still run."""
import contextlib
import filecmp
import gc
import io
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch
from safetensors.torch import load_file as package_load_file, save_file

# The second run (started by (i) at the end of the first): the package is hidden from the converter, as in the Sol container. save_file, imported above, still writes the fixtures.
HIDDEN = os.environ.get("HIDE_SAFETENSORS") == "1"
if HIDDEN:
    sys.modules["safetensors"] = sys.modules["safetensors.torch"] = None

import convert_hf_dinov3_to_meta as converter  # after the hiding above, on purpose
import models
from dinov3.models.vision_transformer import DinoVisionTransformer
from models import DINOV3_VITB16_KWARGS, DINOV3_VITH16PLUS_KWARGS, DINOV3_VITL16_KWARGS, DINOV3_VIT7B16_KWARGS, build_omni_model

torch.manual_seed(0)
assert converter.load_file is (converter.load_file_stdlib if HIDDEN else package_load_file) and ("built-in" in converter.READER) == HIDDEN  # the package when it can be imported, the built-in reader when not

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WEIGHTS = os.environ.get("ARK_WEIGHTS_DIR", os.path.join(REPO, "weights"))
HF_VITB = os.path.abspath(os.path.join(os.environ.get("HF_DINOV3_VITB_DIR", os.path.join(REPO, "..", "dinov3-vitb16-hf")), "model.safetensors"))
STOCK = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
# a fresh interpreter with the package hidden that runs the script's main: what `python convert_hf_dinov3_to_meta.py <src> <dst>` does in the Sol container
HIDE_AND_CONVERT = "import sys; sys.modules['safetensors'] = sys.modules['safetensors.torch'] = None; import convert_hf_dinov3_to_meta as c; c.main(sys.argv[1:])"

# transformers' convert_dinov3_vit_to_hf.py (Apache-2.0), Meta -> HF: its ORIGINAL_TO_CONVERTED_KEY_MAPPING in order, then split_qkv and the skips of convert_and_test_dinov3_checkpoint (to_hf below)
HF_RENAMES = [
    (r"cls_token", r"embeddings.cls_token"),
    (r"mask_token", r"embeddings.mask_token"),
    (r"storage_tokens", r"embeddings.register_tokens"),
    (r"patch_embed.proj", r"embeddings.patch_embeddings"),
    (r"periods", r"inv_freq"),
    (r"rope_embed", r"rope_embeddings"),
    (r"blocks.(\d+).attn.proj", r"layer.\1.attention.o_proj"),
    (r"blocks.(\d+).attn.", r"layer.\1.attention."),
    (r"blocks.(\d+).ls(\d+).gamma", r"layer.\1.layer_scale\2.lambda1"),
    (r"blocks.(\d+).mlp.fc1", r"layer.\1.mlp.up_proj"),
    (r"blocks.(\d+).mlp.fc2", r"layer.\1.mlp.down_proj"),
    (r"blocks.(\d+).mlp", r"layer.\1.mlp"),
    (r"blocks.(\d+).norm", r"layer.\1.norm"),
    (r"w1", r"gate_proj"),
    (r"w2", r"up_proj"),
    (r"w3", r"down_proj"),
]

# The HF tensor names of the real files (their safetensors headers: ViT-B, ViT-L and ViT-H+, read 2026-10-22; the 7B's layout is the HF config's: no q/v/k biases, a gated FFN), written out
# here rather than derived from the mapping above: 7 top-level tensors and per layer the names below. No k bias in any of them, no local CLS norm in any of them.
TOP = ("embeddings.cls_token", "embeddings.mask_token", "embeddings.register_tokens", "embeddings.patch_embeddings.weight", "embeddings.patch_embeddings.bias", "norm.weight", "norm.bias")
COMMON = ("attention.k_proj.weight", "attention.q_proj.weight", "attention.v_proj.weight", "attention.o_proj.weight", "attention.o_proj.bias",
          "layer_scale1.lambda1", "layer_scale2.lambda1", "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias")
QV_BIAS = ("attention.q_proj.bias", "attention.v_proj.bias")
MLP = ("mlp.up_proj.weight", "mlp.up_proj.bias", "mlp.down_proj.weight", "mlp.down_proj.bias")
GATED = MLP + ("mlp.gate_proj.weight", "mlp.gate_proj.bias")
# name -> (the model's kwargs here, the layer prefix of the file, shards, per-layer tensors, HF shapes from the real headers, a --model the file must be refused under)
# "" is the hub's "layer.N", "model." transformers 5.x's state_dict(). The 7B's shapes are the config's (hidden 4096, intermediate_size 8192; its header was not read).
CASES = {
    "vit_base_dinov3": (DINOV3_VITB16_KWARGS, "", 1, COMMON + QV_BIAS + MLP,
                        {"embeddings.cls_token": (1, 1, 768), "embeddings.patch_embeddings.weight": (768, 3, 16, 16), "layer.0.mlp.up_proj.weight": (3072, 768), "layer.0.mlp.down_proj.weight": (768, 3072)},
                        "vit_large_dinov3"),
    "vit_large_dinov3": (DINOV3_VITL16_KWARGS, "", 1, COMMON + QV_BIAS + MLP,
                         {"embeddings.cls_token": (1, 1, 1024), "embeddings.mask_token": (1, 1, 1024), "embeddings.register_tokens": (1, 4, 1024), "embeddings.patch_embeddings.weight": (1024, 3, 16, 16),
                          "layer.0.mlp.up_proj.weight": (4096, 1024), "layer.0.mlp.down_proj.weight": (1024, 4096)},
                         "vit_huge_plus_dinov3"),
    "vit_huge_plus_dinov3": (dict(DINOV3_VITH16PLUS_KWARGS, depth=2), "model.", 1, COMMON + QV_BIAS + GATED,
                             {"embeddings.cls_token": (1, 1, 1280), "layer.0.mlp.gate_proj.weight": (5120, 1280), "layer.0.mlp.gate_proj.bias": (5120,), "layer.0.mlp.up_proj.weight": (5120, 1280),
                              "layer.0.mlp.up_proj.bias": (5120,), "layer.0.mlp.down_proj.weight": (1280, 5120), "layer.0.mlp.down_proj.bias": (1280,)},
                             "vit_7b_dinov3"),
    "vit_7b_dinov3": (dict(DINOV3_VIT7B16_KWARGS, depth=2), "", 2, COMMON + GATED,
                      {"embeddings.cls_token": (1, 1, 4096), "layer.0.mlp.gate_proj.weight": (8192, 4096), "layer.0.mlp.up_proj.weight": (8192, 4096), "layer.0.mlp.down_proj.weight": (4096, 8192),
                       "layer.0.attention.q_proj.weight": (4096, 4096)},
                      "vit_base_dinov3"),
}
KWARGS_NAME = {"vit_base_dinov3": "DINOV3_VITB16_KWARGS", "vit_large_dinov3": "DINOV3_VITL16_KWARGS", "vit_huge_plus_dinov3": "DINOV3_VITH16PLUS_KWARGS", "vit_7b_dinov3": "DINOV3_VIT7B16_KWARGS"}
# tensors in the real files (their headers): the layer lists above reproduce all of them at the real depths. The 7B's 7 + 40 * 17 is the layout's, its header not read.
REAL_TENSORS = {"vit_base_dinov3": 211, "vit_large_dinov3": 415, "vit_huge_plus_dinov3": 615, "vit_7b_dinov3": 687}
HEADER_READ = ("vit_base_dinov3", "vit_large_dinov3", "vit_huge_plus_dinov3")
assert all(len(TOP) + getattr(models, KWARGS_NAME[name])["depth"] * len(case[3]) == REAL_TENSORS[name] for name, case in CASES.items())
TINY = dict(DINOV3_VITB16_KWARGS, embed_dim=64, depth=2, num_heads=4)  # a few thousand parameters for the refusal and file-form checks; the gated one has the same FFN width as the MLP one
TINY_GATED = dict(TINY, ffn_ratio=6.0, ffn_layer="swiglu")


def rejects(exception, call, expected=None):
    """The exception message when call() raises `exception`; fails when it returns. `expected` is a list of strings the message must hold."""
    try:
        call()
    except exception as error:
        message = str(error) if exception is not SystemExit else str(error.code)
        assert all(part in message for part in (expected or ())), (expected, message)
        return message
    raise AssertionError("expected {}".format(exception.__name__))


@contextlib.contextmanager
def patched(obj, attr, value):
    old = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old)


def reference(kwargs, dtype=torch.float32):
    """A Meta-layout state dict: a distinct random value in every parameter (a swapped key or q/k/v order cannot round-trip), real rope periods and k-bias masks, k third of the qkv bias zero."""
    model = DinoVisionTransformer(**kwargs)
    model.init_weights()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_()
        for blk in model.blocks:
            if blk.attn.qkv.bias is not None:
                o = blk.attn.qkv.out_features
                blk.attn.qkv.bias[o // 3 : 2 * o // 3] = 0  # masked off by bias_mask; HF has no k bias, and Meta's init (and so a file saved from it) holds zeros there
    return {k: t if re.fullmatch(converter.KEPT, k) else t.to(dtype) for k, t in model.state_dict().items()}


def to_hf(state, prefix):
    """Meta state dict -> the HF tensors, the way transformers' script does it. `prefix` goes before layer.N."""
    state = dict(state)
    for key in [k for k in state if "qkv" in k]:  # split_qkv; clone: safetensors refuses tensors that share memory
        for part, t in zip("qkv", torch.chunk(state.pop(key), 3, dim=0)):
            state[key.replace("qkv", part + "_proj")] = t.clone()
    hf = {}
    for key, t in state.items():
        new = key
        for pat, rep in HF_RENAMES:
            new = re.sub(pat, rep, new)
        if "bias_mask" in key or "attn.k_proj.bias" in key or "local_cls_norm" in key or "inv_freq" in new:
            continue  # what HF drops
        if "embeddings.mask_token" in new:
            t = t.unsqueeze(1)
        hf[prefix + new if new.startswith("layer.") else new] = t
    return hf


def write_hf(hf, directory, shards=1):
    """model.safetensors, or `shards` shards and model.safetensors.index.json (keys dealt out in turn, so a layer's q, k and v land in different shards). -> the index path or the file path"""
    if shards == 1:
        path = os.path.join(directory, "model.safetensors")
        save_file(hf, path, metadata={"format": "pt"})
        return path
    keys, weight_map = sorted(hf), {}
    for i in range(shards):
        shard = "model-{:05d}-of-{:05d}.safetensors".format(i + 1, shards)
        save_file({k: hf[k] for k in keys[i::shards]}, os.path.join(directory, shard), metadata={"format": "pt"})
        weight_map.update({k: shard for k in keys[i::shards]})
    path = os.path.join(directory, "model.safetensors.index.json")
    with open(path, "w") as f:
        json.dump({"metadata": {"total_size": sum(t.numel() * t.element_size() for t in hf.values())}, "weight_map": weight_map}, f)
    return path


def convert_main(argv, name, kwargs):
    """converter.main(argv) with --model `name` built from `kwargs`. -> what it printed"""
    printed = io.StringIO()
    with patched(converter, "MODELS", dict(converter.MODELS, **{name: kwargs})), contextlib.redirect_stdout(printed):
        converter.main(argv)
    return printed.getvalue()


def convert(src, name, kwargs):
    with patched(converter, "MODELS", dict(converter.MODELS, **{name: kwargs})):
        return converter.convert(src, name)


def guard(name, kwargs, path):
    """build_omni_model with --pretrained_weights `path`, its strict key guard included, for the model built from `kwargs`."""
    args = SimpleNamespace(model_name=name, projector_features=None, use_mlp=False, pretrained_weights=path)
    with patched(models, KWARGS_NAME[name], kwargs):
        return build_omni_model(args, [])


# (a)-(c) per model: the HF export has the real layout; the script converts it back exactly; the result passes the strict guard
for name, (kwargs, prefix, shards, layer_keys, shapes, wrong) in CASES.items():
    ref = reference(kwargs)
    hf = to_hf(ref, prefix)
    layout = lambda depth: set(TOP) | {"{}layer.{}.{}".format(prefix, n, s) for n in range(depth) for s in layer_keys}
    expected = layout(kwargs["depth"])
    assert set(hf) == expected and len(hf) == len(TOP) + kwargs["depth"] * len(layer_keys), name
    real = getattr(models, KWARGS_NAME[name])  # the real config, on the meta device: no tensors, the same names at the real depth
    with torch.device("meta"):
        real_hf = to_hf(DinoVisionTransformer(**real).state_dict(), prefix)
    assert set(real_hf) == layout(real["depth"]) and len(real_hf) == REAL_TENSORS[name], name
    for key, shape in shapes.items():
        hf_key = prefix + key if key.startswith("layer.") else key
        assert tuple(hf[hf_key].shape) == shape, (name, hf_key, tuple(hf[hf_key].shape), shape)
    assert all(t.dtype == torch.float32 for t in hf.values())
    print("OK (a) {}: the export of a {}-block model holds the {} tensors of the real layout ({} top-level + {} per layer, {}layer.N), with the real shapes for {} of them; "
          "at the real depth of {} blocks the config exports {} tensors{}".format(name, kwargs["depth"], len(hf), len(TOP), len(layer_keys), prefix, len(shapes), real["depth"], len(real_hf),
                                                                                  ", as the real file" if name in HEADER_READ else " (the layout's count: the 7B's header was not read)"))

    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "hf")
        os.mkdir(source)
        written = write_hf(hf, source, shards)
        del hf
        dst = os.path.join(tmp, "out", "converted.pth")  # the script makes the directory
        # the ViT-B goes in as one file path with no --model (the old command line); the others as the directory, with --model
        argv = [written, dst] if name == "vit_base_dinov3" else ["--model", name, source, dst]
        summary = convert_main(argv, name, kwargs)
        out = torch.load(dst, map_location="cpu", weights_only=True, mmap=True)
        assert list(out) == list(ref), name  # the same keys, in Meta's order
        assert all(out[k].dtype == ref[k].dtype and out[k].shape == ref[k].shape and torch.equal(out[k], ref[k]) for k in ref), name
        n_kept = sum(bool(re.fullmatch(converter.KEPT, k)) for k in ref)
        n_params = sum(t.numel() for k, t in ref.items() if not re.fullmatch(converter.KEPT, k))
        n_bytes = sum(t.numel() * t.element_size() for t in ref.values())
        kept_names = "rope_embed.periods, blocks.*.attn.qkv.bias_mask" if kwargs["qkv_bias"] else "rope_embed.periods"
        for line in ("safetensors reader: {}".format(converter.READER), "Model:                      {}".format(name), "HF tensors consumed:        {}".format(len(expected)),
                     "Meta params overwritten:    {} ({:,} elements)".format(len(ref) - n_kept, n_params), "Buffers kept from template: {} ({})".format(n_kept, kept_names),
                     "dtype written:              params torch.float32, buffers torch.float32", "Bytes written:              {:,} of tensors, {:,} on disk".format(n_bytes, os.path.getsize(dst)),
                     "Wrote {} ({} keys)".format(dst, len(ref))):
            assert line in summary.splitlines(), (line, summary)
        assert n_kept == (1 + kwargs["depth"] if kwargs["qkv_bias"] else 1)  # the rope periods, and a k-bias mask per block unless there is no qkv bias
        print("OK (b) {}: the script's output has the {} Meta keys of the model in the same order, each torch.equal to the original ({} params written from {} HF tensors; buffers kept: {}; "
              "fp32, {:,} bytes); {} shard(s), --model {}".format(name, len(ref), len(ref) - n_kept, len(expected), n_kept, n_bytes, shards, "left out" if name == "vit_base_dinov3" else "given"))
        del out

        model = guard(name, kwargs, dst)  # raises unless only the (here absent) projector and heads are missing and nothing is unexpected
        loaded = model.state_dict()
        assert list(loaded) == list(ref) and all(torch.equal(loaded[k], ref[k]) for k in ref), name
        assert (model.local_cls_norm is None) and len(model.blocks) == kwargs["depth"]
        print("OK (c) {}: build_omni_model --pretrained_weights the converted file passes the strict guard for the {}-block model and loads every tensor unchanged".format(name, kwargs["depth"]))
        del model, loaded

        # the same file under a --model of another size is refused, naming the HF tensor, the model and both shapes
        width, wrong_width = kwargs["embed_dim"], getattr(models, KWARGS_NAME[wrong])["embed_dim"]
        with contextlib.redirect_stdout(io.StringIO()):  # main prints the reader line before it fails
            message = rejects(SystemExit, lambda: converter.main(["--model", wrong, source, os.path.join(tmp, "refused.pth")]),
                              ["embeddings.cls_token", "(1, 1, {})".format(width), "--model {}".format(wrong), "(1, 1, {})".format(wrong_width), "Meta's cls_token"])
        assert message.startswith("error: ") and not os.path.exists(os.path.join(tmp, "refused.pth"))
        print("OK (d) {}: its HF file under --model {} exits with '{}'; nothing is written".format(name, wrong, message))
    del ref

# (e) the refusals that are about the layout, not the width, and the ways to point at a checkpoint, on tiny models
with tempfile.TemporaryDirectory() as tmp:
    def export(kwargs, folder, shards=1, ref=None, dtype=torch.float32):
        """A tiny model's HF files in tmp/folder. -> (the folder, the file or index path, the Meta state dict they were made from)"""
        directory = os.path.join(tmp, folder)
        os.mkdir(directory)
        ref = reference(kwargs, dtype) if ref is None else ref
        return directory, write_hf(to_hf(ref, ""), directory, shards), ref

    mlp_dir, _, _ = export(TINY, "mlp")
    gated_dir, _, _ = export(TINY_GATED, "gated")
    deep_dir, _, _ = export(dict(TINY, depth=3), "deep")
    nobias_dir, _, _ = export(dict(TINY, qkv_bias=False), "nobias")
    for source, other, model_kwargs, parts in (
            (mlp_dir, "vit_base_dinov3", dict(TINY, depth=3), ["no layer.2.norm1.weight", "Meta's blocks.2.norm1.weight", "--model vit_base_dinov3"]),  # too few blocks in the file
            (deep_dir, "vit_base_dinov3", TINY, ["17 HF tensors not consumed by --model vit_base_dinov3", "layer.2.attention.k_proj.weight"]),  # too many
            (mlp_dir, "vit_huge_plus_dinov3", TINY_GATED, ["no layer.0.mlp.gate_proj.weight", "Meta's blocks.0.mlp.w1.weight", "--model vit_huge_plus_dinov3"]),  # an MLP file for a SwiGLU model
            (gated_dir, "vit_large_dinov3", TINY, ["4 HF tensors not consumed by --model vit_large_dinov3", "layer.0.mlp.gate_proj.bias"]),  # a SwiGLU file for an MLP model
            (nobias_dir, "vit_base_dinov3", TINY, ["no layer.0.attention.q_proj.bias", "Meta's blocks.0.attn.qkv.bias", "--model vit_base_dinov3"]),  # no q/v bias in the file, a qkv bias in the model
            (mlp_dir, "vit_7b_dinov3", dict(TINY, qkv_bias=False), ["4 HF tensors not consumed by --model vit_7b_dinov3", "layer.0.attention.q_proj.bias"])):  # q/v biases in the file, none in the model
        rejects(ValueError, lambda: convert(source, other, model_kwargs), parts)
    print("OK (e1) layout mismatches are refused by name: too few blocks ('{}'), too many, an MLP file for a SwiGLU model and the reverse, a missing and a surplus qkv bias".format(
        rejects(ValueError, lambda: convert(mlp_dir, "vit_base_dinov3", dict(TINY, depth=3)), ["no layer.2.norm1.weight"])))

    # a file that is not a DINOv3 HF layout, and a shard that does not hold what its index says
    other_file = os.path.join(tmp, "other.safetensors")
    save_file({"weight": torch.zeros(2, 2)}, other_file, metadata={"format": "pt"})
    rejects(ValueError, lambda: convert(other_file, "vit_base_dinov3", TINY), ["is not a DINOv3 HF checkpoint layout, keys start with: ['weight']"])
    shard_dir, index, _ = export(TINY, "shards", shards=2)
    with open(index) as f:
        weight_map = json.load(f)["weight_map"]
    weight_map["layer.0.extra.weight"] = "model-00001-of-00002.safetensors"
    with open(index, "w") as f:
        json.dump({"weight_map": weight_map}, f)
    rejects(ValueError, lambda: convert(shard_dir, "vit_base_dinov3", TINY), ["shard model-00001-of-00002.safetensors does not hold exactly the tensors"])
    print("OK (e2) a file without embeddings.cls_token ('is not a DINOv3 HF checkpoint layout') and a shard that lacks a tensor its index lists are both refused")

    # the four ways to point at a checkpoint give the same state dict, the original's: the directory or the file of a single model.safetensors, the directory or the index.json of two shards
    single_dir, single_file, ref = export(TINY, "single")
    sharded_dir, sharded_index, _ = export(TINY, "sharded", shards=2, ref=ref)
    for source in (single_dir, single_file, sharded_dir, sharded_index):
        out = convert(source, "vit_base_dinov3", TINY)[0]
        assert list(out) == list(ref) and all(torch.equal(out[k], ref[k]) for k in ref), source
    print("OK (e3) the directory or the file of a single model.safetensors, and the directory or the index.json of two shards, each convert to the original's {} Meta tensors".format(len(ref)))

    # dtype: written as the HF file has it (bf16 here), no upcast; only the buffers that come from the Meta model are fp32
    half_dir, _, half = export(TINY, "half", dtype=torch.bfloat16)
    half_dst = os.path.join(tmp, "half.pth")
    summary = convert_main(["--model", "vit_base_dinov3", half_dir, half_dst], "vit_base_dinov3", TINY)
    out = torch.load(half_dst, map_location="cpu", weights_only=True)
    kept = [k for k in half if re.fullmatch(converter.KEPT, k)]
    assert list(out) == list(half) and all(out[k].dtype == (torch.float32 if k in kept else torch.bfloat16) and torch.equal(out[k], half[k]) for k in half)
    n_bytes = sum(t.numel() * t.element_size() for t in out.values())
    assert n_bytes == sum(t.numel() * (4 if k in kept else 2) for k, t in half.items())
    assert "dtype written:              params torch.bfloat16, buffers torch.float32" in summary.splitlines() and "Bytes written:              {:,} of tensors".format(n_bytes) in summary
    print("OK (e4) a bf16 HF file converts to bf16 params ({} of them) and fp32 buffers ({}), every tensor torch.equal to the original, {:,} bytes of tensors".format(len(half) - len(kept), len(kept), n_bytes))

# (f) the template of each real config, the 7B's 40 blocks included, is shapes only (torch's meta device: no memory); only the buffers that HF drops are real tensors
COUNTS = {"vit_base_dinov3": (85669632, 13), "vit_large_dinov3": (303154176, 25), "vit_huge_plus_dinov3": (840633600, 33), "vit_7b_dinov3": (6716035072, 1)}  # parameters, buffers kept
for name, (n_params, n_kept) in COUNTS.items():
    template, kept = converter.meta_template(getattr(models, KWARGS_NAME[name]))
    assert len(kept) == n_kept and all(not template[k].is_meta and torch.isfinite(template[k]).all() for k in kept), name
    assert all(t.is_meta for k, t in template.items() if k not in kept) and sum(t.numel() for k, t in template.items() if k not in kept) == n_params, name
print("OK (f) the real configs' templates are shapes only (meta device): {} parameters for the ViT-B / ViT-L / ViT-H+ / 7B, with {} real buffers kept (the rope periods, a k-bias mask per block; the 7B has no qkv bias, so only the periods)".format(
    " / ".join("{:,}".format(n) for n, _ in COUNTS.values()), " / ".join(str(k) for _, k in COUNTS.values())))

# (g) the converter's own safetensors reader, the one it falls back on when the Python has no safetensors package (the Sol container)
def raw_file(path, header, payload=b"", header_len=None):
    """A .safetensors file written by hand (the 8-byte length, the JSON header, the payload), for the files the package would not write. `header_len` lies about the length."""
    encoded = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded) if header_len is None else header_len) + encoded + payload)


def open_fds():
    """The number of file descriptors this process holds after a collection (None where /dev/fd does not exist). Each live mmap holds one."""
    gc.collect()
    return len(os.listdir("/dev/fd")) if os.path.isdir("/dev/fd") else None


read = converter.load_file_stdlib
with tempfile.TemporaryDirectory() as tmp:
    every = {"f64": torch.randn(3, dtype=torch.float64), "f32": torch.randn(2, 3), "f16": torch.randn(4).half(), "bf16": torch.randn(5).bfloat16(), "i64": torch.tensor([-3, 2 ** 40]),
             "i32": torch.tensor([-7, 2 ** 20], dtype=torch.int32), "i16": torch.tensor([300, -2], dtype=torch.int16), "i8": torch.tensor([-1, 5, 127], dtype=torch.int8),
             "u8": torch.tensor([200, 0, 255], dtype=torch.uint8), "flags": torch.tensor([True, False, True]), "scalar": torch.tensor(1.5), "empty": torch.empty(0, 4)}
    path = os.path.join(tmp, "every.safetensors")
    save_file(every, path, metadata={"format": "pt"})
    loaded = read(path)
    assert set(loaded) == set(every) and all(loaded[k].dtype == t.dtype and loaded[k].shape == t.shape and torch.equal(loaded[k], t) for k, t in every.items())  # no __metadata__ key
    assert HIDDEN or list(loaded) == list(package_load_file(path))  # the package's order too: the header's
    loaded["f32"].fill_(7)  # copy-on-write: a write to a loaded tensor does not reach the file
    assert torch.equal(read(path)["f32"], every["f32"])
    print("OK (g1) the built-in reader loads all ten dtypes, a scalar and an empty tensor from a file with metadata, each torch.equal to what was written{}; a write to a loaded tensor leaves the file as it was".format(
        "" if HIDDEN else ", with the keys in the package's order"))
    del loaded

    bad, f32 = os.path.join(tmp, "bad.safetensors"), {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    for header, payload, header_len, parts in (
            ({"w": dict(f32, dtype="F8_E4M3", shape=[8])}, bytes(8), None, [bad, "w has dtype F8_E4M3", "only F64, F32, F16, BF16, I64, I32, I16, I8, U8, BOOL are read"]),  # a dtype it does not read
            ({"w": f32}, bytes(4), None, [bad, "w has data_offsets [0, 8]", "do not hold 2 F32 values of shape [2]"]),  # offsets past the end of the file
            ({"w": dict(f32, data_offsets=[0, 4])}, bytes(8), None, [bad, "w has data_offsets [0, 4]", "do not hold 2 F32 values of shape [2]"]),  # a byte count that is not the shape's
            ({"w": f32}, bytes(8), 10 ** 6, [bad, "header length 1000000 runs past the end of the file"])):  # a header longer than the file
        raw_file(bad, header, payload, header_len)
        rejects(ValueError, lambda: read(bad), parts)
    for short in (b"", b"abc"):  # shorter than the 8-byte length
        with open(bad, "wb") as f:
            f.write(short)
        rejects(ValueError, lambda: read(bad), [bad, "shorter than the 8-byte header length"])
    print("OK (g2) the built-in reader refuses, naming the file and the tensor: a dtype it does not read (F8_E4M3), data_offsets past the end or of the wrong byte count, a header longer than the file, a file under 8 bytes")

    # through read_hf's shard-at-a-time loop: one mapping per shard, alive as long as its tensors and released with them (a closed mapping crashes the reads, a leaked one holds the shard)
    directory = os.path.join(tmp, "sharded")
    os.mkdir(directory)
    expected_hf = to_hf(reference(TINY), "")
    index = write_hf(expected_hf, directory, shards=2)
    before = open_fds()
    with patched(converter, "load_file", read):
        hf = converter.read_hf(index)
    held = open_fds()
    assert set(hf) == set(expected_hf) and all(torch.equal(hf[k], expected_hf[k]) for k in expected_hf)  # still readable after a collection
    del hf
    released = open_fds()
    assert before is None or (held, released) == (before + 2, before), (before, held, released)
    print("OK (g3) read_hf on 2 shards with the built-in reader: every tensor torch.equal to the export after a collection, one mapping per shard held while they live and released with them{}".format(
        "" if before is not None else " (SKIP the descriptor count: no /dev/fd here)"))

# (h) the real ViT-B file, with the package as the reference: the same tensors, then the same converted file byte for byte (the package is hidden in the second run, so nothing to compare with there)
if not HIDDEN:
    if not os.path.exists(HF_VITB):
        print("SKIP (h): {} not found (set HF_DINOV3_VITB_DIR)".format(HF_VITB))
    else:
        package_tensors, stdlib_tensors = package_load_file(HF_VITB), read(HF_VITB)
        assert list(stdlib_tensors) == list(package_tensors) and all(
            stdlib_tensors[k].dtype == t.dtype and stdlib_tensors[k].shape == t.shape and torch.equal(stdlib_tensors[k], t) for k, t in package_tensors.items())
        print("OK (h1) the built-in reader on the real ViT-B file {}: the package's {} keys in the same order, with the same dtypes and shapes and torch.equal tensors".format(HF_VITB, len(package_tensors)))
        del package_tensors, stdlib_tensors

        with tempfile.TemporaryDirectory() as tmp:
            # the same file name in two directories: torch.save writes the name into the zip, so the bytes can only match under one name
            with_package, with_fallback = (os.path.join(tmp, d, STOCK) for d in ("package", "fallback"))
            assert "safetensors reader: safetensors package" in convert_main([HF_VITB, with_package], "vit_base_dinov3", DINOV3_VITB16_KWARGS).splitlines()
            done = subprocess.run([sys.executable, "-W", "ignore", "-c", HIDE_AND_CONVERT, HF_VITB, with_fallback], cwd=HERE, capture_output=True, text=True)
            assert done.returncode == 0 and any(line.startswith("safetensors reader: built-in stdlib reader") for line in done.stdout.splitlines()), (done.returncode, done.stdout, done.stderr)
            assert filecmp.cmp(with_package, with_fallback, shallow=False)
            stock = os.path.join(WEIGHTS, STOCK)
            if os.path.exists(stock):
                assert filecmp.cmp(with_package, stock, shallow=False), stock
            print("OK (h2) the ViT-B converted by a fresh interpreter with the package hidden (the built-in reader, its line printed) is byte-identical (cmp) to the conversion with the package, {:,} bytes{}".format(
                os.path.getsize(with_package), "; both are byte-identical to the stock {}".format(stock) if os.path.exists(stock) else "; SKIP the comparison with the stock file ({} not found, set ARK_WEIGHTS_DIR)".format(stock)))

# (i) all of the above, (a)-(g), once more in a fresh interpreter with the package hidden: the converter then reads every fixture with its built-in reader
if not HIDDEN:
    done = subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__)], env=dict(os.environ, HIDE_SAFETENSORS="1"), capture_output=True, text=True)
    lines = done.stdout.splitlines()
    assert done.returncode == 0 and lines[-1:] == ["ALL OK"], (done.returncode, done.stdout[-3000:], done.stderr[-3000:])
    print("OK (i) this file run again in a subprocess with the safetensors package hidden (the converter on its built-in reader): {} OK lines, then ALL OK".format(sum(line.startswith("OK (") for line in lines)))

print("ALL OK")
