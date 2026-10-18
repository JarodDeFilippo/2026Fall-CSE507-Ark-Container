"""CPU checks for the DINOv2-family backbones (vit_base_dinov2, vit_base_raddino, vit_large_raydino), the raddino normalization preset and
convert_safetensors_to_pth.py. Run from base/: python test_dinov2_backbone.py
Real files: $ARK_WEIGHTS_DIR (default ../weights) with dinov2_vitb14_pretrain.pth, rad_dino_vitb14_pretrain.pth and
rad_dino_backbone_compatible.safetensors; Meta's DINOv2 source for the parity checks: $DINOV2_MAIN (default ../../dinov2-main).
A check whose file is missing prints a SKIP line and the rest still run."""
import importlib.util
import json
import logging
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore", message="xFormers is ")  # DINOv2's layer modules warn once each when xformers is absent

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import engine
import models
from convert_safetensors_to_pth import check_round_trip, convert, read_safetensors
from dataloader import build_transform_classification
from dinov2.models.vision_transformer import DinoVisionTransformer as VendoredDinoV2
from models import ArkViTDinoV2, DINOV2_VITB14_KWARGS, DINOV2_VITL16_KWARGS, build_omni_model, checkpoint_summary, flatten_chunked_blocks, vit_param_groups

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.environ.get("ARK_WEIGHTS_DIR", os.path.join(REPO, "weights"))
DINOV2_MAIN = os.path.abspath(os.environ.get("DINOV2_MAIN", os.path.join(REPO, "..", "dinov2-main")))
FILES = {"vit_base_dinov2": os.path.join(WEIGHTS, "dinov2_vitb14_pretrain.pth"),
         "vit_base_raddino": os.path.join(WEIGHTS, "rad_dino_vitb14_pretrain.pth")}
SAFETENSORS = os.path.join(WEIGHTS, "rad_dino_backbone_compatible.safetensors")
HEAD_PREFIXES = ("projector.", "omni_heads.")
# same ViT with 2 blocks of width 64 on 28x28 inputs, for the key-guard files
TINY = dict(DINOV2_VITB14_KWARGS, img_size=28, embed_dim=64, depth=2, num_heads=2, encoder_features=128)


def ark_args(model_name, projector_features=1376, pretrained_weights=None):
    return SimpleNamespace(model_name=model_name, projector_features=projector_features, use_mlp=False, pretrained_weights=pretrained_weights)


def rejects(exception, call, expected=None):
    """The exception message when call() raises `exception`; fails when it returns."""
    try:
        call()
    except exception as error:
        assert expected is None or expected in str(error), (expected, str(error))
        return str(error)
    raise AssertionError("expected {}".format(exception.__name__))


def backbone_names(model):
    return [n for n, _ in model.named_parameters() if not n.startswith(HEAD_PREFIXES)]


# (a) both model names build the hub's ViT-B/14 in the Ark wrapper; forward shapes; pos_embed interpolation ------------------
torch.manual_seed(0)
x = torch.randn(2, 3, 224, 224)
for name in FILES:
    model = build_omni_model(ark_args(name), [6, 14]).eval()
    assert type(model) is ArkViTDinoV2 and isinstance(model, VendoredDinoV2)
    assert sys.modules[VendoredDinoV2.__module__].__file__.startswith(os.path.join(REPO, "base", "dinov2"))  # the vendored copy
    names = backbone_names(model)
    assert len(names) == 175 == len(model.state_dict()) - 6 and sum(p.numel() for n, p in model.named_parameters() if n in names) == 86580480
    assert not any(n.startswith("register_tokens") for n in names) and model.num_register_tokens == 0 and model.patch_size == 14
    with torch.no_grad():
        assert model(x, return_features=True).shape == (2, 1376)
        feats, outs = model(x, return_all=True)
        assert feats.shape == (2, 1376) and [o.shape for o in outs] == [(2, 6), (2, 14)]
        assert model(x, head_n=1)[1].shape == (2, 14) and [o.shape for o in model(x)] == [(2, 6), (2, 14)]
        embeddings = model.generate_embeddings(x, after_proj=False)
        tokens = super(ArkViTDinoV2, model).forward_features(x)  # DINOv2's dict
    assert embeddings.shape == (2, 1536)
    assert torch.equal(embeddings[:, :768], tokens["x_norm_clstoken"]) and torch.equal(embeddings[:, 768:], tokens["x_norm_patchtokens"].mean(dim=1))
    assert tokens["x_norm_patchtokens"].shape == (2, 256, 768) and tokens["x_norm_regtokens"].shape == (2, 0, 768)
assert model.pos_embed.shape == (1, 1370, 768)  # 37x37 + CLS: learned at 518 px
resampled = model.interpolate_pos_encoding(torch.zeros(2, 257, 768), 224, 224)
assert resampled.shape == (1, 257, 768) and torch.equal(resampled[:, 0], model.pos_embed[:, 0])  # 16x16 + CLS; CLS untouched
probe = build_omni_model(ark_args("vit_base_dinov2", None), [6]).eval()  # what the linear probe will use: no projector, one head
with torch.no_grad():
    assert probe.projector is None and probe.num_features == 1536 and probe(x, return_features=True).shape == (2, 1536)
    assert probe(x, head_n=0)[1].shape == (2, 6) and probe.generate_embeddings(x, after_proj=False).shape == (2, 1536)
print("OK (a) build: both names are the 175-tensor (86,580,480-param) DINOv2 ViT-B/14 in the Ark wrapper; features (2, 1536) = CLS ++ mean patch, "
      "projector (2, 1376), heads; pos_embed (1, 1370, 768) resampled to (1, 257, 768) at 224 px; projector None + one head works")

# (b) strict load of the real files; the key guard; RAD-DINO differs from DINOv2; safetensors round trip ----------------------
stock, loaded_models = {}, {}  # model name -> the file's state dict / the Ark model built from it
for name, path in FILES.items():
    if not os.path.isfile(path):
        print("SKIP (b) {}: {} not found (set ARK_WEIGHTS_DIR)".format(name, path))
        continue
    stock[name] = torch.load(path, map_location="cpu", weights_only=True)
    loaded = build_omni_model(ark_args(name, 1376, path), [6, 14])
    state = loaded.state_dict()
    assert len(stock[name]) == 175 and sorted(backbone_names(loaded)) == sorted(stock[name])
    assert all(torch.equal(state[k], v) for k, v in stock[name].items()), name
    loaded_models[name] = loaded
saved_kwargs = models.DINOV2_VITB14_KWARGS
models.DINOV2_VITB14_KWARGS = TINY  # build_omni_model reads the module global
try:
    with tempfile.TemporaryDirectory() as tmp:
        tiny_backbone = {k: v for k, v in ArkViTDinoV2([], None, False, **TINY).state_dict().items()}
        good = os.path.join(tmp, "good.pth")
        torch.save(tiny_backbone, good)
        build_omni_model(ark_args("vit_base_raddino", 32, good), [6])  # backbone-only file: only projector / heads missing, accepted
        for label, broken in (("missing", {k: v for k, v in tiny_backbone.items() if k != "blocks.0.attn.qkv.weight"}),
                              ("unexpected", dict(tiny_backbone, **{"register_tokens": torch.zeros(1, 4, 64)}))):
            bad = os.path.join(tmp, label + ".pth")
            torch.save(broken, bad)
            for name, label_text in (("vit_base_dinov2", "DINOv2"), ("vit_base_raddino", "RAD-DINO")):
                rejects(RuntimeError, lambda: build_omni_model(ark_args(name, 32, bad), [6]), label_text + " checkpoint does not match the backbone")
finally:
    models.DINOV2_VITB14_KWARGS = saved_kwargs
if len(stock) == 2:
    dinov2_sd, raddino_sd = stock["vit_base_dinov2"], stock["vit_base_raddino"]
    assert set(dinov2_sd) == set(raddino_sd)
    assert all(dinov2_sd[k].shape == raddino_sd[k].shape and dinov2_sd[k].dtype == raddino_sd[k].dtype == torch.float32 for k in dinov2_sd)
    same = [k for k in dinov2_sd if torch.equal(dinov2_sd[k], raddino_sd[k])]
    assert not same, same  # every tensor differs (DINOv2's released mask_token is all zeros, RAD-DINO's is not)
if os.path.isfile(SAFETENSORS) and "vit_base_raddino" in stock:
    header, buffer, from_safetensors = read_safetensors(SAFETENSORS)
    check_round_trip(FILES["vit_base_raddino"], header, buffer)  # keys, order, dtypes, shapes, and bytes
    assert all(torch.equal(from_safetensors[k], v) for k, v in stock["vit_base_raddino"].items())
    print("OK (b) weights: both files load strictly (175 backbone tensors equal to the file); a missing or an extra backbone key raises; "
          "RAD-DINO and DINOv2 share keys/shapes/dtypes and differ in all {} tensors; the RAD-DINO .pth equals its .safetensors byte for byte".format(len(header)))
else:
    print("SKIP (b) partly: the guard ran, the real-file checks need {} and {}".format(FILES["vit_base_raddino"], SAFETENSORS))

# (c) parity with Meta's own hub model, built from dinov2-main in a fresh interpreter (both packages are named dinov2) --------
UPSTREAM = r"""
import sys, torch
source, inputs, out = sys.argv[1:4]
sys.path.insert(0, source)
import dinov2
assert dinov2.__file__.startswith(source), dinov2.__file__
from dinov2.hub.backbones import dinov2_vitb14
model = dinov2_vitb14(pretrained=False).eval()
x = torch.load(inputs)
results = {}
for weights in sys.argv[4:]:
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)  # what the hub does with pretrained=True
    with torch.no_grad():
        o = model.forward_features(x)
    results[weights] = (o["x_norm_clstoken"], o["x_norm_patchtokens"])
torch.save(results, out)
"""
present = [name for name in FILES if name in stock]
if not os.path.isfile(os.path.join(DINOV2_MAIN, "dinov2", "hub", "backbones.py")):
    print("SKIP (c): Meta's dinov2 source not found at {} (set DINOV2_MAIN)".format(DINOV2_MAIN))
elif not present:
    print("SKIP (c): no real weights to compare with")
else:
    with tempfile.TemporaryDirectory() as tmp:
        inputs, out = os.path.join(tmp, "x.pt"), os.path.join(tmp, "out.pt")
        torch.save(x, inputs)
        subprocess.run([sys.executable, "-I", "-W", "ignore", "-c", UPSTREAM, DINOV2_MAIN, inputs, out] + [FILES[n] for n in present],
                       cwd=tmp, check=True)
        upstream = torch.load(out)
    diffs = []
    for name in present:
        model = loaded_models[name].eval()
        with torch.no_grad():
            ours = super(ArkViTDinoV2, model).forward_features(x)
            features = model.generate_embeddings(x, after_proj=False)
        cls, patch = upstream[FILES[name]]
        diff = max((ours["x_norm_clstoken"] - cls).abs().max().item(), (ours["x_norm_patchtokens"] - patch).abs().max().item(),
                   (features - torch.cat([cls, patch.mean(dim=1)], dim=-1)).abs().max().item())
        assert diff <= 1e-5, (name, diff)
        diffs.append("{} {:.3g}".format(name, diff))
    print("OK (c) parity with dinov2-main's hub dinov2_vitb14 (strict load, 224 px): CLS, patch tokens and the 1536-d feature; max abs diff " + ", ".join(diffs))

# (d) normalization presets: raddino added, the old ones unchanged; raddino is refused with the hard-coded training normalization --
PRESETS = {"imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # copied from dataloader.py before the raddino preset
           "chestx-ray": ([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252]),
           "raddino": ([0.5307, 0.5307, 0.5307], [0.2583, 0.2583, 0.2583]),  # rad-dino/preprocessor_config.json image_mean / image_std
           "none": None}
gray = Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8))
for preset, expected in PRESETS.items():
    for kwargs in (dict(mode="train"), dict(mode="valid"), dict(mode="test", test_augment=False)):
        found = [t for t in build_transform_classification(preset, **kwargs).transforms if isinstance(t, transforms.Normalize)]
        if expected is None:
            assert not found, (preset, kwargs)
        else:
            assert len(found) == 1 and list(found[0].mean) == expected[0] and list(found[0].std) == expected[1], (preset, kwargs)
    crops = build_transform_classification(preset, mode="test", test_augment=True)(gray)  # TenCrop: normalize sits in a Lambda
    assert crops.shape == (10, 3, 224, 224)
    for channel in range(3):
        want = 128 / 255 if expected is None else (128 / 255 - expected[0][channel]) / expected[1][channel]
        assert torch.allclose(crops[:, channel], torch.full_like(crops[:, channel], want), atol=1e-6), (preset, channel)
rejects(SystemExit, lambda: build_transform_classification("rad-dino", mode="valid"))  # unknown names still exit
try:
    import main_ark
except ImportError as error:
    print("SKIP (d) main_ark: cannot import it here ({})".format(error))
else:
    base_argv = ["--mode", "train", "--exp_name", "x", "--seed", "1", "--data_set", "VinDrCXR", "--model", "vit_base_raddino", "--init", "raddino",
                 "--pretrained_weights", "/workspace/weights/rad_dino_vitb14_pretrain.pth", "--normalization", "raddino"]
    sys.argv = ["main_ark.py"] + base_argv
    options = main_ark.get_args_parser()
    assert options.model_name == "vit_base_raddino" and options.init == "raddino" and options.normalization == "raddino" and options.train_augment is True
    rejects(ValueError, lambda: main_ark.main(options), "--normalization raddino needs --train_augment false")  # before any dataset is read
    sys.argv = ["main_ark.py"] + base_argv + ["--train_augment", "false"]
    assert main_ark.get_args_parser().train_augment is False  # the combination the guard lets through
print("OK (d) normalization: raddino = 0.5307 / 0.2583 on 3 channels; imagenet / chestx-ray / none unchanged in train, valid, test and TenCrop; "
      "main_ark refuses --normalization raddino unless --train_augment false")

# (e) param groups mirror dinov2/utils/param_groups.py ------------------------------------------------------------------
model = ArkViTDinoV2([6, 14, 14, 14], 1376, False, **DINOV2_VITB14_KWARGS)
named = list(model.named_parameters())
groups = vit_param_groups(model, 0.04, 0.9, 0.2)
group_of = {}
for group in groups:
    assert set(group) == {"params", "weight_decay", "lr_scale"}
    for p in group["params"]:
        assert id(p) not in group_of  # every param in exactly one group
        group_of[id(p)] = group
assert set(group_of) == {id(p) for _, p in named}
assert groups[0]["lr_scale"] == 1.0 and len({(g["lr_scale"], g["weight_decay"]) for g in groups}) == len(groups) == 29


def scale_of(prefix):
    scales = {group_of[id(p)]["lr_scale"] for n, p in named if n.startswith(prefix)}
    assert len(scales) == 1, (prefix, scales)
    return scales.pop()


assert math.isclose(scale_of("patch_embed."), 0.9 ** 13 * 0.2, rel_tol=1e-12)
for token in ("pos_embed", "cls_token", "mask_token"):  # layer 0
    assert math.isclose(scale_of(token), 0.9 ** 13, rel_tol=1e-12)
for i in range(12):  # blocks.0 = 0.9**12 ... blocks.11 = 0.9**1
    assert math.isclose(scale_of("blocks.%d." % i), 0.9 ** (12 - i), rel_tol=1e-12)
for prefix in ("norm.", "projector.", "omni_heads."):  # the top
    assert scale_of(prefix) == 1.0
for n, p in named:  # DINOv2's rule: no decay on ".bias", "norm" (LayerNorms) and "gamma" (LayerScale) only
    assert group_of[id(p)]["weight_decay"] == (0.0 if n.endswith(".bias") or "norm" in n or "gamma" in n else 0.04), n
decayed = {n for n, p in named if group_of[id(p)]["weight_decay"]}
assert {"pos_embed", "cls_token", "mask_token", "patch_embed.proj.weight", "blocks.0.attn.qkv.weight", "projector.weight"} <= decayed
assert not {"blocks.0.ls1.gamma", "blocks.0.norm1.weight", "norm.weight", "blocks.0.mlp.fc1.bias", "omni_heads.0.bias"} & decayed
upstream_file = os.path.join(DINOV2_MAIN, "dinov2", "utils", "param_groups.py")
if os.path.isfile(upstream_file):  # Meta's function on the same model, loaded under another name (it imports only the stdlib)
    spec = importlib.util.spec_from_file_location("upstream_dinov2_param_groups", upstream_file)
    upstream_groups = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream_groups)
    meta = {d["name"]: d for d in upstream_groups.get_params_groups_with_decay(model, lr_decay_rate=0.9, patch_embed_lr_mult=0.2)}
    assert sorted(meta) == sorted(n for n, _ in named)
    for n, p in named:
        assert math.isclose(group_of[id(p)]["lr_scale"], meta[n]["lr_multiplier"], rel_tol=1e-12), n
        assert group_of[id(p)]["weight_decay"] == 0.04 * meta[n]["wd_multiplier"], n
    upstream_note = "; equal, parameter by parameter, to dinov2-main's get_params_groups_with_decay (lr_multiplier, wd_multiplier)"
else:
    upstream_note = "; SKIP the dinov2-main cross-check ({} not found)".format(upstream_file)
for name in FILES:
    built = engine._create_vit_finetune_optimizer(SimpleNamespace(model_name=name, opt="adamw", lr=1e-4, opt_eps=1e-8, momentum=0.9,
                                                                  weight_decay=0.04, layer_decay=0.9, patch_embed_lr_mult=0.2), model)
    assert isinstance(built, torch.optim.AdamW) and len(built.param_groups) == 29
rejects(ValueError, lambda: engine._create_vit_finetune_optimizer(SimpleNamespace(model_name="swin_base", opt="adamw", lr=1e-4, opt_eps=1e-8,
        momentum=0.9, weight_decay=0.04, layer_decay=0.9, patch_embed_lr_mult=0.2), model), "only implemented for --model vit_base_dinov3")
print("OK (e) param groups: 29 groups; patch_embed / pos_embed / cls_token / mask_token at layer 0, blocks.i at i+1, norm / projector / heads at 1.0; "
      "no decay on bias / norm / LayerScale gamma, pos_embed and the tokens decayed" + upstream_note)

# (f) --freeze_backbone freezes exactly the 175 backbone tensors -----------------------------------------------------------
model = ArkViTDinoV2([6, 14], 1376, False, **DINOV2_VITB14_KWARGS)
summary = engine._freeze_backbone(model)
frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
assert len(frozen) == 175 and all(n.startswith(HEAD_PREFIXES) for n in trainable) and len(trainable) == 6
if "vit_base_dinov2" in stock:
    assert sorted(frozen) == sorted(stock["vit_base_dinov2"])  # the frozen set is the weight file's key set
recipe = SimpleNamespace(model_name="vit_base_raddino", opt="adamw", lr=7e-5, opt_eps=1e-8, momentum=0.9, weight_decay=0.01,
                         layer_decay=None, patch_embed_lr_mult=1.0)
optimizer = engine._create_frozen_backbone_optimizer(recipe, model)
assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in model.parameters() if p.requires_grad}
print("OK (f) freeze: " + summary)

# (g) the converter on a hand-written file: every dtype, a scalar, an empty tensor, metadata; refusals --------------------------
def write_safetensors(path, entries, metadata=None, bad_end=None, header_len=None):
    """entries: name -> (dtype name, shape, little-endian bytes), written in order."""
    header = {"__metadata__": metadata} if metadata else {}
    offset = 0
    for name, (dtype, shape, raw) in entries.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    if bad_end is not None:
        header[name]["data_offsets"][1] = bad_end
    encoded = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded) if header_len is None else header_len) + encoded + b"".join(raw for _, _, raw in entries.values()))


bf16 = [1.0, -2.5, 0.15625]  # exact in bfloat16: the upper half of the float32 bits
entries = {
    "f32": ("F32", [2, 3], np.arange(6, dtype="<f4").tobytes()),
    "f16": ("F16", [4], np.array([0.5, -1, 2, 65504], dtype="<f2").tobytes()),
    "bf16": ("BF16", [3], (np.array(bf16, dtype="<f4").view("<u4") >> 16).astype("<u2").tobytes()),
    "f64": ("F64", [1], np.array([math.pi], dtype="<f8").tobytes()),
    "i64": ("I64", [2], np.array([-3, 2 ** 40], dtype="<i8").tobytes()),
    "i32": ("I32", [1], np.array([-7], dtype="<i4").tobytes()),
    "i16": ("I16", [1], np.array([300], dtype="<i2").tobytes()),
    "i8": ("I8", [2], np.array([-1, 5], dtype="i1").tobytes()),
    "u8": ("U8", [], bytes([200])),
    "flags": ("BOOL", [3], bytes([1, 0, 1])),
    "empty": ("F32", [0, 4], b""),
}
expected = {"f32": torch.arange(6, dtype=torch.float32).reshape(2, 3), "f16": torch.tensor([0.5, -1, 2, 65504], dtype=torch.float16),
            "bf16": torch.tensor(bf16, dtype=torch.bfloat16), "f64": torch.tensor([math.pi], dtype=torch.float64),
            "i64": torch.tensor([-3, 2 ** 40]), "i32": torch.tensor([-7], dtype=torch.int32), "i16": torch.tensor([300], dtype=torch.int16),
            "i8": torch.tensor([-1, 5], dtype=torch.int8), "u8": torch.tensor(200, dtype=torch.uint8),
            "flags": torch.tensor([True, False, True]), "empty": torch.empty(0, 4)}
with tempfile.TemporaryDirectory() as tmp:
    src, dst = os.path.join(tmp, "in.safetensors"), os.path.join(tmp, "out.pth")
    write_safetensors(src, entries, metadata={"format": "pt"})
    message = convert(src, dst)
    assert "11 tensors" in message and sorted(os.listdir(tmp)) == ["in.safetensors", "out.pth"]  # no .partial left behind
    saved = torch.load(dst, weights_only=True)
    assert list(saved) == list(expected)
    for k, v in expected.items():
        assert saved[k].dtype == v.dtype and saved[k].shape == v.shape and torch.equal(saved[k], v), k
    before = open(dst, "rb").read()
    rejects(FileExistsError, lambda: convert(src, dst), "refusing to overwrite")
    assert open(dst, "rb").read() == before
    for label, kwargs, text in (("short", dict(bad_end=4), "data_offsets"), ("past", dict(bad_end=10 ** 6), "data_offsets"),
                                ("length", dict(header_len=10 ** 9), "runs past the end")):
        broken = os.path.join(tmp, label + ".safetensors")
        write_safetensors(broken, entries, **kwargs)
        rejects(ValueError, lambda: convert(broken, os.path.join(tmp, label + ".pth")), text)
        assert not os.path.exists(os.path.join(tmp, label + ".pth"))
    write_safetensors(src + "8", {"f8": ("F8_E4M3", [1], b"\x00")})
    rejects(ValueError, lambda: convert(src + "8", os.path.join(tmp, "f8.pth")), "unsupported dtype F8_E4M3")
    # Hugging Face Dinov2Model names (model.safetensors): refused with a pointer to backbone_compatible.safetensors, not remapped
    for label, key, prefix in (("hf_embeddings", "embeddings.cls_token", "embeddings."), ("hf_encoder", "encoder.layer.0.norm1.weight", "encoder.")):
        hf = os.path.join(tmp, label + ".safetensors")
        write_safetensors(hf, {key: ("F32", [4], np.zeros(4, dtype="<f4").tobytes())})
        message = rejects(ValueError, lambda: convert(hf, os.path.join(tmp, label + ".pth")), "backbone_compatible.safetensors")
        assert "'" + prefix + "'" in message, message  # names the offending prefix
        assert not os.path.exists(os.path.join(tmp, label + ".pth")) and not os.path.exists(os.path.join(tmp, label + ".pth.partial"))
print("OK (g) converter: F32 / F16 / BF16 / F64 / I64 / I32 / I16 / I8 / U8 (scalar) / BOOL / an empty tensor convert exactly, metadata ignored; "
      "an existing output is never overwritten; bad offsets, a bad header length and an unknown dtype raise ValueError and leave no file; "
      "Hugging Face-layout names (embeddings. / encoder.) raise ValueError pointing to backbone_compatible.safetensors and leave no file")

# (h) the whole engine on CPU from main_ark's parser: --model vit_base_dinov2 (tiny), backbone-only weights, the layer-decay recipe --
if "main_ark" in sys.modules:
    import contextlib
    import io
    from utils import get_config

    class RandomData(torch.utils.data.Dataset):  # img_list / img_label like dataloader.py's datasets
        def __init__(self, classes, count, seed):
            self.img_list = ["image_{}_{}".format(seed, i) for i in range(count)]
            self.img_label = (np.random.RandomState(seed).rand(count, classes) > 0.5).astype(np.float32)

        def __len__(self):
            return len(self.img_list)

        def __getitem__(self, index):
            return torch.randn(3, 28, 28), torch.randn(3, 28, 28), torch.from_numpy(self.img_label[index])

    config = get_config("datasets_config.yaml")  # relative to base/
    names = ["VinDrCXR", "ChestXray14"]
    models.DINOV2_VITB14_KWARGS = TINY
    try:
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "backbone.pth")
            torch.save(ArkViTDinoV2([], None, False, **TINY).state_dict(), weights)  # like the real files: backbone keys only
            sys.argv = ["main_ark.py", "--mode", "train", "--exp_name", "smoke", "--seed", "1", "--device", "cpu", "--model", "vit_base_dinov2",
                        "--init", "dinov2", "--pretrained_weights", weights, "--projector_features", "32", "--batch_size", "8", "--workers", "0",
                        "--pretrain_epochs", "3", "--warmup-epochs", "2", "--opt", "adamw", "--lr", "1e-4", "--warmup-lr", "1e-6", "--min-lr", "1e-6",
                        "--weight-decay", "0.04", "--layer-decay", "0.9", "--patch-embed-lr-mult", "0.2", "--test_augment", "false",
                        "--data_set", names[0], "--data_set", names[1]]
            splits = [[RandomData(len(config[name]["diseases"]), count, seed + i) for i, name in enumerate(names)]
                      for seed, count in ((10, 16), (20, 8), (30, 8))]  # train, validation, test
            with contextlib.redirect_stdout(io.StringIO()):
                run_dir = os.path.join(tmp, "run")  # omni_engine refuses an existing run directory
                engine.omni_engine(main_ark.get_args_parser(), run_dir, run_dir, names, config, *splits)
            log = open(os.path.join(run_dir, "train.log")).read()
            for line in ("Backbone: vit_base_dinov2", "Initialization: dinov2", "Optimizer param groups: 9 groups",
                         "missing_keys=['projector.weight', 'projector.bias', 'omni_heads.0.weight'", "Cycle 0003: learning rate"):
                assert line in log, line
    finally:
        models.DINOV2_VITB14_KWARGS = saved_kwargs
    print("OK (h) engine: main_ark argv with --model vit_base_dinov2 --init dinov2 --pretrained_weights <backbone file> --layer-decay 0.9 "
          "--patch-embed-lr-mult 0.2 trains 3 cycles on CPU (tiny ViT): only projector / heads missing at load, grouped optimizer logged")

# (i) vit_large_raydino: RayDINO's ViT-L/16 in the same Ark wrapper (DINOV2_VITL16_KWARGS, encoder_features 2048) --------------------------
# (i1) build: DINOv2's vit_large at patch 16 with a 512 px pos_embed, flat blocks; features 2 * 1024 = 2048-d; the kwargs differ from the ViT-B/14's in the sizes only
assert set(DINOV2_VITL16_KWARGS) == set(DINOV2_VITB14_KWARGS)
assert {k for k, v in DINOV2_VITL16_KWARGS.items() if v != DINOV2_VITB14_KWARGS[k]} == {"img_size", "patch_size", "embed_dim", "depth", "num_heads"}
ray = build_omni_model(ark_args("vit_large_raydino"), [6, 14]).eval()
assert type(ray) is ArkViTDinoV2 and isinstance(ray, VendoredDinoV2)
ray_names = backbone_names(ray)
ray_shapes = {k: list(v.shape) for k, v in ray.state_dict().items() if not k.startswith(HEAD_PREFIXES)}  # for (i7)
assert len(ray_names) == 343 == len(ray.state_dict()) - 6 and sum(p.numel() for n, p in ray.named_parameters() if n in ray_names) == 304199680
assert not any(n.startswith("register_tokens") for n in ray_names) and ray.num_register_tokens == 0 and ray.patch_size == 16 and ray.embed_dim == 1024
assert len(ray.blocks) == 24 and not ray.chunked_blocks  # built flat (block_chunks=0): blocks.0 ... blocks.23
with torch.no_grad():
    feats, outs = ray(x, return_all=True)
    embeddings = ray.generate_embeddings(x, after_proj=False)
    tokens = super(ArkViTDinoV2, ray).forward_features(x)
assert feats.shape == (2, 1376) and [o.shape for o in outs] == [(2, 6), (2, 14)]
assert embeddings.shape == (2, 2048) and torch.equal(embeddings[:, :1024], tokens["x_norm_clstoken"]) and torch.equal(embeddings[:, 1024:], tokens["x_norm_patchtokens"].mean(dim=1))
assert tokens["x_norm_patchtokens"].shape == (2, 196, 1024) and tokens["x_norm_regtokens"].shape == (2, 0, 1024)  # 14 x 14 patches at 224 px
assert ray.pos_embed.shape == (1, 1025, 1024)  # 32 x 32 + CLS: learned at 512 px
resampled = ray.interpolate_pos_encoding(torch.zeros(2, 197, 1024), 224, 224)
assert resampled.shape == (1, 197, 1024) and torch.equal(resampled[:, 0], ray.pos_embed[:, 0])  # 14 x 14 + CLS; CLS untouched
print("OK (i1) build: vit_large_raydino is the 343-tensor (304,199,680-param) DINOv2 ViT-L/16 in the Ark wrapper, 24 flat blocks, no registers; features (2, 2048) = CLS ++ mean patch, projector (2, 1376), "
      "heads; pos_embed (1, 1025, 1024) resampled to (1, 197, 1024) at 224 px; its kwargs differ from the ViT-B/14's in img_size, patch_size, embed_dim, depth and num_heads only")

# (i2) param groups for depth 24: blocks.0 ... blocks.23 parse (two-digit indices included), 53 groups = 2 (patch_embed) + 1 (pos_embed, tokens) + 2 x 24 (blocks) + 2 (norm, projector, heads)
ray_named = list(ray.named_parameters())
ray_groups = vit_param_groups(ray, 0.04, 0.9, 0.2)
ray_group_of = {id(p): g for g in ray_groups for p in g["params"]}
assert sum(len(g["params"]) for g in ray_groups) == len(ray_named) == len(ray_group_of) and set(ray_group_of) == {id(p) for _, p in ray_named}  # every param in exactly one group
assert ray_groups[0]["lr_scale"] == 1.0 and len({(g["lr_scale"], g["weight_decay"]) for g in ray_groups}) == len(ray_groups) == 53


def ray_scale_of(prefix):
    scales = {ray_group_of[id(p)]["lr_scale"] for n, p in ray_named if n.startswith(prefix)}
    assert len(scales) == 1, (prefix, scales)
    return scales.pop()


assert math.isclose(ray_scale_of("patch_embed."), 0.9 ** 25 * 0.2, rel_tol=1e-12)
for token in ("pos_embed", "cls_token", "mask_token"):  # layer 0
    assert math.isclose(ray_scale_of(token), 0.9 ** 25, rel_tol=1e-12)
for i in range(24):  # blocks.0 = 0.9**24 ... blocks.23 = 0.9**1
    assert math.isclose(ray_scale_of("blocks.%d." % i), 0.9 ** (24 - i), rel_tol=1e-12)
for prefix in ("norm.", "projector.", "omni_heads."):  # the top
    assert ray_scale_of(prefix) == 1.0
for n, p in ray_named:  # DINOv2's rule: no decay on ".bias", "norm" (LayerNorms) and "gamma" (LayerScale) only
    assert ray_group_of[id(p)]["weight_decay"] == (0.0 if n.endswith(".bias") or "norm" in n or "gamma" in n else 0.04), n
if os.path.isfile(upstream_file):  # Meta's function on the same model (loaded in (e))
    ray_meta = {d["name"]: d for d in upstream_groups.get_params_groups_with_decay(ray, lr_decay_rate=0.9, patch_embed_lr_mult=0.2)}
    assert sorted(ray_meta) == sorted(n for n, _ in ray_named)
    for n, p in ray_named:
        assert math.isclose(ray_group_of[id(p)]["lr_scale"], ray_meta[n]["lr_multiplier"], rel_tol=1e-12), n
        assert ray_group_of[id(p)]["weight_decay"] == 0.04 * ray_meta[n]["wd_multiplier"], n
    ray_upstream_note = "; equal, parameter by parameter, to dinov2-main's get_params_groups_with_decay (n_blocks 24)"
else:
    ray_upstream_note = "; SKIP the dinov2-main cross-check ({} not found)".format(upstream_file)
print("OK (i2) param groups for depth 24: 53 groups; patch_embed 0.9**25 * 0.2, pos_embed / cls_token / mask_token 0.9**25, blocks.i 0.9**(24 - i) (blocks.10 ... blocks.23 included), norm / projector / heads 1.0; "
      "no decay on bias / norm / LayerScale gamma" + ray_upstream_note)

# (i3) --layer-decay accepts the name (DINO_VIT_MODELS) and builds the 53-group AdamW; --freeze_backbone freezes exactly the 343 backbone tensors (last: it changes requires_grad)
ray_recipe = SimpleNamespace(model_name="vit_large_raydino", opt="adamw", lr=1e-4, opt_eps=1e-8, momentum=0.9, weight_decay=0.04, layer_decay=0.9, patch_embed_lr_mult=0.2)
built = engine._create_vit_finetune_optimizer(ray_recipe, ray)
assert isinstance(built, torch.optim.AdamW) and len(built.param_groups) == 53
message = rejects(ValueError, lambda: engine._create_vit_finetune_optimizer(SimpleNamespace(model_name="swin_base", opt="adamw", lr=1e-4, opt_eps=1e-8, momentum=0.9,
                                                                                      weight_decay=0.04, layer_decay=0.9, patch_embed_lr_mult=0.2), ray), "only implemented for --model vit_base_dinov3")
assert message.endswith("vit_base_dinov3 | vit_base_dinov2 | vit_base_raddino | vit_large_raydino"), message
ray_summary = engine._freeze_backbone(ray)
ray_frozen = [n for n, p in ray.named_parameters() if not p.requires_grad]
ray_trainable = [n for n, p in ray.named_parameters() if p.requires_grad]
assert len(ray_frozen) == 343 and all(n.startswith(HEAD_PREFIXES) for n in ray_trainable) and len(ray_trainable) == 6
print("OK (i3) recipe: --layer-decay builds the 53-group AdamW for vit_large_raydino (the refusal for other models lists all four names); --freeze_backbone: " + ray_summary)
del ray, built, embeddings, tokens, feats, outs, ray_named, ray_groups, ray_group_of  # the full-size model (1.2 GB) goes before the next ones are built

# (i4) the key guard, on a backbone of the real width with one block (2048-d features, cheap on the CPU): the same file shapes as above, labelled RayDINO
TINY_RAY = dict(DINOV2_VITL16_KWARGS, img_size=32, depth=1)
saved_ray_kwargs = models.DINOV2_VITL16_KWARGS
models.DINOV2_VITL16_KWARGS = TINY_RAY  # build_omni_model reads the module global
try:
    with tempfile.TemporaryDirectory() as tmp:
        def ray_file(name, state):
            path = os.path.join(tmp, name + ".pth")
            torch.save(state, path)
            return path

        tiny_ray = ArkViTDinoV2([], None, False, encoder_features=2048, **TINY_RAY).state_dict()
        loaded = build_omni_model(ark_args("vit_large_raydino", 32, ray_file("good", tiny_ray)), [6])  # a backbone-only file: only projector / heads missing, accepted
        assert all(torch.equal(loaded.state_dict()[k], v) for k, v in tiny_ray.items()) and loaded.num_features == 32  # the file's tensors, and a projector in front of the heads
        probe = build_omni_model(ark_args("vit_large_raydino", None, ray_file("good2", tiny_ray)), [6]).eval()  # what the extractor builds: no projector
        with torch.no_grad():
            assert probe.projector is None and probe.num_features == 2048 and probe.generate_embeddings(torch.randn(2, 3, 32, 32), after_proj=False).shape == (2, 2048)
        guard = "RayDINO checkpoint does not match the backbone"
        for label, broken, key in (("missing", {k: v for k, v in tiny_ray.items() if k != "blocks.0.attn.qkv.weight"}, "blocks.0.attn.qkv.weight"),
                                   ("no_pos_embed", {k: v for k, v in tiny_ray.items() if k != "pos_embed"}, "['pos_embed']"),
                                   ("unexpected", dict(tiny_ray, register_tokens=torch.zeros(1, 4, 1024)), "register_tokens"),
                                   ("wrapped", {"teacher": dict(tiny_ray)}, "unexpected keys ['teacher']")):
            message = rejects(RuntimeError, lambda: build_omni_model(ark_args("vit_large_raydino", 32, ray_file(label, broken)), [6]), guard)
            assert key in message, (label, key)
        # a shape mismatch: torch's own lines (the offending key and both shapes) under the same label. A 224 px pos_embed, and the likeliest wrong file: a ViT-B's
        message = rejects(RuntimeError, lambda: build_omni_model(ark_args("vit_large_raydino", 32, ray_file("pos224", dict(tiny_ray, pos_embed=torch.zeros(1, 197, 1024)))), [6]), guard)
        assert "size mismatch for pos_embed" in message and "torch.Size([1, 197, 1024])" in message and "torch.Size([1, 5, 1024])" in message, message
        vit_b = ArkViTDinoV2([], None, False, **dict(DINOV2_VITB14_KWARGS, depth=1)).state_dict()
        message = rejects(RuntimeError, lambda: build_omni_model(ark_args("vit_large_raydino", 32, ray_file("vit_b", vit_b)), [6]), guard)
        assert "size mismatch for cls_token" in message and "torch.Size([1, 1, 768])" in message and "torch.Size([1, 1, 1024])" in message, message
        models.DINOV2_VITB14_KWARGS = TINY
        try:
            for name, label_text in (("vit_base_dinov2", "DINOv2"), ("vit_base_raddino", "RAD-DINO")):  # the other names get the label on a shape mismatch too (a ViT-L-width file into the tiny ViT-B)
                rejects(RuntimeError, lambda: build_omni_model(ark_args(name, 32, ray_file("vit_l_" + name, tiny_ray)), [6]), label_text + " checkpoint does not match the backbone: Error(s) in loading state_dict")
        finally:
            models.DINOV2_VITB14_KWARGS = saved_kwargs
finally:
    models.DINOV2_VITL16_KWARGS = saved_ray_kwargs
print("OK (i4) key guard for vit_large_raydino: a one-block ViT-L-width file loads strictly (projector / heads missing only; without a projector 2048-d features); a missing block tensor, a missing pos_embed, "
      "register_tokens and a wrapped checkpoint raise 'RayDINO checkpoint does not match the backbone' naming the keys; a 224 px pos_embed and a ViT-B file raise it with torch's size-mismatch lines "
      "(the key and both shapes)")


# (i5) Meta's released file is in DINOv2's block_chunks=4 layout, blocks.<chunk>.<block>.* with <block> counting over the whole depth: the loader renames it to the flat layout the model is built with, and says so
class Notices(logging.Handler):
    """The warnings and errors the 'ark' logger gets (not its info lines: after (h) the engine has left it at INFO)."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def load_with_notices(name, path, kwargs_for_model):
    """build_omni_model(vit_large_raydino) on `path` with the model sizes `kwargs_for_model`: (the model, the notices it logged on the 'ark' logger)."""
    notices, ark_logger, saved = Notices(), logging.getLogger("ark"), models.DINOV2_VITL16_KWARGS
    ark_logger.addHandler(notices)
    models.DINOV2_VITL16_KWARGS = kwargs_for_model
    try:
        return build_omni_model(ark_args(name, None, path), []).eval(), notices.messages
    finally:
        models.DINOV2_VITL16_KWARGS = saved
        ark_logger.removeHandler(notices)


TINY_CHUNKS = dict(DINOV2_VITL16_KWARGS, img_size=32, embed_dim=64, depth=8, num_heads=2)  # 8 blocks of width 64: 4 chunks of 2 blocks (the real file: 4 chunks of 6)
flat_model = ArkViTDinoV2([], None, False, encoder_features=2048, **TINY_CHUNKS)
chunked_model = ArkViTDinoV2([], None, False, encoder_features=2048, **dict(TINY_CHUNKS, block_chunks=4)).eval()
flat_state, chunked_state = flat_model.state_dict(), chunked_model.state_dict()
assert "blocks.1.2.norm1.weight" in chunked_state and "blocks.3.7.ls2.gamma" in chunked_state and "blocks.1.0.norm1.weight" not in chunked_state  # DINOv2 numbers the blocks over the whole depth
renamed, chunks = flatten_chunked_blocks(chunked_state, 8)
assert chunks == 4 and list(renamed) == list(flat_state) and all(renamed[k] is chunked_state[c] for k, c in zip(renamed, chunked_state))  # same names, same order, the same tensors
within = {}  # the same file with <block> counted within its chunk (blocks.1.0.* ... blocks.1.1.*): placed all the same
for k, v in chunked_state.items():
    m = re.fullmatch(r"blocks\.(\d+)\.(\d+)\.(.+)", k)
    within["blocks.{}.{}.{}".format(m.group(1), int(m.group(2)) - 2 * int(m.group(1)), m.group(3)) if m else k] = v
assert "blocks.1.0.norm1.weight" in within and flatten_chunked_blocks(within, 8)[1] == 4 and list(flatten_chunked_blocks(within, 8)[0]) == list(flat_state)
assert flatten_chunked_blocks(flat_state, 8)[0] is flat_state and flatten_chunked_blocks(flat_state, 8)[1] == 0  # a flat file (and one with no blocks at all) is returned untouched
assert flatten_chunked_blocks({"cls_token": 1}, 8) == ({"cls_token": 1}, 0)
mixed = dict(chunked_state, **{"blocks.0.norm1.weight": flat_state["blocks.0.norm1.weight"]})  # one flat key among chunked ones
rejects(RuntimeError, lambda: flatten_chunked_blocks(mixed, 8), "neither the flat layout")
rejects(RuntimeError, lambda: flatten_chunked_blocks(chunked_state, 6), "4 chunk(s)")  # 4 chunks do not divide 6 blocks
rejects(RuntimeError, lambda: flatten_chunked_blocks(chunked_state, 4), "4 chunk(s)")  # as many chunks as blocks: not a chunked layout
rejects(RuntimeError, lambda: flatten_chunked_blocks({"blocks.0.0.a": 1, "blocks.1.9.a": 1}, 8), "blocks.1.9.a: block 9 is not one of chunk 1's blocks 4..7")  # 2 chunks of 4
rejects(RuntimeError, lambda: flatten_chunked_blocks({"blocks.0.5.a": 1, "blocks.1.4.a": 1}, 8), "blocks.0.5.a: block 5 is not one of chunk 0's blocks 0..3")
rejects(RuntimeError, lambda: flatten_chunked_blocks({"blocks.0.0.a": 1, "blocks.1.4.a": 1, "blocks.1.0.a": 1}, 8), "both become blocks.4.a")  # absolute 4 and within-chunk 0 of chunk 1
# what the loader logs about a RayDINO file before it renames or loads anything: one line with the key count, every non-block key and its shape, the first and last block key and the layout
tiny_head = ("RayDINO checkpoint: 119 keys. Not blocks (7): cls_token [1, 1, 64], pos_embed [1, 5, 64], mask_token [1, 64], patch_embed.proj.weight [64, 3, 16, 16], patch_embed.proj.bias [64], "
             "norm.weight [64], norm.bias [64]. Blocks (112): ")
assert checkpoint_summary(chunked_state) == tiny_head + "first blocks.0.0.attn.proj.bias [64], last blocks.3.7.norm2.weight [64]; layout 4 chunks, 8 blocks"
assert checkpoint_summary(within) == tiny_head + "first blocks.0.0.attn.proj.bias [64], last blocks.3.1.norm2.weight [64]; layout 4 chunks, 8 blocks"  # <block> counted within the chunk: the last key says so
assert checkpoint_summary(flat_state) == tiny_head + "first blocks.0.attn.proj.bias [64], last blocks.7.norm2.weight [64]; layout flat, 8 blocks"
assert checkpoint_summary(mixed).endswith("layout mixed, 112 chunked and 1 flat block keys")
assert checkpoint_summary({"cls_token": torch.zeros(1, 1, 4)}) == "RayDINO checkpoint: 1 keys. Not blocks (1): cls_token [1, 1, 4]; layout no block keys"
assert checkpoint_summary({"teacher": dict(flat_state)}) == "RayDINO checkpoint: 1 keys. Not blocks (1): teacher dict; layout no block keys"  # a wrapped checkpoint shows as its one key
assert checkpoint_summary({}) == "RayDINO checkpoint: 0 keys. Not blocks (0): none; layout no block keys"
assert "Not blocks (8)" in checkpoint_summary(dict(flat_state, register_tokens=torch.zeros(1, 4, 64))) and "register_tokens [1, 4, 64]" in checkpoint_summary(dict(flat_state, register_tokens=torch.zeros(1, 4, 64)))
many = checkpoint_summary(dict(flat_state, **{"head.%d" % i: torch.zeros(2) for i in range(40)}))  # 47 keys that are not blocks: the first 32 are listed
assert "Not blocks (47)" in many and "head.24 [2]" in many and "head.25" not in many and ", ... +15 more. Blocks (112)" in many
assert all("\n" not in s for s in (checkpoint_summary(chunked_state), many))  # one line
with tempfile.TemporaryDirectory() as tmp:
    torch.manual_seed(3)
    small = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        want = chunked_model.generate_embeddings(small, after_proj=False)
    assert want.shape == (2, 128)  # CLS ++ mean patch of width 64 (the stand-in is not 2048-d: no projector is built, so nothing depends on encoder_features)
    for label, state in (("chunked", chunked_state), ("within", within)):
        path = os.path.join(tmp, label + ".pth")
        torch.save(state, path)
        model, notices = load_with_notices("vit_large_raydino", path, TINY_CHUNKS)
        with torch.no_grad():
            assert torch.equal(model.generate_embeddings(small, after_proj=False), want), label  # the renamed file is the chunked model, function for function
        assert len(notices) == 2 and notices[0] == checkpoint_summary(state), notices  # the summary of the file as it is, first
        assert "chunked block layout (4 chunks of 2 blocks" in notices[1] and "renamed its block keys to blocks.<block>.*" in notices[1], notices
    path = os.path.join(tmp, "flat.pth")
    torch.save(flat_state, path)
    assert load_with_notices("vit_large_raydino", path, TINY_CHUNKS)[1] == [checkpoint_summary(flat_state)]  # a flat file: the summary, nothing renamed, nothing said about renaming
    dropped = os.path.join(tmp, "dropped.pth")
    torch.save({k: v for k, v in chunked_state.items() if not k.startswith("blocks.2.4.")}, dropped)  # a block short: renamed, then refused by the key guard
    message = rejects(RuntimeError, lambda: load_with_notices("vit_large_raydino", dropped, TINY_CHUNKS), "RayDINO checkpoint does not match the backbone: missing keys ['blocks.4.")
    other = os.path.join(tmp, "other_model.pth")
    torch.save(chunked_state, other)
    rejects(RuntimeError, lambda: load_with_notices("vit_base_dinov2", other, TINY_CHUNKS), "DINOv2 checkpoint does not match the backbone")  # the renaming is RayDINO's only: another name sees chunked keys as unexpected
print("OK (i5) chunked layout: blocks.<chunk>.<block>.* (4 chunks, <block> over the whole depth, or counted within the chunk) is renamed to blocks.<block>.* in file order and the build_omni_model result "
      "equals the chunked model function for function, after the file's one-line summary (key count, non-block keys with shapes, first and last block key, layout) and with one 'chunked block layout ... renamed' notice; "
      "a flat file is left alone (the summary only); mixed layouts, a chunk count that does not divide the depth, a block outside its chunk and a name collision raise; a block short is refused by the "
      "RayDINO key guard; other model names are not renamed")
del flat_model, chunked_model, flat_state, chunked_state, renamed, within, mixed

# (i6) the same at the real size: a ViT-L/16 built with block_chunks=4 (the layout of Meta's xray_dino_vitl16), its state dict as a float16 file (every weight rounded to float16 first, so none is lost):
# vit_large_raydino loads it, names the renaming, and its (2, 2048) features at 224 px equal the chunked model's
with tempfile.TemporaryDirectory() as tmp:
    torch.manual_seed(4)
    chunked_large = ArkViTDinoV2([], None, False, encoder_features=2048, **dict(DINOV2_VITL16_KWARGS, block_chunks=4)).eval()
    assert chunked_large.chunked_blocks and len(chunked_large.blocks) == 4 and "blocks.3.23.mlp.fc2.weight" in chunked_large.state_dict()
    with torch.no_grad():
        for p in chunked_large.parameters():
            p.copy_(p.half().float())
        want = chunked_large.generate_embeddings(x, after_proj=False)
    path = os.path.join(tmp, "chunked_large.pth")
    torch.save({k: v.half() for k, v in chunked_large.state_dict().items()}, path)
    n_keys = len(chunked_large.state_dict())
    del chunked_large
    model, notices = load_with_notices("vit_large_raydino", path, DINOV2_VITL16_KWARGS)
    with torch.no_grad():
        got = model.generate_embeddings(x, after_proj=False)
    assert n_keys == 343 and got.shape == (2, 2048) and torch.equal(got, want), (got.shape, (got - want).abs().max().item())
    summaries = [n for n in notices if n.startswith("RayDINO checkpoint: ")]
    renames = [n for n in notices if "chunked block layout" in n]
    assert notices == summaries + renames and len(summaries) == 1 and len(renames) == 1, notices  # the summary of the file first, once, then the renaming notice, once
    assert "(4 chunks of 6 blocks" in renames[0] and "pos_embed [1, 1025, 1024]" in summaries[0], notices
    assert summaries[0] == ("RayDINO checkpoint: 343 keys. Not blocks (7): cls_token [1, 1, 1024], pos_embed [1, 1025, 1024], mask_token [1, 1024], patch_embed.proj.weight [1024, 3, 16, 16], "
                            "patch_embed.proj.bias [1024], norm.weight [1024], norm.bias [1024]. Blocks (336): first blocks.0.0.attn.proj.bias [1024], last blocks.3.23.norm2.weight [1024]; layout 4 chunks, 24 blocks")
    del model
print("OK (i6) chunked ViT-L/16 at the real size (4 chunks of 6 blocks, 343 tensors, float16 file): vit_large_raydino renames the keys, says so, and gives (2, 2048) features equal to the chunked model's; "
      "it logs once, before renaming, the line: " + summaries[0])

# (i7) Meta's own xray_dino_vitl16 hub entry (dinov2-main, no weights), built in a fresh interpreter: its 343 key names and shapes, once renamed, are the model's
UPSTREAM_KEYS = r"""
import json, sys
source, out = sys.argv[1:3]
sys.path.insert(0, source)
import dinov2
assert dinov2.__file__.startswith(source), dinov2.__file__
from dinov2.hub.xray_dino.backbones import xray_dino_vitl16
model = xray_dino_vitl16(pretrained=False)
assert (model.patch_size, model.num_register_tokens, model.interpolate_offset, model.interpolate_antialias, model.n_blocks, model.chunked_blocks) == (16, 0, 0.1, False, 24, True)
json.dump({k: list(v.shape) for k, v in model.state_dict().items()}, open(out, "w"))
"""
if not os.path.isfile(os.path.join(DINOV2_MAIN, "dinov2", "hub", "xray_dino", "backbones.py")):
    print("SKIP (i7): Meta's xray_dino_vitl16 hub entry not found in {} (set DINOV2_MAIN)".format(DINOV2_MAIN))
else:
    with tempfile.TemporaryDirectory() as tmp:
        keys_out = os.path.join(tmp, "keys.json")
        subprocess.run([sys.executable, "-I", "-W", "ignore", "-c", UPSTREAM_KEYS, DINOV2_MAIN, keys_out], cwd=tmp, check=True)
        with open(keys_out) as f:
            upstream_shapes = json.load(f)
    upstream_flat, upstream_chunks = flatten_chunked_blocks(upstream_shapes, 24)
    assert len(upstream_shapes) == len(ray_shapes) == 343 and upstream_chunks == 4 and "blocks.1.6.attn.qkv.weight" in upstream_shapes and "blocks.3.23.ls2.gamma" in upstream_shapes
    assert upstream_flat == ray_shapes  # the same names and shapes, whichever order
    print("OK (i7) Meta's xray_dino_vitl16 (patch 16, 512 px, no registers, interpolate_offset 0.1, block_chunks=4): its 343 keys (blocks.<chunk>.<block>.*, <block> over the whole depth) renamed by "
          "flatten_chunked_blocks are, with their shapes, exactly vit_large_raydino's")

print("ALL OK")
