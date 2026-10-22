"""Parity check: a converted Meta-layout DINOv3 .pth (convert_hf_dinov3_to_meta.py) is numerically the same model as the Hugging Face checkpoint it was converted from.

Run from base/:  python check_hf_dinov3_parity.py --model vit_base_dinov3 <hf_dir> <converted.pth> [--input-size 224] [--seed 0] [--tol 1e-4] [--dtype float32]
--model is vit_base_dinov3, vit_large_dinov3, vit_huge_plus_dinov3 or vit_7b_dinov3; <hf_dir> is the HF directory (config.json and its safetensors), <converted.pth> what the converter wrote from it.
Ours is build_omni_model(--model, pretrained_weights=<converted.pth>), strict key guard included; the reference is transformers' DINOv3ViTModel.from_pretrained(<hf_dir>, torch_dtype=...). Both run in eval mode.
What is compared is the raw tokens after the final norm, not Ark's feature: the CLS token, the mean patch token (the half of ArkViT's feature that is not the CLS) and every patch token. Ours comes from the
backbone's own DinoVisionTransformer.forward_features (ArkViT overrides it to concatenate), HF's from last_hidden_state = [CLS, register tokens, patch tokens] (config.num_register_tokens of them).
Two inputs: seeded noise, and a linspace ramp (a bug that happens to cancel on noise still shows on a structured image). PARITY OK when every max |diff| is <= --tol, else PARITY FAIL and exit 1.
"max rel" is max |diff| / max |HF| (an element near zero cannot inflate it). Measured with this script on the real ViT-B checkpoint (2026-10-22, float32): noise 2.4e-6 on the CLS and 5.0e-6 on the patch tokens,
ramp 6.9e-6 and 1.05e-5, so --tol 1e-4 leaves a factor of 10 over the worst (the check of 2026-09-30 had found 2.5e-6 and 5.3e-6 on noise).
One known architectural difference: ours is built with pos_embed_rope_rescale_coords=None (models.py), HF and Meta's hub use 2.0. That is a random rescale of the rope coordinates that only runs in
training mode, so the two eval-mode models must agree; both are asserted to be in eval mode before the forward.
Memory: both models are resident at once (the 7B is about 27 GB per copy in float32); an estimate is printed first. --dtype bfloat16 builds both in bfloat16 (half of that) and then needs an explicit --tol:
1e-4 is a float32 figure, and none has been calibrated for bfloat16 on the real models (two independent 2-block random models already differ by about 0.1 on tokens of magnitude 3; read the table).
Needs `transformers`, which neither the Mac venv nor the Sol container has: on Sol, check_hf_dinov3_parity.sh (repo root) installs the pinned version into /workspace/.pylib-transformers-<pin> and runs this inside the container.
"""
import argparse
import contextlib
import importlib.util
import os
import resource
import sys
import time
from types import SimpleNamespace

import torch

from convert_hf_dinov3_to_meta import MODELS
from dinov3.layers import LinearKMaskedBias
from dinov3.models.vision_transformer import DinoVisionTransformer
from models import build_omni_model

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
FLOAT32_TOL = 1e-4


def say(message):
    print(message, flush=True)  # the job's log is a file: flush, or a 7B load shows nothing for minutes


@contextlib.contextmanager
def default_dtype(dtype):
    """torch's default float dtype while a model is built: its parameters are created in `dtype` straight away, so a bfloat16 7B never exists in float32 (13 GB, not 27).
    The rope periods keep the float32 their constructor asks for."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def build_ours(name, converted, dtype):
    """Ours: build_omni_model for --model `name` with the converted file (None: no weights), in eval mode. With a file it raises unless only the projector and heads are missing and nothing is unexpected."""
    args = SimpleNamespace(model_name=name, projector_features=1376, use_mlp=False, pretrained_weights=converted)
    with default_dtype(dtype):
        return build_omni_model(args, []).eval()


def have_transformers():
    return importlib.util.find_spec("transformers") is not None


def load_hf(hf_dir, dtype):
    """The reference. transformers is imported here, not at the top: this module imports without it (the tests do)."""
    from transformers import DINOv3ViTModel
    return DINOv3ViTModel.from_pretrained(hf_dir, torch_dtype=dtype).eval()


def info(model):
    """(register tokens, the rope coordinate rescale that only training mode applies) of ours or of an HF model."""
    if isinstance(model, DinoVisionTransformer):
        return model.n_storage_tokens, model.rope_embed.rescale_coords
    return model.config.num_register_tokens, model.config.pos_embed_rescale


def n_params(model):
    """Backbone parameters of either model: not Ark's projector and heads, and not the masked k-bias third of each fused qkv.bias (Meta's bias_mask zeroes it, HF has no k bias)."""
    masked = sum(m.out_features // 3 for m in model.modules() if isinstance(m, LinearKMaskedBias) and m.bias is not None)
    return sum(p.numel() for k, p in model.named_parameters() if not k.startswith(("projector.", "omni_heads."))) - masked


def tokens(model, x):
    """(CLS [B, C], patch tokens [B, N, C]) after the final norm. Ours: the backbone's own dict (not ArkViT's override, which concatenates). HF: last_hidden_state is [CLS, registers, patches]."""
    if isinstance(model, DinoVisionTransformer):
        out = DinoVisionTransformer.forward_features(model, x)
        return out["x_norm_clstoken"], out["x_norm_patchtokens"]
    hidden = model(pixel_values=x).last_hidden_state
    return hidden[:, 0], hidden[:, 1 + info(model)[0]:]


def make_inputs(size, seed):
    """Two batches of 2 images: seeded noise, and a linspace ramp over the whole tensor (structured: a bug that cancels on noise still shows)."""
    torch.manual_seed(seed)
    noise = torch.randn(2, 3, size, size)
    return {"noise": noise, "ramp": torch.linspace(-2, 2, noise.numel()).reshape(noise.shape)}


def stats(ref, got):
    """(max |diff|, max |diff| / max |ref|, max |ref|); float32 whatever the models' dtype, so the bfloat16 subtraction does not add its own rounding."""
    ref, got = ref.float(), got.float()
    if ref.shape != got.shape:
        sys.exit("PARITY FAIL: the token tensors differ in shape, ref {} vs ours {}".format(tuple(ref.shape), tuple(got.shape)))
    diff, scale = (ref - got).abs().max().item(), ref.abs().max().item()
    return diff, diff / scale if scale else float("inf"), scale


def compare(ref, ours, inputs, tol):
    """Run both models on every input in `inputs` ({label: images}); print one line per (input, tensor) and the verdict. -> (ok, the largest max |diff|)
    A row passes when its max |diff| <= tol and the reference is not all zeros (a NaN anywhere fails: every comparison with NaN is False)."""
    for model in (ref, ours):
        assert not any(m.training for m in model.modules()), "both models must be in eval mode: the rope rescale is a training-mode augmentation"
    rows = []
    with torch.no_grad():
        for label, x in inputs.items():
            (ref_cls, ref_patches), (our_cls, our_patches) = (tokens(m, x.to(next(m.parameters()).dtype)) for m in (ref, ours))
            for name, a, b in (("CLS", ref_cls, our_cls), ("mean patch", ref_patches.mean(dim=1), our_patches.mean(dim=1)), ("all patches", ref_patches, our_patches)):
                rows.append((label, name) + stats(a, b))
    say("{:<6} {:<12} {:>10} {:>10} {:>10}".format("input", "tensor", "max|diff|", "max rel", "max|ref|"))
    for label, name, diff, rel, scale in rows:
        say("{:<6} {:<12} {:>10.2e} {:>10.2e} {:>10.2e}{}".format(label, name, diff, rel, scale, "" if scale > 0 and diff <= tol else "   FAIL"))
    passed = sum(scale > 0 and diff <= tol for _, _, diff, _, scale in rows)
    diffs = [row[2] for row in rows]
    worst = float("nan") if any(d != d for d in diffs) else max(diffs)
    say("{} of {} comparisons within tol {:.0e}; largest max |diff| {:.2e}".format(passed, len(rows), tol, worst))
    ok = passed == len(rows)
    say("PARITY OK" if ok else "PARITY FAIL")
    return ok, worst


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODELS), help="the build_omni_model name the converted file is for")
    parser.add_argument("hf_dir", help="the HF checkpoint directory (config.json and its safetensors)")
    parser.add_argument("converted", help="the .pth that convert_hf_dinov3_to_meta.py wrote from it")
    parser.add_argument("--input-size", type=int, default=224, help="side of the square test images in pixels, a multiple of the patch size (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the noise input (default: %(default)s)")
    parser.add_argument("--tol", type=float, help="the largest max |diff| that passes (default {} for float32; required for --dtype bfloat16)".format(FLOAT32_TOL))
    parser.add_argument("--dtype", choices=list(DTYPES), default="float32", help="load both models in this dtype (default: %(default)s)")
    args = parser.parse_args(argv)
    dtype, kwargs = DTYPES[args.dtype], MODELS[args.model]
    if args.tol is None and args.dtype != "float32":
        parser.error("--dtype {} needs an explicit --tol: {} is a float32 figure, and no {} tolerance has been measured".format(args.dtype, FLOAT32_TOL, args.dtype))
    tol = FLOAT32_TOL if args.tol is None else args.tol
    if not tol >= 0:
        parser.error("--tol {} is not a non-negative number".format(args.tol))
    if args.input_size <= 0 or args.input_size % kwargs["patch_size"]:
        parser.error("--input-size {} is not a positive multiple of the patch size {}".format(args.input_size, kwargs["patch_size"]))
    for path in (os.path.join(args.hf_dir, "config.json"), args.converted):  # the cheapest signs of the wrong directory, before a 7B takes minutes to load
        if not os.path.isfile(path):
            sys.exit("error: {} not found".format(path))
    if not have_transformers():  # before anything is built: a 7B takes minutes to load
        sys.exit("error: transformers cannot be imported here; on Sol run check_hf_dinov3_parity.sh (it installs the pinned version into /workspace/.pylib-transformers-<pin> and sets PYTHONPATH inside the container)")

    with torch.device("meta"):
        n_meta = sum(p.numel() for p in DinoVisionTransformer(**kwargs).parameters())
    model_gb, file_gb = n_meta * torch.empty((), dtype=dtype).element_size() / 1e9, os.path.getsize(args.converted) / 1e9
    say("model {} | dtype {} | images 2x3x{}x{} (seed {}) | tol {:.0e}".format(args.model, args.dtype, args.input_size, args.input_size, args.seed, tol))
    say("estimated peak RSS ~{:.0f} GB: two {} models of {:.1f} GB ({:,} parameters) resident together; ours is built first, while its {:.1f} GB checkpoint is also in memory; activations are small".format(
        max(2 * model_gb, model_gb + file_gb), args.dtype, model_gb, n_meta, file_gb))

    start = time.perf_counter()
    ours = build_ours(args.model, args.converted, dtype)
    say("ours: build_omni_model({}, pretrained_weights={}) passed its key guard, {:.0f} s".format(args.model, args.converted, time.perf_counter() - start))
    start = time.perf_counter()
    ref = load_hf(args.hf_dir, dtype)
    say("HF: DINOv3ViTModel.from_pretrained({}, torch_dtype={}), {:.0f} s".format(args.hf_dir, dtype, time.perf_counter() - start))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)  # bytes on macOS, KiB on Linux
    say("peak RSS after loading both: {:.1f} GB".format(peak / 1e9))

    n_ours, n_ref = n_params(ours), n_params(ref)
    say("parameters: ours {:,} (backbone only: no projector or heads, no masked k-bias entries) | HF {:,} | {}".format(n_ours, n_ref, "match" if n_ours == n_ref else "DO NOT MATCH"))
    (our_registers, our_rescale), (ref_registers, ref_rescale) = info(ours), info(ref)
    say("register tokens: ours {} | HF {} (config.num_register_tokens, which splits last_hidden_state)".format(our_registers, ref_registers))
    say("rope rescale: ours pos_embed_rope_rescale_coords={} | HF config.pos_embed_rescale={}: a random rescale of the rope coordinates that only runs in training mode; both models are in eval mode, so they must agree".format(
        our_rescale, ref_rescale))
    ok, _ = compare(ref, ours, make_inputs(args.input_size, args.seed), tol)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
