"""CPU checks for check_hf_dinov3_parity.py and check_hf_dinov3_parity.sh without transformers (the Mac venv has none): the comparison runs on ours against ours, against a stand-in with HF's output layout and
against perturbed copies; the real comparison with a Hugging Face checkpoint runs on Sol. Run from base/: python test_check_hf_dinov3_parity.py
(a) the inputs (seeded noise, a linspace ramp); the parameter counts; the HF branch of tokens() on a stand-in with HF's [CLS, registers, patches] layout, whatever its register count
(b) two identical ViT-Bs (full size, built without weights) compare equal: PARITY OK; so do two identical SwiGLU models, and a stand-in with HF's layout around one of them
(c) a perturbed tensor is caught, with a max |diff| above tol: two rows of blocks.0.mlp.fc1 swapped in the ViT-B; w1 and w2 (the gate and up projections, the pair a conversion can swap: w1 and w3 differ in
    shape) swapped in a 2-block SwiGLU model. A model in training mode, a NaN, an all-zero reference and a shape mismatch are refused or fail rather than pass
(d) bfloat16: the model is built in bfloat16 (every tensor the file's float32 rounded once, the rope periods float32) and the default dtype is restored, whatever happens
(e) the command line: an unknown or missing --model, bfloat16 without --tol, a size that is not a multiple of 16, a negative --tol, an HF directory that is missing or has no config.json, a missing .pth, and no transformers are all
    refused before anything is built; and the script in fresh interpreters: it imports without transformers, --help works, an unknown model name exits 2
(f) main() end to end behind a stand-in for the HF loader, on a tiny model whose file is saved the way the converter writes it: PARITY OK and exit 0 for the right file, PARITY FAIL and exit 1 for a file with
    two rows swapped, and the same in bfloat16
(g) check_hf_dinov3_parity.sh: valid bash, executable, no SBATCH lines, the cache directories in .gitignore; in a fake repo behind a stub nvidia.sh: the usage refusal, the one-time staged install into a directory
    named after the pin (numpy dropped, transformers moved in last, no stage left behind, a failed install leaves nothing, a job that loses the mkdir race keeps the other's copy, a cache without transformers is
    reported and left alone, a .pylib or another pin's directory is neither used nor changed), the command it runs inside the container, and its exit status"""
import contextlib
import copy
import fnmatch
import io
import os
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch

import check_hf_dinov3_parity as parity
import models
from dinov3.models.vision_transformer import DinoVisionTransformer
from models import DINOV3_VITB16_KWARGS, DINOV3_VITH16PLUS_KWARGS, DINOV3_VITL16_KWARGS, DINOV3_VIT7B16_KWARGS

HERE = os.path.dirname(os.path.abspath(__file__))
SH = os.path.join(os.path.dirname(HERE), "check_hf_dinov3_parity.sh")
TINY = dict(DINOV3_VITB16_KWARGS, embed_dim=64, depth=2, num_heads=4)  # a few thousand parameters; 4 heads of 16 keep the rope's embed_dim % (4 * heads) == 0
TINY_GATED = dict(TINY, ffn_ratio=6.0, ffn_layer="swiglu")  # the ViT-H+'s FFN
TOL = 1e-4


def rejects(exception, call, expected=None):
    """The message of the exception call() raises (the code of a SystemExit); fails when it returns. `expected` is a string the message must hold."""
    try:
        call()
    except exception as error:
        message = str(error.code) if isinstance(error, SystemExit) else str(error)
        assert expected is None or expected in message, (expected, message)
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


def captured(call):
    """call() with stdout and stderr captured. -> (what it returned, or a SystemExit's code; stdout; stderr)"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            result = call()
        except SystemExit as stop:
            result = stop.code
    return result, out.getvalue(), err.getvalue()


def matter(model):
    """A fresh model's layer scales are 1e-5, so its blocks add next to nothing and a change in one could not show: set them to 1."""
    with torch.no_grad():
        for block in model.blocks:
            block.ls1.gamma.fill_(1.0)
            block.ls2.gamma.fill_(1.0)
    return model


def build(name):
    return matter(parity.build_ours(name, None, torch.float32))


def comparison(ref, ours, inputs, tol=TOL):
    """parity.compare with its table captured. -> (ok, the largest max |diff|, the printed lines)"""
    (ok, worst), out, _ = captured(lambda: parity.compare(ref, ours, inputs, tol))
    return ok, worst, out.splitlines()


class HFShaped(torch.nn.Module):
    """What DINOv3ViTModel looks like from outside: config.num_register_tokens and pos_embed_rescale, and forward(pixel_values=...).last_hidden_state = [CLS, registers, patches] after the final norm.
    `inner` (one of ours) supplies the CLS and the patch tokens, with `laid_out` register tokens (zeros: they are sliced away) between them; `declared` is the register count the config claims."""

    def __init__(self, inner, laid_out, declared=None):
        super().__init__()
        self.inner, self.laid_out = inner, laid_out
        self.config = SimpleNamespace(num_register_tokens=laid_out if declared is None else declared, pos_embed_rescale=2.0)
        self.eval()  # from_pretrained(...).eval(), as parity.load_hf returns it

    def forward(self, pixel_values):
        out = DinoVisionTransformer.forward_features(self.inner, pixel_values)
        cls, patches = out["x_norm_clstoken"], out["x_norm_patchtokens"]
        registers = torch.zeros(cls.shape[0], self.laid_out, cls.shape[1], dtype=cls.dtype)
        return SimpleNamespace(last_hidden_state=torch.cat([cls[:, None], registers, patches], dim=1))


# (a) the inputs, the parameter counts, the HF branch of tokens() ----------------------------------------------------------------------------------------------------------------------------------------------
inputs = parity.make_inputs(224, 0)
torch.manual_seed(0)
assert list(inputs) == ["noise", "ramp"] and all(x.shape == (2, 3, 224, 224) and x.dtype == torch.float32 for x in inputs.values())
assert torch.equal(inputs["noise"], torch.randn(2, 3, 224, 224))  # torch.manual_seed(seed), then torch.randn(2, 3, size, size), as specified
again, other = parity.make_inputs(224, 0), parity.make_inputs(224, 1)
assert torch.equal(again["noise"], inputs["noise"]) and not torch.equal(other["noise"], inputs["noise"]) and torch.equal(other["ramp"], inputs["ramp"])  # the seed moves the noise only
ramp = inputs["ramp"].flatten()
assert ramp[0] == -2 and ramp[-1] == 2 and bool((ramp[1:] > ramp[:-1]).all())  # a ramp over the whole tensor: structured, not random
print("OK (a1) the inputs: 2 x 3 x 224 x 224 seeded noise (torch.manual_seed, then torch.randn: the same for a seed, different for another) and a linspace ramp from -2 to 2 that does not depend on the seed")

# Meta-layout counts (masked k-bias entries included): test_convert_hf_dinov3.py's COUNTS. HF has no k bias, so its count is these less depth x embed_dim where there is a qkv bias; the real HF ViT-B has 85,660,416
# (measured 2026-10-22 on its checkpoint: ours 85,660,416 | HF 85,660,416).
META_COUNTS = {"vit_base_dinov3": (85669632, DINOV3_VITB16_KWARGS), "vit_large_dinov3": (303154176, DINOV3_VITL16_KWARGS), "vit_huge_plus_dinov3": (840633600, DINOV3_VITH16PLUS_KWARGS),
               "vit_7b_dinov3": (6716035072, DINOV3_VIT7B16_KWARGS)}
hf_counts = []
for name, (meta_count, kwargs) in META_COUNTS.items():
    with torch.device("meta"):
        shell = parity.build_ours(name, None, torch.float32)
    masked = kwargs["depth"] * kwargs["embed_dim"] if kwargs["qkv_bias"] else 0
    hf_counts.append(meta_count - masked)
    assert parity.n_params(shell) == hf_counts[-1], name
    assert sum(p.numel() for p in shell.parameters()) > parity.n_params(shell) + masked  # the projector and the heads were left out of the count
assert hf_counts[0] == 85660416
print("OK (a2) parameters without Ark's projector and heads and without the masked k-bias entries: {} for the ViT-B (the real HF ViT-B's count) / ViT-L / ViT-H+ / 7B (no qkv bias, so no entries to leave out)".format(
    " / ".join("{:,}".format(n) for n in hf_counts)))

with patched(models, "DINOV3_VITB16_KWARGS", TINY):
    tiny = build("vit_base_dinov3")
x = torch.randn(2, 3, 32, 32)
with torch.no_grad():
    cls, patches = parity.tokens(tiny, x)
    backbone = DinoVisionTransformer.forward_features(tiny, x)
    assert torch.equal(cls, backbone["x_norm_clstoken"]) and torch.equal(patches, backbone["x_norm_patchtokens"]) and patches.shape == (2, 4, 64)  # the backbone's dict, 2 x 2 patches at 32 px
    assert tiny.forward_features(x).shape == (2, 128)  # ArkViT's own override is CLS ++ mean patch: not what is compared
    for registers in (0, 4, 7):  # the split follows config.num_register_tokens, not a 4 written in the code
        stand_in = HFShaped(tiny, registers)
        assert parity.info(stand_in) == (registers, 2.0) and parity.info(tiny) == (4, None)
        assert stand_in(pixel_values=x).last_hidden_state.shape == (2, 1 + registers + 4, 64)
        stand_cls, stand_patches = parity.tokens(stand_in, x)
        assert torch.equal(stand_cls, cls) and torch.equal(stand_patches, patches), registers  # [CLS, registers, patches] sliced back apart exactly
print("OK (a3) tokens(): ours gives the backbone's CLS and patch tokens (not ArkViT's 2 x 64 concatenation); a stand-in with HF's [CLS, registers, patches] layout is split by config.num_register_tokens "
      "(0, 4 and 7 checked) back into exactly ours; info() reads (register tokens, rope rescale) off either kind")

# (b) identical models pass ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
torch.manual_seed(0)
vit_b = build("vit_base_dinov3")
ok, worst, table = comparison(vit_b, copy.deepcopy(vit_b), inputs)
assert ok and worst < 1e-5 and table[-1] == "PARITY OK" and sum(l.startswith(("noise", "ramp")) for l in table) == 6 and not any(l.endswith("FAIL") for l in table), table
assert table[-2].startswith("6 of 6 comparisons within tol 1e-04; largest max |diff| ")
print("OK (b1) two identical ViT-Bs at 224 px (full size, built without weights): 6 of 6 comparisons within tol {:.0e}, largest max |diff| {:.1e}, last line 'PARITY OK'".format(TOL, worst))
with patched(models, "DINOV3_VITH16PLUS_KWARGS", TINY_GATED):
    gated = build("vit_huge_plus_dinov3")
tiny_inputs = parity.make_inputs(64, 0)
ok, worst, table = comparison(gated, copy.deepcopy(gated), tiny_inputs)
assert ok and worst < 1e-5 and table[-1] == "PARITY OK"
ok, worst, table = comparison(HFShaped(gated, 4), gated, tiny_inputs)  # through the HF branch of tokens(): the registers are sliced away and the rest is ours, exactly
assert ok and worst == 0 and table[-1] == "PARITY OK"
print("OK (b2) two identical 2-block SwiGLU models at 64 px: PARITY OK; so does a stand-in with HF's layout around one of them, with max |diff| exactly 0")

# (c) a perturbed tensor is caught ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
swapped = copy.deepcopy(vit_b)
fc1 = swapped.blocks[0].mlp.fc1
with torch.no_grad():
    fc1.weight[[0, 1]] = fc1.weight[[1, 0]]  # two hidden units' input rows exchanged, their output columns left where they were
    fc1.bias[[0, 1]] = fc1.bias[[1, 0]]
ok, worst, table = comparison(vit_b, swapped, inputs)
assert not ok and worst > TOL and table[-1] == "PARITY FAIL" and table[-2].endswith("{:.2e}".format(worst)) and any(l.endswith("FAIL") for l in table[:-2]), table
print("OK (c1) the ViT-B with two rows of blocks.0.mlp.fc1 swapped: PARITY FAIL, max |diff| {:.2e} > tol {:.0e} ({})".format(worst, TOL, table[-2]))
del swapped, vit_b
gate_up = copy.deepcopy(gated)
with torch.no_grad():
    w1, w2 = gate_up.blocks[0].mlp.w1.weight.clone(), gate_up.blocks[0].mlp.w2.weight.clone()
    gate_up.blocks[0].mlp.w1.weight.copy_(w2)
    gate_up.blocks[0].mlp.w2.weight.copy_(w1)
ok, worst, table = comparison(gated, gate_up, tiny_inputs)
assert not ok and worst > TOL and table[-1] == "PARITY FAIL", table
print("OK (c2) the SwiGLU model with w1 and w2 of block 0 swapped: PARITY FAIL, max |diff| {:.2e} > tol {:.0e}".format(worst, TOL))

training, one_submodule = copy.deepcopy(gated).train(), copy.deepcopy(gated)
one_submodule.blocks[0].attn.train()  # the model itself says eval, one submodule does not
for model in (training, one_submodule):
    rejects(AssertionError, lambda: comparison(gated, model, tiny_inputs), "both models must be in eval mode")
    rejects(AssertionError, lambda: comparison(model, gated, tiny_inputs), "both models must be in eval mode")
nan_model = copy.deepcopy(gated)
with torch.no_grad():
    nan_model.norm.weight[0] = float("nan")
ok, worst, table = comparison(gated, nan_model, tiny_inputs)
assert not ok and worst != worst and table[-1] == "PARITY FAIL"  # a NaN fails, and is reported as one
zeroed = copy.deepcopy(gated)
with torch.no_grad():
    zeroed.norm.weight.zero_()
    zeroed.norm.bias.zero_()
ok, worst, table = comparison(zeroed, copy.deepcopy(zeroed), tiny_inputs)
assert not ok and worst == 0 and table[-1] == "PARITY FAIL"  # an all-zero reference proves nothing: two identical zeros do not pass
code = captured(lambda: parity.compare(HFShaped(gated, 4, declared=3), gated, tiny_inputs, TOL))[0]  # a config that disagrees with the 4 registers laid out: the patch slice is one token too long
assert isinstance(code, str) and code.startswith("PARITY FAIL: the token tensors differ in shape"), code
print("OK (c3) a model in training mode (or with one submodule in it) is refused; a NaN fails and is reported as NaN; an all-zero reference fails even against itself; "
      "a wrong register count (a shape mismatch) exits with PARITY FAIL")
del gate_up, training, one_submodule, nan_model, zeroed, tiny_inputs, gated

# (d) bfloat16: built in bfloat16 straight away, the weights are the file's rounded once, the rope periods stay float32, the default dtype comes back --------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp, patched(models, "DINOV3_VITB16_KWARGS", TINY):
    source = build("vit_base_dinov3")
    backbone_only = {k: v for k, v in source.state_dict().items() if not k.startswith(("projector.", "omni_heads."))}  # what the converter writes
    path = os.path.join(tmp, "tiny.pth")
    torch.save(backbone_only, path)
    half, full = parity.build_ours("vit_base_dinov3", path, torch.bfloat16), parity.build_ours("vit_base_dinov3", path, torch.float32)
    assert torch.get_default_dtype() == torch.float32
    half_state = half.state_dict()
    for key, tensor in backbone_only.items():
        want = torch.float32 if key == "rope_embed.periods" else torch.bfloat16
        assert half_state[key].dtype == want and torch.equal(half_state[key], tensor.to(want)), key  # the file's value, rounded once (the periods not at all)
    assert all(p.dtype == torch.bfloat16 for p in half.parameters()) and all(p.dtype == torch.float32 for p in full.parameters())
    assert not any(m.training for m in half.modules())
    ok, worst, table = comparison(full, half, parity.make_inputs(32, 0), tol=float("inf"))  # the bfloat16 forward runs; with a tolerance of infinity only a NaN fails
    assert ok and 0 < worst < float("inf"), (ok, worst)
    rejects(FileNotFoundError, lambda: parity.build_ours("vit_base_dinov3", os.path.join(tmp, "missing.pth"), torch.bfloat16), "No checkpoint found")
    assert torch.get_default_dtype() == torch.float32  # restored when the build fails too
print("OK (d) bfloat16: every parameter and buffer of the model built from the file is the file's float32 rounded once (the rope periods are float32 and unrounded), the bfloat16 forward runs, "
      "and the default dtype is float32 again after the build, also after a build that raised")
del source, backbone_only, half, full


# (e) the command line: refusals come before anything is built -----------------------------------------------------------------------------------------------------------------------------------------------
def forbidden(*args, **kwargs):
    raise AssertionError("built before the arguments were checked")


with tempfile.TemporaryDirectory() as tmp, patched(parity, "build_ours", forbidden), patched(parity, "load_hf", forbidden), patched(parity, "have_transformers", lambda: True):
    hf_dir, pth = os.path.join(tmp, "hf"), os.path.join(tmp, "x.pth")
    os.mkdir(hf_dir)
    open(os.path.join(hf_dir, "config.json"), "w").close()
    open(pth, "wb").close()
    paths = [hf_dir, pth]
    for argv, expected in ((["--model", "vit_small_dinov3"] + paths, "invalid choice: 'vit_small_dinov3'"),
                           (paths, "the following arguments are required: --model"),
                           (["--model", "vit_base_dinov3", "--dtype", "bfloat16"] + paths, "--dtype bfloat16 needs an explicit --tol"),
                           (["--model", "vit_base_dinov3", "--dtype", "float16"] + paths, "invalid choice: 'float16'"),
                           (["--model", "vit_base_dinov3", "--input-size", "225"] + paths, "--input-size 225 is not a positive multiple of the patch size 16"),
                           (["--model", "vit_base_dinov3", "--input-size", "0"] + paths, "is not a positive multiple of the patch size 16"),
                           (["--model", "vit_base_dinov3", "--tol", "-1"] + paths, "is not a non-negative number")):
        code, out, err = captured(lambda: parity.main(argv))
        assert code == 2 and expected in err and out == "", (argv, code, err)  # argparse's exit status, its message on stderr, nothing on stdout
    for argv, expected in ((["--model", "vit_base_dinov3", os.path.join(tmp, "nope"), pth], os.path.join(tmp, "nope", "config.json") + " not found"),  # no such directory
                           (["--model", "vit_base_dinov3", tmp, pth], os.path.join(tmp, "config.json") + " not found"),  # a directory without a config.json: the wrong one
                           (["--model", "vit_base_dinov3", hf_dir, os.path.join(tmp, "nope.pth")], os.path.join(tmp, "nope.pth") + " not found")):
        code, out, err = captured(lambda: parity.main(argv))
        assert isinstance(code, str) and code.startswith("error: ") and expected in code and out == "", (argv, code)
    with patched(parity, "have_transformers", lambda: False):
        code, out, err = captured(lambda: parity.main(["--model", "vit_base_dinov3"] + paths))
        assert isinstance(code, str) and code.startswith("error: transformers cannot be imported here") and "check_hf_dinov3_parity.sh" in code and out == "", code
print("OK (e) refused with nothing built: an unknown or missing --model, --dtype bfloat16 without --tol, another dtype, --input-size 225 and 0, --tol -1 (argparse's exit 2), an HF directory that is missing "
      "or has no config.json, a missing .pth, and no transformers ('error: ...', pointing at the .sh)")

# the script itself, in fresh interpreters: it imports without transformers (imported only when the HF model is loaded), --help works, and an unknown model name is refused with argparse's exit 2
script = os.path.join(HERE, "check_hf_dinov3_parity.py")
done = subprocess.run([sys.executable, "-W", "ignore", "-c", "import sys, check_hf_dinov3_parity; sys.exit(int('transformers' in sys.modules))"], cwd=HERE, capture_output=True, text=True)
assert done.returncode == 0, done
done = subprocess.run([sys.executable, "-W", "ignore", script, "--help"], cwd=HERE, capture_output=True, text=True)
assert done.returncode == 0 and "Parity check: a converted Meta-layout DINOv3 .pth" in done.stdout and "--model {vit_base_dinov3,vit_large_dinov3,vit_huge_plus_dinov3,vit_7b_dinov3}" in done.stdout, done
done = subprocess.run([sys.executable, "-W", "ignore", script, "--model", "vit_tiny_dinov3", "hf", "converted.pth"], cwd=HERE, capture_output=True, text=True)
assert done.returncode == 2 and "invalid choice: 'vit_tiny_dinov3'" in done.stderr and done.stdout == "", done
print("OK (e2) `python check_hf_dinov3_parity.py` in a fresh interpreter: imports without importing transformers, --help prints the docstring and the four model names, --model vit_tiny_dinov3 exits 2 with 'invalid choice'")

# (f) main() end to end behind a stand-in for the HF loader -----------------------------------------------------------------------------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp, patched(models, "DINOV3_VITB16_KWARGS", TINY), patched(parity, "MODELS", dict(parity.MODELS, vit_base_dinov3=TINY)):
    hf_dir = os.path.join(tmp, "hf")
    os.mkdir(hf_dir)
    open(os.path.join(hf_dir, "config.json"), "w").close()
    original = build("vit_base_dinov3")  # the model that both files are made from; the stand-in for the HF model is built from the right file
    right = {k: v for k, v in original.state_dict().items() if not k.startswith(("projector.", "omni_heads."))}
    wrong = dict(right)
    weight, bias = right["blocks.0.mlp.fc1.weight"].clone(), right["blocks.0.mlp.fc1.bias"].clone()
    weight[[0, 1]], bias[[0, 1]] = weight[[1, 0]], bias[[1, 0]]
    wrong["blocks.0.mlp.fc1.weight"], wrong["blocks.0.mlp.fc1.bias"] = weight, bias
    files = {}
    for label, state in (("right", right), ("wrong", wrong)):
        files[label] = os.path.join(tmp, label + ".pth")
        torch.save(state, files[label])

    def stand_in(directory, dtype):
        return parity.build_ours("vit_base_dinov3", files["right"], dtype)

    def run(label, *extra):
        argv = ["--model", "vit_base_dinov3", hf_dir, files[label], "--input-size", "32"] + list(extra)
        with patched(parity, "load_hf", stand_in), patched(parity, "have_transformers", lambda: True):
            code, out, err = captured(lambda: parity.main(argv))
        return code, out.splitlines()

    code, lines = run("right")
    assert code is None and lines[-1] == "PARITY OK", (code, lines)  # main returns: exit status 0
    assert lines[0] == "model vit_base_dinov3 | dtype float32 | images 2x3x32x32 (seed 0) | tol 1e-04" and lines[1].startswith("estimated peak RSS ~"), lines[:2]
    assert any(l.startswith("ours: build_omni_model(vit_base_dinov3, pretrained_weights={}) passed its key guard".format(files["right"])) for l in lines)
    assert any(l.startswith("peak RSS after loading both: ") and l.endswith(" GB") for l in lines)
    assert any(l.startswith("parameters: ours ") and l.endswith("| match") for l in lines)
    assert "register tokens: ours 4 | HF 4 (config.num_register_tokens, which splits last_hidden_state)" in lines
    assert any(l.startswith("rope rescale: ours pos_embed_rope_rescale_coords=None | HF config.pos_embed_rescale=") for l in lines)
    assert sum(l.startswith(("noise", "ramp")) for l in lines) == 6 and "6 of 6 comparisons within tol 1e-04; largest max |diff| 0.00e+00" in lines
    print("OK (f1) main() on the right file: {} lines, from 'model ... | dtype float32 | images 2x3x32x32 | tol 1e-04' through the estimate, the load lines, parameters '| match', registers and the rope note, "
          "to the table, '6 of 6 comparisons within tol' and, last, PARITY OK; exit 0".format(len(lines)))

    code, lines = run("wrong")
    assert code == 1 and lines[-1] == "PARITY FAIL" and any(l.endswith("FAIL") for l in lines[:-2]) and lines[-2].startswith("0 of 6 comparisons within tol 1e-04"), (code, lines)
    print("OK (f2) main() on the file with two rows of blocks.0.mlp.fc1 swapped: exit 1, '{}', PARITY FAIL last".format(lines[-2]))

    code, lines = run("right", "--dtype", "bfloat16", "--tol", "0.5", "--seed", "3")
    assert code is None and lines[-1] == "PARITY OK" and lines[0] == "model vit_base_dinov3 | dtype bfloat16 | images 2x3x32x32 (seed 3) | tol 5e-01", (code, lines[:1], lines[-1])
    code, lines = run("wrong", "--dtype", "bfloat16", "--tol", "1e-4")
    assert code == 1 and lines[-1] == "PARITY FAIL"
    print("OK (f3) the same in bfloat16 (with an explicit --tol): the right file passes, the wrong one exits 1")
del original, right, wrong

# (g) check_hf_dinov3_parity.sh -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
text = open(SH).read()
subprocess.run(["bash", "-n", SH], check=True)
code_lines = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
assert os.access(SH, os.X_OK) and text.startswith("#!/usr/bin/env bash\n") and not any(l.startswith("#SBATCH") for l in text.splitlines()) and "set -euo pipefail" in code_lines  # the header may say "no #SBATCH lines"
assert "--cleanenv" in text.split("set -euo pipefail")[0]  # the header says why PYTHONPATH is set inside the container command
assert 'exec ./nvidia.sh env PYTHONPATH=/workspace/$CACHE python check_hf_dinov3_parity.py --model "$1" "${@:2}"' in code_lines
assert re.search(r"^TRANSFORMERS=4\.56\.2$", code_lines, re.M) and re.search(r"^CACHE=\.pylib-transformers-\$TRANSFORMERS$", code_lines, re.M)  # the cache directory is named after the pin
CACHE = ".pylib-transformers-4.56.2"
ignored = [l.strip() for l in open(os.path.join(os.path.dirname(HERE), ".gitignore")).read().splitlines()]
assert ".pylib-*/" in ignored and ".pylib.*/" in ignored and fnmatch.fnmatch(CACHE, ".pylib-*") and fnmatch.fnmatch(".pylib.a1B2c3", ".pylib.*")  # the cache and the stage do not dirty the checkout
print("OK (g1) check_hf_dinov3_parity.sh: valid bash (bash -n), executable, no SBATCH lines, set -euo pipefail, transformers pinned to 4.56.2 with the cache directory named after the pin (.pylib-transformers-4.56.2), "
      "the header explains --cleanenv and PYTHONPATH is set inside the container command; .gitignore lists .pylib-*/ and .pylib.*/, which match the cache and the stage")

# a stand-in for nvidia.sh: logs its argv (tab-separated, one line per call) and, for `python -m pip install --target /workspace/<stage> -q transformers==<pin>`, writes what pip would into the stage
STUB = r"""#!/usr/bin/env bash
printf '%s\t' "$@" >> calls.log
printf '\n' >> calls.log
if [[ ${1:-} == python && ${2:-} == -m && ${3:-} == pip ]]; then
    target=${6#/workspace/}
    mkdir -p "$target/transformers" "$target/huggingface_hub" "$target/yaml" "$target/numpy" "$target/numpy.libs" "$target/numpy-9.9.9.dist-info"
    touch "$target/transformers/__init__.py" "$target/numpy/__init__.py" "$target/.hidden-lib" "$target/typing_extensions.py"
    if [[ -n ${STUB_RACE:-} ]]; then mkdir -p .pylib-transformers-4.56.2/transformers; touch .pylib-transformers-4.56.2/other-job; fi
    exit "${STUB_PIP_EXIT:-0}"
fi
exit "${STUB_RUN_EXIT:-0}"
"""
# the commands the script runs on the host are put on a PATH in front of the system's: sleep returns at once (and is counted), mv is logged (tab-separated) and then runs the real one
SLEEP = "#!/usr/bin/env bash\necho \"$*\" >> sleep.log\n"
MV = "#!/usr/bin/env bash\nprintf '%s\\t' \"$@\" >> mv.log\nprintf '\\n' >> mv.log\nexec /bin/mv \"$@\"\n"
ENTRIES = [".hidden-lib", "huggingface_hub", "transformers", "typing_extensions.py", "yaml"]  # what the stub's stage holds once numpy is dropped
RUN = ["env", "PYTHONPATH=/workspace/" + CACHE, "python", "check_hf_dinov3_parity.py", "--model", "vit_7b_dinov3", "/workspace/weights/hf dir", "/workspace/weights/7b.pth", "--dtype", "bfloat16", "--tol", "0.5"]
ARGS = ["vit_7b_dinov3", "/workspace/weights/hf dir", "/workspace/weights/7b.pth", "--dtype", "bfloat16", "--tol", "0.5"]  # a path with a space in it: one argument all the way through

with tempfile.TemporaryDirectory() as tmp:
    def fake_repo(name):
        root = os.path.join(tmp, name)
        os.makedirs(os.path.join(root, "bin"))
        for path, content in (("nvidia.sh", STUB), ("bin/sleep", SLEEP), ("bin/mv", MV)):
            with open(os.path.join(root, path), "w") as stub:
                stub.write(content)
            os.chmod(os.path.join(root, path), 0o755)
        return root

    def helper(root, args, **stub_env):
        env = {k: v for k, v in os.environ.items() if not k.startswith("STUB_")}
        env.update(ARK_ROOT=root, PATH=os.path.join(root, "bin") + os.pathsep + os.environ["PATH"], **stub_env)
        done = subprocess.run(["bash", SH] + args, cwd=tmp, env=env, capture_output=True, text=True)  # not from the fake repo: ARK_ROOT decides where it works
        return done, tab_lines(root, "calls.log")

    def tab_lines(root, name):
        path = os.path.join(root, name)
        return [line.rstrip("\t").split("\t") for line in open(path).read().splitlines()] if os.path.exists(path) else []

    def pylib_entries(root):
        return sorted(n for n in os.listdir(root) if n.startswith(".pylib"))  # the caches and any stage

    root = fake_repo("usage")
    for args in ([], ["vit_base_dinov3"], ["vit_base_dinov3", "/workspace/hf"]):
        done, calls = helper(root, args)
        assert done.returncode == 2 and "Usage:" in done.stderr and calls == [] and sorted(os.listdir(root)) == ["bin", "nvidia.sh"], (args, done)
    print("OK (g2) fewer than three arguments: 'Usage:' on stderr, exit 2, nvidia.sh never called, nothing created")

    root = fake_repo("first")
    done, calls = helper(root, ARGS)
    assert done.returncode == 0 and len(calls) == 2, (done, calls)
    install, run = calls
    assert install[:5] == ["python", "-m", "pip", "install", "--target"] and re.fullmatch(r"/workspace/\.pylib\.\w{6}", install[5]) and install[6:] == ["-q", "transformers==4.56.2"], install
    assert run == RUN, run
    assert pylib_entries(root) == [CACHE]  # the stage was published and is gone
    assert sorted(os.listdir(os.path.join(root, CACHE))) == ENTRIES  # numpy, its libs and its dist-info are gone; a dotfile came along
    moves = tab_lines(root, "mv.log")
    assert [m[1] for m in moves] == [CACHE + "/"] * 5 and moves[-1][0].endswith("/transformers") and sorted(os.path.basename(m[0]) for m in moves) == ENTRIES, moves  # transformers is moved LAST although two entries sort after it
    assert tab_lines(root, "sleep.log") == [] and not os.path.exists(os.path.join(root, "sleep.log"))
    print("OK (g3) first run: the pip install into a staged /workspace/.pylib.XXXXXX at transformers==4.56.2, numpy / numpy.libs / numpy-*.dist-info removed, mkdir .pylib-transformers-4.56.2, the stage's five entries "
          "moved in with transformers LAST (a dotfile included, and two entries that sort after it), no stage left; then 'env PYTHONPATH=/workspace/.pylib-transformers-4.56.2 python check_hf_dinov3_parity.py --model ...' with every argument passed through, "
          "the path with a space intact")
    done, calls = helper(root, ARGS)
    assert done.returncode == 0 and len(calls) == 3 and calls[2] == RUN and pylib_entries(root) == [CACHE] and len(tab_lines(root, "mv.log")) == 5, (done, calls)
    print("OK (g4) second run: no install (the cache directory has transformers), only the check is run")

    root = fake_repo("failed")
    done, calls = helper(root, ARGS, STUB_PIP_EXIT="3")
    assert done.returncode == 3 and len(calls) == 1 and calls[0][:4] == ["python", "-m", "pip", "install"] and pylib_entries(root) == [] and not os.path.exists(os.path.join(root, "mv.log")), (done, calls, os.listdir(root))
    print("OK (g5) a pip that fails: its exit status (3) is the script's, the check is not run, and neither a cache directory nor a stage is left behind")

    root = fake_repo("race")
    done, calls = helper(root, ARGS, STUB_RACE="1")  # another job publishes the cache while this one's pip runs: its mkdir fails
    assert done.returncode == 0 and len(calls) == 2 and calls[1] == RUN, (done, calls)
    assert pylib_entries(root) == [CACHE] and sorted(os.listdir(os.path.join(root, CACHE))) == ["other-job", "transformers"], os.listdir(root)  # the other's copy, not merged with this job's
    assert not os.path.exists(os.path.join(root, "mv.log")) and not os.path.exists(os.path.join(root, "sleep.log"))  # it moved nothing, and transformers was already there: no waiting
    print("OK (g6) a job that finishes its install after another has published: mkdir fails, it discards its own stage (nothing merged, nothing moved, no wait) and runs the check on the other's copy")

    root = fake_repo("partial")
    os.makedirs(os.path.join(root, CACHE))
    open(os.path.join(root, CACHE, "junk"), "w").close()  # what a publisher killed between its mkdir and its last mv leaves: a cache without transformers
    done, calls = helper(root, ARGS)
    assert done.returncode == 1 and "ERROR: " + CACHE + " has no transformers" in done.stderr and len(calls) == 1, (done, calls)  # the install was tried, the check was not run
    assert pylib_entries(root) == [CACHE] and os.listdir(os.path.join(root, CACHE)) == ["junk"]  # untouched: no merge, and the stage is gone
    assert len(open(os.path.join(root, "sleep.log")).read().splitlines()) == 30 and not os.path.exists(os.path.join(root, "mv.log"))  # it waited its 30 s (the stub sleep returns at once) and moved nothing
    print("OK (g7) a cache directory without transformers (a killed publisher): the job waits its 30 s, discards its stage, leaves the directory as it was, says 'has no transformers', exits 1 and does not run the check")

    root = fake_repo("other")
    os.makedirs(os.path.join(root, ".pylib"))
    open(os.path.join(root, ".pylib", "x"), "w").close()  # the old shared directory, without transformers
    os.makedirs(os.path.join(root, ".pylib-transformers-4.55.0", "transformers"))  # the cache of another pin
    done, calls = helper(root, ARGS)
    assert done.returncode == 0 and len(calls) == 2 and calls[0][:4] == ["python", "-m", "pip", "install"] and calls[1] == RUN, (done, calls)  # neither is taken for 4.56.2: it installs, and runs on its own directory
    assert pylib_entries(root) == [".pylib", ".pylib-transformers-4.55.0", CACHE] and os.listdir(os.path.join(root, ".pylib")) == ["x"] and os.listdir(os.path.join(root, ".pylib-transformers-4.55.0")) == ["transformers"]
    assert sorted(os.listdir(os.path.join(root, CACHE))) == ENTRIES
    print("OK (g8) a .pylib without transformers, or a transformers of another pin (.pylib-transformers-4.55.0), is neither used nor changed: 4.56.2 gets its own directory and the check runs on that")

    root = fake_repo("status")
    os.makedirs(os.path.join(root, CACHE, "transformers"))
    done, calls = helper(root, ARGS, STUB_RUN_EXIT="7")
    assert done.returncode == 7 and calls == [RUN], (done, calls)
    print("OK (g9) the check's exit status (7) is the script's (exec), and a cache directory with transformers means no pip call")

print("ALL OK")
