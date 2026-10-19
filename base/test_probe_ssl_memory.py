"""CPU self-test for probe_ssl_memory.py, on the --tiny ViT (2 blocks, 64-dim) and small heads through the trainer's own build path.
Run with a Python with torch, numpy and PyYAML installed, from any directory: python <repo>/base/test_probe_ssl_memory.py"""
import atexit
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True  # importing the probe must not leave a .pyc in base/__pycache__
BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
sys.path.insert(0, BASE)

import numpy as np
import torch
from PIL import Image

import probe_resolution_memory as res
import probe_ssl_memory as probe
from accelerator import Accelerator
from ssl_dinov3.data import DataAugmentationCXR, make_collate_fn
from ssl_dinov3.meta import SSLMetaArch

TMP = tempfile.mkdtemp()
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
SCRIPT = os.path.join(BASE, "probe_ssl_memory.py")


def cli(sizes="32:16:64", batches="2", out=None):  # the spec's CPU run: 32:16:64, batch 2, one timed step, on the tiny ViT
    return ["--device", "cpu", "--sizes", sizes, "--batches", batches, "--steps", "1", "--tiny"] + (["--out", out] if out else [])


def rows_of(path):
    return [(r["sizes"], r["batch"], r["fits"]) for r in json.load(open(path))["rows"]]


# (a) the CLI, default --out in the cwd: one row, it fits, positive step time, the spec's line format, only the JSON written ------------------
cwd = tempfile.mkdtemp(dir=TMP)
run = subprocess.run([sys.executable, SCRIPT] + cli(), cwd=cwd, capture_output=True, text=True)
assert run.returncode == 0, run.stdout + run.stderr
assert os.listdir(cwd) == ["probe_ssl_memory.json"], os.listdir(cwd)
data = json.load(open(os.path.join(cwd, "probe_ssl_memory.json")))
assert [(r["sizes"], r["batch"], r["fits"]) for r in data["rows"]] == [("32:16:64", 2, True)], data["rows"]
assert all(r["step_s"] > 0 and r["img_per_s"] > 0 for r in data["rows"]), data["rows"]
assert {"device", "torch", "timestamp", "args", "trainer"} <= set(data) and data["args"]["tiny"] is True
assert data["trainer"] == dict(local_crops_number=8, gram_weight=2.0, weight_decay=0.04, momentum_teacher=0.999, bf16=False, lr=2e-3,
                               teacher_temp=0.07, clip_grad=3.0), data["trainer"]  # the trainer's defaults
lines = [line for line in run.stdout.splitlines() if line.startswith("sizes=")]
assert len(lines) == 1 and re.fullmatch(r"sizes=32:16:64 batch=2 fits=yes peak_GB=\S+ step_s=\d+\.\d\d img_per_s=\d+\.\d", lines[0]), lines
assert "memory API: n/a, peak reset: n/a" in run.stdout.splitlines()[0], run.stdout  # no hpu or cuda: no attempt answers
print("OK (a) CLI: 32:16:64 batch 2 fits, step_s > 0, the trainer's defaults recorded, only the JSON written; " + lines[0])

# (b)-(d) failures, raised from inside the step (SSLMetaArch.forward_backward) ----------------------------------------------------------------
real_forward_backward = SSLMetaArch.forward_backward


def run_with(fail, sizes, batches):
    out = os.path.join(TMP, "patched.json")

    def forward_backward(self, data, *, teacher_temp):
        fail(data["collated_local_crops"].shape[0] // self.n_local_crops, data["collated_global_crops"].shape[-1])
        return real_forward_backward(self, data, teacher_temp=teacher_temp)

    SSLMetaArch.forward_backward = forward_backward
    try:
        code = probe.main(cli(sizes, batches, out))
    finally:
        SSLMetaArch.forward_backward = real_forward_backward
    return code, rows_of(out), json.load(open(out))["rows"]


def raise_from_4(message):
    def fail(batch, size):
        if batch >= 4:
            raise RuntimeError(message)
    return fail


TWO = "32:16:64,64:32:128"
code, rows, full = run_with(raise_from_4("[Rank:0] FATAL ERROR :: device ran out of memory"), TWO, "2,4,8")
assert code == 0 and rows == [("32:16:64", 2, True), ("32:16:64", 4, False), ("64:32:128", 2, True), ("64:32:128", 4, False)], rows
assert "out of memory" in full[1]["error"] and full[1]["step_s"] is None and full[0]["step_s"] > 0, full
print("OK (b) forced OOM: fits=false at batch 4, batch 8 not tried, both triples: {}".format(rows))

LAZY = "Habana: no storage tensor attached to lazy tensor"
code, rows, full = run_with(raise_from_4(LAZY), TWO, "2,4,8")
assert code == 0 and rows == [("32:16:64", 2, True), ("32:16:64", 4, False), ("64:32:128", 2, True), ("64:32:128", 4, False)], rows
assert full[1]["error"] == "RuntimeError: " + LAZY, full[1]
for text in (LAZY, "NO STORAGE TENSOR ATTACHED TO LAZY TENSOR", "out of memory", "[Rank:0] FATAL ERROR :: MODULE:PT_DEVMEM Allocation failed for size::1024"):
    assert probe.is_oom(RuntimeError(text)), text
assert not res.is_oom(RuntimeError(LAZY))  # the fine-tune probe's matcher alone would call it "error": the extra clause is what catches it
assert not probe.is_oom(ValueError(LAZY)) and not probe.is_oom(RuntimeError("shape mismatch")) and not probe.is_oom(RuntimeError("no storage"))
print("OK (c) the Habana lazy-tensor error is OOM class: fits=false at batch 4, batch 8 not tried; the matcher: {}".format(rows))


def boom_at_32(batch, size):
    if size == 32:
        raise ValueError("boom")


code, rows, full = run_with(boom_at_32, TWO, "2,4")
assert code == 0 and rows == [("32:16:64", 2, None), ("64:32:128", 2, True), ("64:32:128", 4, True)], rows
assert full[0]["error"] == "ValueError: boom"
print("OK (d) other error: recorded at 32:16:64 (no batch 4), 64:32:128 still ran: {}".format(rows))

# (d2) the exit status needs a conclusive row (fits=yes, or fits=no from an OOM): all other errors, or nothing tried -> exit 1 and a one-line summary
def always(error):
    def fail(batch, size):
        raise error
    return fail


with contextlib.redirect_stderr(io.StringIO()) as err:
    code, rows, _ = run_with(always(ValueError("boom")), TWO, "2,4")
assert code == 1 and rows == [("32:16:64", 2, None), ("64:32:128", 2, None)], (code, rows)
assert err.getvalue() == "no conclusive row (fits=yes, or fits=no from an OOM): 2 attempted, all errors; first: ValueError: boom\n", err.getvalue()
with contextlib.redirect_stderr(io.StringIO()) as err:
    code, rows, _ = run_with(always(RuntimeError(LAZY)), TWO, "2,4")  # every row an OOM: conclusive (nothing fits), exit 0
assert code == 0 and rows == [("32:16:64", 2, False), ("64:32:128", 2, False)] and err.getvalue() == "", (code, rows, err.getvalue())
with contextlib.redirect_stderr(io.StringIO()) as err:
    code = probe.main(["--device", "cpu", "--sizes", "", "--tiny", "--out", os.path.join(TMP, "none.json")])
assert code == 1 and err.getvalue() == "no conclusive row (fits=yes, or fits=no from an OOM): 0 attempted, all errors\n", err.getvalue()
print("OK (d2) exit status: only non-OOM errors -> exit 1 + '{}'; only OOM rows -> exit 0; nothing tried -> exit 1".format(
    "no conclusive row (fits=yes, or fits=no from an OOM): 2 attempted, all errors; first: ValueError: boom"))

# (e) defaults and the trainer's crop rule on --sizes ----------------------------------------------------------------------------------------
defaults = probe.parse_args([])
assert (defaults.sizes, defaults.batches, defaults.steps, defaults.out) == (
    [(224, 96, 448), (448, 192, 896), (512, 224, 1024), (512, 224, 512)], [4, 8, 16, 32, 64], 3, "probe_ssl_memory.json"), defaults
assert defaults.device == os.environ.get("PROJECT_ACCELERATOR", "cuda")
for bad in ("500:96:448", "224:224:448", "224:96:464", "512:224:448", "32:16", "a:b:c", "224:96:448:1"):
    try:
        probe.parse_args(["--sizes", bad])
    except SystemExit as error:
        assert error.code == 2, bad
    else:
        raise AssertionError("--sizes {} was accepted".format(bad))
print("OK (e) defaults (both Gram sizes at 512) and refused triples (not x16, local >= global, Gram > 2 x global, Gram < global, malformed)")

# (f) one step is the trainer's, in the trainer's order, on a batch shaped exactly as the trainer's collate of DataAugmentationCXR output ------
events, seen = [], []


def spied(owner, name, tag, check=None):
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        if check:
            check(args, kwargs)
        events.append(tag)
        return original(*args, **kwargs)

    setattr(owner, name, wrapper)
    return lambda: setattr(owner, name, original)


def check_fb(args, kwargs):
    assert kwargs["teacher_temp"] == 0.07
    seen.append(args[1])


def check_clip(args, kwargs):
    assert kwargs["max_norm"] == 3.0


def check_zero_grad(args, kwargs):
    assert kwargs["set_to_none"] is True


def check_ema(args, kwargs):
    assert args[1] == 0.999


restore = [spied(SSLMetaArch, "forward_backward", "fb", check_fb), spied(SSLMetaArch, "update_ema", "ema", check_ema),
           spied(SSLMetaArch, "update_gram", "gram"), spied(Accelerator, "mark_step", "mark"),
           spied(torch.nn.utils, "clip_grad_norm_", "clip", check_clip), spied(torch.optim.AdamW, "step", "opt.step"),
           spied(torch.optim.AdamW, "zero_grad", "zero_grad", check_zero_grad)]
try:
    assert probe.main(cli("32:16:64", "2", os.path.join(TMP, "trace.json"))) == 0
finally:
    for undo in restore:
        undo()
one_step = ["zero_grad", "fb", "mark", "clip", "clip", "clip", "opt.step", "mark", "ema", "mark"]  # clip: backbone, dino_head, ibot_head
assert events == one_step * 2, events  # one warm-up and one timed step; no Gram refresh
image = Image.fromarray(np.random.RandomState(0).randint(0, 255, (300, 400), dtype=np.uint8))  # 2-D uint8: mode L
augment = DataAugmentationCXR(global_crops_size=32, local_crops_size=16, gram_teacher_crops_size=64)
real = make_collate_fn(global_crops_size=32)([(augment(image), ()) for _ in range(2)])
shapes = lambda batch: {k: tuple(v.shape) if torch.is_tensor(v) else v for k, v in batch.items()}
assert shapes(seen[0]) == shapes(real), (shapes(seen[0]), shapes(real))
assert shapes(real)["collated_global_crops"] == (4, 3, 32, 32) and shapes(real)["collated_local_crops"] == (16, 3, 16, 16)
assert shapes(real)["collated_gram_teacher_crops"] == (4, 3, 64, 64)
print("OK (f) step: zero_grad(set_to_none) -> forward_backward(teacher_temp 0.07) -> mark_step -> clip 3.0 x3 -> AdamW.step -> mark_step -> "
      "update_ema(0.999) -> mark_step; the batch's keys and shapes == the trainer's collate of DataAugmentationCXR at 32:16:64: {}".format(shapes(real)))

# (g) the sbatch wrapper -------------------------------------------------------------------------------------------------------------------
WRAPPER = os.path.join(REPO, "run_probe_ssl_gaudi.sh")
subprocess.run(["bash", "-n", WRAPPER], check=True)
assert os.access(WRAPPER, os.X_OK)
text = open(WRAPPER).read()
header = re.findall(r"^#SBATCH (.+)$", text, re.M)
assert header == ["--job-name=d1-8ssl-probe", "-A grp_jliang12", "-p gaudi", "-q public", "-t 01:00:00", "-N 1", "--gres=gpu:hl225:1", "-c 18", "--mem=64G"], header
resolution = re.findall(r"^#SBATCH (.+)$", open(os.path.join(REPO, "run_probe_resolution_gaudi.sh")).read(), re.M)
assert header[1:] == resolution[1:], (header, resolution)  # the same lines as the fine-tune probe's, the job name aside
lines = text.splitlines()
assert "export PT_HPU_LAZY_MODE=1" in lines and "./gaudi_check_cards.sh || exit 1" in lines
assert lines[-1] == 'exec ./gaudi.sh python /workspace/base/probe_ssl_memory.py --out /workspace/probe_ssl_memory.json "$@"', lines[-1]
print("OK (g) run_probe_ssl_gaudi.sh: bash -n, executable, header = run_probe_resolution_gaudi.sh's but --job-name=d1-8ssl-probe, card check, exec line")

print("ALL OK")
