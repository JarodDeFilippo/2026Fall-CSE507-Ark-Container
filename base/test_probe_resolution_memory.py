"""CPU self-test for probe_resolution_memory.py, on the --tiny ViT (2 blocks, 64-dim) through the same build_omni_model path.
Run with the Mac interpreter, from anywhere: /Users/jaroddefilippo/code/CSE494-IP/.venv-ark/bin/python base/test_probe_resolution_memory.py"""
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
import types

sys.dont_write_bytecode = True  # importing the probe must not leave a .pyc in base/__pycache__
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import models
import probe_resolution_memory as probe
import torch

TMP = tempfile.mkdtemp()
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
SCRIPT = os.path.join(BASE, "probe_resolution_memory.py")


def cli(batches="2", out=None):  # the spec's CPU run: sizes 32 and 64, one batch, one timed step, on the tiny ViT
    return ["--device", "cpu", "--input_sizes", "32,64", "--batches", batches, "--steps", "1", "--tiny"] + (["--out", out] if out else [])


def rows_of(path):
    return [(r["size"], r["batch"], r["fits"]) for r in json.load(open(path))["rows"]]


# (a) the CLI, default --out in the cwd, once with the teacher forward (the default) and once with --no_teacher_forward: one row per size, both
# fit, positive step time, the spec's line format plus teacher_forward=, and the JSON header and rows say which step ran ---------------------
for flag, teacher_forward in (([], True), (["--no_teacher_forward"], False)):
    cwd, label = tempfile.mkdtemp(dir=TMP), str(teacher_forward).lower()
    run = subprocess.run([sys.executable, SCRIPT] + cli() + flag, cwd=cwd, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert os.listdir(cwd) == ["probe_resolution_memory.json"], os.listdir(cwd)  # the JSON is all that was written
    data = json.load(open(os.path.join(cwd, "probe_resolution_memory.json")))
    assert [(r["size"], r["batch"], r["fits"]) for r in data["rows"]] == [(32, 2, True), (64, 2, True)], data["rows"]
    assert all(r["step_s"] > 0 and r["img_per_s"] > 0 and r["teacher_forward"] is teacher_forward for r in data["rows"]), data["rows"]
    assert data["teacher_forward"] is teacher_forward and data["args"]["no_teacher_forward"] is not teacher_forward
    assert data["num_classes_list"] == [6, 14, 14, 14]  # read from datasets_config.yaml: VinDrCXR, ChestXray14, MIMIC, CheXpert
    assert {"device", "torch", "timestamp", "args"} <= set(data) and data["args"]["tiny"] is True
    lines = [line for line in run.stdout.splitlines() if line.startswith("size=")]
    pattern = r"size=\d+ batch=2 fits=yes peak_GB=\S+ step_s=\d+\.\d\d img_per_s=\d+ teacher_forward=" + label
    assert len(lines) == 2 and all(re.fullmatch(pattern, line) for line in lines), lines
    assert "memory API: n/a, peak reset: n/a" in run.stdout.splitlines()[0], run.stdout  # no hpu or cuda: no attempt answers
    print("OK (a) CLI, teacher_forward={}: 2 rows fit, step_s > 0, heads [6, 14, 14, 14], only the JSON written; {}".format(label, lines[0]))

# (b) a forced OOM stops that size: batch 8 is never tried, the next size still runs -----------------------------------------
real_forward = models.ArkDinoHeads.forward


def patched(fail):
    def forward(self, x, *args, **kwargs):
        fail(x)
        return real_forward(self, x, *args, **kwargs)
    return forward


def oom_from_4(x):
    if x.shape[0] >= 4:
        raise RuntimeError("[Rank:0] FATAL ERROR :: device ran out of memory")


def boom_at_32(x):
    if x.shape[-1] == 32:
        raise ValueError("boom")


def run_with(fail, batches):
    out = os.path.join(TMP, "patched.json")
    models.ArkDinoHeads.forward = patched(fail)
    try:
        code = probe.main(cli(batches, out))
    finally:
        models.ArkDinoHeads.forward = real_forward
    return code, rows_of(out), json.load(open(out))["rows"]


code, rows, full = run_with(oom_from_4, "2,4,8")
assert code == 0
assert rows == [(32, 2, True), (32, 4, False), (64, 2, True), (64, 4, False)], rows
assert "out of memory" in full[1]["error"] and full[1]["step_s"] is None and full[0]["step_s"] > 0, full
print("OK (b) forced OOM: fits=false at batch 4, batch 8 not tried, both sizes: {}".format(rows))

# (c) any other error is recorded and the next size goes on --------------------------------------------------------------------
code, rows, full = run_with(boom_at_32, "2,4")
assert code == 0
assert rows == [(32, 2, None), (64, 2, True), (64, 4, True)], rows
assert full[0]["error"] == "ValueError: boom"
print("OK (c) other error: recorded at size 32 (no batch 4), size 64 still ran: {}".format(rows))

# (d) the exit status needs a conclusive row (fits=yes, or fits=no from an OOM): only other errors, or nothing tried -> exit 1 and a one-line summary
def always(error):
    def fail(x):
        raise error
    return fail


with contextlib.redirect_stderr(io.StringIO()) as err:
    code, rows, _ = run_with(always(ValueError("boom")), "2,4")
assert code == 1 and rows == [(32, 2, None), (64, 2, None)], (code, rows)
assert err.getvalue() == "no conclusive row (fits=yes, or fits=no from an OOM): 2 attempted, all errors; first: ValueError: boom\n", err.getvalue()
with contextlib.redirect_stderr(io.StringIO()) as err:
    code, rows, _ = run_with(always(RuntimeError("device ran out of memory")), "2,4")  # every row an OOM: conclusive, exit 0
assert code == 0 and rows == [(32, 2, False), (64, 2, False)] and err.getvalue() == "", (code, rows, err.getvalue())
with contextlib.redirect_stderr(io.StringIO()) as err:
    code = probe.main(["--device", "cpu", "--input_sizes", "", "--tiny", "--out", os.path.join(TMP, "none.json")])
assert code == 1 and err.getvalue() == "no conclusive row (fits=yes, or fits=no from an OOM): 0 attempted, all errors\n", err.getvalue()
print("OK (d) exit status: only non-OOM errors -> exit 1 + a one-line summary; only OOM rows -> exit 0; nothing tried -> exit 1")

# (e) the OOM matcher and the defaults --------------------------------------------------------------------------------------------
for text in ("out of memory", "[Rank:0] FATAL ERROR :: MODULE:PT_DEVMEM Allocation failed for size::1024", "Failed to allocate", "OOM when running"):
    assert probe.is_oom(RuntimeError(text)), text
assert not probe.is_oom(ValueError("out of memory")) and not probe.is_oom(RuntimeError("no room for that")) and not probe.is_oom(RuntimeError("shape mismatch"))
defaults = probe.parse_args([])
assert (defaults.input_sizes, defaults.batches, defaults.steps, defaults.out) == (
    [224, 448, 512, 768], [8, 16, 25, 32, 50, 100], 3, "probe_resolution_memory.json")
assert defaults.device == os.environ.get("PROJECT_ACCELERATOR", "cuda") and defaults.pretrained_weights is None
print("OK (e) OOM matcher and defaults")

# (f) the memory-API chains: ordered attempts, each guarded on its own, on stand-ins for torch.hpu and for htcore.hpu (the real ones only
# exist on Gaudi; the probe finds htcore through sys.modules[probe.HTCORE]) ---------------------------------------------------------------
GiB = 2 ** 30
hpu, cuda, cpu = torch.device("hpu"), torch.device("cuda"), torch.device("cpu")  # constructible without the Habana plugin or a GPU
calls = []


def stand_in(torch_hpu=None, htcore_hpu=None):  # install torch.hpu / htcore.hpu from dicts of attributes; None removes it
    if hasattr(torch, "hpu"):
        del torch.hpu
    sys.modules.pop(probe.HTCORE, None)
    if torch_hpu is not None:
        torch.hpu = types.SimpleNamespace(**torch_hpu)
    if htcore_hpu is not None:
        sys.modules[probe.HTCORE] = types.ModuleType(probe.HTCORE)
        sys.modules[probe.HTCORE].hpu = types.SimpleNamespace(**htcore_hpu)


def broken():
    raise RuntimeError("not implemented")


def htcore_stats(stats):
    return {"memory_stats": lambda: stats, "reset_peak_memory_stats": lambda: calls.append("htcore.hpu")}


# both answer: torch.hpu wins, for the peak and for the reset
stand_in({"max_memory_allocated": lambda: 5 * GiB, "reset_peak_memory_stats": lambda: calls.append("torch.hpu"), "get_device_name": lambda: "GAUDI2"},
         htcore_stats({"Limit": 96 * GiB, "MaxInUse": 3 * GiB}))
assert probe.attempt(probe.PEAK, hpu) == ("torch.hpu.max_memory_allocated", 5.0 * GiB) and probe.peak_gb(hpu) == 5.0
assert probe.reset_peak(hpu) == "torch.hpu.reset_peak_memory_stats" and calls == ["torch.hpu"]
assert probe.device_name(hpu) == "GAUDI2"
# the reported case: torch.hpu has neither max_memory_allocated nor reset_peak_memory_stats, htcore.hpu has memory_stats (and a reset)
calls.clear()
stand_in({"get_device_name": lambda: "GAUDI2"}, htcore_stats({"Limit": 96 * GiB, "MaxInUse": 3 * GiB}))
assert probe.attempt(probe.PEAK, hpu) == ("htcore.hpu.memory_stats[MaxInUse]", 3.0 * GiB) and probe.peak_gb(hpu) == 3.0
assert probe.reset_peak(hpu) == "htcore.hpu.reset_peak_memory_stats" and calls == ["htcore.hpu"]
# torch.hpu has the names but they raise: on to htcore.hpu as well
calls.clear()
stand_in({"max_memory_allocated": broken, "reset_peak_memory_stats": broken}, htcore_stats({"MaxInUse": 3 * GiB}))
assert probe.peak_gb(hpu) == 3.0 and probe.reset_peak(hpu) == "htcore.hpu.reset_peak_memory_stats" and calls == ["htcore.hpu"]
# no "MaxInUse": any key that names a max and a use (case-insensitive); InUse and MaxAllocSize name only one of them
stand_in({}, htcore_stats({"InUse": 1 * GiB, "MaxAllocSize": 2 * GiB, "MAX_IN_USE": 4 * GiB}))
assert probe.peak_gb(hpu) == 4.0
# nothing answers (no such key; then no torch.hpu and no htcore at all, as on the Mac): n/a, no error
stand_in({}, htcore_stats({"Limit": 96 * GiB, "InUse": 1 * GiB}))
assert probe.attempt(probe.PEAK, hpu) == ("n/a", None) and probe.peak_gb(hpu) is None
stand_in()
assert probe.peak_gb(hpu) is None and probe.reset_peak(hpu) == "n/a" and probe.device_name(hpu) == "hpu"
# torch.cuda answers for a cuda device only (on the Mac torch.cuda.max_memory_allocated() would read 0 for a cpu)
calls.clear()
real_cuda = torch.cuda.max_memory_allocated, torch.cuda.reset_peak_memory_stats
torch.cuda.max_memory_allocated, torch.cuda.reset_peak_memory_stats = lambda: 2 * GiB, lambda: calls.append("torch.cuda")
try:
    assert probe.attempt(probe.PEAK, cuda) == ("torch.cuda.max_memory_allocated", 2.0 * GiB) and probe.peak_gb(cuda) == 2.0
    assert probe.reset_peak(cuda) == "torch.cuda.reset_peak_memory_stats" and calls == ["torch.cuda"]
    assert probe.peak_gb(cpu) is None and probe.reset_peak(cpu) == "n/a" and calls == ["torch.cuda"]
finally:
    torch.cuda.max_memory_allocated, torch.cuda.reset_peak_memory_stats = real_cuda
print("OK (f) memory API chains: torch.hpu first, htcore.hpu when torch.hpu cannot answer (absent or raising), key fallback, torch.cuda for cuda only, n/a")

# (g) what one step does, counted on the tiny ViT: with the teacher forward, a no-grad teacher forward comes before each student forward, the
# MSE term takes the two 1376-d feature vectors and the grads are clipped at 3.0; with --no_teacher_forward, none of the three --------------
def trace(extra):
    grad_modes, mse_shapes, clips = [], [], []
    real_mse, real_clip = torch.nn.MSELoss.forward, probe.clip_or_measure_grad_norm

    def forward(self, x, *args, **kwargs):
        grad_modes.append(torch.is_grad_enabled())
        return real_forward(self, x, *args, **kwargs)

    def mse(self, student, teacher):
        mse_shapes.append((tuple(student.shape), tuple(teacher.shape)))
        return real_mse(self, student, teacher)

    def clip(params, clip_grad=None):
        clips.append(clip_grad)
        return real_clip(params, clip_grad)

    models.ArkDinoHeads.forward, torch.nn.MSELoss.forward, probe.clip_or_measure_grad_norm = forward, mse, clip
    try:
        assert probe.main(cli("2", os.path.join(TMP, "trace.json")) + extra) == 0
    finally:
        models.ArkDinoHeads.forward, torch.nn.MSELoss.forward, probe.clip_or_measure_grad_norm = real_forward, real_mse, real_clip
    return grad_modes, mse_shapes, clips


# sizes 32 and 64 at batch 2, one warm-up and one timed step each: 4 steps
assert trace([]) == ([False, True] * 4, [((2, 1376), (2, 1376))] * 4, [3.0] * 4)
assert trace(["--no_teacher_forward"]) == ([True] * 4, [], [])
print("OK (g) step: no-grad teacher forward before each student forward, MSE on (2, 1376) features, clip 3.0; all three gone with --no_teacher_forward")

print("ALL OK")
