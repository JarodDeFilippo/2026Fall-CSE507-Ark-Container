"""Input-resolution probe for the Ark+ fine-tune model on ONE card: for each input size, the step time and the peak device memory at
increasing per-card batch sizes, up to the first batch that does not fit. Model and step follow the sweep: vit_base_dinov3 through
build_omni_model with the four data sets' heads (read from datasets_config.yaml, as engine.py does) and --projector_features 1376, AdamW
lr 7e-5 wd 0.01, a second model copy as the EMA teacher. One step is trainer.py's: no-grad teacher forward, student forward, (1 - coff) *
BCE on head 0 + coff * MSE of the two feature vectors, backward, grad clip 3.0, AdamW step, EMA update, with the engine's mark_step()
calls (--no_teacher_forward drops the teacher forward, the MSE term and the clip). Random data: no data set, weights file (unless
--pretrained_weights) or run directory is touched, and nothing is written but the --out JSON (rewritten after every row, so a time-out
keeps what was measured). On Sol it runs through run_probe_resolution_gaudi.sh; --tiny swaps in a 2-block, 64-dim ViT for the CPU
self-test (test_probe_resolution_memory.py).
"""
import sys

sys.dont_write_bytecode = True  # nothing is written but --out: no .pyc into base/__pycache__
import argparse
import copy
import datetime
import gc
import importlib
import json
import os
import re
import time
from types import SimpleNamespace

import torch

import models
from accelerator import Accelerator
from trainer import ema_update_teacher
from utils import clip_or_measure_grad_norm, get_config

DATASETS = ["VinDrCXR", "ChestXray14", "MIMIC", "CheXpert"]  # the sweep's --data_set order (start_all4_cyclic_dinov3_sweep.sh)
PROJECTOR_FEATURES = 1376  # --projector_features of experiment_scripts/common.sh
USE_MLP = False  # main_ark.py's --use_mlp default (the sweep does not pass it)
MOMENTUM_TEACHER = 0.9  # main_ark.py / common.sh --momentum_teacher
LR, WEIGHT_DECAY = 7e-5, 0.01  # the sweep's centre recipe
CLIP_GRAD = 3.0  # --clip-grad of start_all4_cyclic_dinov3_sweep.sh
HTCORE = "habana_frameworks.torch.core"  # "htcore": Accelerator has already imported it on hpu
COFF = 0.25  # trainer.py: coff = (momentum_schedule[it] - 0.9) * 5, i.e. 0 at the first cycle up to 0.5 at the last; a mid value keeps the MSE term live


def ints(text):
    return [int(s) for s in text.split(",") if s.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Input-resolution memory/throughput probe (one card)")
    p.add_argument("--input_sizes", type=ints, default="224,448,512,768")
    p.add_argument("--batches", type=ints, default="8,16,25,32,50,100", help="per-card batch sizes, tried in increasing order")
    p.add_argument("--steps", type=int, default=3, help="timed steps after 1 warm-up step")
    p.add_argument("--device", default=os.environ.get("PROJECT_ACCELERATOR", "cuda"))
    p.add_argument("--pretrained_weights", default=None, help="optional; random init does for memory")
    p.add_argument("--out", default="probe_resolution_memory.json")
    p.add_argument("--no_teacher_forward", action="store_true", help="student-only step: no no-grad teacher forward, MSE term or grad clip")
    p.add_argument("--tiny", action="store_true", help="CPU self-test: a 2-block, 64-dim ViT")
    return p.parse_args(argv)


def is_oom(error):  # Habana's wording varies: "Allocation failed ...", "out of memory", "OOM"
    return isinstance(error, RuntimeError) and re.search(r"memory|alloc|\boom\b", str(error), re.I) is not None


def max_in_use():  # htcore.hpu.memory_stats(): "MaxInUse" (bytes), else any key that names a max and a use
    stats = importlib.import_module(HTCORE).hpu.memory_stats()
    return float(stats["MaxInUse"] if "MaxInUse" in stats else next(v for k, v in stats.items() if "max" in k.lower() and "use" in k.lower()))


# Ordered attempts per device type: attempt() runs them one by one, each guarded on its own, and the first that answers wins.
PEAK = {"hpu": [("torch.hpu.max_memory_allocated", lambda: float(torch.hpu.max_memory_allocated())),
                ("htcore.hpu.memory_stats[MaxInUse]", max_in_use)],
        "cuda": [("torch.cuda.max_memory_allocated", lambda: float(torch.cuda.max_memory_allocated()))]}
RESET = {"hpu": [("torch.hpu.reset_peak_memory_stats", lambda: torch.hpu.reset_peak_memory_stats()),
                 ("htcore.hpu.reset_peak_memory_stats", lambda: importlib.import_module(HTCORE).hpu.reset_peak_memory_stats())],
         "cuda": [("torch.cuda.reset_peak_memory_stats", lambda: torch.cuda.reset_peak_memory_stats())]}


def attempt(table, device):  # (name, result) of the first attempt that answers, else ("n/a", None)
    for name, source in table.get(device.type, ()):
        try:
            return name, source()
        except Exception:  # a missing name, a raise, a result that is no number: on to the next attempt
            pass
    return "n/a", None


def peak_gb(device):  # GB = 2**30 bytes, as hl-smi's MiB; None when no attempt answers (that must not cost the step time just measured)
    peak = attempt(PEAK, device)[1]
    return None if peak is None else peak / 2 ** 30


def reset_peak(device):  # the name of the attempt that answered; without one the peaks only grow
    return attempt(RESET, device)[0]


def device_name(device):
    try:
        return getattr(torch, device.type).get_device_name()  # torch.hpu exists once habana_frameworks is imported
    except Exception:
        return str(device)


def build(acc, num_classes, pretrained_weights):
    config = SimpleNamespace(model_name="vit_base_dinov3", projector_features=PROJECTOR_FEATURES, use_mlp=USE_MLP,
                             pretrained_weights=pretrained_weights)
    model = models.build_omni_model(config, num_classes)
    teacher = copy.deepcopy(model).requires_grad_(False).eval()
    model, teacher = model.train().to(acc.device), teacher.to(acc.device)
    return model, teacher, torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def run_batch(acc, model, teacher, opt, size, batch, steps, head_classes, teacher_forward):
    """One warm-up step (the HPU graph compile) and `steps` timed ones at this shape; the peak is over all of them."""
    device = acc.device
    x = torch.randn(batch, 3, size, size, device=device)
    y = (torch.rand(batch, head_classes, device=device) > 0.5).float()
    criterion, mse, params = torch.nn.BCEWithLogitsLoss(), torch.nn.MSELoss(), list(model.parameters())

    def step():  # trainer.py train_one_epoch; without the teacher forward there is no MSE term and no clip either
        if teacher_forward:
            with torch.no_grad():
                feat_t, _ = teacher(x, 0)  # same batch, task head 0
        feat_s, pred = model(x, 0)
        loss = criterion(pred, y)
        if teacher_forward:
            loss = (1 - COFF) * loss + COFF * mse(feat_s, feat_t)
        opt.zero_grad()
        loss.backward()
        acc.mark_step()
        if teacher_forward:
            clip_or_measure_grad_norm(params, CLIP_GRAD)
        opt.step()
        acc.mark_step()
        ema_update_teacher(model, teacher, (MOMENTUM_TEACHER,), 0, acc)
        loss.item()  # host sync: the step has really run when the clock is read
        if device.type == "cuda":
            torch.cuda.synchronize()

    reset_peak(device)
    step()
    start = time.perf_counter()
    for _ in range(steps):
        step()
    seconds = (time.perf_counter() - start) / steps
    return dict(fits=True, peak_GB=peak_gb(device), step_s=seconds, img_per_s=batch / seconds)


def conclusive(rows):
    """0 if a row says something about memory (fits=yes, or fits=no from an OOM), else 1 with a one-line summary on stderr: rows that are all
    other errors (fits=None) measured nothing."""
    if any(row["fits"] is not None for row in rows):
        return 0
    first = next((row["error"] for row in rows if row.get("error")), None)
    print("no conclusive row (fits=yes, or fits=no from an OOM): {} attempted, all errors{}".format(
        len(rows), "; first: " + first if first else ""), file=sys.stderr, flush=True)
    return 1


def show(row):
    teacher_forward = str(row["teacher_forward"]).lower()
    if row["fits"]:
        peak = "n/a" if row["peak_GB"] is None else "{:.1f}".format(row["peak_GB"])
        return "size={size} batch={batch} fits=yes peak_GB={} step_s={step_s:.2f} img_per_s={img_per_s:.0f} teacher_forward={}".format(peak, teacher_forward, **row)
    return "size={size} batch={batch} fits={} teacher_forward={} error={error}".format("no" if row["fits"] is False else "error", teacher_forward, **row)


def main(argv=None):
    args = parse_args(argv)
    if args.tiny:  # same build_omni_model path; encoder_features is the CLS ++ mean-patch width, 2 * embed_dim
        models.DINOV3_VITB16_KWARGS = dict(models.DINOV3_VITB16_KWARGS, embed_dim=64, depth=2, num_heads=2, encoder_features=128)
    acc, teacher_forward = Accelerator(args.device), not args.no_teacher_forward
    config = get_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets_config.yaml"))
    num_classes = [len(config[name]["diseases"]) for name in DATASETS]  # engine.py: num_classes_list
    info = dict(device=device_name(acc.device), torch=torch.__version__, timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                teacher_forward=teacher_forward, args=vars(args), num_classes_list=num_classes, rows=[])
    print("device {} ({}), torch {}, heads {}, memory API: {}, peak reset: {}".format(
        info["device"], acc.device, torch.__version__, num_classes, attempt(PEAK, acc.device)[0], reset_peak(acc.device)), flush=True)

    def record(row=None):  # add and print a row, then rewrite the JSON; no row: only the write, up front, so a bad --out fails early
        if row:
            info["rows"].append(row)
            print(show(row), flush=True)
        with open(args.out, "w") as f:
            json.dump(info, f, indent=2)

    record()
    for size in args.input_sizes:
        model = teacher = opt = None
        row = dict(size=size, batch=None, teacher_forward=teacher_forward)  # the attempt in flight
        try:
            model, teacher, opt = build(acc, num_classes, args.pretrained_weights)
            for batch in sorted(set(args.batches)):
                row = dict(size=size, batch=batch, teacher_forward=teacher_forward)
                row.update(run_batch(acc, model, teacher, opt, size, batch, args.steps, num_classes[0], teacher_forward))
                record(row)
        except Exception as error:
            # an OOM stops this size (a larger batch will not fit either); any other error is recorded and the next size goes on.
            # ponytail: sizes share one process, so allocator fragmentation can carry over; rerun a size alone (--input_sizes N) if its fit looks low
            message = (str(error).strip().splitlines() or [""])[0][:300]
            row.update(fits=False if is_oom(error) else None, peak_GB=None, step_s=None, img_per_s=None,
                       error="{}: {}".format(type(error).__name__, message))
            record(row)
        finally:  # the next size starts from a fresh model, optimizer and teacher
            model = teacher = opt = None
            gc.collect()
            getattr(getattr(torch, acc.device.type, None), "empty_cache", lambda: None)()  # where the API exists
    return conclusive(info["rows"])


if __name__ == "__main__":
    sys.exit(main())
