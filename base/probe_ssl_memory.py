"""SSL memory probe for the Stage A trainer (ssl_pretrain.py) on ONE card: for each crop-size triple global:local:gram, the step time and the peak
device memory at increasing per-card batch sizes, up to the first batch that does not fit. Models and step are the trainer's: build_models (student,
EMA teacher and Gram teacher, heads with 65,536 prototypes), SSLMetaArch with the trainer's defaults (8 local crops, Gram weight 2.0), AdamW over
get_params_groups, and per step apply_optim_scheduler -> zero_grad -> forward_backward -> mark_step -> grad clip 3.0 per student submodule ->
optimizer.step -> mark_step -> update_ema -> mark_step. The batch is the trainer's collate (make_collate_fn) over synthetic crops of the triple's
sizes, so its tensors (2 global, 8 local and 2 Gram teacher crops per image, the iBOT masks) have the trainer's shapes. One card: no DDP wrapper (the
4-card run adds DDP's gradient buckets, about one more copy of the student's gradients). Random data, random init: no data set, weights file or run
directory is touched, and nothing is written but the --out JSON (rewritten after every row, so a time-out keeps what was measured). Peak memory, the
device name, the OOM matcher and the exit status (0 only with a conclusive row: fits=yes, or fits=no from an OOM) are probe_resolution_memory.py's;
Habana's "no storage tensor attached to lazy tensor" RuntimeError counts as an OOM too (the fine-tune probe hit it where the card ran out, 2026-10-18).
On Sol it runs through run_probe_ssl_gaudi.sh; --tiny swaps in a 2-block, 64-dim ViT with small heads for the CPU self-test (test_probe_ssl_memory.py).
"""
import sys

sys.dont_write_bytecode = True  # nothing is written but --out: no .pyc into base/__pycache__
import argparse
import datetime
import gc
import json
import os
import re
import time

import torch

import models
import probe_resolution_memory as res
import ssl_pretrain
from accelerator import Accelerator
from ssl_dinov3 import model as ssl_model
from ssl_dinov3.data import make_collate_fn
from ssl_dinov3.meta import SSLMetaArch
from ssl_dinov3.optim import apply_optim_scheduler, build_optimizer, get_params_groups

# the trainer's defaults (local crops, Gram weight, weight decay, teacher momentum); the weights path is never read
TRAINER = ssl_pretrain.parse_args(["--data_set", "MIMIC", "--pretrained_weights", "unused", "--output_dir", "unused"])
LR = 2e-3  # the hi-res stage's flat LR (chain/hires_lr.txt); memory does not depend on it as long as it is not 0
TEACHER_TEMP = 0.07  # the teacher temperature after its warmup
LAZY_OOM = re.compile(r"no storage tensor attached to lazy tensor", re.I)


def triples(text):
    found = []
    for item in text.split(","):
        if item.strip():
            parts = item.split(":")
            if len(parts) != 3 or not all(p.strip().isdigit() for p in parts):
                raise argparse.ArgumentTypeError("expected global:local:gram, got {!r}".format(item))
            found.append(tuple(int(p) for p in parts))
    return found


def ints(text):
    return [int(s) for s in text.split(",") if s.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SSL (Stage A trainer) memory/throughput probe (one card)")
    p.add_argument("--sizes", type=triples, default="224:96:448,448:192:896,512:224:1024,512:224:512", help="global:local:gram crop sizes, comma-separated")
    p.add_argument("--batches", type=ints, default="4,8,16,32,64", help="per-card batch sizes, tried in increasing order")
    p.add_argument("--steps", type=int, default=3, help="timed steps after 1 warm-up step")
    p.add_argument("--device", default=os.environ.get("PROJECT_ACCELERATOR", "cuda"))
    p.add_argument("--out", default="probe_ssl_memory.json")
    p.add_argument("--tiny", action="store_true", help="CPU self-test: a 2-block, 64-dim ViT and small heads")
    args = p.parse_args(argv)
    for g, l, gram in args.sizes:  # the trainer's own rule: multiples of 16, local < global <= gram <= 2 x global
        try:
            ssl_pretrain.check_args(argparse.Namespace(**dict(vars(TRAINER), global_crops_size=g, local_crops_size=l, gram_teacher_size=gram)))
        except ValueError as error:
            p.error("--sizes {}:{}:{}: {}".format(g, l, gram, error))
    return args


def is_oom(error):
    return res.is_oom(error) or (isinstance(error, RuntimeError) and LAZY_OOM.search(str(error)) is not None)


def make_batch(sizes, batch, n_local):
    """The trainer's collate over `batch` synthetic samples shaped as DataAugmentationCXR's output at these sizes."""
    g, l, gram = sizes
    crops = lambda size, n: [torch.randn(3, size, size) for _ in range(n)]
    samples = []
    for _ in range(batch):
        global_crops = crops(g, 2)
        samples.append((dict(global_crops=global_crops, global_crops_teacher=global_crops, gram_teacher_crops=crops(gram, 2),
                             local_crops=crops(l, n_local), offsets=()), ()))
    return make_collate_fn(global_crops_size=g)(samples)


def build(acc):
    student, teacher, gram_teacher = ssl_model.build_models(models.DINOV3_VITB16_KWARGS, None, ssl_model.HEAD_KWARGS)
    meta = SSLMetaArch(student.to(acc.device), teacher.to(acc.device), gram_teacher.to(acc.device), n_local_crops=TRAINER.local_crops_number,
                       gram_loss_weight=TRAINER.gram_weight, autocast_dtype=torch.bfloat16 if TRAINER.bf16 else None).to(acc.device)
    meta.train()
    return meta, build_optimizer(get_params_groups(meta.student))


def run_batch(acc, meta, opt, sizes, batch, steps):
    """One warm-up step (the HPU graph compile) and `steps` timed ones at this shape; the peak is over all of them."""
    device, data = acc.device, make_batch(sizes, batch, meta.n_local_crops)

    def step():  # ssl_pretrain.run's inner loop, minus the logging
        apply_optim_scheduler(opt, LR, TRAINER.weight_decay, LR, freeze_backbone=False)
        opt.zero_grad(set_to_none=True)
        loss, _ = meta.forward_backward(data, teacher_temp=TEACHER_TEMP)
        acc.mark_step()
        for module in meta.student.values():
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=ssl_pretrain.CLIP_GRAD)
        opt.step()
        acc.mark_step()
        meta.update_ema(TRAINER.momentum_teacher)
        acc.mark_step()
        loss.item()  # host sync: the step has really run when the clock is read
        if device.type == "cuda":
            torch.cuda.synchronize()

    res.reset_peak(device)
    step()
    start = time.perf_counter()
    for _ in range(steps):
        step()
    seconds = (time.perf_counter() - start) / steps
    return dict(fits=True, peak_GB=res.peak_gb(device), step_s=seconds, img_per_s=batch / seconds)


def show(row):
    if row["fits"]:
        peak = "n/a" if row["peak_GB"] is None else "{:.1f}".format(row["peak_GB"])
        return "sizes={sizes} batch={batch} fits=yes peak_GB={} step_s={step_s:.2f} img_per_s={img_per_s:.1f}".format(peak, **row)
    return "sizes={sizes} batch={batch} fits={} error={error}".format("no" if row["fits"] is False else "error", **row)


def main(argv=None):
    args = parse_args(argv)
    if args.tiny:  # the same build path on test_ssl_pretrain.py's tiny ViT and heads
        models.DINOV3_VITB16_KWARGS = dict(models.DINOV3_VITB16_KWARGS, embed_dim=64, depth=2, num_heads=2, ffn_ratio=2)
        ssl_model.HEAD_KWARGS = dict(out_dim=128, hidden_dim=64, bottleneck_dim=32, nlayers=3)
    acc = Accelerator(args.device)
    ssl_pretrain.ensure_process_group(acc)  # SSLMetaArch's Sinkhorn-Knopp all_reduces even on one process
    trainer = dict(local_crops_number=TRAINER.local_crops_number, gram_weight=TRAINER.gram_weight, weight_decay=TRAINER.weight_decay,
                   momentum_teacher=TRAINER.momentum_teacher, bf16=TRAINER.bf16, lr=LR, teacher_temp=TEACHER_TEMP, clip_grad=ssl_pretrain.CLIP_GRAD)
    info = dict(device=res.device_name(acc.device), torch=torch.__version__, timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                args=vars(args), trainer=trainer, rows=[])
    print("device {} ({}), torch {}, memory API: {}, peak reset: {}".format(
        info["device"], acc.device, torch.__version__, res.attempt(res.PEAK, acc.device)[0], res.reset_peak(acc.device)), flush=True)

    def record(row=None):  # add and print a row, then rewrite the JSON; no row: only the write, up front, so a bad --out fails early
        if row:
            info["rows"].append(row)
            print(show(row), flush=True)
        with open(args.out, "w") as f:
            json.dump(info, f, indent=2)

    record()
    for sizes in args.sizes:
        label = ":".join(str(s) for s in sizes)
        meta = opt = None
        row = dict(sizes=label, batch=None)  # the attempt in flight
        try:
            meta, opt = build(acc)
            for batch in sorted(set(args.batches)):
                row = dict(sizes=label, batch=batch)
                row.update(run_batch(acc, meta, opt, sizes, batch, args.steps))
                record(row)
        except Exception as error:
            # an OOM stops this triple (a larger batch will not fit either); any other error is recorded and the next triple goes on.
            # ponytail: triples share one process, so allocator state can carry over; rerun one alone (--sizes 512:224:1024) if its fit looks low
            message = (str(error).strip().splitlines() or [""])[0][:300]
            row.update(fits=False if is_oom(error) else None, peak_GB=None, step_s=None, img_per_s=None,
                       error="{}: {}".format(type(error).__name__, message))
            record(row)
        finally:  # the next triple starts from fresh models and a fresh optimizer
            meta = opt = None
            gc.collect()
            getattr(getattr(torch, acc.device.type, None), "empty_cache", lambda: None)()  # where the API exists
    return res.conclusive(info["rows"])  # exit 1 unless a row is conclusive (fits=yes, or an OOM)


if __name__ == "__main__":
    sys.exit(main())
