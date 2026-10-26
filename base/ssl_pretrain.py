"""Stage A: DINOv3-style SSL continued pretraining of the Ark+ ViT-B/16 backbone on chest X-rays (student, EMA teacher
and a Gram-anchor teacher; see ssl_dinov3/). One step mirrors dinov3/train/train.py do_train: schedules ->
apply_optim_scheduler -> zero_grad -> forward_backward -> per-submodule grad clip -> optimizer.step -> EMA update ->
Gram refresh at the configured steps.

Run from base/:
    python -m torch.distributed.run --standalone --nproc-per-node 4 ssl_pretrain.py --device hpu \
        --data_set VinDrCXR --data_set ChestXray14 --data_set MIMIC --data_set CheXpert \
        --pretrained_weights <dinov3_vitb16 .pth> --output_dir <run dir> --resume

Crop sizes: --global_crops_size / --local_crops_size / --gram_teacher_size (224 / 96 / 2 x global; chain/hires.sh runs 512 / 224 / chain/hires_gram_size.txt).
--init_from_checkpoint <run>/ckpt/epoch_NNNN.pth starts a NEW run (fresh optimizer and schedule, step 0) from a finished run's full checkpoint:
student and EMA teacher, heads included; the Gram teacher is re-anchored to that EMA teacher (DINOv3 Sec. 5, high-resolution adaptation).

Under --output_dir: args.json, ssl_losses.csv, ckpt/epoch_NNNN.pth (full state, what --resume reads) and
export/teacher_backbone_epoch_NNNN.pth (flat stock-key backbone, loads into Ark's vit_base_dinov3).

The loss columns of ssl_losses.csv are global means over the ranks, built the way Ark's trainer.py builds its losses: each
rank reads its own on the host with .item(), a tensor made from those floats is all-reduced, then divided by world_size.

--no_gram trains with no Gram teacher at all: none is built (build_models gram=False), moved to the device, loaded (--init_from_checkpoint
included) or run; no Gram loss is computed, nothing is refreshed, and the batches carry no Gram crops (the global crops are still cut at
2 x --global_crops_size and resized, so the student / teacher crops are the Gram runs'). An explicit --gram_weight, --gram_refresh_steps or
--gram_teacher_size is refused. The flag is locked on --resume, args.json records it, and the checkpoint's gram_teacher is None. ssl_losses.csv
keeps all 15 columns and writes gram and gram_weight as 0.0 in every row, never blank: the DINO-X report builder parses them as numbers and
counts a blank as unreadable.

--branch_from <no-Gram run>/ckpt/epoch_NNNN.pth starts a NEW run dir that continues that --no_gram run exactly, with the Gram teacher on: student,
EMA teacher, optimizer, step, epoch and the python / numpy / torch RNG states are restored as a --resume of the trunk would restore them (same
schedule position, same data order), and the Gram teacher is anchored to that checkpoint's EMA teacher backbone (the mapping --init_from_checkpoint
uses); its Gram refreshes start empty. --gram_refresh_steps is required and absolute (global steps, each after the branch step); --gram_weight and
--gram_teacher_size keep their defaults (2.0, 2 x global) unless given. An explicit --gram_teacher_size other than the source's (its 2 x global)
also changes the size the global crops are cut at, so the crops are then no longer the trunk's. Epochs continue the trunk's numbering (a branch from
epoch 10 first writes ckpt/epoch_0011.pth and export/teacher_backbone_epoch_0011.pth) and ssl_losses.csv holds only the branch's rows, from the
branch step on. The source is read only when the branch starts: a --resume restores the branch's own epoch checkpoint, so the trunk pruning the
source cannot break it, and branch_from is locked on --resume. Refused before anything on disk is touched: a source that is not a file, not a full
epoch checkpoint or not a --no_gram run's; --no_gram or --init_from_checkpoint with it; a source that differs on any --resume-locked setting but the
Gram ones and its own --init_from_checkpoint, or in its image list, world size or epoch length; a refresh step at or before the branch step; a stop
epoch at or before the source's.

--drop_path_rate R, --rope_rescale_coords S and --untie_local_cls_norm are three settings of Meta's DINOv3 training recipe (ssl_default_config's
drop_path_rate 0.3; pos_embed_rope_rescale_coords 2 and untie_global_and_local_cls_norm true as in the released ViT-7B configs), passed to the student
backbone: stochastic depth (each of a block's two residual branches runs on a random 1 - R of the batch, rescaled), RoPE coordinates scaled by a
log-uniform factor in [1/S, S] (drawn anew per block, crop group and forward), and a second LayerNorm local_cls_norm for the CLS and register tokens
of the local crops (entry 1 of the student's list forward). All three act in train mode only: the EMA and Gram teachers are copies of the student but
run in eval, so none of the three reaches the teacher forwards, the probe or the export. The draws come from torch's generator on the training device:
on CPU the checkpoint's torch RNG state covers them, so a resumed run is bitwise the uninterrupted one; a device generator (HPU) is not in the
checkpoint, so there a resumed run draws other (equally distributed) factors and subsets. Every rank builds the models from the same --seed (the
heads' init and the teachers' copies are then alike on all ranks), then re-seeds python / numpy / torch with --seed + rank (Meta seeds seed + rank), so
each rank draws its own drop-path subsets and RoPE factors; rank 0 keeps --seed. A multi-rank checkpoint holds every rank's RNG states (rng_per_rank,
entry r = rank r's; rng is rank 0's as before, and all a one-rank checkpoint has) and a --resume or --branch_from restores each rank its own; one
written before that holds a single state, which every rank restores (one line says so).
The defaults (0.0, none, off) are the behaviour before the flags existed, so every earlier checkpoint resumes and
branches unchanged: the three are locked on --resume (a checkpoint without the keys reads as the defaults), recorded in args.json, and a --branch_from
source must match them. With --untie_local_cls_norm the stock --pretrained_weights file loads strictly and local_cls_norm starts as a copy of its
norm (one line says so); the epoch checkpoints carry local_cls_norm (student, EMA teacher, Gram teacher) and the exports leave it out (the stock 188
keys, as before). An --init_from_checkpoint source must match the flag: its student and teacher load strictly.
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from accelerator import Accelerator
from models import DINOV3_VITB16_KWARGS
from ssl_dinov3.data import (DATASET_NAMES, DataAugmentationCXR, build_data_loader, build_dataset, make_collate_fn,
                             set_epoch)
from ssl_dinov3.meta import SSLMetaArch
from ssl_dinov3.model import HEAD_KWARGS, build_models, export_backbone
from ssl_dinov3.optim import apply_optim_scheduler, build_optimizer, build_schedules, get_params_groups
from utils import get_config

CLIP_GRAD = 3.0  # dinov3 optim.clip_grad, applied per student submodule
FREEZE_LAST_LAYER_EPOCHS = 1  # dinov3 optim.freeze_last_layer_epochs
# dino_global .. total: global means over ranks (Ark-style: host-built tensor all-reduced); step_time / data_time: see run()
CSV_HEADER = ["step", "epoch", "dino_global", "dino_local", "koleo", "ibot", "gram", "total", "lr", "wd", "momentum",
              "teacher_temp", "gram_weight", "step_time", "data_time"]
LOSS_KEYS = ("dino_global_crops_loss", "dino_local_crops_loss", "koleo_loss", "ibot_loss", "gram_loss")
# what shapes the run: a --resume must match its checkpoint on these (the other flags may change between invocations).
# init_from_checkpoint: a run started from one resumes only with the same one (not re-applied then); any other run's checkpoint is refused
# no_gram: a no-Gram run resumes only as one (its checkpoint has no Gram teacher), and a Gram run never as one
# branch_from: a branch resumes only as the same branch (its source is read only at its start); a trunk or any other run never as one
# drop_path_rate / rope_rescale_coords / untie_local_cls_norm: the student's training recipe (its forward; untied, its parameters too)
RESUME_MATCH = ("data_set", "epochs", "batch_size_per_gpu", "lr", "min_lr", "no_lr_scaling", "warmup_epochs",
                "weight_decay", "momentum_teacher", "teacher_temp_warmup_epochs", "gram_weight", "global_crops_size",
                "local_crops_size", "gram_teacher_size", "gram_refresh_steps", "freeze_backbone_epochs", "local_crops_number",
                "seed", "bf16", "world_size", "init_from_checkpoint", "no_gram", "branch_from", "drop_path_rate",
                "rope_rescale_coords", "untie_local_cls_norm")
# checkpoints written before the crop-size flags existed (pilots, main, constlr) were all trained at these
CROP_DEFAULTS = dict(global_crops_size=224, local_crops_size=96, gram_teacher_size=448)
# keys a checkpoint written before these flags lacks (check_resume reads them as): the crop sizes above, --no_gram off, no --branch_from, and no
# recipe setting (drop path 0, no RoPE rescale, tied CLS norm)
ARG_DEFAULTS = dict(CROP_DEFAULTS, no_gram=False, branch_from=None, drop_path_rate=0.0, rope_rescale_coords=None, untie_local_cls_norm=False)
# what a --branch_from start may differ from its source in (check_resume skips these): the Gram switch and knobs the branch turns on, the path itself,
# and the source's own init provenance (its full state is restored; a branch cannot pass --init_from_checkpoint, so it would always differ)
BRANCH_FREE = ("no_gram", "gram_weight", "gram_refresh_steps", "gram_teacher_size", "branch_from", "init_from_checkpoint")
BRANCH_SOURCE_KEYS = ("student", "teacher", "optimizer", "step", "epoch", "rng", "args", "world_size")  # a full epoch checkpoint (save_state)
PATCH_SIZE = 16  # DINOV3_VITB16_KWARGS patch_size: every crop side must be a multiple of it
DEFAULT_GRAM_REFRESH_STEPS = "10000,20000,30000"  # --gram_refresh_steps when not given (Meta's three refreshes); --no_gram: none


def parse_steps(text):
    if text.strip().lower() == "none":  # no refresh: the Gram teacher stays the initial one
        return []
    try:
        steps = sorted({int(s) for s in text.split(",") if s.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}")
    if any(s <= 0 for s in steps):
        raise argparse.ArgumentTypeError(f"refresh steps must be positive, got {text!r}")
    return steps


def parse_rescale(text):
    if text.strip().lower() == "none":  # no rescale: the RoPE coordinates as in eval
        return None
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number above 1 or none, got {text!r}")
    if not (math.isfinite(value) and value > 1):  # 1 would draw a factor of exactly 1; below 1 the range [1/S, S] is upside down
        raise argparse.ArgumentTypeError(f"the rescale must be above 1 (a log-uniform factor in [1/S, S]) or none, got {text!r}")
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage A DINOv3-style SSL continued pretraining")
    p.add_argument("--data_set", action="append", choices=DATASET_NAMES, required=True)
    p.add_argument("--datasets_config", default="datasets_config.yaml")
    p.add_argument("--pretrained_weights", default=None,
                   help="flat DINOv3 backbone .pth; required unless --init_from_checkpoint or --branch_from (which then win, with a warning)")
    p.add_argument("--init_from_checkpoint", default=None,
                   help="a full checkpoint (<run>/ckpt/epoch_NNNN.pth) to start a NEW run from: student + teacher, heads included")
    p.add_argument("--branch_from", default=None,
                   help="a --no_gram run's full checkpoint (<run>/ckpt/epoch_NNNN.pth) to continue exactly in a NEW run dir with the Gram teacher on, "
                        "anchored to its EMA teacher; needs --gram_refresh_steps (absolute global steps, each after the branch step)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=40, help="schedule length (LR warmup + cosine, teacher temperature)")
    p.add_argument("--stop_after_epochs", type=int, default=None,
                   help="stop after this many completed epochs of the --epochs schedule (default: all of them)")
    p.add_argument("--batch_size_per_gpu", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--no_lr_scaling", action="store_true",
                   help="--lr/--min_lr are absolute (default: scaled by dinov3's sqrt_wrt_1024 rule)")
    p.add_argument("--warmup_epochs", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--momentum_teacher", type=float, default=0.999)
    p.add_argument("--teacher_temp_warmup_epochs", type=int, default=12)
    p.add_argument("--no_gram", action="store_true",
                   help="train with no Gram teacher at all (none built or loaded, no Gram loss, no refreshes); the 2026-10-23 trunk run")
    p.add_argument("--gram_weight", type=float, default=None,
                   help="Gram loss weight (default 2.0; refused with --no_gram, where it is 0.0)")
    p.add_argument("--global_crops_size", type=int, default=224)
    p.add_argument("--local_crops_size", type=int, default=96)
    p.add_argument("--gram_teacher_size", type=int, default=None,
                   help="default 2 x --global_crops_size (448 at the default 224): DINOv3's 2x Gram teacher, Sec. 4.3. Refused with --no_gram, "
                        "where it still resolves to 2 x global: it only sets the size the global crops are cut at before the resize to the "
                        "global size, so the crops match the Gram runs'")
    p.add_argument("--gram_refresh_steps", type=parse_steps, default=None,
                   help=f"comma-separated global steps after which the Gram teacher is reloaded from the EMA teacher (default "
                        f"{DEFAULT_GRAM_REFRESH_STEPS}; none: no refresh). Refused with --no_gram, where there are none")
    p.add_argument("--freeze_backbone_epochs", type=int, default=1, help="backbone LR 0 (heads train) for this many epochs")
    p.add_argument("--local_crops_number", type=int, default=8)
    p.add_argument("--drop_path_rate", type=float, default=0.0,
                   help="student stochastic depth, in [0, 1) (Meta's DINOv3 recipe: 0.3); train mode only")
    p.add_argument("--rope_rescale_coords", type=parse_rescale, default=None,
                   help="student RoPE coordinates scaled by a log-uniform factor in [1/S, S], S above 1, or none (default; Meta's DINOv3 recipe: 2); "
                        "train mode only")
    p.add_argument("--untie_local_cls_norm", action="store_true",
                   help="a second student LayerNorm for the local crops' CLS and register tokens (Meta's untie_global_and_local_cls_norm); "
                        "train mode only, initialized from the stock norm, not exported")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=None, help="stop after this many global steps (timing runs)")
    p.add_argument("--print_freq", type=int, default=50)
    p.add_argument("--device", default=os.environ.get("PROJECT_ACCELERATOR", "cuda"))
    p.add_argument("--bf16", action="store_true", help="bf16 autocast for the forward passes (default fp32)")
    args = p.parse_args(argv)
    for name in ("init_from_checkpoint", "branch_from"):  # an empty path (an unset variable in a launch line) would pass the presence checks below but not run()'s
        if getattr(args, name) == "":
            p.error(f"--{name} is empty: pass a checkpoint path, or leave the flag out")
    if not 0.0 <= args.drop_path_rate < 1.0:  # 1 would drop every sample of every residual branch (and NaN fails both)
        p.error(f"--drop_path_rate must be in [0, 1), got {args.drop_path_rate}")
    if args.pretrained_weights is None and args.init_from_checkpoint is None and args.branch_from is None:
        p.error("--pretrained_weights is required unless --init_from_checkpoint is given (or --branch_from)")
    if args.branch_from is not None:  # a branch continues a no-Gram run exactly with the Gram teacher on; these would contradict that
        for name, why in (("no_gram", "a branch is the no-Gram trunk continued with the Gram teacher on"),
                          ("init_from_checkpoint", "a branch continues its source's optimizer, step and schedule; --init_from_checkpoint starts afresh at step 0")):
            if getattr(args, name):
                p.error(f"--branch_from cannot be combined with --{name} ({why})")
        if args.gram_refresh_steps is None:
            p.error("--branch_from needs --gram_refresh_steps: absolute global steps, each after the branch step (the launcher computes them, so the "
                    "list, which --resume locks, is the same on every resubmit); none: no refresh")
    if args.no_gram:  # no Gram teacher: the Gram knobs have nothing to act on, so an explicit one is a mistake, not a no-op
        for name, why in (("gram_weight", "there is no Gram loss to weight"), ("gram_refresh_steps", "there is no Gram teacher to refresh"),
                          ("gram_teacher_size", "there is no Gram teacher to size; the global crops are cut at 2 x --global_crops_size")):
            if getattr(args, name) is not None:
                p.error(f"--{name} cannot be combined with --no_gram ({why})")
        args.gram_weight, args.gram_refresh_steps = 0.0, []
    else:
        if args.gram_weight is None:
            args.gram_weight = 2.0
        if args.gram_refresh_steps is None:
            args.gram_refresh_steps = parse_steps(DEFAULT_GRAM_REFRESH_STEPS)
    if args.gram_teacher_size is None:
        args.gram_teacher_size = 2 * args.global_crops_size
    return args


def check_args(args):
    """Clear errors where the schedules would only assert. Returns the resolved --stop_after_epochs."""
    if args.warmup_epochs > args.epochs:
        raise ValueError(f"--warmup_epochs {args.warmup_epochs} exceeds --epochs {args.epochs}")
    if args.teacher_temp_warmup_epochs > args.epochs:
        raise ValueError(f"--teacher_temp_warmup_epochs {args.teacher_temp_warmup_epochs} exceeds --epochs {args.epochs}")
    stop = args.epochs if args.stop_after_epochs is None else args.stop_after_epochs
    if not 1 <= stop <= args.epochs:
        raise ValueError(f"--stop_after_epochs {stop} must be between 1 and --epochs {args.epochs}")
    if args.print_freq < 1:
        raise ValueError(f"--print_freq {args.print_freq} must be >= 1")
    sizes = dict(global_crops_size=args.global_crops_size, local_crops_size=args.local_crops_size,
                 gram_teacher_size=args.gram_teacher_size)
    for name, size in sizes.items():
        if size % PATCH_SIZE:
            raise ValueError(f"--{name} {size} is not a multiple of the patch size {PATCH_SIZE}")
    local, global_, gram = args.local_crops_size, args.global_crops_size, args.gram_teacher_size
    if not 0 < local < global_ <= gram <= 2 * global_:
        raise ValueError(f"crop sizes must satisfy 0 < local < global <= Gram teacher <= 2 x global; got local {local}, "
                         f"global {global_}, Gram teacher {gram}")
    return stop


def ensure_process_group(accelerator):
    """The vendored iBOT Sinkhorn-Knopp all_reduces unconditionally, so a one-process run needs a default group too."""
    if dist.is_initialized():
        return
    backend = {"cpu": "gloo", "hpu": "hccl", "cuda": "nccl"}[accelerator.name]
    if backend == "hccl":
        import habana_frameworks.torch.distributed.hccl  # noqa: F401  (registers the backend)
    dist.init_process_group(backend, store=dist.HashStore(), rank=0, world_size=1)


def epoch_checkpoints(ckpt_dir):
    """[(epoch, path)] of ckpt/epoch_NNNN.pth, ascending; maxsteps_*.pth and *.tmp are not epoch checkpoints."""
    found = []
    for path in glob.glob(os.path.join(ckpt_dir, "epoch_*.pth")):
        match = re.fullmatch(r"epoch_(\d+)\.pth", os.path.basename(path))
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def prune_checkpoints(ckpt_dir, keep_last=3, keep_every=10):
    for epoch, path in epoch_checkpoints(ckpt_dir)[:-keep_last]:
        if epoch % keep_every:
            os.remove(path)


def to_cpu(obj):
    """Tensors to CPU, recursively through dicts / lists / tuples: HPU tensors must not go into torch.save."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def save_atomic(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def check_resume(ckpt, args, world_size, epoch_length, data_fingerprint, skip=(), action="--resume"):
    saved = dict(ARG_DEFAULTS, **ckpt["args"], world_size=ckpt["world_size"])  # no crop / no_gram / branch_from keys: an old checkpoint, 224 / 96 / 448, Gram on, no branch
    now = dict(vars(args), world_size=world_size)
    diffs = [f"{k}: checkpoint {saved.get(k)!r}, now {now[k]!r}" for k in RESUME_MATCH if k not in skip and saved.get(k) != now[k]]
    if ckpt.get("data_fingerprint") != data_fingerprint:  # same names and length, different (or reordered) image list
        diffs.append(f"training image list: checkpoint {ckpt.get('data_fingerprint')}, now {data_fingerprint}")
    if ckpt["step"] != ckpt["epoch"] * epoch_length:
        diffs.append(f"step {ckpt['step']} != epoch {ckpt['epoch']} x epoch_length {epoch_length} (dataset changed?)")
    if diffs:
        raise ValueError(f"cannot {action}: this run differs from the checkpoint in " + "; ".join(diffs))


def load_branch_source(args, stop_epoch, world_size, epoch_length, data_fingerprint):
    """The --branch_from checkpoint, loaded and checked: a full epoch checkpoint of a --no_gram run that this run continues exactly (raises otherwise)."""
    src = torch.load(args.branch_from, map_location="cpu", weights_only=False)
    if not isinstance(src, dict) or not set(BRANCH_SOURCE_KEYS) <= set(src):
        raise ValueError(f"--branch_from {args.branch_from} is not a full epoch checkpoint ({', '.join(BRANCH_SOURCE_KEYS)}); "
                         "an export/teacher_backbone_*.pth holds the backbone only")
    if src["args"].get("no_gram") is not True:
        raise ValueError(f"--branch_from {args.branch_from} is not a --no_gram run's checkpoint (its no_gram is {src['args'].get('no_gram')!r}): "
                         "a branch adds the Gram teacher to a no-Gram trunk")
    # every other --resume-locked setting, the image list, world size and epoch length (step = epoch x epoch_length) as a --resume of the trunk checks them
    check_resume(src, args, world_size, epoch_length, data_fingerprint, skip=BRANCH_FREE, action="--branch_from")
    early = [s for s in args.gram_refresh_steps if s <= src["step"]]
    if early:
        raise ValueError(f"--gram_refresh_steps {early} not after the branch step {src['step']} (epoch {src['epoch']} of {args.branch_from}): "
                         "refresh steps are absolute global steps")
    if stop_epoch <= src["epoch"]:
        raise ValueError(f"--stop_after_epochs {stop_epoch} is not after the source's epoch {src['epoch']}: the branch would train nothing")
    return src


def run(args, vit_kwargs=None, head_kwargs=None, datasets_config=None, check_exists=True):
    """Train (or resume) as args say; returns dict(step, epoch, gram_refreshes, peak_lr, epoch_length). The keyword
    arguments are the test seam: a tiny ViT / heads, an in-memory datasets config, no image-existence check."""
    args = argparse.Namespace(**vars(args))
    stop_epoch = args.stop_after_epochs = check_args(args)
    # a --resume of a run that has its own epoch checkpoint restores that one (check_resume compares the recorded init / branch path), so the
    # init or branch source file is read, and must exist, only when the run starts
    resuming_own = bool(args.resume and epoch_checkpoints(os.path.join(args.output_dir, "ckpt")))
    if args.init_from_checkpoint and not resuming_own and not os.path.isfile(args.init_from_checkpoint):
        raise FileNotFoundError(f"--init_from_checkpoint {args.init_from_checkpoint} is not a file")
    branching = bool(args.branch_from) and not resuming_own  # likewise the --branch_from source: read, and needed, only at the branch's start
    if branching and not os.path.isfile(args.branch_from):
        raise FileNotFoundError(f"--branch_from {args.branch_from} is not a file")
    accelerator = Accelerator(args.device)
    accelerator.initialize_distributed()
    ensure_process_group(accelerator)
    rank, world_size, device, main = accelerator.rank, accelerator.world_size, accelerator.device, accelerator.is_main_process

    def say(message):
        if main:
            print(message, flush=True)

    # same seed on every rank before the models exist: identical init everywhere
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = args.output_dir
    ckpt_dir, export_dir = os.path.join(out, "ckpt"), os.path.join(out, "export")
    if epoch_checkpoints(ckpt_dir) and not args.resume:
        raise FileExistsError(f"{ckpt_dir} already holds epoch checkpoints: pass --resume to continue that run, "
                              "or use a new --output_dir")

    weights_from = "--init_from_checkpoint" if args.init_from_checkpoint else "--branch_from" if args.branch_from else None
    if weights_from and args.pretrained_weights:
        say(f"WARNING: --pretrained_weights {args.pretrained_weights} ignored: {weights_from} supplies every weight")
    # the student's training recipe; the EMA and Gram teachers are its copies (same parameters), and run in eval, where none of it acts
    recipe = dict(drop_path_rate=args.drop_path_rate, pos_embed_rope_rescale_coords=args.rope_rescale_coords,
                  untie_global_and_local_cls_norm=args.untie_local_cls_norm)
    student, teacher, gram_teacher = build_models(dict(vit_kwargs or DINOV3_VITB16_KWARGS, **recipe),
                                                  None if weights_from else args.pretrained_weights,
                                                  head_kwargs or HEAD_KWARGS, gram=not args.no_gram)
    # the init and the EMA / Gram copies are done: from here each rank draws its own model-side randomness (drop path, RoPE rescale), as Meta seeds
    # seed + rank; rank 0 keeps args.seed. The data loaders seed per rank themselves (build_data_loader, set_epoch)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if (args.drop_path_rate, args.rope_rescale_coords, args.untie_local_cls_norm) != (0.0, None, False):  # the defaults print nothing new
        say(f"recipe (student, train mode only; the EMA / Gram teachers run in eval): drop path {args.drop_path_rate:g}, RoPE coordinate rescale "
            f"{'none' if args.rope_rescale_coords is None else format(args.rope_rescale_coords, 'g')}, local-crop CLS norm "
            f"{'untied' if args.untie_local_cls_norm else 'tied'}")
    if args.untie_local_cls_norm and args.pretrained_weights and not weights_from and not resuming_own:  # a resume / init / branch restores it instead
        say(f"local_cls_norm: initialized as a copy of norm.weight / norm.bias from {args.pretrained_weights} (the stock file has no local_cls_norm)")
    if args.no_gram:
        say("Gram: OFF (--no_gram): no Gram teacher built or loaded, no Gram loss, no Gram refreshes; "
            "ssl_losses.csv gram and gram_weight columns are 0.0")
    # a resume of this run restores its own checkpoint below instead (check_resume: same init_from_checkpoint or refused)
    if args.init_from_checkpoint and not resuming_own:
        init = torch.load(args.init_from_checkpoint, map_location="cpu", weights_only=False)
        if not {"student", "teacher", "gram_teacher", "args"} <= set(init):
            raise ValueError(f"--init_from_checkpoint {args.init_from_checkpoint} is not a full checkpoint (student, teacher, "
                             "gram_teacher, args); an export/teacher_backbone_*.pth holds the backbone only")
        student.load_state_dict(init["student"], strict=True)
        teacher.load_state_dict(init["teacher"], strict=True)
        # DINOv3 re-anchors the Gram teacher to the previous stage's EMA teacher (not to that stage's own, older Gram teacher)
        if gram_teacher is not None:  # --no_gram: nothing to anchor
            gram_teacher.load_state_dict({k: v for k, v in init["teacher"].items() if k.startswith("backbone.")}, strict=True)
        gram_init_note = "no Gram teacher (--no_gram)" if args.no_gram else "Gram teacher <- its EMA teacher backbone"
        was = dict(CROP_DEFAULTS, **init["args"])
        say(f"initialized from {args.init_from_checkpoint}: its step {init.get('step')}, epoch {init.get('epoch')}, crops global "
            f"{was['global_crops_size']} / local {was['local_crops_size']} / Gram teacher {was['gram_teacher_size']}; student and "
            f"teacher (heads included) loaded strictly, {gram_init_note}; fresh optimizer, step 0")
        del init
    student, teacher = student.to(device), teacher.to(device)
    if gram_teacher is not None:
        gram_teacher = gram_teacher.to(device)
    # Meta calls student.backbone / dino_head / ibot_head separately, so each is its own DDP module
    student = nn.ModuleDict({k: accelerator.wrap_model(m) for k, m in student.items()})
    student_unwrapped = nn.ModuleDict({k: accelerator.unwrap_model(m) for k, m in student.items()})
    meta = SSLMetaArch(student, teacher, gram_teacher, n_local_crops=args.local_crops_number,
                       gram_loss_weight=args.gram_weight,
                       autocast_dtype=torch.bfloat16 if args.bf16 else None).to(device)
    meta.train()
    optimizer = build_optimizer(get_params_groups(student))

    transform = DataAugmentationCXR(local_crops_number=args.local_crops_number, global_crops_size=args.global_crops_size,
                                    local_crops_size=args.local_crops_size, gram_teacher_crops_size=args.gram_teacher_size,
                                    emit_gram_teacher_crops=not args.no_gram)
    decode_side = transform.decode_min_side  # follows the crop sizes: 1024 for 224 / 96 / 448 (as before), 2368 for 512 / 224 / 1024
    gram_crops_note = (f"no Gram teacher (global crops cut at {args.gram_teacher_size} and resized to {args.global_crops_size}, as in the "
                       "Gram runs)" if args.no_gram else f"Gram teacher {args.gram_teacher_size}")
    say(f"crops: global {args.global_crops_size}, local {args.local_crops_size} (x{args.local_crops_number}), "
        f"{gram_crops_note}; JPEG decode min side {decode_side}")
    if datasets_config is None:
        datasets_config = get_config(args.datasets_config)
    dataset = build_dataset(args.data_set, datasets_config, transform, check_exists=check_exists and main, decode_side=decode_side)
    loader, sampler = build_data_loader(dataset, args.batch_size_per_gpu, args.workers, rank, world_size, args.seed,
                                        accelerator.pin_memory, make_collate_fn(global_crops_size=args.global_crops_size))
    epoch_length = len(loader)
    data_fingerprint = hashlib.sha256("\n".join(dataset.paths).encode()).hexdigest()  # resume guard
    if epoch_length == 0:
        raise ValueError(f"{len(dataset)} images give no full batch of {args.batch_size_per_gpu} x {world_size} ranks")

    global_batch = args.batch_size_per_gpu * world_size
    sch = build_schedules(lr=args.lr, min_lr=args.min_lr, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
                          epoch_length=epoch_length, global_batch_size=global_batch, scale_lr=not args.no_lr_scaling,
                          teacher_temp_warmup_epochs=args.teacher_temp_warmup_epochs,
                          freeze_last_layer_epochs=FREEZE_LAST_LAYER_EPOCHS)
    total_steps = args.epochs * epoch_length
    if args.no_lr_scaling:
        say(f"LR: --no_lr_scaling, peak = --lr = {args.lr:g}, end {sch['end_lr']:g}")
    else:
        say(f"LR: --lr {args.lr:g} x sqrt_wrt_1024 factor 4*sqrt({global_batch}/1024) = {sch['factor']:.4f} "
            f"-> peak {sch['peak_lr']:g}, end {sch['end_lr']:g}")
    say(f"epoch_length {epoch_length} steps, {total_steps} steps in the {args.epochs}-epoch schedule, "
        f"stopping after epoch {stop_epoch} (step {stop_epoch * epoch_length})")
    for s in args.gram_refresh_steps:
        say(f"Gram refresh scheduled after step {s} (epoch {s / epoch_length:.2f} of {args.epochs})")

    step, epochs_done, gram_refreshes, ckpt = 0, 0, [], None
    if args.resume:
        found = epoch_checkpoints(ckpt_dir)
        if found:
            ckpt = torch.load(found[-1][1], map_location="cpu", weights_only=False)
            check_resume(ckpt, args, world_size, epoch_length, data_fingerprint)  # raises before anything on disk is touched
        elif not branching:  # a branch's start prints its "branched from" line below instead
            say(f"no checkpoint in {ckpt_dir}, starting from scratch")
    if branching:  # its every refusal raises here, before anything on disk is touched
        ckpt = load_branch_source(args, stop_epoch, world_size, epoch_length, data_fingerprint)
    if main:  # after the resume / branch checks: a rejected run creates nothing and must not overwrite the record of the real run
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(export_dir, exist_ok=True)
        info = dict(vars(args), decode_min_side=decode_side, effective_peak_lr=sch["peak_lr"], effective_end_lr=sch["end_lr"],
                    lr_scaling_factor=sch["factor"], world_size=world_size, global_batch_size=global_batch,
                    epoch_length=epoch_length, total_steps=total_steps)
        with open(os.path.join(out, "args.json"), "w") as f:
            json.dump(info, f, indent=2)
    accelerator.barrier()
    if ckpt is not None:
        student_unwrapped.load_state_dict(ckpt["student"])
        teacher.load_state_dict(ckpt["teacher"])
        if branching:  # the anchor: the source's EMA teacher backbone, mapped as --init_from_checkpoint maps it (the source has no Gram teacher)
            gram_teacher.load_state_dict({k: v for k, v in ckpt["teacher"].items() if k.startswith("backbone.")}, strict=True)
        elif gram_teacher is not None:  # --no_gram: the checkpoint's gram_teacher is None
            gram_teacher.load_state_dict(ckpt["gram_teacher"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, epochs_done = ckpt["step"], ckpt["epoch"]
        gram_refreshes = [] if branching else list(ckpt["gram_refreshes"])  # a branch's refreshes are its own
        if "rng_per_rank" in ckpt:  # entry r = rank r's states: the ranks keep drawing differently across a resume / branch
            states = ckpt["rng_per_rank"]
            assert len(states) == world_size, f"the checkpoint holds {len(states)} per-rank RNG states, this run has {world_size} ranks"
            rng = states[rank]
        else:  # one rank, or written before the per-rank states: its one state (rank 0's), on every rank
            rng = ckpt["rng"]
            if world_size > 1:
                say(f"RNG: the checkpoint predates the per-rank RNG states and holds one: it is restored on every rank ({world_size})")
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if branching:
            say(f"branched from {args.branch_from}: its step {step}, epoch {epochs_done}; student/teacher/optimizer/RNG restored; Gram teacher <- its "
                f"EMA teacher backbone; Gram ON, weight {args.gram_weight:g}, refreshes after steps {args.gram_refresh_steps}")
        else:
            restored = "student/teacher/optimizer/RNG restored" if args.no_gram else "student/teacher/Gram teacher/optimizer/RNG restored"
            say(f"resumed from {found[-1][1]}: {epochs_done} epochs / {step} steps done, Gram refreshes {gram_refreshes}, {restored}")
        del ckpt
    summary = lambda: dict(step=step, epoch=epochs_done, gram_refreshes=list(gram_refreshes),
                           peak_lr=sch["peak_lr"], epoch_length=epoch_length)
    csv_path = os.path.join(out, "ssl_losses.csv")
    if main:  # fresh: overwrite. Resume: keep the rows of finished steps, drop those of the interrupted epoch
        kept = []
        if step and os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                kept = [r for r in list(csv.reader(f))[1:] if len(r) == len(CSV_HEADER) and r[0].isdigit() and int(r[0]) < step]
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerows([CSV_HEADER] + kept)

    if args.max_steps and step >= args.max_steps:
        say(f"nothing to do: {step} steps done, --max_steps {args.max_steps}")
        return summary()
    if epochs_done >= stop_epoch:
        say(f"nothing to do: {epochs_done} epochs done, --stop_after_epochs {stop_epoch}")
        return summary()

    def save_state(ckpt_path, export_path):
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
        per_rank = {}  # one rank: rng says it all, and its checkpoint keeps the layout it had before the per-rank states (and runs no extra collective)
        if world_size > 1:  # every rank's states, so --resume / --branch_from hand each rank its own back; a collective: every rank calls it, rank 0 writes
            per_rank["rng_per_rank"] = [None] * world_size
            dist.all_gather_object(per_rank["rng_per_rank"], rng)
        if main:  # export first: the checkpoint is the commit record --resume looks for
            save_atomic(export_backbone(teacher.backbone), export_path)
            save_atomic(to_cpu(dict(
                student=student_unwrapped.state_dict(), teacher=teacher.state_dict(),
                gram_teacher=None if gram_teacher is None else gram_teacher.state_dict(),  # --no_gram: the key stays, holding None
                optimizer=optimizer.state_dict(), step=step, epoch=epochs_done,
                gram_refreshes=list(gram_refreshes), args=vars(args), world_size=world_size, data_fingerprint=data_fingerprint,
                rng=rng, **per_rank)), ckpt_path)  # rng: rank 0's, as before
        accelerator.barrier()

    refresh_steps, first_step = set(args.gram_refresh_steps), step
    end_step = stop_epoch * epoch_length if not args.max_steps else min(stop_epoch * epoch_length, args.max_steps)
    pending = None  # (k, S, gram loss of the row before) of a Gram refresh whose "after" row is not logged yet

    def gram_line(k, s, before, after):
        tail = f"after (step {s} row) {after:.6f}" if after is not None else "run ended before the next row"
        return (f"Gram refresh {k}/{len(args.gram_refresh_steps)} after step {s}: Gram teacher <- EMA teacher backbone; "
                f"gram loss before (step {s - 1} row) {before:.6f}, {tail}")

    t_start = time.time()
    for epoch in range(epochs_done, stop_epoch):
        set_epoch(loader, sampler, epoch, args.seed, rank)
        freeze = epoch < args.freeze_backbone_epochs  # ours: backbone LR 0 while the heads warm up
        t_epoch = t_mark = t_data = time.time()
        n_mark, data_wait = 0, 0.0
        for data in loader:
            data_wait += time.time() - t_data
            it = step
            lr, temp = float(sch["lr"][it]), float(sch["teacher_temp"][it])
            apply_optim_scheduler(optimizer, lr, args.weight_decay, float(sch["last_layer_lr"][it]), freeze_backbone=freeze)
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = meta.forward_backward(data, teacher_temp=temp)
            accelerator.mark_step()
            for module in student.values():
                torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=CLIP_GRAD)
            optimizer.step()
            accelerator.mark_step()
            meta.update_ema(args.momentum_teacher)
            accelerator.mark_step()
            step += 1
            n_mark += 1
            refreshed = step in refresh_steps and step not in gram_refreshes
            if refreshed:  # dinov3: update_gram(m=0) after iteration it when (it + 1) hits the refresh cadence
                meta.update_gram()
                accelerator.mark_step()
                gram_refreshes.append(step)
            # no host sync outside these steps: every print_freq, around a Gram refresh, the first step of the run
            # (compile time), the last step of every epoch (so a NaN is caught before a checkpoint) and of the run
            if (step % args.print_freq == 0 or refreshed or (step - 1) in gram_refreshes or step == first_step + 1
                    or step % epoch_length == 0 or step == end_step):
                # Ark's trainer.py pattern: host reads (.item(), one per scalar) of tensors the mark_step() calls above already
                # materialized, then a tensor built from those Python floats is all-reduced and divided by world_size on the
                # host. Never a torch.stack of the lazy loss tensors: all_reduce(stack(...)) died on Gaudi in HPU lazy mode
                # (synNodeCreateWithId failed for node: concat).
                # --no_gram: no gram_loss key; its column is 0.0, never blank (the report builder parses the CSV as numbers)
                local = [0.0 if args.no_gram and k == "gram_loss" else loss_dict[k].item() for k in LOSS_KEYS] + [loss.item()]
                stats = torch.tensor(local, dtype=torch.float32, device=device)
                if accelerator.distributed:
                    accelerator.all_reduce(stats)
                dg, dl, ko, ib, gr, total = (v / world_size for v in stats.cpu().tolist())  # the host sync
                now = time.time()
                # under HPU lazy mode one step can't be timed without a sync, so these are per-step means over the
                # steps since the previous row (this sync closes the window)
                step_time, data_time = (now - t_mark) / n_mark, data_wait / n_mark
                if main:
                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([it, epoch, dg, dl, ko, ib, gr, total, lr, args.weight_decay,
                                                args.momentum_teacher, temp, args.gram_weight, step_time, data_time])
                    print(f"step {step}/{total_steps} epoch {epoch} | dino_g {dg:.4f} dino_l {dl:.4f} koleo {ko:.4f} "
                          f"ibot {ib:.4f} gram {gr:.5f} total {total:.4f} | lr {lr:.3e} temp {temp:.4f} | "
                          f"{step_time:.3f}s/step data {data_time:.3f}s | elapsed {now - t_start:.0f}s", flush=True)
                if pending and step - 1 == pending[1]:
                    say(gram_line(*pending, gr))
                    pending = None
                if refreshed:
                    pending = (len(gram_refreshes), step, gr)
                if not math.isfinite(total):  # NaN or inf on any rank (the sum carries it): abort before a checkpoint saves it
                    raise RuntimeError(f"non-finite loss {total} at step {it} on rank {rank} "
                                       f"(all-reduced over {world_size} ranks); row written to {csv_path}")
                t_mark, n_mark, data_wait = time.time(), 0, 0.0
            t_data = time.time()
            if args.max_steps and step >= args.max_steps:
                break
        if step == (epoch + 1) * epoch_length:  # epoch complete (its last step was a synced log step)
            epoch_time = time.time() - t_epoch
            epochs_done = epoch + 1
            save_state(os.path.join(ckpt_dir, "epoch_%04d.pth" % epochs_done),
                       os.path.join(export_dir, "teacher_backbone_epoch_%04d.pth" % epochs_done))
            if main:
                prune_checkpoints(ckpt_dir)
            say(f"epoch {epochs_done}/{args.epochs} done in {epoch_time:.0f}s ({epoch_time / epoch_length:.3f} s/step); "
                f"projected remaining: {(stop_epoch - epochs_done) * epoch_time / 3600:.2f} h to epoch {stop_epoch} "
                f"(--stop_after_epochs), {(args.epochs - epochs_done) * epoch_time / 3600:.2f} h to epoch {args.epochs}")
        if args.max_steps and step >= args.max_steps:
            save_state(os.path.join(ckpt_dir, "maxsteps_%07d.pth" % step),
                       os.path.join(export_dir, "teacher_backbone_maxsteps_%07d.pth" % step))
            say(f"--max_steps {args.max_steps} reached at step {step}")
            break
    if pending:
        say(gram_line(*pending, None))
    return summary()


if __name__ == "__main__":
    run(parse_args())
    if dist.is_initialized():
        dist.destroy_process_group()
