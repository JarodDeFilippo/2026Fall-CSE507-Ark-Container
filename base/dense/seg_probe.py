"""Frozen-backbone segmentation probe: Foundation X's UPerNet-style head (upernet.py) trained on the patch tokens of a frozen ViT (pyramid.py) for CANDID-PTX pneumothorax or
JSRT lungs (data.py), under RayDINO's budget: 5,000 iterations of batch 16, AdamW (lr 1e-4, weight decay 1e-4) with Foundation X's cosine step_decay schedule, a validation pass every
250 iterations, and the best validation-loss state tested once. One run is one (backbone, dataset, seed).

Run from base/:
    python -m dense.seg_probe --model vit_base_dinov3 --weights ../weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth --normalization imagenet \
        --dataset jsrt_lung --tag stock_dinov3 --out results/dense

It writes <out>/<tag>/<dataset>/seed<S>/result.json (the settings, the validation losses, the test Dice, the list sizes, the git sha and the times) and head_best.pt (what trained: the
pyramid's level layers and the head; the frozen backbone is the weights file), S being --seed so that the seeds of one (tag, dataset) sit side by side, and refuses to run when that seed<S>
directory exists. Both files are written (and fsynced) into a private seed<S>.partial-<pid>-<hex>/ beside it, which one rename makes seed<S> at the end: a seed<S> that exists is a whole run, and
a second job of the same (tag, dataset, seed) that finishes later fails at the rename, the first one's directory untouched and its own run left in its .partial- directory. Dice is x100 at the
input size: "pooled" over every pixel of the test list, and "mean" over its images, which is Foundation X's CSV column. Progress goes to stderr; stdout is the one SEG line. --weights random
builds the backbone without loading anything, for the CPU tests (it needs --device cpu).
"""
import argparse
import datetime
import json
import math
import os
import random
import secrets
import shutil
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from ark_finetune.extract_embeddings import TAG_RE, load_backbone, sha256_file

from .data import DATASETS, NORMALIZATION, make_dataset
from .pyramid import ViTPyramid, layers_for
from .upernet import SegHead, dice_score, mean_dice_coef, step_decay, torch_dice_coef_loss

MODELS = ("vit_base_dinov3", "vit_base_dinov2", "vit_base_raddino", "vit_large_raydino")
SPLITS = ("train", "val", "test")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train Foundation X's segmentation head on a frozen ViT backbone and test the best validation-loss state once")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--weights", required=True, help="the backbone's weights file, as extract_embeddings loads it, or 'random' (tests only: an untrained backbone, --device cpu)")
    p.add_argument("--normalization", required=True, choices=list(NORMALIZATION), help="the backbone's own: the one its linear probe used")
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--tag", required=True, help="names the backbone, e.g. stock_dinov3: the result is <out>/<tag>/<dataset>/seed<seed>/")
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--batch", type=int, default=16, help="images per batch, at least 2: the head's pyramid pooling reduces a map to 1 x 1, where training-mode BatchNorm needs more than one image")
    p.add_argument("--val_every", type=int, default=250, help="iterations between validation passes (and one at the last iteration)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0, help="torch, numpy and random, the train loader's shuffle and its augmentation draws; names the output directory, seed<seed>")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--size", type=int, default=224, help="input side in pixels; a multiple of the backbone's patch size")
    p.add_argument("--out", default="results/dense")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--data_root", help="tests: replaces the dataset's image root")
    p.add_argument("--list_dir", help="tests: a directory with train.txt, val.txt and test.txt in place of the lab's lists")
    args = p.parse_args(argv)
    if not TAG_RE.match(args.tag):  # the tag is a directory name
        p.error("--tag {!r}: letters, digits, . _ - only, starting with a letter or digit".format(args.tag))
    if args.weights == "random":
        if args.device != "cpu":
            p.error("--weights random is an untrained backbone, for tests: it needs --device cpu")
    elif not os.path.isfile(args.weights):
        p.error("--weights {} is not a file".format(args.weights))
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("--device cuda, but CUDA is not available")
    # iters: step_decay divides by iters - step_inc, so one iteration is too few; batch: the PSP bin of 1 x 1 gives training-mode BatchNorm2d one value per channel from a batch of one; seed: it names a directory
    for name, least in (("iters", 2), ("batch", 2), ("val_every", 1), ("size", 1), ("workers", 0), ("seed", 0)):
        if getattr(args, name) < least:
            p.error("--{} must be at least {}".format(name, least))
    return args


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha():
    """HEAD of the repository this file is in; None when git or the repository is not there (a container's copy may have no .git)."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, check=True,
                              timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def log(message):
    print("seg_probe: " + message, file=sys.stderr, flush=True)


def autocast(device):
    """bf16 autocast on CUDA, nothing anywhere else."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def build_model(model_name, weights, device):
    """(ViTPyramid, SegHead) on `device` over the container's backbone for `model_name`, loaded from `weights` by load_backbone ('random': built, not loaded)."""
    backbone = load_backbone(model_name, None if weights == "random" else weights)
    pyramid = ViTPyramid(backbone, layers_for(backbone))
    return pyramid.to(device), SegHead(pyramid.feature_channels, 1).to(device)


def trainable_state(pyramid, head):
    """A copy, on the CPU, of what trains: the pyramid's level layers and the head. The frozen backbone is the weights file, so it is not in here."""
    return {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()} for name, module in (("pyramid", pyramid.levels), ("head", head))}


def forever(loader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(pyramid, head, loader, device):
    """(mean Dice loss, pooled Dice x100, per-image mean Dice x100) over the loader, with the head in eval mode (restored afterwards). The loss is the batch losses averaged by
    batch size, as Foundation X's evaluation_SEGMENTATION does; the Dice values are computed on the sigmoid maps and masks of the whole loader, as its test_SEGMENTATION gathers them."""
    # ponytail: the whole list's sigmoid maps and masks are held in memory (float32, 400 KB per image at 224 px); count intersections per batch instead if a list outgrows the node
    was_training = head.training
    head.eval()
    loss_sum, seen, probs, masks = 0.0, 0, [], []
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        with autocast(device):
            logits = head(pyramid(img), img.shape[-2:])
        prob = torch.sigmoid(logits.float())
        loss_sum += torch_dice_coef_loss(mask, prob).item() * len(img)
        seen += len(img)
        probs.append(prob.cpu().numpy())
        masks.append(mask.cpu().numpy())
    head.train(was_training)
    p, y = np.concatenate(probs), np.concatenate(masks)
    return loss_sum / seen, 100 * float(dice_score(p, y)), 100 * float(mean_dice_coef(y > 0.5, p > 0.5))


def step_inc_for(iters):
    """step_decay's warm-up length: Foundation X's 250 iterations at the 5,000-iteration budget; a shorter run gets a twentieth of its iterations (at least one), so step_inc stays below iters."""
    return min(250, max(1, iters // 20))


def make_loader(dataset, args, device, **kwargs):
    return DataLoader(dataset, batch_size=args.batch, num_workers=args.workers, pin_memory=device.type == "cuda", **kwargs)


def train_into(args, out_dir):
    """One run: train on the train list, validate every --val_every iterations (and at the last), test the best validation-loss state once, write head_best.pt and result.json (both
    fsynced) into `out_dir`, which exists. Returns the result dict."""
    started = utc_now()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    sets = {split: make_dataset(args.dataset, split, args.size, args.normalization, root=args.data_root,
                                list_path=os.path.join(args.list_dir, split + ".txt") if args.list_dir else None) for split in SPLITS}
    counts = {split: len(dataset) for split, dataset in sets.items()}
    if counts["train"] < args.batch or not counts["val"] or not counts["test"]:  # the train loader drops its last partial batch (as Foundation X's does) and would never yield
        raise ValueError("the train list needs at least --batch ({}) images and the val and test lists one: {} / {} / {}".format(args.batch, *counts.values()))
    sha = None if args.weights == "random" else sha256_file(args.weights)  # of the file the model loads next
    pyramid, head = build_model(args.model, args.weights, device)
    if args.size % pyramid.backbone.patch_size:  # DINOv3 would drop the remainder pixels without a word
        raise ValueError("--size {} is not a multiple of {}'s patch size {}".format(args.size, args.model, pyramid.backbone.patch_size))
    trainable = [p for p in pyramid.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    train_loader = make_loader(sets["train"], args, device, shuffle=True, drop_last=True, persistent_workers=args.workers > 0, generator=torch.Generator().manual_seed(args.seed))
    val_loader, test_loader = make_loader(sets["val"], args, device), make_loader(sets["test"], args, device)
    step_inc = step_inc_for(args.iters)
    log("{} on {}: train {} / val {} / test {} images, {:.1f} M trainable parameters, device {}".format(
        args.tag, args.dataset, counts["train"], counts["val"], counts["test"], sum(p.numel() for p in trainable) / 1e6, device))

    pyramid.train()  # the backbone stays in eval
    head.train()
    batches = forever(train_loader)
    val_losses, recent = [], []
    best_val, best_iter, best_state = math.inf, None, None
    for it in range(1, args.iters + 1):
        for group in optimizer.param_groups:
            group["lr"] = float(step_decay(it, args.lr, args.iters, step_inc=step_inc))
        img, mask = next(batches)
        img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        with autocast(device):
            logits = head(pyramid(img), img.shape[-2:])
        loss = torch_dice_coef_loss(mask, torch.sigmoid(logits.float()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        recent.append(loss.detach())
        if it % args.val_every == 0 or it == args.iters:
            val_loss, val_pooled, val_mean = evaluate(pyramid, head, val_loader, device)
            if not math.isfinite(val_loss):  # the weights have diverged: stop the job here, not at its time limit
                raise RuntimeError("the validation loss is {} at iteration {}".format(val_loss, it))
            val_losses.append(val_loss)
            if val_loss < best_val:
                best_val, best_iter, best_state = val_loss, it, trainable_state(pyramid, head)
            log("iteration {}/{}: lr {:.2e}, train loss {:.4f}, val loss {:.4f}, val Dice pooled {:.2f} / mean {:.2f} (best val loss {:.4f} at {})".format(
                it, args.iters, optimizer.param_groups[0]["lr"], torch.stack(recent).mean().item(), val_loss, val_pooled, val_mean, best_val, best_iter))
            recent.clear()

    pyramid.levels.load_state_dict(best_state["pyramid"])  # the test list sees the best validation-loss state, not the last one
    head.load_state_dict(best_state["head"])
    _, test_pooled, test_mean = evaluate(pyramid, head, test_loader, device)
    # ponytail: head_best.pt and result.json are written once, here; a Slurm time-limit kill loses the run (and leaves its .partial- directory). Save on each improvement if that ever costs one.
    with open(os.path.join(out_dir, "head_best.pt"), "wb") as f:
        torch.save(best_state, f)
        f.flush()
        os.fsync(f.fileno())
    result = {"tag": args.tag, "model": args.model, "weights": args.weights, "weights_sha256": sha, "normalization": args.normalization, "dataset": args.dataset,
              "seed": args.seed, "iters": args.iters, "batch": args.batch, "lr": args.lr, "wd": args.wd, "size": args.size, "step_inc": step_inc,
              "best_iter": best_iter, "best_val_loss": best_val, "val_losses": val_losses, "test_dice_pooled": test_pooled, "test_dice_mean": test_mean,
              "n_train": counts["train"], "n_val": counts["val"], "n_test": counts["test"], "git_sha": git_sha(), "started": started, "finished": utc_now()}
    text = json.dumps(result, indent=2, allow_nan=False)  # serialized first: a NaN raises before result.json exists
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())
    return result


def publish(staging, final_dir):
    """Rename the finished run in `staging` to `final_dir`. One rename, so `final_dir` appears whole or not at all, and it fails when anything but an empty directory is already there (a job of
    the same tag, dataset and seed that published first): what is there is not touched, and this run stays in `staging`, for inspection."""
    try:
        os.rename(staging, final_dir)  # staging is beside final_dir, so on its filesystem: no copy fallback
    except OSError as error:
        raise RuntimeError("could not publish {} as {} ({}): the run stays in {} and {} was not touched; another job with this tag, dataset and seed may have published first".format(
            staging, final_dir, error.strerror or error, staging, final_dir)) from error


def train(args):
    """One run (train_into), written into a private seed<S>.partial-<pid>-<hex> directory beside <out>/<tag>/<dataset>/seed<S> and published by renaming that directory to seed<S> once both
    files are complete. Returns the result dict."""
    final_dir = os.path.join(args.out, args.tag, args.dataset, "seed{}".format(args.seed))
    if os.path.lexists(final_dir):  # a run costs hours: say so before the hash, the model and the data (lexists: a dangling symlink at that name refuses too)
        raise FileExistsError("refusing to overwrite {}; delete it to run again".format(final_dir))
    staging = "{}.partial-{}-{}".format(final_dir, os.getpid(), secrets.token_hex(4))  # makedirs, not mkdtemp: the umask sets its mode, as it did for seed<S> itself
    os.makedirs(staging)  # now and not at the end, so that an output directory that cannot be written fails before the hours
    try:
        result = train_into(args, staging)
    except BaseException:  # best effort; a failed publish, below this try, keeps the finished run
        shutil.rmtree(staging, ignore_errors=True)
        raise
    publish(staging, final_dir)
    return result


def main(argv=None):
    result = train(parse_args(argv))
    print("SEG {tag} {dataset} seed{seed}: best iteration {best_iter} (val loss {best_val_loss:.4f}), test Dice mean {test_dice_mean:.2f}, pooled {test_dice_pooled:.2f}".format(**result))


if __name__ == "__main__":
    main()
