"""Embedding extractor for B2, the Ark+ lab's official linear probe on pre-generated embeddings (CSE 494 IP, Step VIII).

Freezes a backbone and writes, for every image of the train and test lists of ChestXray14, CheXpert, MIMIC and VinDrCXR, the vector Ark's projector receives (the container's generate_embeddings(after_proj=False)):
1536-d (ViT-B) or 2048-d (ViT-L), the CLS token ++ the mean patch token; 1024-d (Swin-B) or 1536-d (Swin-L), the pooled output of an Ark+ teacher's backbone, before its projector. The lab's own Zeroshot notebook
embeds with generate_embeddings' default after_proj=True (the 1376-d projector output), and because the lab's projector is one nn.Linear (no --use_mlp in the README recipes) that vector is an exact affine map of the cached one.
The images go through the vendored official transform
(build_transform_classification, mode "valid"): resize to 256 and center-crop 224 with no augmentation, for both splits: ONE embedding per image, as the lab's embedding linear probe
(ark_linearprobe/) expects. There is no validation split: the probe has none. Lists and data directories are base/datasets_config.yaml's, the splits every arm uses
(the full official train list of each dataset, VinDrCXR's 15,000 included, and its official test list; dataset_lists.py).
The backbone is built by the container's build_omni_model from --model, passed through as it is, so any ViT model name base/models.py knows works. The Swin names (swin_base, swin_large, swin_large_384,
swin_large_768) are Ark+ checkpoints (.pth.tar): they load the TEACHER weights through build_omni_model_from_checkpoint (see load_backbone), and a Swin is built for ONE input size, so --input_size must be
it (run_emb_gaudi.sh: ARK_EMB_INPUT_SIZE) and main() refuses otherwise. The width of the vectors is the built backbone's (model.num_features), and meta.json records it as feature_dim.

Run from base/ (sbatch run_emb_gaudi.sh <tag> <model> <in-container weights path> <normalization> on Sol):
    python -m ark_finetune.extract_embeddings --device hpu --tag stock --model vit_base_dinov3 --normalization imagenet \
        --weights ../weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

Per dataset it writes <embeddings_root>/<tag>/<dataset>/{train,test}.npy (float16, [N, 1536 (ViT-B or Swin-L), 2048 (ViT-L) or 1024 (Swin-B)]), {split}_paths.txt (the
dataset's img_list, one path per line, row i of the .npy being img_list[i]) and, last, meta.json (tag, model, weights path, size and sha256, normalization, transform,
counts, timings, device). It refuses to overwrite a file that exists, before it loads the model: to redo a dataset (or finish after a job that died in the
middle of one), delete its directory and run again with --datasets.

stdout is one "EMB ..." line per dataset; progress and diagnostics go to stderr: a line at the start and end of each phase (model load, extract <dataset> <split>)
and, while extracting, one per 20 batches with the images/s and the device the features are on. The run refuses (RuntimeError) if the parameters or the features
are not on the requested device. --max_images N is the smoke switch: only the first N list entries of each split (meta.json records it, and the fit then uses
the first N entries of each list too).
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from accelerator import Accelerator
from models import ArkSwinTransformer, build_omni_model, build_omni_model_from_checkpoint

from . import dataloader
from .dataset_lists import BASE, DATASETS, REPO, dataset_entry
from .embedding_mode import EMBEDDING_DIM, SPLITS, split_files, write_paths

NORMALIZATIONS = ("imagenet", "chestx-ray", "none", "raddino")
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")  # the tag is a directory name: no path separators, no spaces


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract the embeddings (1536-d ViT-B or Swin-L, 2048-d ViT-L, 1024-d Swin-B) of a frozen backbone for the B2 linear probe")
    p.add_argument("--tag", required=True, help="names the backbone, e.g. stock_dinov3: the cache is <embeddings_root>/<tag>/<dataset>/")
    p.add_argument("--model", required=True, help="base/models.py's model name, e.g. vit_base_dinov3 (passed through); swin_base, swin_large, swin_large_384 and swin_large_768 embed the TEACHER of an Ark+ .pth.tar")
    p.add_argument("--weights", required=True, help="the backbone's weights file, as the container's builder loads it: the stock .pth, a Stage A export/teacher_backbone_epoch_NNNN.pth, or (Swin names) an Ark+ .pth.tar")
    p.add_argument("--normalization", required=True, choices=NORMALIZATIONS)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--embeddings_root", default=os.path.join(REPO, "embeddings"))
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--img_resize", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64, help="images per batch")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default=os.environ.get("PROJECT_ACCELERATOR", "cuda"), help="cpu|cuda|hpu, as main_ark.py")
    p.add_argument("--max_images", type=int, default=None, help="smoke runs: only the first N list entries of each split")
    args = p.parse_args(argv)
    if not TAG_RE.match(args.tag):
        p.error("--tag {!r}: letters, digits, . _ - only, starting with a letter or digit".format(args.tag))
    if not os.path.isfile(args.weights):
        p.error("--weights {} is not a file".format(args.weights))
    if args.max_images is not None and args.max_images < 1:
        p.error("--max_images must be at least 1")
    if len(set(args.datasets)) != len(args.datasets):
        p.error("--datasets lists a dataset twice")
    return args


def make_log():
    """Progress to stderr, stamped with the seconds since this call: stdout stays one EMB line per dataset."""
    start = time.time()
    return lambda message: print("extract_embeddings [{:7.1f}s] {}".format(time.time() - start, message), file=sys.stderr, flush=True)


def no_log(message):
    pass


@contextlib.contextmanager
def phase(log, name):
    """A start line and, when the block finishes, a done line with its seconds."""
    began = time.time()
    log(name + ": start")
    yield
    log("{}: done in {:.1f}s".format(name, time.time() - began))


def sha256_file(path, chunk=1 << 20):
    """The sha256 hex digest of a file, read `chunk` bytes at a time (a Stage A export is hundreds of MB): the fit refuses caches of one tag whose digests differ."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_backbone(model_name, weights):
    """The container's backbone with no projector and no heads. The ViT names load through build_omni_model, whose strict key guard raises on any missing or unexpected backbone key.
    The Swin names are the Ark+ lab's checkpoints and load through build_omni_model_from_checkpoint with the TEACHER weights, the lab's convention for the encoder of an Ark checkpoint (Ark_Plus README, "Load pre-trained
    encoder": state_dict = checkpoint['teacher']; the Zeroshot notebook, key = "teacher"; the Finetuning --init ark_plus --key teacher). build_omni_model would take checkpoint['state_dict'], the Ark STUDENT, into a Swin with no key guard.
    The load refuses a file with no 'teacher' entry (KeyError), whose Swin embed_dim, window or input size is not the model's, or whose keys do not match the backbone (RuntimeError)."""
    args = SimpleNamespace(model_name=model_name, projector_features=None, use_mlp=False, pretrained_weights=weights)
    if model_name.startswith("swin"):
        return build_omni_model_from_checkpoint(args, [], "teacher")
    return build_omni_model(args, [])


def to_device(model, accelerator):
    """model.to(accelerator.device) as omni_engine does, then check where a parameter really is: a silent CPU fallback must stop here, not run for an hour."""
    model = model.to(accelerator.device)
    placed = next(model.parameters()).device.type
    if placed != accelerator.device.type:
        raise RuntimeError("the model's parameters are on '{}' after .to({}), but device '{}' was requested".format(placed, accelerator.device, accelerator.device.type))
    return model


def extract_features(model, loader, accelerator, log=no_log, split="", dim=EMBEDDING_DIM):
    """Float16 numpy [N, dim] for a loader of images, in loader order: eval mode, no grad, fp32 forward, the vector Ark's
    projector receives. `dim` is its width: main() passes the built backbone's (1536 for a ViT-B or a Swin-L, 2048 for a ViT-L, 1024 for a Swin-B), the default is the ViT-B width.
    Raises RuntimeError if a batch's features are not on the requested device or not `dim` wide, or if any feature is not finite once cast to float16.
    Logs a line for the first two batches (the first compiles the HPU graph, the second is steady state) and one per 20 batches: images done, images/s since the
    previous line, and the device the features are on."""
    model.eval()
    total = len(loader.dataset)
    features = None
    done = reported = 0
    last = time.time()
    with torch.no_grad():
        for i, (samples, _) in enumerate(loader):
            out = model.generate_embeddings(samples.float().to(accelerator.device), after_proj=False)
            if out.device.type != accelerator.device.type:
                raise RuntimeError("the features landed on '{}' but device '{}' was requested: the forward is not running where it was asked to".format(
                    out.device.type, accelerator.device.type))
            accelerator.mark_step()  # flush the HPU lazy graph per batch, as trainer.evaluate does
            where = out.device
            out = out.float().cpu()  # and bring the features to the host per batch (evaluate's loss.item()): in lazy mode this is what executes the batch
            if out.shape[-1] != dim:
                raise RuntimeError("the backbone returned {}-d features, the probe's head takes {}".format(out.shape[-1], dim))
            if features is None:
                features = np.empty((total, dim), dtype=np.float16)
            features[done:done + len(samples)] = out.numpy()
            done += len(samples)
            if i < 2 or (i + 1) % 20 == 0:
                now = time.time()
                log("extract {} {}/{} imgs, {:.1f} img/s, device={}".format(split, done, total, (done - reported) / max(now - last, 1e-9), where))
                last, reported = now, done
    if features is None or done != total:
        raise RuntimeError("extracted {} of {} images".format(done, total))
    if not np.isfinite(features).all():
        raise RuntimeError("{}: {} of {} features are not finite once cast to float16".format(split, int((~np.isfinite(features)).sum()), features.size))
    return features


def build_splits(name, args):
    """{split: official dataset} for one dataset, constructed as the official main_classification branch constructs them but with the extraction transform:
    'valid' (resize, center crop, no augmentation) for both splits."""
    entry = dataset_entry(name)
    splits = {}
    for split in SPLITS:
        transform = dataloader.build_transform_classification(normalize=args.normalization, crop_size=args.input_size, resize=args.img_resize, mode="valid")
        splits[split] = getattr(dataloader, name)(images_path=entry["data_dir"], file_path=entry[split + "_list"], augment=transform)
    return entry, splits


def main(argv=None):
    args = parse_args(argv)
    start = time.time()
    log = make_log()
    out_dirs = {name: os.path.join(args.embeddings_root, args.tag, name) for name in args.datasets}
    taken = [path for folder in out_dirs.values() for path in [os.path.join(folder, "meta.json")] + [f for split in SPLITS for f in split_files(folder, split)] if os.path.exists(path)]
    if taken:  # before the model loads and before an image is read
        raise FileExistsError("refusing to overwrite: {} exist(s); delete the dataset's directory to extract it again".format(", ".join(taken)))
    datasets = {name: build_splits(name, args) for name in args.datasets}  # parses every list now: a bad list fails before the accelerator is acquired
    log("{} on {}: {} requested".format(", ".join("{} train {} / test {}".format(n, *(len(datasets[n][1][s]) for s in SPLITS)) for n in args.datasets), args.tag, args.device))
    with phase(log, "weights sha256"):  # of the file the model loads next; the same record goes into every dataset's meta.json
        weights = {"path": args.weights, "size_bytes": os.path.getsize(args.weights), "sha256": sha256_file(args.weights)}
    with phase(log, "model load"):
        accelerator = Accelerator(args.device)
        model = to_device(load_backbone(args.model, args.weights), accelerator)
        dim = model.num_features  # no projector was built: the encoder's width (CLS ++ mean patch, or a Swin's pooled output), which sizes the arrays, the shape check and meta.json's feature_dim
        if isinstance(model, ArkSwinTransformer) and tuple(model.patch_embed.img_size) != (args.input_size, args.input_size):  # timm 0.5.4's PatchEmbed would assert at the first batch, after the output directories were made
            size = model.patch_embed.img_size[0]
            raise ValueError("--model {} takes {}x{} input, but the crop is {} px: set ARK_EMB_INPUT_SIZE={} (run_emb_gaudi.sh), i.e. --input_size {}".format(args.model, size, size, args.input_size, size, size))
        log("model parameters on {} (lazy_mode={})".format(next(model.parameters()).device, getattr(accelerator, "lazy_mode", None)))
    for name, (entry, splits) in datasets.items():
        os.makedirs(out_dirs[name])
        counts, seconds, shapes, transforms = {}, {}, {}, {}
        for split, dataset in splits.items():
            began = time.time()
            keep = len(dataset) if args.max_images is None else min(args.max_images, len(dataset))
            with phase(log, "extract {} {}".format(name, split)):
                loader = DataLoader(Subset(dataset, range(keep)), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=accelerator.pin_memory)
                features = extract_features(model, loader, accelerator, log, "{} {}".format(name, split), dim)
            npy, paths = split_files(out_dirs[name], split)
            with open(npy, "xb") as f:  # "x": never overwrites
                np.save(f, features)
            write_paths(paths, dataset.img_list[:keep])
            counts[split], seconds[split], shapes[split] = keep, round(time.time() - began, 1), list(features.shape)
            transforms[split] = [type(t).__name__ for t in dataset.augment.transforms]
        meta = {
            "tag": args.tag, "dataset": name, "model": args.model, "weights": weights,
            "normalization": args.normalization, "transform": {"resize": args.img_resize, "crop": args.input_size, "pipelines": transforms},
            "feature_dim": dim, "dtype": "float16", "counts": counts, "shapes": shapes, "max_images": args.max_images,
            "data_dir": entry["data_dir"], "lists": {split: os.path.relpath(entry[split + "_list"], BASE) for split in SPLITS},
            "seconds": seconds, "device": accelerator.name, "torch": torch.__version__, "workers": args.workers, "batch_size": args.batch_size,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(os.path.join(out_dirs[name], "meta.json"), "x") as f:
            json.dump(meta, f, indent=2, allow_nan=False)
        print("EMB {} {}: train {} / test {}, dim {}, {:.0f}s".format(args.tag, name, counts["train"], counts["test"], dim, sum(seconds.values())))
    log("done in {:.1f}s".format(time.time() - start))


if __name__ == "__main__":
    main()
