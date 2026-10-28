"""B2's fit stage: the lab's embedding linear probe (Ark_Plus/Linearprobing/main_linearprobing.py), as shipped, on the embeddings extract_embeddings.py cached (CSE 494 IP, Step VIII).

For one backbone tag it runs the vendored main() once per dataset, on the CPU, with the shipped settings: a float64 nn.Linear on the 1536-d (ViT-B) or 2048-d (ViT-L) embedding (the cache's feature_dim), sigmoid + BCELoss, Adam lr 1e-4,
batch 512, 100 epochs, CosineAnnealingLR(T_max=1000, eta_min=1) stepped per epoch (eta_min is above the base lr, so the lr RISES: 1e-4 at epoch 0, 0.00626 at epoch 50, 0.02409 at epoch 99),
--runs 10 independent runs, and in every run the epoch with the best TEST mAUC is the one reported (the script has no validation split; early stopping is commented out in it).
It trains on the full official train list of the dataset and is scored on its official test list (CheXpert: the 5 competition findings), with the metrics of the script's utils.py.
Only the columns with both a positive and a negative example among the test images it is scored on count: the others are dropped and recorded (a smoke cache can make a column single-valued).
It writes, for each dataset,
  <results_root>/<tag>/<dataset>.json    per run the best test epoch (0-based, as the script prints it), its mAUC and per-class AUCs; mean and s.d. over the runs; the weights and the
                                         protocol it ran; flags; for a RayDINO backbone also pretraining_overlap_train (informational: not a flag)
                                         (written whole or not at all: a synced .part in probe/<dataset>/, hard-linked into place, or renamed where hard links are refused)
  <results_root>/<tag>/probe/<dataset>/  the script's own files: log.txt (a line per batch, large), linear_eval/<dataset>/<tag>/b2/log.txt (the average over the runs) and
                                         run_<i>/ckpt.pth (each run's best epoch)
Every number in the JSON comes from the script's own run (its returned records) and is checked against the checkpoints (best epoch) and the average line the script wrote to its log.

Before anything is fitted it refuses: results that already exist for the tag (the same two paths are checked by run_emb_gaudi.sh before it starts the extraction); a missing cache; caches under
the tag that were not extracted from the same backbone the same way (model, weights path, size and sha256, normalization, transform, max_images); a dataset none of whose columns can be scored;
and anything but the shipped protocol: the probe's settings (optimizer, lr, epochs, batch size, multiclass, ...) and the device (cpu only) cannot be changed from here.

Run from base/:   python -m ark_linearprobe.run_linearprobe --tag stock [--datasets MIMIC] [--runs 10] [--seed 0]
Only --runs, --seed and the paths (--tag, --datasets, --embeddings_root, --results_root) may vary. A smoke subset is chosen at extraction (extract_embeddings --max_images), which writes it to
the cache's meta.json: the fit uses the first N entries of each list and the JSON says so.
stdout is one "PROBE ..." line per dataset; the script prints a line per improved epoch and per run to stdout too.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

from ark_finetune.dataset_lists import DATASETS, REPO, dataset_entry
from ark_finetune.embedding_mode import SPLITS, feature_dim, read_meta, split_files
from . import main_linearprobing
from .embedding_datasets import evaluable_columns, test_labels

EXP_NAME = "b2"  # the script's --exp_name; with --arch <tag> it names the run directory linear_eval/<dataset>/<tag>/b2
SELECTION = "best epoch on test (as shipped)"
AS_SHIPPED = {"optimizer": "adam", "lr": 1e-4, "epochs": 100, "batch_size": 512, "multiclass": False, "mode": "train", "few_shot": -1, "device": "cpu"}  # the shipped script's settings, as its parser has them
RAD_DINO_OVERLAP = {"ChestXray14": "train+val+test"}  # RAD-DINO's pretraining data holds all of ChestXray14. CheXpert's overlap was its valid split only, which the probe does not use;
                                                       # MIMIC train is in it, but MIMIC test is clean: a flag on the JSON, not a correction
# RayDINO's pretraining corpus (Table 2: PadChest, CheXpert, MIMIC-CXR, NIH ChestX-ray14) holds the TRAIN splits of three of the four probe datasets and none of their test lists, VinDr-CXR not at all:
# no test-set overlap, so rad_dino_pretraining_overlap stays None for it. The train-split overlap is recorded apart, as the JSON's informational pretraining_overlap_train (outside flags, which drive the dagger)
RAYDINO_TRAIN_OVERLAP = "CheXpert, MIMIC-CXR, ChestX-ray14 training splits (RayDINO Table 2); VinDr-CXR not in its corpus"
# what two caches under one tag must agree on to be results of one backbone: (name in the message, path into meta.json)
PROVENANCE = (("model", ("model",)), ("weights path", ("weights", "path")), ("weights size", ("weights", "size_bytes")), ("weights sha256", ("weights", "sha256")),
              ("normalization", ("normalization",)), ("transform", ("transform",)), ("max_images", ("max_images",)))
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="The lab's embedding linear probe on cached embeddings", allow_abbrev=False)
    p.add_argument("--tag", required=True, help="the backbone tag the embeddings were extracted under")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--embeddings_root", default=os.path.join(REPO, "embeddings"))
    p.add_argument("--results_root", default=os.path.join(REPO, "results"))
    p.add_argument("--runs", type=int, default=10, help="the script's --runs (its default is 1): independent runs per dataset")
    p.add_argument("--seed", type=int, default=0, help="run i is seeded with seed + i (a deviation from the shipped script, for reproducibility)")
    p.add_argument("--device", default="cpu", type=main_linearprobing.cpu_only, help="cpu only: the fit runs on the CPU by design")
    args, extra = p.parse_known_args(argv)
    if len(set(args.datasets)) != len(args.datasets):  # as the extractor: a repeated name would fit the dataset twice, and the second fit would stop at the first one's results directory
        p.error("--datasets lists a dataset twice")
    if args.runs < 1:  # the script averages over its runs: none would crash it after the results directory exists
        p.error("--runs must be at least 1")
    if extra:
        p.error("{}: not accepted. The probe runs exactly as shipped (optimizer, lr, epochs, batch size, multiclass, workers, ...): only --runs, --seed and the paths may vary. "
                "A smoke subset is chosen at extraction (extract_embeddings --max_images) and read from the cache's meta.json".format(" ".join(extra)))
    return args


def results_of(results_root, tag, name):
    """What fitting a (tag, dataset) leaves, and what blocks fitting it again: its JSON and the directory of the script's own files (a half-done fit cannot be resumed: log.txt appends
    and ckpt.pth is overwritten). run_emb_gaudi.sh checks these two paths for the four datasets before it starts the extraction."""
    folder = os.path.join(results_root, tag)
    return [os.path.join(folder, name + ".json"), os.path.join(folder, "probe", name)]


def cache_files(embeddings_root, tag, name):
    folder = os.path.join(embeddings_root, tag, name)
    return [os.path.join(folder, "meta.json")] + [path for split in SPLITS for path in split_files(folder, split)]


def rad_dino_flag(meta, tag, dataset):
    """'train+val+test' (ChestXray14) when the backbone is RAD-DINO, else None. By the model name or, failing that, by the tag."""
    is_raddino = meta["model"] == "vit_base_raddino" or "raddino" in re.sub(r"[^a-z0-9]", "", tag.lower())
    return RAD_DINO_OVERLAP.get(dataset) if is_raddino else None


def raydino_train_overlap(meta):
    """RAYDINO_TRAIN_OVERLAP when the backbone is RayDINO, else None. By the model name alone: only vit_large_raydino builds RayDINO's ViT-L/16, so (unlike RAD-DINO, whose weights load into vit_base_dinov2 too) the tag adds nothing."""
    return RAYDINO_TRAIN_OVERLAP if meta["model"] == "vit_large_raydino" else None


def check_provenance(embeddings_root, tag):
    """Refuse unless every dataset cache under the tag that has a meta.json (whichever datasets are being fit: the wrapper fits one per process) was extracted from the same backbone the same way:
    the PROVENANCE fields of their meta.json are identical (the sha256 of the weights file included, which must be a digest). Returns {dataset: meta}."""
    metas = {name: read_meta(os.path.join(embeddings_root, tag, name)) for name in DATASETS if os.path.isfile(os.path.join(embeddings_root, tag, name, "meta.json"))}
    for field, keys in PROVENANCE:
        values = {}
        for name, meta in metas.items():
            value = meta
            for key in keys:
                if not isinstance(value, dict) or key not in value:
                    raise RuntimeError("{}'s meta.json under tag {} lacks {}".format(name, tag, ".".join(keys)))
                value = value[key]
            values[name] = value
        if len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1:
            raise RuntimeError("the caches under tag {} were not extracted from the same backbone the same way: {} differs: {}".format(
                tag, field, "; ".join("{} = {}".format(name, json.dumps(v, sort_keys=True)) for name, v in values.items())))
    for name, meta in metas.items():  # equal is not enough: null or "" in every cache would be equal
        if not (isinstance(meta["weights"]["sha256"], str) and SHA256_RE.fullmatch(meta["weights"]["sha256"])):
            raise RuntimeError("{}'s meta.json under tag {}: weights.sha256 {!r} is not a sha256 hex digest".format(name, tag, meta["weights"]["sha256"]))
    return metas


def scoring_plan(name, entry, meta):
    """({'columns': the module's test_columns (None: all), 'scored': class names, 'dropped': [{'column', 'reason', 'name'}]}) for dataset `name` on the test images its cache serves, by the rule
    the dataset module applies (embedding_datasets.evaluable_columns); collect() checks the script's own run against it. Raises ValueError if no column can be scored: before anything is fitted."""
    columns, dropped = evaluable_columns(name, test_labels(name, entry["data_dir"], entry["test_list"], meta.get("max_images")))
    diseases = entry["diseases"]
    return {"columns": columns, "scored": [diseases[c] for c in (range(len(diseases)) if columns is None else columns)], "dropped": [dict(d, name=diseases[d["column"]]) for d in dropped]}


def check_protocol(options):
    """Refuse unless the settings the script will run with are the shipped ones: the runner builds the options from fixed values and refuses any other, so this fails only if the vendored
    defaults were edited."""
    changed = ["{} = {!r} (shipped {!r})".format(key, getattr(options, key), value) for key, value in AS_SHIPPED.items() if getattr(options, key) != value]
    if changed:
        raise RuntimeError("the effective settings are not the shipped probe's: " + ", ".join(changed))


def protocol_text(options, meta):
    """The protocol this run used, from the options the script ran with (and, when the cache is a smoke subset, the max_images it records)."""
    text = ("the lab's embedding linear probe (Ark_Plus/Linearprobing/main_linearprobing.py) as shipped: float64 nn.Linear on the {}-d embedding, {}, {} lr {}, batch {}, {} epochs, "
            "CosineAnnealingLR(T_max=1000, eta_min=1) stepped per epoch (the lr rises), {} runs seeded {}..{}, device {}; full train list, official test list, best epoch on test".format(
                options.emb_dim, "CrossEntropyLoss" if options.multiclass else "sigmoid + BCELoss", options.optimizer, options.lr, options.batch_size, options.epochs, options.runs,
                options.seed, options.seed + options.runs - 1, options.device))
    if meta.get("max_images") is not None:
        text += "; SMOKE cache: the first {} entries of each list".format(meta["max_images"])
    return text


def collect(options, probe_dir, records, meta, plan, tag):
    """The dataset's JSON from the script's returned `records` (one per run: best epoch, [mAUC, mMCC, mAP, mF1], [AUC, MCC, AP, F1 per class] at that epoch), checked against the files it wrote."""
    exp_path = os.path.join(probe_dir, "linear_eval", options.dataset, options.arch, options.exp_name)
    if len(records) != options.runs:
        raise RuntimeError("{} runs returned, --runs {}".format(len(records), options.runs))
    for record in records:
        if record["best_epoch"] < 0 or record["scores"] is None:  # the test mAUC was never a number (a failed measurement is not a measurement)
            raise RuntimeError("run {}: no epoch had a defined test mAUC (best epoch {})".format(record["run"], record["best_epoch"]))
    scored = plan["scored"]
    per_run = []
    for record in records:
        aucs = record["scores"][0]
        if len(aucs) != len(scored) or abs(float(np.mean(aucs)) - record["performance"][0]) > 1e-12:
            raise RuntimeError("run {}: {} class AUCs for {} scored classes, or their mean is not the run's mAUC".format(record["run"], len(aucs), len(scored)))
        if record["columns"] != plan["columns"] or record["dropped_columns"] != [{"column": d["column"], "reason": d["reason"]} for d in plan["dropped"]]:  # the script scored what was planned
            raise RuntimeError("run {}: the script scored columns {} and dropped {}, the plan was {} and {}".format(record["run"], record["columns"], record["dropped_columns"], plan["columns"], plan["dropped"]))
        checkpoint = torch.load(os.path.join(exp_path, "run_{}".format(record["run"]), "ckpt.pth"), map_location="cpu", weights_only=False)  # saved at the best epoch; the script's own file, so no allow-list
        if checkpoint["epoch"] != record["best_epoch"]:
            raise RuntimeError("run {}: best epoch {}, its checkpoint is epoch {}".format(record["run"], record["best_epoch"], checkpoint["epoch"]))
        per_run.append({"run": record["run"], "seed": record["seed"], "best_epoch": record["best_epoch"], "test_mAUC": float(record["performance"][0]),
                        "test_AUC": dict(zip(scored, map(float, aucs)))})
    maucs = np.array([r["test_mAUC"] for r in per_run])
    class_aucs = np.array([[r["test_AUC"][name] for name in scored] for r in per_run])
    with open(os.path.join(exp_path, "log.txt")) as f:  # the script's own mean(s.d.) over the runs, for the four metrics
        average = re.findall(r"^Average performance from \d+ runs: \['(\d\.\d{4})\((\d\.\d{4})\)'", f.read(), re.M)
    if average != [("{:.4f}".format(maucs.mean()), "{:.4f}".format(maucs.std()))]:
        raise RuntimeError("mean (s.d.) mAUC {:.4f} ({:.4f}) recomputed, {} in the script's log".format(maucs.mean(), maucs.std(), average))
    warnings = ["{}: {} among the evaluated test images, so no AUC: not scored (left out of the means)".format(d["name"], d["reason"]) for d in plan["dropped"]]
    result = {
        "tag": tag, "dataset": options.dataset, "model": meta["model"], "weights": {"path": meta["weights"]["path"], "size_bytes": meta["weights"]["size_bytes"], "sha256": meta["weights"]["sha256"]},
        "protocol": protocol_text(options, meta),
        "embeddings_dir": options.emb_path, "normalization": meta["normalization"], "max_images": meta.get("max_images"),
        "n": {split: meta["counts"][split] for split in SPLITS}, "runs": options.runs, "seed": options.seed, "epochs": options.epochs,
        "classes": scored, "dropped_columns": plan["dropped"], "per_run": per_run, "best_epoch_per_run": [r["best_epoch"] for r in per_run],
        "test_mAUC_mean": float(maucs.mean()), "test_mAUC_std": float(maucs.std()),  # np.average / np.std (ddof 0), as the script prints them
        "test_AUC_mean": dict(zip(scored, class_aucs.mean(0).tolist())), "test_AUC_std": dict(zip(scored, class_aucs.std(0).tolist())),
        "flags": {"selection": SELECTION, "rad_dino_pretraining_overlap": rad_dino_flag(meta, tag, options.dataset)}, "warnings": warnings,
        "script_log": os.path.relpath(os.path.join(exp_path, "log.txt"), os.path.dirname(os.path.dirname(probe_dir))),  # relative to results/<tag>
    }
    train_overlap = raydino_train_overlap(meta)
    if train_overlap:  # informational, for the ledger: kept out of flags
        result["pretraining_overlap_train"] = train_overlap
    return result


def fit(name, args, meta, plan):
    cache = os.path.join(args.embeddings_root, args.tag, name)
    if (meta["tag"], meta["dataset"]) != (args.tag, name):
        raise RuntimeError("{} holds the embeddings of {} / {}, not {} / {}".format(cache, meta["tag"], meta["dataset"], args.tag, name))
    entry = dataset_entry(name)
    out = os.path.abspath(os.path.join(args.results_root, args.tag))
    probe_dir = os.path.join(out, "probe", name)
    argv = ["--dataset", name, "--num_class", str(len(entry["diseases"])), "--emb_dim", str(feature_dim(meta)), "--emb_path", os.path.abspath(cache),
            "--train_list", entry["train_list"], "--test_list", entry["test_list"], "--data_dir", entry["data_dir"],
            "--runs", str(args.runs), "--seed", str(args.seed), "--device", args.device, "--num_workers", "0",
            "--results_dir", probe_dir, "--arch", args.tag, "--exp_name", EXP_NAME]
    options = main_linearprobing.get_args_parser(argv)
    check_protocol(options)
    os.makedirs(probe_dir)
    began = time.time()
    records = main_linearprobing.main(options)
    result = collect(options, probe_dir, records, meta, plan, args.tag)
    result["elapsed_s"] = round(time.time() - began, 1)
    text = json.dumps(result, indent=2, allow_nan=False)  # serialized first: a value json cannot write (NaN) raises here, before the JSON or its .part exists
    final, part = os.path.join(out, name + ".json"), os.path.join(probe_dir, name + ".json.part")  # the .part is inside probe/<dataset>/: run_emb_fit_class.sh's resume rule removes that directory with a half-done fit
    with open(part, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())  # the text is on disk before the JSON has a name; no directory fsync: a link lost in a crash means no JSON, so the dataset is fitted again, the safe direction
    try:
        os.link(part, final)  # atomic, and FileExistsError if final exists (what open(final, "x") gave): the JSON is there whole or not at all, which is why its existence means a finished fit to run_emb_fit_class.sh's resume rule
    except FileExistsError:
        raise
    except OSError:  # a filesystem that refuses a hard link across directories: rename instead. Not race-free (the check, then the rename); what keeps two fits of one dataset apart there is the makedirs(probe_dir) above
        if os.path.exists(final):
            raise FileExistsError("{} exists already: refusing to overwrite it".format(final))
        os.replace(part, final)
    else:
        os.remove(part)  # the JSON is the same file under its own name now
    for warning in result["warnings"]:
        print("run_linearprobe: WARNING {} {}: {}".format(args.tag, name, warning), file=sys.stderr)
    overlap = result["flags"]["rad_dino_pretraining_overlap"]
    print("PROBE {} {}: test mAUC {:.4f} ({:.4f}) over {} runs, {}, best epochs {}{}".format(
        args.tag, name, result["test_mAUC_mean"], result["test_mAUC_std"], result["runs"], SELECTION, result["best_epoch_per_run"],
        ", rad_dino_pretraining_overlap " + overlap if overlap else ""))
    return result


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)  # the loop is bound by Python overhead on small tensors
    taken = [path for name in args.datasets for path in results_of(args.results_root, args.tag, name) if os.path.exists(path)]
    if taken:  # first, before the cache is looked at: results that exist are the reason to stop, whatever state the cache is in
        raise FileExistsError("results already exist for tag {}: {}; refusing to overwrite (a half-done fit cannot be resumed). Remove them to fit again, or use another tag".format(
            args.tag, ", ".join(taken)))
    missing = [path for name in args.datasets for path in cache_files(args.embeddings_root, args.tag, name) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("no embeddings for tag {}: {} missing (run extract_embeddings first)".format(args.tag, ", ".join(missing)))
    metas = check_provenance(args.embeddings_root, args.tag)
    plans = {name: scoring_plan(name, dataset_entry(name), metas[name]) for name in args.datasets}  # before anything is fitted: a dataset with nothing to score stops the run
    for name in args.datasets:
        fit(name, args, metas[name], plans[name])


if __name__ == "__main__":
    main()
