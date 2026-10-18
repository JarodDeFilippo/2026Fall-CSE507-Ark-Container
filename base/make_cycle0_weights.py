"""Rebuild the cycle-0 weights of a finished Ark+ run as models/weights/epoch_0000/, in the layout engine.py writes for every cycle (CPU only).

Runs trained before --eval_cycle0 never saved their initial weights. They are a function of the run's seed, its model and its backbone file, so they are rebuilt by replaying
what the trainer does between its start and its first cycle: main_ark.main seeds random, numpy and torch with --seed, then omni_engine builds the student and, right after it,
the teacher with build_omni_model. Both get the backbone file and each its OWN random projector and heads, because the second build starts from the RNG state the first one
left (student and teacher agree on the backbone only). Nothing draws from the RNG in between (the datasets use their own random.Random objects, a DataLoader draws only when it
is iterated), so the replay is
    random.seed(s); np.random.seed(s); torch.manual_seed(s); student = build_omni_model(...); teacher = build_omni_model(...)
and both are saved by engine._save_cycle_weights, the helper the trainer saves a cycle with, plus reconstructed_from_seed.json (seed, model, backbone path and sha256, timestamp)
so that a ledger can say these weights were reconstructed, not saved during training.
This replays the init procedure; it is bit-identical to what the trainer held only if its RNG consumption really was the same. --verify-against EPOCH_DIR checks that against
a natively saved epoch_0000 (a run trained with --eval_cycle0 true has one): it rebuilds in memory, writes nothing, prints max |diff| per tensor for student and teacher and
exits 1 unless every tensor is identical.

The recipe (model, projector features, heads, data sets, run name, seed) is read from the run's lowest-numbered models/weights/epoch_*/manifest.json: JSON that engine.py writes
and validates with the loader these weights go through again, unlike train.log (free text) or the checkpoints (the same metadata plus optimizer state). None of them records the
backbone file (train.log says only "Pretrained Weights: True"), so it comes from --pretrained-weights or, for a sweep run, from its row of sweep/wave1.tsv (STAGEA_FINAL rows
are refused: pass the file the run started from). Refuses to overwrite an existing epoch_0000; the new one is built next to it and renamed into place, so a killed run leaves none.

usage (inside the image, where nvidia.sh starts in /workspace/base):
    python make_cycle0_weights.py /workspace/<run dir> [--pretrained-weights FILE] [--verify-against EPOCH_DIR]
Exit 0: written (or verified identical). 1: refused, or --verify-against found a difference. 2: usage error.
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

import engine
from models import build_omni_model

SWEEP_PREFIX = "all4_cyclic_dinov3_sweep_"  # experiment_scripts/all4_cyclic_dinov3_sweep: the run name is this + the sweep id, the run dir adds _seed_<seed>


class Refused(Exception):
    pass


def find_recipe(run_dir, avoid=None):
    """(epoch dir, manifest) of the lowest-numbered saved epoch: the run's model, heads, data sets, run name and seed, as engine.py wrote and validates them.
    `avoid`: the epoch dir being verified, passed over when the run has another one, so that its manifest is compared with the recipe of a different epoch."""
    epochs = sorted((engine._weight_cycle(path), path) for path in glob.glob(os.path.join(run_dir, "models", "weights", "epoch_*"))
                    if engine._weight_cycle(path) is not None and os.path.isfile(os.path.join(path, "manifest.json")))
    others = [(cycle, path) for cycle, path in epochs if avoid is None or os.path.realpath(path) != os.path.realpath(avoid)]
    epochs = others or epochs
    if not epochs:
        raise Refused("{} has no models/weights/epoch_*/manifest.json to read the run's recipe from".format(run_dir))
    try:
        return epochs[0][1], engine._load_weight_manifest(epochs[0][1])
    except (ValueError, FileNotFoundError) as error:
        raise Refused(str(error))


def table_weights(run_dir, run_name):
    """(the path as the trainer got it, the file to open, where that came from) of a sweep run, from its row of sweep/wave1.tsv. The repo root is the run dir's parent: /workspace in the image."""
    if not run_name.startswith(SWEEP_PREFIX):
        raise Refused("{} is not a sweep run, and a run dir does not record its backbone file: pass --pretrained-weights".format(run_name))
    root, sweep_id = os.path.dirname(os.path.abspath(run_dir)), run_name[len(SWEEP_PREFIX):]
    table = os.path.join(root, "sweep", "wave1.tsv")
    try:
        with open(table, newline="") as handle:
            rows = [row for row in csv.DictReader(handle, delimiter="\t") if row.get("id") == sweep_id]
    except OSError as error:
        raise Refused("cannot read {} for the backbone file of {} ({}): pass --pretrained-weights".format(table, run_name, error))
    if len(rows) != 1:
        raise Refused("{} has {} rows for sweep id {}, not exactly one: pass --pretrained-weights".format(table, len(rows), sweep_id))
    weights = rows[0].get("weights") or ""
    if weights == "STAGEA_FINAL":
        raise Refused("sweep {} started from STAGEA_FINAL, which chain/sweep_run.sh resolved when it launched (chain/main_epochs.txt may say something else now): "
                      "pass the file it started from with --pretrained-weights".format(sweep_id))
    if not weights.startswith("/workspace/"):
        raise Refused("{}: weights '{}' of sweep {} is not an in-container path under /workspace/: pass --pretrained-weights".format(table, weights, sweep_id))
    return weights, os.path.join(root, weights[len("/workspace/"):]), "the row of sweep {} in {}".format(sweep_id, table)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_init(manifest, weights_file):
    """The trainer's init, replayed: main_ark.main's seeding, then omni_engine's student build and teacher build, in that order. Returns (student, teacher)."""
    args = SimpleNamespace(model_name=manifest["model_name"], projector_features=manifest["projector_features"], use_mlp=manifest["use_mlp"],
                           pretrained_weights=weights_file)
    random.seed(manifest["seed"])
    np.random.seed(manifest["seed"])
    torch.manual_seed(manifest["seed"])
    student = build_omni_model(args, manifest["num_classes_list"])
    teacher = build_omni_model(args, manifest["num_classes_list"])
    return student, teacher


def cycle0_metadata(manifest):
    """(args, manifest.json content) of cycle 0, from the same helper the trainer uses; what the run's own manifest does not say takes engine._run_metadata's default."""
    sizes = {"input_size": "crop_size", "img_resize": "resize"}  # manifest key -> the trainer's args attribute
    args = SimpleNamespace(exp_name=manifest["run_name"], seed=manifest["seed"], model_name=manifest["model_name"], projector_features=manifest["projector_features"],
                           use_mlp=manifest["use_mlp"], **{key: manifest[key] for key in ("training_strategy", "joint_sampling", "freeze_backbone") if key in manifest},
                           **{attribute: manifest[key] for key, attribute in sizes.items() if key in manifest})
    metadata = engine._run_metadata(args, manifest["dataset_list"], manifest["num_classes_list"], 0)
    for key in sizes:  # a run from before the input size was recorded: its epoch_0000 matches its other manifests (no key = 224 / 256 to the engine)
        if key not in manifest:
            del metadata[key]
    return args, metadata


def write_epoch0(weights_root, args, metadata, student, teacher, record):
    """models/weights/epoch_0000 from a scratch dir next to it (same filesystem), renamed into place at the end: all or nothing."""
    final = os.path.join(weights_root, "epoch_0000")
    scratch = tempfile.mkdtemp(prefix=".cycle0-", dir=weights_root)
    try:
        engine._save_cycle_weights(scratch, args, metadata, student, teacher)
        with open(os.path.join(scratch, "epoch_0000", "reconstructed_from_seed.json"), "w") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
        try:
            os.rename(os.path.join(scratch, "epoch_0000"), final)
        except OSError as error:
            raise Refused("{} appeared while the weights were being built ({}): not overwriting it, nothing written".format(final, error))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return final


def verify(native_dir, metadata, student, teacher):
    """Compare the replay with a natively saved epoch dir: max |diff| per tensor of student and teacher, then the manifests. True when everything is identical."""
    identical = True
    for role, network in (("student", student), ("teacher", teacher)):
        try:
            native = torch.load(engine._find_saved_weight(native_dir, role), map_location="cpu", weights_only=False)
        except (FileNotFoundError, OSError) as error:
            raise Refused("cannot read the {} weights of {}: {}".format(role, native_dir, error))
        replay, differing, largest = network.state_dict(), 0, 0.0
        for key in sorted(set(native) | set(replay)):
            if key not in native or key not in replay:
                print("{:7s} {}: only in the {}".format(role, key, "replay" if key in replay else "native file"))
                differing += 1
                continue
            a, b = native[key], replay[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                print("{:7s} {}: native {} {} vs replay {} {}".format(role, key, tuple(a.shape), a.dtype, tuple(b.shape), b.dtype))
                differing += 1
                continue
            diff = 0.0 if torch.equal(a, b) else float((a.double() - b.double()).abs().max())
            print("{:7s} max|diff| {:<14.8g} {} {}".format(role, diff, key, list(a.shape)))
            if diff != 0.0:
                differing += 1
            largest = max(largest, diff)
        print("{}: {} tensors compared, {} differ, largest max|diff| {:.8g}".format(role, len(set(native) | set(replay)), differing, largest))
        identical = identical and differing == 0
    try:
        with open(os.path.join(native_dir, "manifest.json")) as handle:
            native_manifest = json.load(handle)
    except (OSError, ValueError) as error:
        raise Refused("cannot read the manifest of {}: {}".format(native_dir, error))
    changed = sorted(key for key in set(native_manifest) | set(metadata) if native_manifest.get(key) != metadata.get(key))
    print("manifest.json: " + ("identical" if not changed else "differs in " + ", ".join(changed)))
    return identical and not changed


def run(options):
    run_dir = os.path.abspath(options.run_dir)
    if not os.path.isdir(os.path.join(run_dir, "models", "weights")):
        raise Refused("{} has no models/weights directory: not a run dir".format(run_dir))
    weights_root, final = os.path.join(run_dir, "models", "weights"), os.path.join(run_dir, "models", "weights", "epoch_0000")
    if options.verify_against is None and os.path.lexists(final):
        raise Refused("{} exists: not overwriting it (move it away first if it really has to be rebuilt)".format(final))
    recipe_dir, manifest = find_recipe(run_dir, options.verify_against)
    if options.pretrained_weights:
        recorded, weights_file, source = options.pretrained_weights, options.pretrained_weights, "--pretrained-weights"
    else:
        recorded, weights_file, source = table_weights(run_dir, manifest["run_name"])
    if not os.path.isfile(weights_file):
        raise Refused("backbone file {} is missing (from {})".format(weights_file, source))
    print("run {}: recipe from {} ({}, projector {}, data sets {}, seed {})".format(
        manifest["run_name"], os.path.relpath(recipe_dir, run_dir), manifest["model_name"], manifest["projector_features"], " ".join(manifest["dataset_list"]), manifest["seed"]))
    digest = sha256_of(weights_file)
    print("backbone file {} (from {}), sha256 {}".format(weights_file, source, digest))
    student, teacher = replay_init(manifest, weights_file)
    args, metadata = cycle0_metadata(manifest)
    if options.verify_against is not None:
        same = verify(options.verify_against, metadata, student, teacher)
        print("IDENTICAL: the replay reproduces the native epoch_0000 bit for bit" if same else "DIFFERENT: the replay does not reproduce the native epoch_0000")
        return 0 if same else 1
    record = {"reconstructed": True, "run": manifest["run_name"], "seed": manifest["seed"], "model": manifest["model_name"], "weights": recorded,
              "weights_sha256": digest, "weights_source": source, "recipe_from": os.path.relpath(recipe_dir, run_dir), "torch": torch.__version__,
              "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "procedure": "random.seed, np.random.seed, torch.manual_seed(seed); student = build_omni_model; teacher = build_omni_model (main_ark.main, then omni_engine)"}
    written = write_epoch0(weights_root, args, metadata, student, teacher, record)
    print("wrote {} (student, teacher, manifest.json, reconstructed_from_seed.json); the projector and heads of student and teacher differ, as the trainer's two builds make them".format(written))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rebuild models/weights/epoch_0000 of a finished run from its seed (see the module docstring).")
    parser.add_argument("run_dir", help="the run directory, e.g. /workspace/all4_cyclic_dinov3_sweep_S2_seed_100")
    parser.add_argument("--pretrained-weights", metavar="FILE", help="the backbone file the run started from (its --pretrained_weights); default: the run's row of sweep/wave1.tsv")
    parser.add_argument("--verify-against", metavar="EPOCH_DIR", help="compare the replay with this natively saved epoch_0000 and write nothing")
    try:
        return run(parser.parse_args(argv))
    except Refused as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
