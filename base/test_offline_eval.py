"""CPU checks for the offline evaluation of saved cycles: eval_d1_6.sh's protocol switch, engine.py's result-file marker and base/make_cycle0_weights.py.
Run from base/:  python test_offline_eval.py
(a) eval_d1_6.sh in a fake repo with a stub nvidia.sh: ten-crop by default with exactly the old argv; ARK_TEST_AUGMENT=false adds --test_output_suffix .centercrop; other spellings are refused
(b) engine._test_output_path: the three weight layouts with and without the suffix; the suffix is validated
(c) engine.omni_engine in test mode (tiny ViT, CPU): the two protocols write two files side by side, same columns, and each still refuses to overwrite itself
(d) make_cycle0_weights.py: layout and loader acceptance, the trainer's student-then-teacher replay, bit-identical reruns, refusals, --verify-against (incl. failing on a changed RNG order)
(e) the input size in test mode: main_ark.main builds the test transforms at the saved cycle's recorded input_size / img_resize (its manifest, or a cycle checkpoint's
    own metadata; 224 / 256 without them), refuses a passed --input_size / --img_resize that differs, and eval_d1_6.sh needs no flag for it
"""
import contextlib
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

import engine
import main_ark
import make_cycle0_weights
import models
from models import ArkViT, DINOV3_VITB16_KWARGS, build_omni_model
from utils import get_config

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HEAD_PREFIXES = ("projector.", "omni_heads.")
# same ViT as the container's with 2 blocks of width 64 on 32x32 inputs (test_freeze_backbone.py's tiny model)
TINY = dict(DINOV3_VITB16_KWARGS, img_size=32, embed_dim=64, depth=2, num_heads=2, encoder_features=128)
config = get_config(os.path.join(HERE, "datasets_config.yaml"))
NAMES = ["VinDrCXR", "ChestXray14"]
CLASSES = [len(config[name]["diseases"]) for name in NAMES]  # 6 and 14
OLD_ARGV = ["python", "main_ark.py", "--mode", "test", "--test_augment", "true", "--pretrained_weights", "/workspace/{}/models/weights/epoch_{}",
            "--data_set", "VinDrCXR", "--data_set", "ChestXray14", "--data_set", "MIMIC", "--data_set", "CheXpert",
            "--model", "swin_base", "--projector_features", "1376", "--device", "cuda"]  # eval_d1_6.sh's argv before ARK_TEST_AUGMENT existed


def rejects(exception, call, expected=None):
    """The exception message when call() raises `exception`; fails when it returns."""
    try:
        call()
    except exception as error:
        assert expected is None or expected in str(error), (expected, str(error))
        return str(error)
    raise AssertionError("expected {}".format(exception.__name__))


def same_bits(a, b):
    return (a.dtype, a.shape) == (b.dtype, b.shape) and torch.equal(a.detach().reshape(-1).contiguous().view(torch.uint8), b.detach().reshape(-1).contiguous().view(torch.uint8))


def same_state(a, b):
    return list(a) == list(b) and all(same_bits(a[key], b[key]) for key in a)


# (a) eval_d1_6.sh: the protocol switch ------------------------------------------------------------------------------------------
EVAL = os.path.join(REPO, "eval_d1_6.sh")
subprocess.run(["bash", "-n", EVAL], check=True)
assert os.access(EVAL, os.X_OK)
eval_text = open(EVAL).read()
CD = "cd /scratch/$USER/2026Fall-CSE507-Ark-Container"
assert eval_text.count(CD) == 1
assert re.findall(r"^#SBATCH (.+)$", eval_text, re.M) == ["--job-name=d1-6-eval", "-A grp_jliang12", "-p public", "-q public", "-t 01:30:00", "-G a100:1",
                                                           "--constraint=a100_80", "-c 16", "--mem=64G"]  # the Slurm header: -t cut to 01:30:00 on 2026-10-17 (measured 26 min per eval; MaxGRESRunMinsPerUser counts the requested limit), the rest untouched


def run_eval(args, **env):
    """eval_d1_6.sh from a fake repo (its cd to the cluster checkout points at the fake one) with a stub nvidia.sh that prints its argv, one per line.
    Returns (process, the argv lines or None when the stub never ran)."""
    with tempfile.TemporaryDirectory() as root:
        with open(os.path.join(root, "eval_d1_6.sh"), "w") as handle:
            handle.write(eval_text.replace(CD, "cd " + root))
        with open(os.path.join(root, "nvidia.sh"), "w") as handle:
            handle.write('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
        os.chmod(os.path.join(root, "nvidia.sh"), 0o755)
        environment = {key: value for key, value in os.environ.items() if not key.startswith("ARK_")}
        environment.update(env)
        done = subprocess.run(["bash", os.path.join(root, "eval_d1_6.sh")] + list(args), cwd=root, env=environment, capture_output=True, text=True)
    return done, (done.stdout.splitlines() if done.returncode == 0 and done.stdout else None)


RUN_DIR = "all4_cyclic_dinov3_sweep_S2_seed_100"
done, argv = run_eval([RUN_DIR, "12"])
assert done.returncode == 0 and done.stderr == "", done
assert argv == [part.format(RUN_DIR, "0012") for part in OLD_ARGV], argv  # ten-crop is the default and the argv is exactly the old one (bash 3.2 included)
done, argv = run_eval([RUN_DIR, "12"], ARK_TEST_AUGMENT="true")
assert argv == [part.format(RUN_DIR, "0012") for part in OLD_ARGV], argv
done, argv = run_eval([RUN_DIR, "0"], ARK_TEST_AUGMENT="false", ARK_MODEL="vit_base_dinov3")
center = [part.format(RUN_DIR, "0000") for part in OLD_ARGV]
center[center.index("--test_augment") + 1] = "false"
center[center.index("--test_augment") + 2:center.index("--test_augment") + 2] = ["--test_output_suffix", ".centercrop"]
center[center.index("--model") + 1] = "vit_base_dinov3"
assert done.returncode == 0 and argv == center, argv  # cycle 0 is zero-padded to epoch_0000, the center-crop protocol carries its marker
done, argv = run_eval([RUN_DIR, "012", "VinDrCXR", "ChestXray14"], ARK_TEST_AUGMENT="false")
assert argv.count("--data_set") == 2 and argv[argv.index("--pretrained_weights") + 1] == "/workspace/{}/models/weights/epoch_0012".format(RUN_DIR), argv  # 012 is decimal
assert argv[argv.index("--test_augment") + 1:argv.index("--test_augment") + 4] == ["false", "--test_output_suffix", ".centercrop"], argv
for spelling in ("False", "0", "no", "f", "n", "banana", "TRUE", "t", "1", " true", "true "):  # main_ark.py reads these as booleans or as false: the marker would be wrong
    done, argv = run_eval([RUN_DIR, "5"], ARK_TEST_AUGMENT=spelling)
    assert done.returncode == 2 and argv is None and done.stdout == "" and "ARK_TEST_AUGMENT must be true" in done.stderr, (spelling, done)
done, argv = run_eval([RUN_DIR, "5"], ARK_TEST_AUGMENT="")
assert done.returncode == 0 and argv == [part.format(RUN_DIR, "0005") for part in OLD_ARGV], argv  # set but empty = unset = ten-crop, like ${VAR:-true} everywhere
done, argv = run_eval([RUN_DIR])
assert done.returncode == 2 and argv is None and "Usage:" in done.stderr and "ARK_TEST_AUGMENT" in done.stderr, done
for env, test_augment, suffix in (({}, True, ""), ({"ARK_TEST_AUGMENT": "false"}, False, ".centercrop")):  # main_ark.py's own parser takes the argv
    sys.argv = ["main_ark.py"] + run_eval([RUN_DIR, "3"], **env)[1][2:]
    options = main_ark.get_args_parser()
    assert (options.mode, options.test_augment, options.test_output_suffix, options.dataset_list) == ("test", test_augment, suffix, NAMES + ["MIMIC", "CheXpert"]), vars(options)
print("OK (a) eval_d1_6.sh: Slurm header as expected (-t 01:30:00); no ARK_TEST_AUGMENT (or true, or empty) = the old argv exactly; false = --test_augment false --test_output_suffix .centercrop; "
      "cycles zero-padded as decimal; every other spelling (False, 0, no, banana, ...) -> exit 2 before anything runs; main_ark.py parses both argvs")

# (b) engine._test_output_path --------------------------------------------------------------------------------------------------
manifest_args = SimpleNamespace(exp_name="tiny", seed=1, model_name="vit_base_dinov3", projector_features=32, use_mlp=False)
with tempfile.TemporaryDirectory() as tmp:
    weights = os.path.join(tmp, "run", "models", "weights", "epoch_0007")
    os.makedirs(weights)
    with open(os.path.join(weights, "manifest.json"), "w") as handle:
        json.dump(engine._run_metadata(manifest_args, NAMES, CLASSES, 7), handle)
    for name in ("student_cycle_0007_tiny_seed_1.pth", "teacher_cycle_0007_tiny_seed_1.pth"):
        torch.save({}, os.path.join(weights, name))
    out = os.path.join(tmp, "out")
    for suffix in ("", ".centercrop", "_v2", "-a.b"):
        expected = os.path.join(out, "cycle_0007_tiny_seed_1{}.csv".format(suffix))
        assert engine._test_output_path(out, weights, suffix) == expected  # a weights directory
        assert engine._test_output_path(out, os.path.join(weights, "student_cycle_0007_tiny_seed_1.pth"), suffix) == expected  # a saved weight file
        assert engine._test_output_path(out, os.path.join(tmp, "cycle_0010_tiny_seed_1.pth.tar"), suffix) == os.path.join(out, "cycle_0010_tiny_seed_1{}.csv".format(suffix))
        assert engine._test_output_path(out, os.path.join(tmp, "legacy.pth"), suffix) == os.path.join(out, "legacy{}.csv".format(suffix))
    assert engine._test_output_path(out, weights) == engine._test_output_path(out, weights, "") == os.path.join(out, "cycle_0007_tiny_seed_1.csv")  # today's name, no suffix given
    for bad in ("/x", "../x", "a b", "é", ".centercrop/", "a\nb"):
        rejects(ValueError, lambda: engine._test_output_path(out, weights, bad), "--test_output_suffix may contain only letters, numbers, '.', '_' and '-'")
print("OK (b) engine._test_output_path: weights directory, saved weight file, .pth.tar and plain .pth names all take the suffix before .csv; no suffix = today's names; "
      "a suffix with a slash, space, newline or non-ASCII is refused")

# (c) the engine in test mode: two protocols, two files --------------------------------------------------------------------------
class FixedData(torch.utils.data.Dataset):
    """Stands in for a CXR test set: the same images on every pass; `crops` 10 gives ten-crop's [10, 3, H, W] samples, 1 gives center crop's [3, H, W]."""

    def __init__(self, classes, count, seed, crops):
        self.labels = (np.random.RandomState(seed).rand(count, classes) > 0.5).astype(np.float32)
        self.seed, self.crops = seed, crops

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.randn(3, 32, 32, generator=torch.Generator().manual_seed(1000 * self.seed + index))
        image = image.unsqueeze(0).repeat(10, 1, 1, 1) if self.crops == 10 else image
        return image, image.clone(), torch.from_numpy(self.labels[index])


def test_mode(weights_dir, out_dir, crops, suffix=None):
    """main_ark.py --mode test on weights_dir, as eval_d1_6.sh calls it (on the CPU, on FixedData). Returns the parsed options."""
    sys.argv = ["main_ark.py", "--mode", "test", "--device", "cpu", "--model", "vit_base_dinov3", "--projector_features", "32", "--batch_size", "8", "--workers", "0",
                "--test_augment", "true" if crops == 10 else "false", "--pretrained_weights", weights_dir, "--data_set", NAMES[0], "--data_set", NAMES[1]]
    if suffix is not None:
        sys.argv += ["--test_output_suffix", suffix]
    options = main_ark.get_args_parser()
    split = [FixedData(CLASSES[k], 8, 30 + k, crops) for k in range(2)]
    with contextlib.redirect_stdout(io.StringIO()):
        engine.omni_engine(options, out_dir, out_dir, NAMES, config, [], [], split)
    return options


def make_run(root, name="all4_cyclic_dinov3_sweep_S2", seed=100, cycles=(1, 2)):
    """A run dir with only the manifests of `cycles` under models/weights (all that make_cycle0_weights.py reads) and the args the trainer would have had."""
    run_dir = os.path.join(root, "{}_seed_{}".format(name, seed))
    args = SimpleNamespace(exp_name=name, seed=seed, model_name="vit_base_dinov3", projector_features=32, use_mlp=False, training_strategy="cyclic",
                           joint_sampling="proportional", freeze_backbone=False)
    for cycle in cycles:
        directory = os.path.join(run_dir, "models", "weights", "epoch_%04d" % cycle)
        os.makedirs(directory)
        with open(os.path.join(directory, "manifest.json"), "w") as handle:
            json.dump(engine._run_metadata(args, NAMES, CLASSES, cycle), handle, indent=2, sort_keys=True)
    return run_dir, args


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.reader(handle))


models.DINOV3_VITB16_KWARGS = TINY
try:
    with tempfile.TemporaryDirectory() as tmp:
        stock = os.path.join(tmp, "weights", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")  # the DINOv3 file's shape: the backbone only, no projector or heads
        os.makedirs(os.path.dirname(stock))
        torch.manual_seed(1234)
        torch.save({k: v for k, v in ArkViT(CLASSES, 32, False, **TINY).state_dict().items() if not k.startswith(HEAD_PREFIXES)}, stock)
        run_dir, run_args = make_run(tmp)
        assert make_cycle0_weights.main([run_dir, "--pretrained-weights", stock]) == 0
        epoch0 = os.path.join(run_dir, "models", "weights", "epoch_0000")
        out = os.path.join(tmp, "results")
        os.makedirs(out)
        stem = "cycle_0000_all4_cyclic_dinov3_sweep_S2_seed_100"
        ten = test_mode(epoch0, out, 10)  # the ten-crop protocol: no marker, today's file name
        assert ten.test_augment is True and ten.test_output_suffix == "" and os.listdir(out) == [stem + ".csv"], os.listdir(out)
        centre = test_mode(epoch0, out, 1, ".centercrop")  # the center-crop protocol next to it
        assert centre.test_augment is False and centre.test_output_suffix == ".centercrop" and sorted(os.listdir(out)) == [stem + ".centercrop.csv", stem + ".csv"], os.listdir(out)
        ten_rows, centre_rows = read_rows(os.path.join(out, stem + ".csv")), read_rows(os.path.join(out, stem + ".centercrop.csv"))
        assert ten_rows[0] == centre_rows[0] == ["dataset", "model", "metric", "class", "value"]  # content and columns are the same in both
        assert [row[:4] for row in ten_rows] == [row[:4] for row in centre_rows] and {row[1] for row in ten_rows[1:]} == {"student", "teacher"}
        assert {(row[0], row[2]) for row in ten_rows[1:]} >= {(name, "mAUC") for name in NAMES}
        before = {name: open(os.path.join(out, name), "rb").read() for name in os.listdir(out)}
        rejects(FileExistsError, lambda: test_mode(epoch0, out, 1, ".centercrop"), "Test results file already exists: " + os.path.join(out, stem + ".centercrop.csv"))
        rejects(FileExistsError, lambda: test_mode(epoch0, out, 10), "Test results file already exists: " + os.path.join(out, stem + ".csv"))
        assert {name: open(os.path.join(out, name), "rb").read() for name in os.listdir(out)} == before  # a refused run touches nothing
        rejects(ValueError, lambda: test_mode(epoch0, out, 1, "../escape"), "--test_output_suffix may contain only")
        assert sorted(os.listdir(out)) == sorted(before) and not os.path.exists(os.path.join(tmp, "escape.csv"))
    print("OK (c) engine test mode on a tiny ViT: ten-crop writes cycle_<NNNN>_<run>.csv, center crop writes cycle_<NNNN>_<run>.centercrop.csv next to it, the same "
          "dataset/model/metric/class/value columns, each protocol still refuses to overwrite its own file, a bad suffix writes nothing; the weights are the reconstructed cycle 0")

    # (d) make_cycle0_weights.py -----------------------------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("a", "b", "c"):
            os.makedirs(os.path.join(tmp, sub, "weights"))
        torch.manual_seed(1234)
        backbone = {k: v for k, v in ArkViT(CLASSES, 32, False, **TINY).state_dict().items() if not k.startswith(HEAD_PREFIXES)}
        for sub in ("a", "b", "c"):
            torch.save(backbone, os.path.join(tmp, sub, "weights", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"))
            os.makedirs(os.path.join(tmp, sub, "ssl_dinov3_cxr_constlr_seed_100", "export"))
            torch.save(backbone, os.path.join(tmp, sub, "ssl_dinov3_cxr_constlr_seed_100", "export", "teacher_backbone_epoch_0030.pth"))
            os.makedirs(os.path.join(tmp, sub, "sweep"))
            with open(os.path.join(REPO, "sweep", "wave1.tsv")) as source, open(os.path.join(tmp, sub, "sweep", "wave1.tsv"), "w") as target:
                target.write(source.read())  # the real table: S2 stock weights, S13 STAGEA_FINAL, S16 the constant-LR export
        stock_a = os.path.join(tmp, "a", "weights", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
        digest = hashlib.sha256(open(stock_a, "rb").read()).hexdigest()
        run_a, args_a = make_run(os.path.join(tmp, "a"))
        run_b, _ = make_run(os.path.join(tmp, "b"))
        weights_a = os.path.join(run_a, "models", "weights")
        before = sorted(os.listdir(weights_a))
        assert before == ["epoch_0001", "epoch_0002"]

        # the run's row of the table gives the backbone file (the repo root is the run dir's parent); the stdout says what was read
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            assert make_cycle0_weights.main([run_a]) == 0
        epoch0 = os.path.join(weights_a, "epoch_0000")
        assert sorted(os.listdir(weights_a)) == ["epoch_0000", "epoch_0001", "epoch_0002"], os.listdir(weights_a)  # no scratch dir left behind
        stem = "cycle_0000_all4_cyclic_dinov3_sweep_S2_seed_100"
        assert sorted(os.listdir(epoch0)) == ["manifest.json", "reconstructed_from_seed.json", "student_" + stem + ".pth", "teacher_" + stem + ".pth"], os.listdir(epoch0)
        assert "recipe from models/weights/epoch_0001" in printed.getvalue() and digest in printed.getvalue() and "row of sweep S2" in printed.getvalue(), printed.getvalue()
        # the layout is the trainer's own: the manifest is what engine._run_metadata writes for cycle 0, and engine's loaders take the directory
        metadata = json.load(open(os.path.join(epoch0, "manifest.json")))
        assert metadata == engine._run_metadata(args_a, NAMES, CLASSES, 0) and metadata["cycle"] == 0, metadata
        assert engine._load_weight_manifest(epoch0) == metadata
        student_state, teacher_state, loaded_manifest = engine._load_test_state_dicts(epoch0)
        assert loaded_manifest == metadata
        record = json.load(open(os.path.join(epoch0, "reconstructed_from_seed.json")))
        assert (record["seed"], record["model"], record["weights"], record["weights_sha256"], record["reconstructed"]) == (
            100, "vit_base_dinov3", "/workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", digest, True), record
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", record["timestamp"]) and record["recipe_from"] == "models/weights/epoch_0001", record

        # what the trainer would hold: seeded as main_ark.main does, the student built first and the teacher right after it, from the same backbone file
        replay = SimpleNamespace(model_name="vit_base_dinov3", projector_features=32, use_mlp=False, pretrained_weights=stock_a)
        random.seed(100)
        np.random.seed(100)
        torch.manual_seed(100)
        expected_student, expected_teacher = build_omni_model(replay, CLASSES).state_dict(), build_omni_model(replay, CLASSES).state_dict()
        assert same_state(student_state, expected_student) and same_state(teacher_state, expected_teacher)
        differing = [key for key in student_state if not same_bits(student_state[key], teacher_state[key])]
        assert differing and all(key.startswith(HEAD_PREFIXES) for key in differing), differing  # student and teacher share the backbone only: the heads differ
        assert len(differing) == 2 + 2 * len(NAMES) and sorted(differing) == sorted(k for k in student_state if k.startswith(HEAD_PREFIXES)), differing
        assert not same_state(student_state, expected_teacher)  # the order matters: a teacher-first replay would swap the two

        # bit-identical on a rerun from the same inputs, whatever the caller's RNG state or thread count was
        random.seed(5)
        np.random.seed(5)
        torch.manual_seed(5)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            assert make_cycle0_weights.main([run_b]) == 0
        finally:
            torch.set_num_threads(threads)
        epoch0_b = os.path.join(run_b, "models", "weights", "epoch_0000")
        for name in ("student_" + stem + ".pth", "teacher_" + stem + ".pth"):
            assert same_state(torch.load(os.path.join(epoch0, name), weights_only=False), torch.load(os.path.join(epoch0_b, name), weights_only=False)), name
        assert open(os.path.join(epoch0, "manifest.json")).read() == open(os.path.join(epoch0_b, "manifest.json")).read()

        # never overwrites: a finished epoch_0000, an empty directory, even a file of that name; nothing changes
        snapshot = {name: open(os.path.join(epoch0, name), "rb").read() for name in os.listdir(epoch0)}
        for _ in range(2):
            with contextlib.redirect_stderr(io.StringIO()) as problems:
                assert make_cycle0_weights.main([run_a]) == 1
            assert "ERROR: " in problems.getvalue() and "not overwriting" in problems.getvalue(), problems.getvalue()
        assert {name: open(os.path.join(epoch0, name), "rb").read() for name in os.listdir(epoch0)} == snapshot and sorted(os.listdir(weights_a)) == ["epoch_0000", "epoch_0001", "epoch_0002"]
        for blocker in ("directory", "file"):
            run_c, _ = make_run(os.path.join(tmp, "c"), name="all4_cyclic_dinov3_sweep_S2", seed=100)
            target = os.path.join(run_c, "models", "weights", "epoch_0000")
            if blocker == "directory":
                os.makedirs(target)
            else:
                open(target, "w").close()
            with contextlib.redirect_stderr(io.StringIO()) as problems:
                assert make_cycle0_weights.main([run_c]) == 1 and "not overwriting" in problems.getvalue()
            assert (os.listdir(target) == [] if blocker == "directory" else os.path.getsize(target) == 0) and sorted(os.listdir(os.path.dirname(target))) == ["epoch_0000", "epoch_0001", "epoch_0002"]
            shutil.rmtree(run_c)

        # where the backbone file comes from: an explicit file wins; STAGEA_FINAL rows and runs outside the sweep need one; S16's row names its export
        def attempt(argv):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as problems:
                code = make_cycle0_weights.main(argv)
            return code, problems.getvalue()

        run_13, _ = make_run(os.path.join(tmp, "a"), name="all4_cyclic_dinov3_sweep_S13", cycles=(1,))
        code, problem = attempt([run_13])
        assert code == 1 and "STAGEA_FINAL" in problem and "--pretrained-weights" in problem and not os.path.exists(os.path.join(run_13, "models", "weights", "epoch_0000")), problem
        assert attempt([run_13, "--pretrained-weights", stock_a])[0] == 0
        assert json.load(open(os.path.join(run_13, "models", "weights", "epoch_0000", "reconstructed_from_seed.json")))["weights_source"] == "--pretrained-weights"
        run_16, _ = make_run(os.path.join(tmp, "a"), name="all4_cyclic_dinov3_sweep_S16", cycles=(1,))
        assert attempt([run_16])[0] == 0
        assert json.load(open(os.path.join(run_16, "models", "weights", "epoch_0000", "reconstructed_from_seed.json")))["weights"] == (
            "/workspace/ssl_dinov3_cxr_constlr_seed_100/export/teacher_backbone_epoch_0030.pth")
        other, _ = make_run(os.path.join(tmp, "a"), name="some_other_run", cycles=(1,))
        code, problem = attempt([other])
        assert code == 1 and "not a sweep run" in problem and "--pretrained-weights" in problem, problem
        unknown, _ = make_run(os.path.join(tmp, "a"), name="all4_cyclic_dinov3_sweep_S99", cycles=(1,))
        code, problem = attempt([unknown])
        assert code == 1 and "not exactly one" in problem, problem
        missing, _ = make_run(os.path.join(tmp, "a"), name="all4_cyclic_dinov3_sweep_S4", cycles=(1,))  # S4's row names the stock file, which is there; point at a missing one
        code, problem = attempt([missing, "--pretrained-weights", os.path.join(tmp, "nope.pth")])
        assert code == 1 and "is missing" in problem and not os.path.exists(os.path.join(missing, "models", "weights", "epoch_0000"))
        assert sorted(os.listdir(os.path.join(missing, "models", "weights"))) == ["epoch_0001"]  # nothing, not even a scratch dir
        empty = os.path.join(tmp, "a", "all4_cyclic_dinov3_sweep_S5_seed_100")
        os.makedirs(os.path.join(empty, "models", "weights"))
        code, problem = attempt([empty])
        assert code == 1 and "no models/weights/epoch_*/manifest.json" in problem, problem
        code, problem = attempt([os.path.join(tmp, "a", "no_such_run")])
        assert code == 1 and "not a run dir" in problem, problem

        # --verify-against: the replay against a natively saved epoch_0000, here simulated by the trainer's own sequence and its own saver
        def native_run(name, order="student first", extra_draw=False):
            """A run dir whose epoch_0000 is what the trainer would have saved: seeded, then the two builds, saved by engine._save_cycle_weights."""
            run, args = make_run(os.path.join(tmp, "a"), name=name, cycles=(1,))
            random.seed(100)
            np.random.seed(100)
            torch.manual_seed(100)
            if extra_draw:
                torch.rand(1)  # one more draw from the global RNG before the builds
            if order == "student first":
                student = build_omni_model(replay, CLASSES)
                teacher = build_omni_model(replay, CLASSES)
            else:
                teacher = build_omni_model(replay, CLASSES)
                student = build_omni_model(replay, CLASSES)
            engine._save_cycle_weights(os.path.join(run, "models", "weights"), args, engine._run_metadata(args, NAMES, CLASSES, 0), student, teacher)
            return run, os.path.join(run, "models", "weights", "epoch_0000")

        def verify(run, native):
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                code = make_cycle0_weights.main([run, "--pretrained-weights", stock_a, "--verify-against", native])
            return code, printed.getvalue()

        native, native_dir = native_run("native_a")
        listing = sorted(os.listdir(os.path.join(native, "models", "weights")))
        code, printed = verify(native, native_dir)
        tensor_lines = lambda output: [line.split() for line in output.splitlines() if re.match(r"(student|teacher) +max\|diff\| ", line)]  # [role, "max|diff|", diff, key, shape...]
        assert code == 0 and "IDENTICAL" in printed and "manifest.json: identical" in printed, printed
        assert len(tensor_lines(printed)) == 2 * len(student_state) and all(float(line[2]) == 0.0 for line in tensor_lines(printed))  # one line per tensor of both networks
        assert sorted(os.listdir(os.path.join(native, "models", "weights"))) == listing  # and nothing was written
        for flawed, why in ((native_run("native_b", extra_draw=True), "one extra RNG draw before the builds"), (native_run("native_c", order="teacher first"), "teacher built first")):
            code, printed = verify(*flawed)
            moved = [line[3] for line in tensor_lines(printed) if float(line[2]) != 0.0]
            assert code == 1 and "DIFFERENT" in printed and moved and all(key.startswith(HEAD_PREFIXES) for key in moved), (why, printed)  # the heads move, the backbone file's tensors do not
        native2, native2_dir = native_run("native_d")  # a changed backbone tensor in the native teacher is named, with its max |diff|
        teacher_file = os.path.join(native2_dir, "teacher_cycle_0000_native_d_seed_100.pth")
        tampered = torch.load(teacher_file, weights_only=False)
        key = "blocks.0.attn.proj.weight"
        tampered[key] = tampered[key] + 0.25
        torch.save(tampered, teacher_file)
        code, printed = verify(native2, native2_dir)
        line = [line for line in tensor_lines(printed) if line[0] == "teacher" and line[3] == key][0]
        assert code == 1 and abs(float(line[2]) - 0.25) < 1e-6 and [l[3] for l in tensor_lines(printed) if float(l[2]) != 0.0] == [key], printed
        native3, native3_dir = native_run("native_e")  # identical tensors but a manifest that says something else
        manifest_file = os.path.join(native3_dir, "manifest.json")
        changed = json.load(open(manifest_file))
        changed["freeze_backbone"] = True
        json.dump(changed, open(manifest_file, "w"))
        code, printed = verify(native3, native3_dir)
        assert code == 1 and "manifest.json: differs in freeze_backbone" in printed and "DIFFERENT" in printed, printed
        # the module also runs as a script (the form used in the image)
        shown = subprocess.run([sys.executable, os.path.join(HERE, "make_cycle0_weights.py"), "--help"], capture_output=True, text=True, cwd=tmp)
        assert shown.returncode == 0 and "--verify-against" in shown.stdout, shown
    print("OK (d) make_cycle0_weights.py: epoch_0000 = manifest (engine._run_metadata, cycle 0) + student + teacher in the trainer's layout, taken by engine's loaders; "
          "weights equal an independent seed -> student build -> teacher build replay (student and teacher share the backbone only: the {} head tensors differ); a rerun "
          "is bit-identical whatever the caller's RNG or threads; an existing epoch_0000 (dir, empty dir or file) is never overwritten and a failed run leaves no scratch dir; "
          "backbone file from --pretrained-weights or the sweep row (STAGEA_FINAL / other runs need the flag); --verify-against says IDENTICAL for a native replay and "
          "DIFFERENT (exit 1, per-tensor max|diff|) for an extra RNG draw, a teacher-first build, a changed tensor or a changed manifest".format(len(differing)))
finally:
    models.DINOV3_VITB16_KWARGS = DINOV3_VITB16_KWARGS

# (e) test mode scores a cycle at its own input size -------------------------------------------------------------------------------
import torchvision.transforms as T


RECORDED = {"built": [], "engine": []}  # what the last test_main call built and handed the engine, kept when main raises


def test_main(weights, *flags, crops=10):
    """main_ark.main in test mode as eval_d1_6.sh calls it, with the dataset classes and omni_engine replaced by recorders: returns (args omni_engine got,
    [(dataset, the kwargs its test set was built with)]). The test transforms are built by main_ark.main itself."""
    sys.argv = ["main_ark.py", "--mode", "test", "--test_augment", "true" if crops == 10 else "false", "--pretrained_weights", weights,
                "--data_set", NAMES[0], "--data_set", NAMES[1], "--model", "vit_base_dinov3", "--projector_features", "32", "--device", "cpu"] + list(flags)
    built, engine_args = RECORDED["built"], RECORDED["engine"]
    del built[:], engine_args[:]
    stock_loaders, stock_engine, cwd = main_ark.dict_dataloarder, main_ark.omni_engine, os.getcwd()
    main_ark.dict_dataloarder = {name: (lambda name: lambda **kwargs: built.append((name, kwargs)) or name)(name) for name in stock_loaders}
    main_ark.omni_engine = lambda args, *rest: engine_args.append(args)
    try:
        os.chdir(HERE)  # main reads datasets_config.yaml from the working directory, base/
        main_ark.main(main_ark.get_args_parser())
    finally:
        main_ark.dict_dataloarder, main_ark.omni_engine = stock_loaders, stock_engine
        os.chdir(cwd)
    return engine_args[0], built


def sizes(transform):
    """(resize, crop) of a test transform: Resize then TenCrop (ten-crop) or CenterCrop (center crop)."""
    steps = transform.transforms
    assert isinstance(steps[0], T.Resize) and isinstance(steps[1], (T.TenCrop, T.CenterCrop)), steps
    return tuple(steps[0].size), tuple(steps[1].size)


def saved_cycle(root, cycle, **size):
    """models/weights/epoch_<cycle> of a run with the trainer's manifest (args crop_size / resize as given; none = an args object from before them)."""
    directory = os.path.join(root, "run_seed_1", "models", "weights", "epoch_%04d" % cycle)
    os.makedirs(directory)
    metadata = engine._run_metadata(SimpleNamespace(exp_name="run", seed=1, model_name="vit_base_dinov3", projector_features=32, use_mlp=False, **size), NAMES, CLASSES, cycle)
    with open(os.path.join(directory, "manifest.json"), "w") as handle:
        json.dump(metadata, handle)
    for role in ("student", "teacher"):
        torch.save({}, os.path.join(directory, "{}_cycle_{:04d}_run_seed_1.pth".format(role, cycle)))
    return directory, metadata


with tempfile.TemporaryDirectory() as tmp:
    big, big_manifest = saved_cycle(os.path.join(tmp, "big"), 3, crop_size=448, resize=512)
    assert (big_manifest["input_size"], big_manifest["img_resize"]) == (448, 512)
    old, old_manifest = saved_cycle(os.path.join(tmp, "old"), 3)
    assert (old_manifest["input_size"], old_manifest["img_resize"]) == (224, 256)  # engine._run_metadata's defaults for args without the attributes
    with open(os.path.join(old, "manifest.json"), "w") as handle:  # a manifest written before the keys existed
        json.dump({k: v for k, v in old_manifest.items() if k not in ("input_size", "img_resize")}, handle)
    # eval_d1_6.sh's own argv passes no size: the parser leaves both unset in test mode (and fills 224 / 256 in train mode)
    sys.argv = ["main_ark.py"] + run_eval([RUN_DIR, "3"])[1][2:]
    assert (main_ark.get_args_parser().crop_size, main_ark.get_args_parser().resize) == (None, None)
    sys.argv = ["main_ark.py", "--mode", "train"]
    assert (main_ark.get_args_parser().crop_size, main_ark.get_args_parser().resize) == (224, 256)
    for weights in (big, os.path.join(big, "student_cycle_0003_run_seed_1.pth")):  # the weights directory or a saved weight file in it
        for crops, crop_type in ((10, T.TenCrop), (1, T.CenterCrop)):
            for flags in ((), ("--input_size", "448"), ("--img_resize", "512"), ("--input_size", "448", "--img_resize", "512")):  # unset, or passed and equal
                args, built = test_main(weights, *flags, crops=crops)
                assert (args.crop_size, args.resize) == (448, 512) and [name for name, _ in built] == NAMES, (weights, flags, vars(args))
                for name, kwargs in built:
                    assert (kwargs["crop_size"], kwargs["resize"]) == (448, 512) and sizes(kwargs["augment"]) == ((512, 512), (448, 448)), (name, flags)
                    assert isinstance(kwargs["augment"].transforms[1], crop_type), kwargs["augment"]
    args, built = test_main(old)  # no keys: what every run trained at before them
    assert (args.crop_size, args.resize) == (224, 256) and all(sizes(kwargs["augment"]) == ((256, 256), (224, 224)) for _, kwargs in built), vars(args)
    # a cycle checkpoint carries the manifest's fields (engine: checkpoint = metadata.copy()); one from before the keys trained at 224 / 256
    tar_big, tar_old = os.path.join(tmp, "cycle_0003_big_seed_1.pth.tar"), os.path.join(tmp, "cycle_0003_old_seed_1.pth.tar")
    torch.save(dict(big_manifest, epoch=2, state_dict={}, teacher={}), tar_big)
    torch.save(dict({k: v for k, v in old_manifest.items() if k not in ("input_size", "img_resize")}, epoch=2, state_dict={}, teacher={}), tar_old)
    for crops in (10, 1):
        args, built = test_main(tar_big, crops=crops)  # no flags: the checkpoint's 448 / 512
        assert (args.crop_size, args.resize) == (448, 512) and all(sizes(kwargs["augment"]) == ((512, 512), (448, 448)) for _, kwargs in built) and len(built) == 2, vars(args)
        args, built = test_main(tar_old, crops=crops)
        assert (args.crop_size, args.resize) == (224, 256) and all(sizes(kwargs["augment"]) == ((256, 256), (224, 224)) for _, kwargs in built) and len(built) == 2, vars(args)
    assert (test_main(tar_big, "--input_size", "448", "--img_resize", "512")[0].crop_size, test_main(tar_old, "--input_size", "224")[0].resize) == (448, 256)  # passed and equal
    for weights, flags, message in ((big, ("--input_size", "224"), "--input_size 224 differs from input_size=448 recorded in '{}'".format(big)),
                                    (big, ("--img_resize", "256"), "--img_resize 256 differs from img_resize=512 recorded in '{}'".format(big)),
                                    (big, ("--input_size", "448", "--img_resize", "585"), "--img_resize 585 differs from img_resize=512"),
                                    (old, ("--input_size", "448"), "--input_size 448 differs from input_size=224 recorded in '{}'".format(old)),
                                    (tar_big, ("--input_size", "224"), "--input_size 224 differs from input_size=448 recorded in '{}'".format(tar_big)),
                                    (tar_big, ("--img_resize", "256"), "--img_resize 256 differs from img_resize=512 recorded in '{}'".format(tar_big)),
                                    (tar_old, ("--input_size", "448"), "--input_size 448 differs from input_size=224 recorded in '{}'".format(tar_old))):
        rejects(ValueError, lambda: test_main(weights, *flags), message)
        assert RECORDED == {"built": [], "engine": []}, (flags, RECORDED)  # refused before a dataset is built or the engine starts
    # a file this trainer did not write (no run metadata) records nothing: the value passed, else 224 / 256
    outside = os.path.join(tmp, "outside.pth.tar")
    torch.save({"state_dict": {}, "teacher": {}}, outside)
    assert (test_main(outside)[0].crop_size, test_main(outside)[0].resize) == (224, 256)
    assert (test_main(outside, "--input_size", "448", "--img_resize", "512")[0].crop_size, test_main(outside, "--input_size", "448")[0].resize) == (448, 256)
# make_cycle0_weights.py rebuilds epoch_0000 with the run's own size (the lowest saved manifest's), and none for a run from before the keys
_, rebuilt = make_cycle0_weights.cycle0_metadata(big_manifest)
assert (rebuilt["input_size"], rebuilt["img_resize"], rebuilt["cycle"]) == (448, 512, 0), rebuilt
_, rebuilt = make_cycle0_weights.cycle0_metadata({k: v for k, v in old_manifest.items() if k not in ("input_size", "img_resize")})
assert "input_size" not in rebuilt and "img_resize" not in rebuilt, rebuilt
print("OK (e) test mode: a manifest with input_size 448 / img_resize 512 makes main_ark.main build Resize(512) + TenCrop(448) (center crop: CenterCrop(448)) for every test set, "
      "from the weights directory or a saved weight file; a manifest without the keys -> 224 / 256; --input_size / --img_resize passed and equal pass, passed and different "
      "-> ValueError naming both values, before any dataset or the engine; a cycle .pth.tar the same from its own metadata (448 / 512 recorded, or none: 224 / 256); "
      "eval_d1_6.sh passes neither; a file the trainer did not write keeps the flag or 224 / 256; "
      "make_cycle0_weights.py carries the run's size into epoch_0000")

print("ALL OK")
