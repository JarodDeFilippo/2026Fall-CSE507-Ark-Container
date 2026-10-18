"""Checks for the chain phase scripts (chain/armD.sh, probe_a.sh, probe_b.sh, gate.sh) and the standalone chain/armD_long.sh in fake repo
layouts in a temp dir.
Run from base/: python test_chain_scripts.py (torch is only needed for the parsed-options checks of (e), skipped without it).

The phase scripts call their start / resume scripts and gaudi.sh by relative path from the working directory (the chain driver has
already cd'ed to the repo root), so no DRY_RUN switch is needed: (a)-(d) and (f) give the fake repo stubs at those paths. A stub prints how
it was called ($0, the arguments, the ARK_* environment) and exits with $STUB_EXIT (default 0). A phase script must never change the tree.
(e) runs the REAL start / resume scripts and common.sh behind a gaudi.sh that prints its argv.
(f) armD_long.sh (D-long: arm D's recipe for 50 cycles on one card; not a chain phase) and its sbatch wrapper run_d1_8dl_gaudi.sh.
(g) the Step VIII fine-tuning sweep: the parameterized start / resume pair, the table sweep/wave1.tsv and the standalone chain/sweep_run.sh
(not a chain phase either), run through the real scripts and main_ark.py's option parser. (h) its sbatch wrappers run_d1_8sw_gaudi.sh (2 cards),
run_d1_8sw4_gaudi.sh and run_d1_8sw8_gaudi.sh (4 / 8 cards) and sweep/submit_wave.sh, with a stub sbatch."""
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_START = "experiment_scripts/all4_cyclic_dinov3_frozen/start_all4_cyclic_dinov3_frozen.sh"
FROZEN_RESUME = "experiment_scripts/all4_cyclic_dinov3_frozen/resume_all4_cyclic_dinov3_frozen.sh"
PROBE_START = "experiment_scripts/all4_cyclic_dinov3_probe/start_all4_cyclic_dinov3_probe.sh"
PROBE_RESUME = "experiment_scripts/all4_cyclic_dinov3_probe/resume_all4_cyclic_dinov3_probe.sh"
FROZEN_LONG_START = "experiment_scripts/all4_cyclic_dinov3_frozen_long/start_all4_cyclic_dinov3_frozen_long.sh"
FROZEN_LONG_RESUME = "experiment_scripts/all4_cyclic_dinov3_frozen_long/resume_all4_cyclic_dinov3_frozen_long.sh"
SWEEP_START = "experiment_scripts/all4_cyclic_dinov3_sweep/start_all4_cyclic_dinov3_sweep.sh"
SWEEP_RESUME = "experiment_scripts/all4_cyclic_dinov3_sweep/resume_all4_cyclic_dinov3_sweep.sh"
STOCK = "weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
STUB = """#!/usr/bin/env bash
printf 'ARGV0=%s\\n' "$0"
for argument in "$@"; do printf 'ARG=%s\\n' "$argument"; done
env | grep '^ARK_' | sort | sed 's/^/ENV /'
exit "${STUB_EXIT:-0}"
"""
PRINTING_WRAPPER = '#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n'
EXPORT = "ssl_dinov3_cxr_pilot_{}_seed_100/export/teacher_backbone_epoch_0004.pth"  # the pilots stop after epoch 4 (end of the LR warmup)
wrapper_env = dict(re.findall(r"^export (ARK_\w+)=(\S+)$", open(os.path.join(REPO, "run_d1_8d_gaudi.sh")).read(), re.M))
assert len(wrapper_env) == 9, wrapper_env  # batch, lr, test augment, eval every, warmup, workers, model, init, weights


def populate(root, files):
    """Create `files` in root: relative paths (content "x"), "dir/" for a directory, or (path, content)."""
    for entry in files:
        if isinstance(entry, str) and entry.endswith("/"):
            os.makedirs(os.path.join(root, entry), exist_ok=True)
            continue
        path, content = entry if isinstance(entry, tuple) else (entry, "x")
        os.makedirs(os.path.dirname(os.path.join(root, path)), exist_ok=True)
        with open(os.path.join(root, path), "w") as handle:
            handle.write(content)


def snapshot(root):
    return sorted(os.path.join(directory, name) for directory, directories, names in os.walk(root) for name in directories + names)


def run(script, files=(), stub_exit=0, args=()):
    """Run chain/<script> `args` from the root of a fresh fake repo (stubs for the start / resume scripts and gaudi.sh, plus `files`).
    Returns (completed process, what the stub recorded or None when none ran): {"argv0": $0 as typed, "file": the stub's path
    relative to the fake repo (bash's exec builtin shows an absolute $0), "args", "env", "count": how many stubs ran}."""
    with tempfile.TemporaryDirectory() as root:
        for stub in (FROZEN_START, FROZEN_RESUME, FROZEN_LONG_START, FROZEN_LONG_RESUME, PROBE_START, PROBE_RESUME, SWEEP_START, SWEEP_RESUME,
                     "gaudi.sh"):
            populate(root, [(stub, STUB)])
            os.chmod(os.path.join(root, stub), 0o755)
        os.makedirs(os.path.join(root, "chain"))
        shutil.copy(os.path.join(REPO, "chain", script), os.path.join(root, "chain"))  # keeps the executable bit
        populate(root, files)
        before = snapshot(root)
        env = {k: v for k, v in os.environ.items() if not k.startswith("ARK_")}
        env["STUB_EXIT"] = str(stub_exit)
        done = subprocess.run([os.path.join(root, "chain", script)] + list(args), cwd=root, env=env, capture_output=True, text=True)
        assert snapshot(root) == before, "{} changed the tree".format(script)  # nothing created, nothing deleted
        lines = done.stdout.splitlines()
        called = None
        if any(line.startswith("ARGV0=") for line in lines):
            argv0 = [l[6:] for l in lines if l.startswith("ARGV0=")][0]
            called = {"argv0": argv0,
                      "file": os.path.relpath(os.path.realpath(os.path.join(root, argv0)), os.path.realpath(root)),
                      "args": [l[4:] for l in lines if l.startswith("ARG=")],
                      "env": dict(l[4:].split("=", 1) for l in lines if l.startswith("ENV ")),
                      "count": sum(1 for l in lines if l.startswith("ARGV0="))}
    return done, called


# (a) static: valid bash, executable, plain scripts that leave the driver's job alone and delete nothing ----------------
for script in ("armD.sh", "armD_long.sh", "sweep_run.sh", "probe_a.sh", "probe_b.sh", "gate.sh"):
    path = os.path.join(REPO, "chain", script)
    subprocess.run(["bash", "-n", path], check=True)
    assert os.access(path, os.X_OK), script
    text = open(path).read()
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert text.startswith("#!/usr/bin/env bash\n") and "set -uo pipefail" in code and not re.search(r"set -\w*e", code), script
    for forbidden in ("gaudi_check_cards", "gaudi_verify_busy", "PT_HPU_LAZY_MODE", "SBATCH", "sbatch", "srun"):
        assert forbidden not in code, (script, forbidden)
    assert not re.search(r"^\s*cd\s", code, re.M), script  # the driver already did the cd
    assert not re.search(r"\b(rm|rmdir|mv|unlink|truncate)\b|-delete", code), script  # a leftover run dir is left for a human
    for called_script in re.findall(r"\./experiment_scripts/\S+\.sh", code):
        assert os.access(os.path.join(REPO, called_script), os.X_OK), (script, called_script)  # the scripts it calls exist
for kind in ("frozen", "frozen_long", "probe", "sweep"):
    for verb in ("start", "resume"):
        path = os.path.join(REPO, "experiment_scripts", "all4_cyclic_dinov3_" + kind, "{}_all4_cyclic_dinov3_{}.sh".format(verb, kind))
        subprocess.run(["bash", "-n", path], check=True)
        assert os.access(path, os.X_OK), path
print("OK (a) static: bash -n, executable, set -uo pipefail, no SBATCH header, no cd / card check / busy verify / lazy-mode export / rm; "
      "the start and resume scripts they call exist")

# (b) armD: start, resume, skip, fail --------------------------------------------------------------------------------------
RUN_D = "all4_cyclic_dinov3_frozen_seed_100"
CHECKPOINT_D = RUN_D + "/models/checkpoints/cycle_0002_" + RUN_D + ".pth.tar"
DONE_D = RUN_D + "/models/weights/epoch_0005/student_cycle_0005_" + RUN_D + ".pth"
NO_CHECKPOINT = "exists but holds no resumable checkpoint ({}/models/checkpoints/*.pth.tar)".format(RUN_D)
# fresh start: no run dir
done, called = run("armD.sh", [STOCK])
assert done.returncode == 0 and done.stderr == "", done.stderr
assert called["argv0"] == "./" + FROZEN_START and called["file"] == FROZEN_START and called["count"] == 1, called
assert called["args"] == ["100", "gaudi"], called
assert called["env"] == wrapper_env, called["env"]  # the same ARK_* recipe as run_d1_8d_gaudi.sh
start_command = " ".join([called["argv0"]] + called["args"])
assert run("armD.sh", [STOCK], stub_exit=7)[0].returncode == 7  # the start script's status comes through
for missing in ([], [(STOCK, "")]):  # missing or empty weights
    done, called = run("armD.sh", missing)
    assert done.returncode == 1 and called is None and "DINOv3 weights not found at ./" + STOCK in done.stderr, done
# interrupted attempt: the run dir exists with a checkpoint -> resume, never start
done, called = run("armD.sh", [STOCK, CHECKPOINT_D])
assert done.returncode == 0 and done.stderr == "", done.stderr
assert called["argv0"] == "./" + FROZEN_RESUME and called["file"] == FROZEN_RESUME and called["count"] == 1, called
assert called["args"] == ["100", "gaudi"] and called["env"] == wrapper_env, called
assert "armD: resuming ./" + RUN_D + " from its latest checkpoint" in done.stdout, done.stdout
resume_command = " ".join([called["argv0"]] + called["args"])
assert run("armD.sh", [STOCK, CHECKPOINT_D], stub_exit=7)[0].returncode == 7  # the resume script's status comes through
done, called = run("armD.sh", [CHECKPOINT_D])  # a resume builds the model from the stock weights too: same fail-fast guard
assert done.returncode == 1 and called is None and "DINOv3 weights not found at ./" + STOCK in done.stderr, done
# a run dir without a resumable checkpoint: an error that names the dir, nothing invoked, nothing deleted (run() checks the tree)
for partial in ([RUN_D + "/"],                                              # just the dir
                [RUN_D + "/models/checkpoints/"],                           # an empty checkpoint dir
                [DONE_D.replace("epoch_0005", "epoch_0004").replace("cycle_0005", "cycle_0004")],  # cycle 4's student only
                [RUN_D + "/models/weights/epoch_0005/teacher_cycle_0005_" + RUN_D + ".pth"],        # cycle 5's teacher only
                [RUN_D + "/models/checkpoints/cycle_0002_" + RUN_D + ".pth"]):                      # not a .pth.tar
    done, called = run("armD.sh", [STOCK] + partial)
    assert done.returncode == 1 and called is None and "./" + RUN_D + " " + NO_CHECKPOINT in done.stderr, (partial, done)
    assert "neither resuming nor starting over it, nothing deleted" in done.stderr
# done wins over everything: no weights check, no resume, nothing launched
for present in ([DONE_D], [DONE_D, CHECKPOINT_D], [STOCK, DONE_D, CHECKPOINT_D]):
    done, called = run("armD.sh", present)
    assert done.returncode == 0 and done.stdout.strip() == "armD already done" and called is None, (present, done)
# another run's dir is none of its business
other = "all4_cyclic_dinov3_probe_a_seed_100/models/checkpoints/cycle_0002_all4_cyclic_dinov3_probe_a_seed_100.pth.tar"
done, called = run("armD.sh", [STOCK, other, "all4_cyclic_dinov3_probe_a_seed_100/models/weights/epoch_0005/student_cycle_0005_x.pth"])
assert done.returncode == 0 and called["file"] == FROZEN_START, (done, called)
print("OK (b) armD: fresh -> `{}`; run dir + checkpoint -> `{}`, both with the 8d ARK_* env; done (epoch_0005/student_cycle_0005_*.pth) skips first; "
      "empty/missing weights or a run dir without a *.pth.tar -> exit 1 naming it; exit statuses pass through".format(start_command, resume_command))

# (c) probe_a / probe_b ----------------------------------------------------------------------------------------------------
probe_commands = {}
for letter in ("a", "b"):
    script = "probe_{}.sh".format(letter)
    export = EXPORT.format(letter)
    other_export = EXPORT.format("b" if letter == "a" else "a")
    run_dir = "all4_cyclic_dinov3_probe_{}_seed_100".format(letter)
    exp_name = "all4_cyclic_dinov3_probe_" + letter
    checkpoint = run_dir + "/models/checkpoints/cycle_0002_" + run_dir + ".pth.tar"
    finished = run_dir + "/models/weights/epoch_0005/student_cycle_0005_" + run_dir + ".pth"
    wanted_env = dict(wrapper_env, ARK_PRETRAINED_WEIGHTS="/workspace/" + export)  # 8d's recipe, the pilot's weights
    wanted_args = ["100", "gaudi", exp_name, "/workspace/" + export]
    missing_text = "pilot {} export ./{} is missing or empty".format(letter.upper(), export)
    # fresh start
    done, called = run(script, [export])
    assert done.returncode == 0 and done.stderr == "", done.stderr
    assert called["argv0"] == "./" + PROBE_START and called["count"] == 1 and called["args"] == wanted_args and called["env"] == wanted_env, called
    assert run(script, [export], stub_exit=3)[0].returncode == 3
    # interrupted attempt -> resume with the very same arguments
    done, called = run(script, [export, checkpoint])
    assert done.returncode == 0 and done.stderr == "", done.stderr
    assert called["argv0"] == "./" + PROBE_RESUME and called["count"] == 1 and called["args"] == wanted_args and called["env"] == wanted_env, called
    assert "probe_{}: resuming ./{} from its latest checkpoint".format(letter, run_dir) in done.stdout, done.stdout
    probe_commands[script] = (" ".join(["./" + PROBE_START] + wanted_args), " ".join(["./" + PROBE_RESUME] + wanted_args))
    assert run(script, [export, checkpoint], stub_exit=3)[0].returncode == 3
    # the export is needed to start and to resume; missing, empty, or only the other pilot's
    for broken in ([], [(export, "")], [other_export]):
        for with_checkpoint in ([], [checkpoint]):
            done, called = run(script, broken + with_checkpoint)
            assert done.returncode == 1 and called is None and missing_text in done.stderr, (script, broken, with_checkpoint, done)
    # a run dir without a resumable checkpoint
    for partial in ([run_dir + "/"], [run_dir + "/models/checkpoints/"], [run_dir + "/models/weights/epoch_0002/student_cycle_0002_x.pth"],
                    [run_dir + "/models/weights/epoch_0003/student_cycle_0003_x.pth"],  # the old 3-cycle done marker: now just an unfinished cycle
                    [run_dir + "/models/weights/epoch_0004/student_cycle_0004_x.pth"],  # cycle 4's student only
                    [run_dir + "/models/weights/epoch_0005/teacher_cycle_0005_x.pth"],  # cycle 5's teacher only
                    [run_dir + "/models/checkpoints/cycle_0002_x.pth"]):                # not a .pth.tar
        done, called = run(script, [export] + partial)
        assert done.returncode == 1 and called is None and "./{} exists but holds no resumable checkpoint ({}/models/checkpoints/*.pth.tar)".format(
            run_dir, run_dir) in done.stderr, (script, partial, done)
    # done: skipped before anything else, with or without the export, the run dir contents or a checkpoint
    for present in ([export, finished], [finished], [finished, checkpoint]):
        done, called = run(script, present)
        assert done.returncode == 0 and done.stdout.strip() == "probe_{} already done".format(letter) and called is None, (script, present, done)
    # the other probe's run dir (checkpoint or finished) is none of its business
    other_dir = "all4_cyclic_dinov3_probe_{}_seed_100".format("b" if letter == "a" else "a")
    done, called = run(script, [export, other_dir + "/models/checkpoints/cycle_0002_x.pth.tar", other_dir + "/models/weights/epoch_0005/student_cycle_0005_x.pth"])
    assert done.returncode == 0 and called["argv0"] == "./" + PROBE_START, (script, done)
print("OK (c) probes: fresh -> `{}` / `{}`; run dir + checkpoint -> `{}` / `{}`; done (epoch_0005/student_cycle_0005_*.pth) skips first; "
      "missing or empty export, or a run dir without a *.pth.tar -> exit 1".format(
          probe_commands["probe_a.sh"][0], probe_commands["probe_b.sh"][0], probe_commands["probe_a.sh"][1], probe_commands["probe_b.sh"][1]))

# (d) gate ---------------------------------------------------------------------------------------------------------------
GATE_ARGS = ["python", "/workspace/chain/gate.py",
             "--arm_d", "/workspace/all4_cyclic_dinov3_frozen_seed_100",
             "--probe_a", "/workspace/all4_cyclic_dinov3_probe_a_seed_100", "--probe_b", "/workspace/all4_cyclic_dinov3_probe_b_seed_100",
             "--pilot_a", "/workspace/ssl_dinov3_cxr_pilot_a_seed_100", "--pilot_b", "/workspace/ssl_dinov3_cxr_pilot_b_seed_100",
             "--out", "/workspace/chain"]
done, called = run("gate.sh", [("chain/gate.py", "# stub\n")])
assert done.returncode == 0 and called["file"] == "gaudi.sh" and called["args"] == GATE_ARGS, (done, called)
assert "exec ./gaudi.sh python /workspace/chain/gate.py" in open(os.path.join(REPO, "chain", "gate.sh")).read()  # exec, relative to the repo root
for code in (1, 2, 5):  # gate.py's exit codes come through unchanged: 0 decision, 2 STOP, 1 parse error
    assert run("gate.sh", [("chain/gate.py", "# stub\n")], stub_exit=code)[0].returncode == code
for missing in ([], [("chain/gate.py", "")]):
    done, called = run("gate.sh", missing)
    assert done.returncode == 1 and called is None and "chain/gate.py is missing or empty" in done.stderr, done
print("OK (d) gate: exec ./gaudi.sh {}; exit status passes through (0 / 2 / 1); missing gate.py -> exit 1 before any container starts".format(
    " ".join(GATE_ARGS)))


# (e) the real start / resume scripts: resume == start apart from the resume flag, end to end through the chain scripts -----
def real_run(script, args=(), files=(), without=()):
    """Run `script` (relative to the repo root) in a fake repo made of the REAL common.sh, start / resume and chain scripts, with a
    gaudi.sh that prints its argv instead of launching, 4 cards visible and the 8d ARK_* env (minus the names in `without`, whatever the caller exports).
    Returns (process, wrapper argv or None)."""
    with tempfile.TemporaryDirectory() as root:
        for rel in ("experiment_scripts/common.sh", FROZEN_START, FROZEN_RESUME, FROZEN_LONG_START, FROZEN_LONG_RESUME, PROBE_START, PROBE_RESUME,
                    SWEEP_START, SWEEP_RESUME, "chain/armD.sh", "chain/armD_long.sh", "chain/sweep_run.sh", "chain/probe_a.sh", "chain/probe_b.sh"):
            populate(root, [(rel, "")])
            shutil.copy(os.path.join(REPO, rel), os.path.join(root, rel))
        populate(root, [("gaudi.sh", PRINTING_WRAPPER)])
        os.chmod(os.path.join(root, "gaudi.sh"), 0o755)
        populate(root, files)
        env = dict(os.environ, HABANA_VISIBLE_MODULES="0,1,2,3", **wrapper_env)
        for name in without:  # out of the inherited environment too: the result must not depend on what the caller's shell exports
            env.pop(name, None)
        done = subprocess.run([os.path.join(root, script)] + list(args), cwd=root, env=env, capture_output=True, text=True)
    lines = done.stdout.splitlines()
    return done, (lines[lines.index("python"):] if "python" in lines else None)


def main_argv(wrapper_argv):
    return wrapper_argv[wrapper_argv.index("main_ark.py") + 1:]


def with_resume(argv):
    """argv as common.sh builds it with resume=true: --resume true goes right after --seed <n>."""
    k = argv.index("--seed") + 2
    return argv[:k] + ["--resume", "true"] + argv[k:]


try:
    import main_ark
except ImportError as error:
    main_ark = None
    print("SKIP the parsed-options checks of (e): cannot import main_ark here ({})".format(error))


def only_resume_differs(start_argv, resume_argv):
    if main_ark is None:
        return
    sys.argv = ["main_ark.py"] + start_argv
    start = main_ark.get_args_parser()
    sys.argv = ["main_ark.py"] + resume_argv
    resume = main_ark.get_args_parser()
    assert start.resume is False and resume.resume is True and resume.resume_from is None and resume.freeze_backbone is True
    assert {k for k in vars(start) if vars(start)[k] != vars(resume)[k]} == {"resume"}, "resume and start options differ beyond --resume"


export_a = "/workspace/" + EXPORT.format("a")
launched = {}  # kind -> (start argv, resume argv) as main_ark.py receives them
for kind, start, resume, args in (("frozen", FROZEN_START, FROZEN_RESUME, ["100", "gaudi"]),
                                  ("frozen_long", FROZEN_LONG_START, FROZEN_LONG_RESUME, ["100", "gaudi"]),
                                  ("probe", PROBE_START, PROBE_RESUME, ["100", "gaudi", "all4_cyclic_dinov3_probe_a", export_a])):
    start_text, resume_text = open(os.path.join(REPO, start)).read(), open(os.path.join(REPO, resume)).read()
    assert resume_text != start_text and resume_text == start_text.replace("    false \\\n", "    true \\\n"), kind  # one line: resume=true
    (started, start_wrapper), (resumed, resume_wrapper) = real_run(start, args), real_run(resume, args)
    assert started.returncode == 0 == resumed.returncode, (started.stderr, resumed.stderr)
    assert "--resume" not in main_argv(start_wrapper) and main_argv(resume_wrapper) == with_resume(main_argv(start_wrapper)), kind
    assert resume_wrapper[:-len(main_argv(resume_wrapper))] == start_wrapper[:-len(main_argv(start_wrapper))]  # same launcher
    only_resume_differs(main_argv(start_wrapper), main_argv(resume_wrapper))
    launched[kind] = (main_argv(start_wrapper), main_argv(resume_wrapper))
# frozen_long is the frozen pair with two values changed, start and resume alike: the experiment name and the cycle count
# (the last --pretrain_epochs wins over common.sh's 200); --warmup-epochs 2, --freeze_backbone true, adamw, the four data sets are unchanged
for frozen_argv, long_argv in zip(launched["frozen"], launched["frozen_long"]):
    assert len(long_argv) == len(frozen_argv) and long_argv[-4:] == ["--pretrain_epochs", "50", "--warmup-epochs", "2"], long_argv
    assert long_argv[long_argv.index("--exp_name") + 1] == "all4_cyclic_dinov3_frozen_long", long_argv
    assert [(a, b) for a, b in zip(frozen_argv, long_argv) if a != b] == [("all4_cyclic_dinov3_frozen", "all4_cyclic_dinov3_frozen_long"), ("5", "50")]
usage_args = ["100", "gaudi", "all4_cyclic_dinov3_probe_a", export_a, "extra"]
for count in (0, 1, 2, 3, 5):  # the probe resume script takes the probe start script's four arguments
    usage, wrapper_argv = real_run(PROBE_RESUME, usage_args[:count])
    assert usage.returncode == 2 and wrapper_argv is None and "Usage:" in usage.stderr, (count, usage.returncode, usage.stderr)
usage, wrapper_argv = real_run(FROZEN_RESUME, ["100", "gaudi", "extra"])
assert usage.returncode == 2 and wrapper_argv is None and "Usage:" in usage.stderr
resume_lines = {}
for script, run_dir, exp_name, needs in (("chain/armD.sh", RUN_D, "all4_cyclic_dinov3_frozen", [STOCK]),
                                         ("chain/probe_a.sh", "all4_cyclic_dinov3_probe_a_seed_100", "all4_cyclic_dinov3_probe_a", [EXPORT.format("a")]),
                                         ("chain/probe_b.sh", "all4_cyclic_dinov3_probe_b_seed_100", "all4_cyclic_dinov3_probe_b", [EXPORT.format("b")])):
    (fresh, fresh_wrapper) = real_run(script, files=needs)
    (interrupted, resume_wrapper) = real_run(script, files=needs + ["{0}/models/checkpoints/cycle_0002_{0}.pth.tar".format(run_dir)])
    assert fresh.returncode == 0 == interrupted.returncode, (script, fresh.stderr, interrupted.stderr)
    assert "--resume" not in main_argv(fresh_wrapper) and main_argv(resume_wrapper) == with_resume(main_argv(fresh_wrapper)), script
    assert main_argv(resume_wrapper)[main_argv(resume_wrapper).index("--exp_name") + 1] == exp_name
    only_resume_differs(main_argv(fresh_wrapper), main_argv(resume_wrapper))
    resume_lines[script] = " ".join(resume_wrapper)
print("OK (e) real scripts: each resume script is its start script with resume=true (one line); the argv differs only by `--resume true` after --seed; "
      "{}; the chain scripts launch exactly that on an interrupted run dir; frozen_long is frozen with only the experiment name and `--pretrain_epochs 50` changed".format(
          "parsed options differ only in `resume`" if main_ark is not None else "parsed-options check skipped"))
for script, line in resume_lines.items():
    print("   " + script + " (interrupted run) -> <repo>/gaudi.sh " + line)

# (f) armD_long: D-long, arm D's recipe for 50 cycles on one card -- a standalone phase script, not a chain phase -----------------
RUN_LONG = "all4_cyclic_dinov3_frozen_long_seed_100"
CHECKPOINT_LONG = RUN_LONG + "/models/checkpoints/cycle_0010_" + RUN_LONG + ".pth.tar"
DONE_LONG = RUN_LONG + "/models/weights/epoch_0050/student_cycle_0050_" + RUN_LONG + ".pth"
EPOCH5_LONG = RUN_LONG + "/models/weights/epoch_0005/student_cycle_0005_" + RUN_LONG + ".pth"  # arm D's done marker, 45 cycles short here
NO_CHECKPOINT_LONG = "./{0} exists but holds no resumable checkpoint ({0}/models/checkpoints/*.pth.tar)".format(RUN_LONG)
armd_env = run("armD.sh", [STOCK])[1]["env"]
assert armd_env["ARK_EVAL_EVERY"] == "1", armd_env  # arm D evaluates every cycle; D-long every 5th and the last, like arms B and C
long_env = dict(armd_env, ARK_EVAL_EVERY="5")
# fresh start: no run dir
done, called = run("armD_long.sh", [STOCK])
assert done.returncode == 0 and done.stderr == "", done.stderr
assert called["argv0"] == "./" + FROZEN_LONG_START and called["file"] == FROZEN_LONG_START and called["count"] == 1, called
assert called["args"] == ["100", "gaudi"] and called["env"] == long_env, called  # armD.sh's ARK_* env, with ARK_EVAL_EVERY=5 only
long_start_command = " ".join([called["argv0"]] + called["args"])
assert run("armD_long.sh", [STOCK], stub_exit=7)[0].returncode == 7  # the start script's status comes through
# interrupted attempt: the run dir exists with a checkpoint -> resume, never start
done, called = run("armD_long.sh", [STOCK, CHECKPOINT_LONG])
assert done.returncode == 0 and done.stderr == "", done.stderr
assert called["argv0"] == "./" + FROZEN_LONG_RESUME and called["file"] == FROZEN_LONG_RESUME and called["count"] == 1, called
assert called["args"] == ["100", "gaudi"] and called["env"] == long_env, called
assert "armD_long: resuming ./" + RUN_LONG + " from its latest checkpoint" in done.stdout, done.stdout
long_resume_command = " ".join([called["argv0"]] + called["args"])
assert run("armD_long.sh", [STOCK, CHECKPOINT_LONG], stub_exit=7)[0].returncode == 7  # the resume script's status comes through
# the stock weights are needed to start and to resume: missing or empty -> exit 1 before anything is called
for broken in ([], [(STOCK, "")]):
    for with_checkpoint in ([], [CHECKPOINT_LONG]):
        done, called = run("armD_long.sh", broken + with_checkpoint)
        assert done.returncode == 1 and called is None and "DINOv3 weights not found at ./" + STOCK in done.stderr, (broken, with_checkpoint, done)
# a run dir without a resumable checkpoint: an error that names the dir, nothing invoked, nothing deleted (run() checks the tree)
for partial in ([RUN_LONG + "/"],                                                                     # just the dir
                [RUN_LONG + "/models/checkpoints/"],                                                  # an empty checkpoint dir
                [EPOCH5_LONG],                                                                        # cycle 5's student only: not done
                [DONE_LONG.replace("epoch_0050", "epoch_0049").replace("cycle_0050", "cycle_0049")],  # cycle 49's student only
                [RUN_LONG + "/models/weights/epoch_0050/teacher_cycle_0050_" + RUN_LONG + ".pth"],    # cycle 50's teacher only
                [RUN_LONG + "/models/checkpoints/cycle_0010_" + RUN_LONG + ".pth"]):                  # not a .pth.tar
    done, called = run("armD_long.sh", [STOCK] + partial)
    assert done.returncode == 1 and called is None and NO_CHECKPOINT_LONG in done.stderr, (partial, done)
    assert "neither resuming nor starting over it, nothing deleted" in done.stderr
# done = epoch_0050's student; it wins over everything: no weights check, no resume, nothing launched
for present in ([DONE_LONG], [DONE_LONG, CHECKPOINT_LONG], [STOCK, DONE_LONG, CHECKPOINT_LONG]):
    done, called = run("armD_long.sh", present)
    assert done.returncode == 0 and done.stdout.strip() == "armD_long already done" and called is None, (present, done)
# epoch_0005 without epoch_0050 is an unfinished 50-cycle run, not a done one: with a checkpoint it resumes
done, called = run("armD_long.sh", [STOCK, EPOCH5_LONG, CHECKPOINT_LONG])
assert done.returncode == 0 and called["file"] == FROZEN_LONG_RESUME and "already done" not in done.stdout, (done, called)
# arm D's run dir is none of D-long's business, and D-long's is none of arm D's
done, called = run("armD_long.sh", [STOCK, DONE_D, CHECKPOINT_D])
assert done.returncode == 0 and called["file"] == FROZEN_LONG_START, (done, called)
done, called = run("armD.sh", [STOCK, DONE_LONG, CHECKPOINT_LONG])
assert done.returncode == 0 and called["file"] == FROZEN_START, (done, called)
# a standalone script: the chain driver runs what chain/phases.txt lists, so it must not list armD_long
phases = [line.split("#")[0].strip() for line in open(os.path.join(REPO, "chain", "phases.txt"))]
assert "armD" in phases and "armD_long" not in phases, phases
# end to end through the real start / resume scripts: what main_ark.py receives
(fresh, fresh_wrapper) = real_run("chain/armD_long.sh", files=[STOCK])
(interrupted, resume_wrapper) = real_run("chain/armD_long.sh", files=[STOCK, CHECKPOINT_LONG])
assert fresh.returncode == 0 == interrupted.returncode, (fresh.stderr, interrupted.stderr)
fresh_argv = main_argv(fresh_wrapper)
assert "--resume" not in fresh_argv and main_argv(resume_wrapper) == with_resume(fresh_argv)
assert fresh_argv[fresh_argv.index("--exp_name") + 1] == "all4_cyclic_dinov3_frozen_long" and fresh_argv[fresh_argv.index("--eval_every") + 1] == "5"
assert fresh_argv[-4:] == ["--pretrain_epochs", "50", "--warmup-epochs", "2"], fresh_argv
only_resume_differs(fresh_argv, main_argv(resume_wrapper))
# the sbatch wrapper: one card, and the phase script owns the ARK_* recipe
wrapper_path = os.path.join(REPO, "run_d1_8dl_gaudi.sh")
subprocess.run(["bash", "-n", wrapper_path], check=True)
assert os.access(wrapper_path, os.X_OK)
wrapper_text = open(wrapper_path).read()
assert re.findall(r"^#SBATCH (.+)$", wrapper_text, re.M) == ["--job-name=d1-8dl-g1", "-A grp_jliang12", "-p gaudi", "-q public", "-t 3-00:00:00",
                                                           "-N 1", "--gres=gpu:hl225:1", "-c 18", "--mem=64G"]
wrapper_code = [line for line in wrapper_text.splitlines() if line and not line.startswith("#")]
assert wrapper_code[-1] == "exec ./chain/armD_long.sh" and "ARK_" not in "\n".join(wrapper_code), wrapper_code
print("OK (f) armD_long: fresh -> `{}`; run dir + checkpoint -> `{}`, both with armD.sh's ARK_* env except ARK_EVAL_EVERY=5; done (epoch_0050/student_cycle_0050_*.pth) "
      "skips first, epoch_0005 alone does not; empty/missing weights or a run dir without a *.pth.tar -> exit 1; exit statuses pass through; "
      "not in chain/phases.txt; run_d1_8dl_gaudi.sh: 1 card, no ARK_* exports, ends in exec ./chain/armD_long.sh".format(long_start_command, long_resume_command))
print("   chain/armD_long.sh (interrupted run) -> <repo>/gaudi.sh " + " ".join(resume_wrapper))

# (g) sweep: wave 1 of the Step VIII fine-tuning sweep -- ONE parameterized start / resume pair, the table sweep/wave1.tsv and the standalone
# chain/sweep_run.sh (not a chain phase) that turns a row of the table into the ARK_* env plus the pair's arguments -----------------------------
TABLE = "sweep/wave1.tsv"
COLUMNS = "id opt lr wd layer_decay patch_embed_mult warmup cycles model init weights normalization seed eval_every eval_cycle0 input_size".split()
TABLE_TEXT = open(os.path.join(REPO, TABLE)).read()
STAGE_A_RUN = "ssl_dinov3_cxr_main_seed_100"  # chain/main.sh's run; its exports are <run>/export/teacher_backbone_epoch_<epochs counted from 1>.pth
MAIN_EPOCHS, MAIN_DONE = "chain/main_epochs.txt", "chain/done/main"  # what a finished Stage A main phase leaves: chain/gate.py's epoch count, the driver's marker
CENTER = dict(opt="adamw", lr="7e-5", wd="0.01", layer_decay="0.75", patch_embed_mult="1.0", warmup="2", cycles="30", model="vit_base_dinov3",
              init="dinov3", weights="/workspace/" + STOCK, normalization="imagenet",
              seed="100", eval_every="2", eval_cycle0="false", input_size="224")  # what every wave-1 arm ran with: seed 100, evaluated every 2nd cycle, no cycle 0, 224 px
MOMENTUM = dict(opt="momentum", lr="1e-3", wd="0")  # SGD with main_ark.py's default momentum 0.9
CHANGES = [("S2", {}), ("S4", dict(wd="0")), ("S10", MOMENTUM), ("S11", dict(MOMENTUM, lr="3e-3")), ("S12", dict(MOMENTUM, lr="3e-3", patch_embed_mult="0")),
           ("S1", dict(lr="2.5e-5")), ("S3", dict(lr="2e-4")), ("S5", dict(wd="0.05")), ("S6", dict(layer_decay="0.65")), ("S7", dict(layer_decay="0.9")),
           ("S13", dict(weights="STAGEA_FINAL")),
           ("S16", dict(weights="/workspace/ssl_dinov3_cxr_constlr_seed_100/export/teacher_backbone_epoch_0030.pth")),
           # the input-size arms: the center recipe (30 cycles; S2 is the 224 px control), evaluated every cycle, at 448 and 512 px
           ("S18", dict(eval_every="1", input_size="448")), ("S19", dict(eval_every="1", input_size="512"))]  # file order = submission order
IMG_RESIZE = {"224": "256", "448": "512", "512": "585", "320": "366"}  # --img_resize = round(input_size * 8 / 7), worked out by hand (320: the synthetic row below)
ARMS = [dict(CENTER, id=sweep_id, **change) for sweep_id, change in CHANGES]
table = [line.split("\t") for line in TABLE_TEXT.split("\n")]
assert table[-1] == [""] and table[0] == COLUMNS, table[:1]  # the file ends in a newline and its header names the columns
assert all(len(row) == len(COLUMNS) and all(row) for row in table[1:-1])  # every row parses: 16 non-empty tab-separated fields
assert [dict(zip(COLUMNS, row)) for row in table[1:-1]] == ARMS, "sweep/wave1.tsv is not the wave-1 definition"


def host_path(path):
    return path[len("/workspace/"):]  # the same file as seen from the repo root on the host


def sweep_dir(sweep_id, seed="100"):
    return "all4_cyclic_dinov3_sweep_{}_seed_{}".format(sweep_id, seed)


def sweep_checkpoint(sweep_id, cycle=10, seed="100"):
    return "{0}/models/checkpoints/cycle_{1:04d}_{0}.pth.tar".format(sweep_dir(sweep_id, seed), cycle)


def sweep_student(sweep_id, cycle, seed="100"):
    return "{0}/models/weights/epoch_{1:04d}/student_cycle_{1:04d}_{0}.pth".format(sweep_dir(sweep_id, seed), cycle)


def sweep_args(arm):  # what chain/sweep_run.sh hands the start / resume script: the id, the row's seed, container, then the row's hyperparameters
    return [arm["id"], arm["seed"], "gaudi", arm["opt"], arm["wd"], arm["layer_decay"], arm["patch_embed_mult"], arm["warmup"], arm["cycles"], arm["normalization"]]


def sweep_initial(sweep_id, seed="100", parts=("manifest", "student", "teacher")):
    """The files an eval_cycle0 run has in models/weights/epoch_0000 once cycle 0 is evaluated (the engine writes the manifest, then the student, then the teacher): `parts` of them."""
    run = sweep_dir(sweep_id, seed)
    names = {"manifest": "manifest.json", "student": "student_cycle_0000_{}.pth".format(run), "teacher": "teacher_cycle_0000_{}.pth".format(run)}
    return [("{}/models/weights/epoch_0000/{}".format(run, names[part]), "{}") for part in parts]


def stage_a_export(epochs):  # the Stage A main run's export after `epochs` epochs, as the container sees it
    return "/workspace/{}/export/teacher_backbone_epoch_{:04d}.pth".format(STAGE_A_RUN, int(epochs))


def resolved_weights(arm, epochs="30"):  # the weights the phase script settles on: the row's path, or STAGEA_FINAL resolved with main_epochs.txt = `epochs`
    return stage_a_export(epochs) if arm["weights"] == "STAGEA_FINAL" else arm["weights"]


def sweep_env(arm, epochs="30"):  # the ARK_* env it exports: the nine names of the other phase scripts (evaluate every N cycles is the row's), ARK_EVAL_CYCLE0 and the input size, the row's
    return dict(wrapper_env, ARK_LR=arm["lr"], ARK_EVAL_EVERY=arm["eval_every"], ARK_EVAL_CYCLE0=arm["eval_cycle0"], ARK_WARMUP_EPOCHS=arm["warmup"],
                ARK_MODEL=arm["model"], ARK_INIT=arm["init"], ARK_PRETRAINED_WEIGHTS=resolved_weights(arm, epochs),
                ARK_INPUT_SIZE=arm["input_size"], ARK_IMG_RESIZE=IMG_RESIZE[arm["input_size"]])


def arm_files(arm, epochs="30", table_text=TABLE_TEXT):
    """The table and the arm's weights file; for STAGEA_FINAL also what a finished Stage A main phase leaves: main_epochs.txt = `epochs`, the done marker."""
    files = [(TABLE, table_text), host_path(resolved_weights(arm, epochs))]
    return files + ([(MAIN_EPOCHS, "{}\n".format(epochs)), (MAIN_DONE, "")] if arm["weights"] == "STAGEA_FINAL" else [])


def edited(fn, sweep_id="S2"):
    """TABLE_TEXT with the row of `sweep_id` replaced by fn(its fields): a list (joined by tabs) or a raw string."""
    lines = TABLE_TEXT.split("\n")
    k = [line.split("\t")[0] for line in lines].index(sweep_id)
    new = fn(lines[k].split("\t"))
    lines[k] = new if isinstance(new, str) else "\t".join(new)
    return "\n".join(lines)


def with_field(column, value):
    k = COLUMNS.index(column)
    return lambda row: row[:k] + [value] + row[k + 1:]


def launched(done, called, script):  # exit 0 with exactly this start / resume stub called (a refused run has called = None: report it, don't crash on it)
    return done.returncode == 0 and called is not None and called["file"] == script


S2 = ARMS[0]
T = [(TABLE, TABLE_TEXT)]
assert sweep_env(S2) == dict(armd_env, ARK_EVAL_EVERY="2", ARK_EVAL_CYCLE0="false", ARK_INPUT_SIZE="224", ARK_IMG_RESIZE="256")  # the center is arm D's / D-long's stock-DINOv3 ARK_* recipe, evaluated every 2nd cycle, no cycle 0, 224 / 256
for sweep_id, crop, resize in (("S2", "224", "256"), ("S18", "448", "512"), ("S19", "512", "585")):  # what the input-size arms export
    arm = ARMS[[a["id"] for a in ARMS].index(sweep_id)]
    done, called = run("sweep_run.sh", arm_files(arm), args=[sweep_id])
    assert done.returncode == 0 and (called["env"]["ARK_INPUT_SIZE"], called["env"]["ARK_IMG_RESIZE"]) == (crop, resize), (sweep_id, done, called)
# every arm: a fresh start calls the start script, a run dir with a checkpoint the resume script, both with that arm's arguments and ARK_* env
for arm in ARMS:
    sweep_id = arm["id"]
    done, called = run("sweep_run.sh", arm_files(arm), args=[sweep_id])
    assert done.returncode == 0 and done.stderr == "", (sweep_id, done.stderr)
    assert called["argv0"] == "./" + SWEEP_START and called["file"] == SWEEP_START and called["count"] == 1, called
    assert called["args"] == sweep_args(arm) and called["env"] == sweep_env(arm), (sweep_id, called)
    done, called = run("sweep_run.sh", arm_files(arm) + [sweep_checkpoint(sweep_id, seed=arm["seed"])], args=[sweep_id])
    assert done.returncode == 0 and done.stderr == "", (sweep_id, done.stderr)
    assert called["argv0"] == "./" + SWEEP_RESUME and called["file"] == SWEEP_RESUME and called["count"] == 1, called
    assert called["args"] == sweep_args(arm) and called["env"] == sweep_env(arm), (sweep_id, called)
    assert "sweep_run {}: resuming ./{} from its latest checkpoint".format(sweep_id, sweep_dir(sweep_id, arm["seed"])) in done.stdout, done.stdout
S2_RUN, S2_FILES = sweep_dir("S2"), arm_files(S2)
S2_NO_CHECKPOINT = "./{0} exists but holds no resumable checkpoint ({0}/models/checkpoints/*.pth.tar)".format(S2_RUN)
# the start / resume script's exit status comes through
assert run("sweep_run.sh", S2_FILES, stub_exit=7, args=["S2"])[0].returncode == 7
assert run("sweep_run.sh", S2_FILES + [sweep_checkpoint("S2")], stub_exit=7, args=["S2"])[0].returncode == 7
# the arm's weights are needed to start and to resume: missing or empty -> exit 1 before anything is called
for broken in (T, T + [(STOCK, "")]):
    for with_checkpoint in ([], [sweep_checkpoint("S2")]):
        done, called = run("sweep_run.sh", broken + with_checkpoint, args=["S2"])
        assert done.returncode == 1 and called is None and "weights file ./" + STOCK + " is missing or empty" in done.stderr, (broken, with_checkpoint, done)
# S13's weights are STAGEA_FINAL: the Stage A main run's final export, <chain/main_epochs.txt zero-padded to 4 digits, read as decimal>. It only exists once the
# main phase is done, so chain/main_epochs.txt AND the chain driver's marker chain/done/main have to be there (exit 1 with a message otherwise)
S13 = next(arm for arm in ARMS if arm["id"] == "S13")
export_30, export_40 = host_path(stage_a_export(30)), host_path(stage_a_export(40))
for text, epochs in (("30", 30), ("30\n", 30), ("40\n", 40), (" 40 \n\n", 40), ("030\n", 30), ("0008", 8)):  # however the file is spelled, read as decimal
    exports = [export_30, export_40, host_path(stage_a_export(8))]  # all there: only the epoch count decides which one is used
    done, called = run("sweep_run.sh", T + [(MAIN_EPOCHS, text), (MAIN_DONE, "")] + exports, args=["S13"])
    assert launched(done, called, SWEEP_START) and done.stderr == "" and called["args"] == sweep_args(S13), (text, done, called)
    assert called["env"]["ARK_PRETRAINED_WEIGHTS"] == "/workspace/ssl_dinov3_cxr_main_seed_100/export/teacher_backbone_epoch_{:04d}.pth".format(epochs), (text, called)
    assert called["env"] == sweep_env(S13, epochs), (text, called)
done, called = run("sweep_run.sh", T + [(MAIN_EPOCHS, "40\n"), (MAIN_DONE, ""), export_30, export_40, sweep_checkpoint("S13")], args=["S13"])  # an interrupted S13 resumes
assert launched(done, called, SWEEP_RESUME) and called["env"] == sweep_env(S13, 40), (done, called)
S13_REFUSED = {  # label -> (files, what the message says)
    "no epochs file": (T + [(MAIN_DONE, ""), export_30], "chain/main_epochs.txt (the main run's epoch count, written by chain/gate.py) is missing or empty"),
    "empty epochs file": (T + [(MAIN_DONE, ""), (MAIN_EPOCHS, ""), export_30], "chain/main_epochs.txt (the main run's epoch count, written by chain/gate.py) is missing or empty"),
    "no done marker": (T + [(MAIN_EPOCHS, "30\n"), export_30], "the main phase is not done: chain/done/main is missing"),
    "another phase's marker": (T + [(MAIN_EPOCHS, "30\n"), ("chain/done/gate", ""), ("chain/done/probe_b", ""), export_30], "chain/done/main is missing"),
    "neither": (T + [export_30], "chain/done/main is missing"),
    "epochs a word": (T + [(MAIN_EPOCHS, "thirty\n"), (MAIN_DONE, ""), export_30], "chain/main_epochs.txt holds 'thirty'"),
    "epochs a fraction": (T + [(MAIN_EPOCHS, "30.5\n"), (MAIN_DONE, ""), export_30], "chain/main_epochs.txt holds '30.5'"),
    "epochs too long": (T + [(MAIN_EPOCHS, "12345\n"), (MAIN_DONE, ""), export_30], "chain/main_epochs.txt holds '12345'"),
    "epochs zero": (T + [(MAIN_EPOCHS, "0\n"), (MAIN_DONE, ""), export_30], "chain/main_epochs.txt holds 0, which is not a positive epoch count"),
    "no export": (T + [(MAIN_EPOCHS, "30\n"), (MAIN_DONE, "")], "weights file ./" + export_30 + " is missing or empty"),
    "empty export": (T + [(MAIN_EPOCHS, "30\n"), (MAIN_DONE, ""), (export_30, "")], "weights file ./" + export_30 + " is missing or empty"),
    "export of another epoch count": (T + [(MAIN_EPOCHS, "40\n"), (MAIN_DONE, ""), export_30], "weights file ./" + export_40 + " is missing or empty"),
}
for label, (files, message) in S13_REFUSED.items():
    done, called = run("sweep_run.sh", files, args=["S13"])
    assert done.returncode == 1 and called is None and message in done.stderr and "S13" in done.stderr, (label, done)
# done wins over all of it: a finished S13 needs neither the main phase's files nor its export
done, called = run("sweep_run.sh", T + [sweep_student("S13", 30), sweep_checkpoint("S13", 30)], args=["S13"])
assert done.returncode == 0 and done.stdout.strip() == "sweep_run S13 already done" and called is None, done
# a run dir without a resumable checkpoint: an error that names the dir, nothing invoked, nothing deleted (run() checks the tree)
for partial in ([S2_RUN + "/"],                                                                          # just the dir
                [S2_RUN + "/models/checkpoints/"],                                                       # an empty checkpoint dir
                [sweep_student("S2", 10)],                                                               # cycle 10's student only: not done
                [sweep_student("S2", 29)],                                                               # cycle 29's student only
                [sweep_student("S2", 30)],                                                               # the final student, but no checkpoint at all
                [S2_RUN + "/models/weights/epoch_0030/teacher_cycle_0030_" + S2_RUN + ".pth"],           # cycle 30's teacher only
                [sweep_student("S2", 30), S2_RUN + "/models/checkpoints/cycle_0030_" + S2_RUN + ".pth"],  # the final student, a checkpoint that is not a .pth.tar
                [S2_RUN + "/models/checkpoints/cycle_0010_" + S2_RUN + ".pth"],                          # not a .pth.tar
                [S2_RUN + "/models/weights/epoch_0000/"],                                                # initial weights killed before the manifest: an empty epoch_0000
                sweep_initial("S2", parts=("manifest",)),                                                # a manifest only
                sweep_initial("S2", parts=("manifest", "student")),                                      # killed between the student and the teacher
                sweep_initial("S2", parts=("manifest", "teacher")),
                sweep_initial("S2", parts=("student", "teacher")),                                       # no manifest
                [(sweep_initial("S2")[0][0], "")] + sweep_initial("S2", parts=("student", "teacher")),   # an empty manifest
                [(S2_RUN + "/models/weights/epoch_0000", "x")],                                          # a file, not a directory
                [S2_RUN + "/models/weights/epoch_0001/manifest.json"]):                                  # another epoch's manifest is not the initial weights
    done, called = run("sweep_run.sh", S2_FILES + partial, args=["S2"])
    assert done.returncode == 1 and called is None and S2_NO_CHECKPOINT in done.stderr, (partial, done)
    assert "neither resuming nor starting over it, nothing deleted" in done.stderr and "no complete saved initial weights" in done.stderr
# the engine saves the initial weights (eval_cycle0 true) once cycle 0 is evaluated and resumes from them when no cycle has finished yet: complete epoch_0000 and no
# checkpoint -> the resume script with the arm's own arguments; a checkpoint as well -> the latest checkpoint decides, as before
done, called = run("sweep_run.sh", S2_FILES + sweep_initial("S2"), args=["S2"])
assert launched(done, called, SWEEP_RESUME) and called["args"] == sweep_args(S2) and called["env"] == sweep_env(S2) and done.stderr == "", (done, called)
assert "sweep_run S2: resuming ./{} from its saved initial weights ({}/models/weights/epoch_0000".format(S2_RUN, S2_RUN) in done.stdout and "latest checkpoint" not in done.stdout, done.stdout
assert run("sweep_run.sh", S2_FILES + sweep_initial("S2"), stub_exit=7, args=["S2"])[0].returncode == 7  # the resume script's status comes through
done, called = run("sweep_run.sh", S2_FILES + sweep_initial("S2") + [sweep_checkpoint("S2", 10)], args=["S2"])
assert launched(done, called, SWEEP_RESUME) and "from its latest checkpoint" in done.stdout and "initial weights" not in done.stdout, (done, called)
for broken in (T, T + [(STOCK, "")]):  # the weights guard still comes first: the resume builds the model from the arm's weights too
    done, called = run("sweep_run.sh", broken + sweep_initial("S2"), args=["S2"])
    assert done.returncode == 1 and called is None and "weights file ./" + STOCK + " is missing or empty" in done.stderr, (broken, done)
for other_dir in (sweep_initial("S4"), sweep_initial("S2", "7")):  # another arm's initial weights, or this arm's under another seed, are none of its business: it starts
    done, called = run("sweep_run.sh", S2_FILES + other_dir, args=["S2"])
    assert launched(done, called, SWEEP_START), (other_dir[0], done, called)
done, called = run("sweep_run.sh", T + [(MAIN_EPOCHS, "30\n"), (MAIN_DONE, ""), export_30] + sweep_initial("S13"), args=["S13"])  # STAGEA_FINAL's checks hold for it too
assert launched(done, called, SWEEP_RESUME) and called["env"] == sweep_env(S13, 30) and "saved initial weights" in done.stdout, (done, called)
done, called = run("sweep_run.sh", T + [sweep_student("S2", 30), sweep_checkpoint("S2", 30)] + sweep_initial("S2"), args=["S2"])  # done wins over everything
assert done.returncode == 0 and done.stdout.strip() == "sweep_run S2 already done" and called is None, done
# done = the final cycle's student weights AND its checkpoint, the last thing engine.py writes in a cycle (manifest, student, teacher, then the checkpoint);
# it wins over everything: no weights check, no resume, nothing launched
final = [sweep_student("S2", 30), sweep_checkpoint("S2", 30)]
for present in (final, final + [sweep_checkpoint("S2", 10), sweep_checkpoint("S2", 20)], S2_FILES[1:] + final):
    done, called = run("sweep_run.sh", T + present, args=["S2"])
    assert done.returncode == 0 and done.stdout.strip() == "sweep_run S2 already done" and called is None, (present, done)
# either one alone is not done, and an earlier cycle's pair is an unfinished run: each resumes (the final student without its checkpoint is a kill between
# the two writes; the final checkpoint without its student cannot be written by engine.py, and its resume would stop at once: "Cannot continue training after cycle 30")
for present in ([sweep_student("S2", 30), sweep_checkpoint("S2", 10)], [sweep_student("S2", 30), sweep_checkpoint("S2", 29)], [sweep_checkpoint("S2", 30)],
                [sweep_student("S2", 29), sweep_checkpoint("S2", 30)], [sweep_student("S2", 10), sweep_checkpoint("S2", 10)],
                [sweep_student("S2", 29), sweep_checkpoint("S2", 29)]):
    done, called = run("sweep_run.sh", S2_FILES + present, args=["S2"])
    assert launched(done, called, SWEEP_RESUME) and "already done" not in done.stdout, (present, done, called)
# the rule rests on engine.py's write order at the end of a cycle: student, teacher (_save_cycle_weights writes the manifest, then the two state dicts in that
# order), then the cycle checkpoint, and no other write after it in that cycle
engine_text = open(os.path.join(REPO, "base", "engine.py")).read()
saver = engine_text[engine_text.index("def _save_cycle_weights("):]
saver = saver[:saver.index("\n\n\ndef ")]
assert re.search(r'for role, network in \(\("student", student_model\), \("teacher", teacher\)\):', saver) and saver.count("torch.save(") == 1, saver
write_order = [re.search(pattern, engine_text).start() for pattern in (r"teacher_state_dict = _save_cycle_weights\(", r"save_checkpoint\(\s*checkpoint,")]
assert write_order == sorted(write_order), write_order
after = engine_text[write_order[-1]:engine_text.index("accelerator.barrier()", write_order[-1])]
assert not re.search(r"torch\.save|open\(|\.write|json\.dump|\bsave_checkpoint\(\s*(?!checkpoint,)", after[len("save_checkpoint("):]), after
# one arm's run dir is none of another arm's business: not S4's (done), not S10's for S1 (S1's id is a prefix of it), nor the other way round
for sweep_id, other in (("S2", "S4"), ("S4", "S2"), ("S1", "S10"), ("S10", "S1")):
    arm = ARMS[[a["id"] for a in ARMS].index(sweep_id)]
    done, called = run("sweep_run.sh", arm_files(arm) + [sweep_student(other, 30), sweep_checkpoint(other, 30)], args=[sweep_id])
    assert launched(done, called, SWEEP_START) and called["args"][0] == sweep_id, (sweep_id, other, done, called)
# the zero-padded done marker follows the row's cycle count, however it is spelled: a leading 0 is octal to bash $(( )) and to main_ark.py's optparse
for spelled, cycles, other in (("20", 20, 30), ("030", 30, 24), ("08", 8, 10)):  # `other`: the cycle a wrong (octal) reading would look for
    table_text = edited(with_field("cycles", spelled))
    done, called = run("sweep_run.sh", [(TABLE, table_text), STOCK, sweep_student("S2", cycles), sweep_checkpoint("S2", cycles)], args=["S2"])
    assert done.returncode == 0 and done.stdout.strip() == "sweep_run S2 already done" and called is None, (spelled, done)
    done, called = run("sweep_run.sh", [(TABLE, table_text), STOCK, sweep_student("S2", other), sweep_checkpoint("S2", other)], args=["S2"])
    assert launched(done, called, SWEEP_RESUME) and called["args"][-2] == str(cycles), (spelled, done, called)  # not done: resumes, decimal
done, called = run("sweep_run.sh", [(TABLE, edited(with_field("warmup", "02"))), STOCK], args=["S2"])
assert called["args"][-3] == "2" and called["env"]["ARK_WARMUP_EPOCHS"] == "2", called
# standalone: the chain driver runs what chain/phases.txt lists, so it must not list sweep_run
phases = [line.split("#")[0].strip() for line in open(os.path.join(REPO, "chain", "phases.txt"))]
assert "sweep_run" not in phases, phases
# what STAGEA_FINAL leans on: the driver's done marker is chain/done/<phase> for the phase "main", and chain/main.sh runs ssl_dinov3_cxr_main_seed_100 for
# `--epochs $EPOCHS` read from chain/main_epochs.txt
driver_text, main_text = open(os.path.join(REPO, "run_d1_8_chain_gaudi.sh")).read(), open(os.path.join(REPO, "chain", "main.sh")).read()
assert re.search(r"^CHAIN=chain$", driver_text, re.M) and 'touch "$CHAIN/done/$phase"' in driver_text and "main" in phases and MAIN_DONE == "chain/done/main"
assert 'ssl_launch {} --lr "$LR" --no_lr_scaling --epochs "$EPOCHS"'.format(STAGE_A_RUN) in main_text and "chain/main_epochs.txt" in main_text
# usage, unknown ids and malformed tables / rows: exit 2 before anything is called, the tree untouched, the message names the problem
for args in ([], ["S2", "S4"]):
    done, called = run("sweep_run.sh", T + [STOCK], args=args)
    assert done.returncode == 2 and called is None and "Usage:" in done.stderr, (args, done)
done, called = run("sweep_run.sh", T + [STOCK], args=["S99"])
assert done.returncode == 2 and called is None, done
assert "sweep id 'S99' is not a row of sweep/wave1.tsv (rows: S2 S4 S10 S11 S12 S1 S3 S5 S6 S7 S13 S16 S18 S19)" in done.stderr, done.stderr
for weird in ("", "../S2", "S2;ls", "S 2", "-S2", "*", "S", "s2", "S02", "id"):  # not ids, not rows, or the header's first cell
    done, called = run("sweep_run.sh", T + [STOCK], args=[weird])
    assert done.returncode == 2 and called is None, (weird, done)
done, called = run("sweep_run.sh", [STOCK], args=["S2"])
assert done.returncode == 2 and called is None and "sweep/wave1.tsv is missing or unreadable" in done.stderr, done
row_s2 = TABLE_TEXT.split("\n")[1]
MALFORMED = {  # label -> (table text, what the message says)
    "empty field": (edited(with_field("lr", "")), "needs exactly 16 non-empty tab-separated fields"),
    "missing field": (edited(lambda row: row[:-1]), "needs exactly 16 non-empty tab-separated fields"),
    "extra field": (edited(lambda row: row + ["x"]), "needs exactly 16 non-empty tab-separated fields"),
    "trailing tab": (edited(lambda row: "\t".join(row) + "\t"), "needs exactly 16 non-empty tab-separated fields"),
    "doubled tab": (edited(lambda row: "\t".join(row[:2]) + "\t\t" + "\t".join(row[2:])), "needs exactly 16 non-empty tab-separated fields"),
    "sgd": (edited(with_field("opt", "sgd")), "opt 'sgd' is not adamw or momentum"),
    "AdamW": (edited(with_field("opt", "AdamW")), "is not adamw or momentum"),
    "lr word": (edited(with_field("lr", "fast")), "lr 'fast' is not a number"),
    "lr junk": (edited(with_field("lr", "7e-5x")), "lr '7e-5x' is not a number"),
    "lr negative": (edited(with_field("lr", "-7e-5")), "is not a number"),
    "lr zero": (edited(with_field("lr", "0")), "lr '0' is not positive"),
    "wd negative": (edited(with_field("wd", "-0.01")), "wd '-0.01' is not a number"),
    "layer decay comma": (edited(with_field("layer_decay", "0,75")), "is not a number"),
    "mult word": (edited(with_field("patch_embed_mult", "none")), "is not a number"),
    "warmup fraction": (edited(with_field("warmup", "2.5")), "warmup '2.5' is not a whole number"),
    "cycles word": (edited(with_field("cycles", "thirty")), "cycles 'thirty' is not a whole number"),
    "cycles too long": (edited(with_field("cycles", "12345")), "is not a whole number of at most 4 digits"),
    "model space": (edited(with_field("model", "vit base")), "is not a model name"),
    "init slash": (edited(with_field("init", "a/b")), "is not an init name"),
    "weights on the host": (edited(with_field("weights", "weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")), "is not STAGEA_FINAL or an in-container path under /workspace/"),
    "weights elsewhere": (edited(with_field("weights", "/data/x.pth")), "is not STAGEA_FINAL or an in-container path under /workspace/"),
    "weights token typo": (edited(with_field("weights", "stagea_final")), "weights 'stagea_final' is not STAGEA_FINAL or an in-container path"),
    "normalization chestx-ray": (edited(with_field("normalization", "chestx-ray")), "normalization 'chestx-ray' is not supported: only imagenet is accepted"),
    "normalization none": (edited(with_field("normalization", "none")), "normalization 'none' is not supported: only imagenet is accepted"),
    "normalization case": (edited(with_field("normalization", "ImageNet")), "normalization 'ImageNet' is not supported: only imagenet is accepted"),
    "normalization typo": (edited(with_field("normalization", "chestxray")), "normalization 'chestxray' is not supported: only imagenet is accepted"),
    "warmup 1": (edited(with_field("warmup", "1")), "warmup 1 is below 2 (WarmupCosineScheduler needs a warmup of at least 2 cycles)"),
    "warmup 0": (edited(with_field("warmup", "0")), "warmup 0 is below 2"),
    "cycles 2": (edited(with_field("cycles", "2")), "cycles 2 is below 3 (WarmupCosineScheduler needs at least 3 cycles)"),
    "cycles 1": (edited(with_field("cycles", "1")), "cycles 1 is below 3"),
    "cycles 0": (edited(with_field("cycles", "0")), "cycles 0 is below 3"),
    "warmup equals cycles": (edited(lambda row: row[:6] + ["30", "30"] + row[8:]), "warmup 30 is not below cycles 30"),
    "warmup 3 of 3 cycles": (edited(lambda row: row[:6] + ["3", "3"] + row[8:]), "warmup 3 is not below cycles 3"),
    "warmup above cycles": (edited(lambda row: row[:6] + ["31", "30"] + row[8:]), "warmup 31 is not below cycles 30"),
    "lr below the fixed warmup-lr / min-lr": (edited(with_field("lr", "9e-7")), "lr '9e-7' is below 1e-6, the start script's fixed --warmup-lr and --min-lr"),
    "swapped header columns": (TABLE_TEXT.replace("lr\twd", "wd\tlr", 1), "does not start with the tab-separated header"),
    "no header": ("\n".join(TABLE_TEXT.split("\n")[1:]), "does not start with the tab-separated header"),
    "CRLF line ends": (TABLE_TEXT.replace("\n", "\r\n"), "does not start with the tab-separated header"),
    "duplicate row": (TABLE_TEXT.rstrip("\n") + "\n" + row_s2 + "\n", "sweep id 'S2' is listed more than once in sweep/wave1.tsv"),
    "header only": (TABLE_TEXT.split("\n")[0] + "\n", "sweep id 'S2' is not a row of sweep/wave1.tsv"),
    "empty file": ("", "does not start with the tab-separated header"),
    "seed word": (edited(with_field("seed", "x")), "seed 'x' is not a whole number of at most 9 digits"),
    "seed negative": (edited(with_field("seed", "-1")), "seed '-1' is not a whole number of at most 9 digits"),
    "seed fraction": (edited(with_field("seed", "7.5")), "seed '7.5' is not a whole number of at most 9 digits"),
    "seed too long": (edited(with_field("seed", "1234567890")), "seed '1234567890' is not a whole number of at most 9 digits"),
    "eval_every word": (edited(with_field("eval_every", "often")), "eval_every 'often' is not a whole number of at most 4 digits"),
    "eval_every negative": (edited(with_field("eval_every", "-2")), "eval_every '-2' is not a whole number of at most 4 digits"),
    "eval_every fraction": (edited(with_field("eval_every", "1.5")), "eval_every '1.5' is not a whole number of at most 4 digits"),
    "eval_every too long": (edited(with_field("eval_every", "12345")), "eval_every '12345' is not a whole number of at most 4 digits"),
    "eval_every 0": (edited(with_field("eval_every", "0")), "eval_every 0 is below 1"),
    "eval_every 000": (edited(with_field("eval_every", "000")), "eval_every 0 is below 1"),
    "eval_cycle0 True": (edited(with_field("eval_cycle0", "True")), "eval_cycle0 'True' is not true or false"),
    "eval_cycle0 yes": (edited(with_field("eval_cycle0", "yes")), "eval_cycle0 'yes' is not true or false"),
    "eval_cycle0 1": (edited(with_field("eval_cycle0", "1")), "eval_cycle0 '1' is not true or false"),
    "no seed column": (TABLE_TEXT.replace("\tseed\t", "\tsd\t", 1), "the header has 0 columns named 'seed'"),
    "no eval_every column": (TABLE_TEXT.replace("\teval_every\t", "\tevery\t", 1), "the header has 0 columns named 'eval_every'"),
    "no eval_cycle0 column": (TABLE_TEXT.replace("\teval_cycle0\t", "\tcycle0\t", 1), "the header has 0 columns named 'eval_cycle0'"),
    "no input_size column": (TABLE_TEXT.replace("\tinput_size\n", "\tsize\n", 1), "the header has 0 columns named 'input_size'"),
    "the fifteen-column table": ("\n".join("\t".join(line.split("\t")[:15]) for line in TABLE_TEXT.split("\n")), "the header has 0 columns named 'input_size'"),
    "input_size 450": (edited(with_field("input_size", "450")), "input_size 450 is not a multiple of 16 (the ViT-B/16 patch size)"),
    "input_size 96": (edited(with_field("input_size", "96")), "input_size '96' is not a whole number of 3 or 4 digits (pixels)"),
    "input_size 096": (edited(with_field("input_size", "096")), "input_size 96 is below 112"),
    "input_size abc": (edited(with_field("input_size", "abc")), "input_size 'abc' is not a whole number of 3 or 4 digits"),
    "input_size fraction": (edited(with_field("input_size", "448.0")), "input_size '448.0' is not a whole number of 3 or 4 digits"),
    "input_size too long": (edited(with_field("input_size", "12345")), "input_size '12345' is not a whole number of 3 or 4 digits"),
    "input_size negative": (edited(with_field("input_size", "-448")), "input_size '-448' is not a whole number of 3 or 4 digits"),
    "seed column twice": (TABLE_TEXT.replace("\teval_every\t", "\tseed\t", 1), "the header has 2 columns named 'seed'"),
    "the twelve-column table": ("\n".join("\t".join(line.split("\t")[:12]) for line in TABLE_TEXT.split("\n")), "the header has 0 columns named 'seed'"),
    "empty column name in the header": (TABLE_TEXT.replace("\tseed\t", "\t\t", 1), "the header row has an empty column name"),
}
for label, (table_text, message) in MALFORMED.items():
    done, called = run("sweep_run.sh", [(TABLE, table_text), STOCK], args=["S2"])
    assert done.returncode == 2 and called is None and message in done.stderr and done.stderr.startswith("ERROR: "), (label, done)
done, called = run("sweep_run.sh", [(TABLE, MALFORMED["normalization chestx-ray"][0]), STOCK], args=["S2"])  # the refusal says why
assert "--train_augment true" in done.stderr and "hard-code ImageNet mean/std for training" in done.stderr, done.stderr
# seed, eval_every and eval_cycle0 are read by header NAME: in another order, with a column of another name between them, and with S2's own values changed
# (the other rows keep theirs), the arms still get their own numbers; the first twelve columns stay positional
cells = [line.split("\t") for line in TABLE_TEXT.split("\n")[:-1]]
shuffled = "\n".join(["\t".join(cells[0][:12] + ["input_size", "eval_cycle0", "note", "eval_every", "seed"])] + [
    "\t".join(row[:12] + ([row[15], row[14], "n", row[13], row[12]] if row[0] != "S2" else ["448", "true", "n", "1", "7"])) for row in cells[1:]]) + "\n"
done, called = run("sweep_run.sh", [(TABLE, shuffled), STOCK], args=["S2"])
assert launched(done, called, SWEEP_START) and called["args"][:2] == ["S2", "7"], (done, called)
assert called["env"]["ARK_EVAL_EVERY"] == "1" and called["env"]["ARK_EVAL_CYCLE0"] == "true", called["env"]
assert (called["env"]["ARK_INPUT_SIZE"], called["env"]["ARK_IMG_RESIZE"]) == ("448", "512"), called["env"]
done, called = run("sweep_run.sh", [(TABLE, shuffled), STOCK], args=["S4"])
assert launched(done, called, SWEEP_START) and called["args"][:2] == ["S4", "100"], (done, called)
assert called["env"]["ARK_EVAL_EVERY"] == "2" and called["env"]["ARK_EVAL_CYCLE0"] == "false", called["env"]
assert (called["env"]["ARK_INPUT_SIZE"], called["env"]["ARK_IMG_RESIZE"]) == ("224", "256"), called["env"]
done, called = run("sweep_run.sh", [(TABLE, edited(with_field("input_size", "0448"))), STOCK], args=["S2"])  # a leading zero is decimal here too
assert launched(done, called, SWEEP_START) and (called["env"]["ARK_INPUT_SIZE"], called["env"]["ARK_IMG_RESIZE"]) == ("448", "512"), called["env"]
# an extra column after the four is ignored too
done, called = run("sweep_run.sh", [(TABLE, "\n".join(line + ("\tnote" if k == 0 else "\tx") for k, line in enumerate(TABLE_TEXT.split("\n")[:-1])) + "\n"), STOCK], args=["S2"])
assert launched(done, called, SWEEP_START) and called["args"][:2] == ["S2", "100"], (done, called)
# a second seed of the center arm (the S2b row of sweep/wave1.tsv's comment): seed 7, evaluated every cycle and at cycle 0. Its run dir is ..._S2b_seed_7: that
# dir is what it resumes and what marks it done, the seed-100 dir of the same id is another run, and S2's seed-100 dir is none of its business
S2B = dict(S2, id="S2b", seed="7", eval_every="1", eval_cycle0="true")
table_s2b = TABLE_TEXT + "\t".join(S2B[column] for column in COLUMNS) + "\n"
assert table_s2b.split("\n")[-2] == "\t".join(["S2b", "adamw", "7e-5", "0.01", "0.75", "1.0", "2", "30", "vit_base_dinov3", "dinov3", "/workspace/" + STOCK,
                                                "imagenet", "7", "1", "true", "224"])
S2B_RUN, S2B_ARGS = sweep_dir("S2b", "7"), ["S2b", "7", "gaudi", "adamw", "0.01", "0.75", "1.0", "2", "30", "imagenet"]
assert S2B_RUN == "all4_cyclic_dinov3_sweep_S2b_seed_7"
done, called = run("sweep_run.sh", [(TABLE, table_s2b), STOCK], args=["S2b"])
assert launched(done, called, SWEEP_START) and called["args"] == S2B_ARGS and done.stderr == "", (done, called)
assert called["env"] == dict(sweep_env(S2), ARK_EVAL_EVERY="1", ARK_EVAL_CYCLE0="true"), called["env"]
done, called = run("sweep_run.sh", [(TABLE, table_s2b), STOCK, sweep_checkpoint("S2b", 10, "7")], args=["S2b"])
assert launched(done, called, SWEEP_RESUME) and called["args"] == S2B_ARGS and "resuming ./" + S2B_RUN + " from its latest checkpoint" in done.stdout, (done, called)
done, called = run("sweep_run.sh", [(TABLE, table_s2b), STOCK, sweep_checkpoint("S2b", 10, "100"), sweep_student("S2b", 30, "100"), sweep_checkpoint("S2b", 30, "100")],
                   args=["S2b"])
assert launched(done, called, SWEEP_START) and called["args"] == S2B_ARGS, (done, called)  # the seed-100 dir (even a finished one) is not this run
done, called = run("sweep_run.sh", [(TABLE, table_s2b), sweep_student("S2b", 30, "7"), sweep_checkpoint("S2b", 30, "7")], args=["S2b"])
assert done.returncode == 0 and done.stdout.strip() == "sweep_run S2b already done" and called is None, done
done, called = run("sweep_run.sh", [(TABLE, table_s2b), STOCK] + sweep_initial("S2b", "7"), args=["S2b"])  # stopped after cycle 0: the seed-7 initial weights
assert launched(done, called, SWEEP_RESUME) and called["args"] == S2B_ARGS and "resuming ./" + S2B_RUN + " from its saved initial weights" in done.stdout, (done, called)
done, called = run("sweep_run.sh", [(TABLE, table_s2b), STOCK] + sweep_initial("S2b", "100"), args=["S2b"])  # the seed-100 ones are another run's
assert launched(done, called, SWEEP_START), (done, called)
done, called = run("sweep_run.sh", [(TABLE, table_s2b), STOCK, sweep_student("S2b", 30, "7"), sweep_checkpoint("S2b", 30, "7")], args=["S2"])
assert launched(done, called, SWEEP_START) and called["args"][:2] == ["S2", "100"], (done, called)  # S2 (seed 100) next to a finished S2b (seed 7)
# a leading zero is decimal here, as it is for the cycle counts (optparse would read 010 as 8): the seed and the run dir are the decimal number
for spelled, seed in (("007", "7"), ("010", "10"), ("0", "0"), ("100", "100")):
    table_text = edited(with_field("seed", spelled))
    done, called = run("sweep_run.sh", [(TABLE, table_text), STOCK], args=["S2"])
    assert launched(done, called, SWEEP_START) and called["args"][:2] == ["S2", seed], (spelled, done, called)
    done, called = run("sweep_run.sh", [(TABLE, table_text), STOCK, sweep_checkpoint("S2", 10, seed)], args=["S2"])
    assert launched(done, called, SWEEP_RESUME) and "resuming ./" + sweep_dir("S2", seed) + " " in done.stdout, (spelled, done, called)
    done, called = run("sweep_run.sh", [(TABLE, table_text), sweep_student("S2", 30, seed), sweep_checkpoint("S2", 30, seed)], args=["S2"])
    assert done.returncode == 0 and done.stdout.strip() == "sweep_run S2 already done" and called is None, (spelled, done)
# the schedule rules are the engine's, exactly: every (cycles, warmup, peak LR) is accepted by chain/sweep_run.sh if and only if WarmupCosineScheduler accepts
# it (with the pair's fixed --warmup-lr and --min-lr of 1e-6); both exit 0 / 2 before anything is created or launched. Without torch only the cases above run.
if main_ark is not None:
    scheduler = sys.modules["engine"].WarmupCosineScheduler

    def engine_accepts(cycles, warmup, lr):
        try:
            scheduler(None, total_cycles=cycles, warmup_cycles=warmup, start_lr=1e-6, peak_lr=float(lr), end_lr=1e-6)
            return True
        except ValueError:
            return False

    grid = [(c, w, "7e-5") for c in (1, 2, 3, 4, 30) for w in (0, 1, 2, 3, 4, 29, 30, 31)]
    grid += [(30, 2, lr) for lr in ("9e-7", "9.99e-7", "1e-6", "1.0e-06", "0.000001", "1.1e-6", "2e-5")]
    accepted = 0
    for cycles, warmup, lr in grid:
        table_text = edited(lambda row: row[:2] + [lr] + row[3:6] + [str(warmup), str(cycles)] + row[8:])
        done, called = run("sweep_run.sh", [(TABLE, table_text), STOCK], args=["S2"])
        assert done.returncode == (0 if engine_accepts(cycles, warmup, lr) else 2), (cycles, warmup, lr, done.returncode, done.stderr)
        assert (called is not None) == (done.returncode == 0), (cycles, warmup, lr)
        accepted += done.returncode == 0
    assert 0 < accepted < len(grid), (accepted, len(grid))  # the grid has both kinds
# a bad row stops its own arm only: S4's row is broken, S2 still starts (and S4 does not)
broken_s4 = edited(with_field("lr", "oops"), "S4")
done, called = run("sweep_run.sh", [(TABLE, broken_s4), STOCK], args=["S2"])
assert launched(done, called, SWEEP_START), (done, called)
done, called = run("sweep_run.sh", [(TABLE, broken_s4), STOCK], args=["S4"])
assert done.returncode == 2 and called is None and "row 'S4' of sweep/wave1.tsv: lr 'oops' is not a number" in done.stderr, done


# end to end through the real common.sh, start / resume scripts and chain/sweep_run.sh: what main_ark.py receives, and what its option parser makes of it
def option_values(argv, flag):
    return [argv[i + 1] for i, token in enumerate(argv) if token == flag]


def parsed(argv):
    """main_ark.py's own option parser on argv (None without torch): every flag in argv has to be a real option, or it exits with usage."""
    if main_ark is None:
        return None
    sys.argv = ["main_ark.py"] + argv
    try:
        return main_ark.get_args_parser()
    except SystemExit:
        raise AssertionError("main_ark.py's option parser rejected: " + " ".join(argv))


start_text, resume_text = open(os.path.join(REPO, SWEEP_START)).read(), open(os.path.join(REPO, SWEEP_RESUME)).read()
assert resume_text != start_text and resume_text == start_text.replace("    false \\\n", "    true \\\n")  # one line: resume=true


def check_sweep_arm(arm, table_text):
    """One row through the real chain/sweep_run.sh, start / resume script and common.sh: what main_ark.py receives for a fresh and for an
    interrupted run, and what its option parser (when torch is importable) makes of it. Returns (the resume run's gaudi.sh argv, parsed start options)."""
    sweep_id = arm["id"]
    files = arm_files(arm, table_text=table_text)
    (fresh, fresh_wrapper) = real_run("chain/sweep_run.sh", [sweep_id], files=files)
    (interrupted, resume_wrapper) = real_run("chain/sweep_run.sh", [sweep_id], files=files + [sweep_checkpoint(sweep_id, seed=arm["seed"])])
    assert fresh.returncode == 0 == interrupted.returncode, (sweep_id, fresh.stderr, interrupted.stderr)
    start_argv, resume_argv = main_argv(fresh_wrapper), main_argv(resume_wrapper)
    assert "--resume" not in start_argv and resume_argv == with_resume(start_argv), sweep_id
    assert resume_wrapper[:-len(resume_argv)] == fresh_wrapper[:-len(start_argv)]  # same launcher
    (initial, initial_wrapper) = real_run("chain/sweep_run.sh", [sweep_id], files=files + sweep_initial(sweep_id, arm["seed"]))  # stopped after cycle 0, before any checkpoint
    assert initial.returncode == 0 and "from its saved initial weights" in initial.stdout and initial_wrapper == resume_wrapper, (sweep_id, initial.stderr)  # the same resume
    # optparse keeps the last occurrence of an option: common.sh's defaults come first and lose to the arm's values
    last = {"--opt": arm["opt"], "--lr": arm["lr"], "--weight-decay": arm["wd"], "--layer-decay": arm["layer_decay"],
            "--patch-embed-lr-mult": arm["patch_embed_mult"], "--warmup-epochs": arm["warmup"], "--pretrain_epochs": arm["cycles"],
            "--eval_every": arm["eval_every"], "--eval_cycle0": arm["eval_cycle0"], "--batch_size": "200", "--clip-grad": "3.0", "--warmup-lr": "1e-6", "--min-lr": "1e-6", "--test_augment": "false",
            "--workers": "16", "--model": arm["model"], "--init": arm["init"], "--pretrained_weights": resolved_weights(arm),
            "--normalization": arm["normalization"], "--exp_name": "all4_cyclic_dinov3_sweep_" + sweep_id, "--seed": arm["seed"], "--device": "hpu",
            "--input_size": arm["input_size"], "--img_resize": IMG_RESIZE[arm["input_size"]]}
    for flag, value in last.items():
        assert option_values(start_argv, flag) and option_values(start_argv, flag)[-1] == value, (sweep_id, flag, option_values(start_argv, flag))
    assert option_values(start_argv, "--opt") == ["sgd", arm["opt"]] and option_values(start_argv, "--pretrain_epochs") == ["200", arm["cycles"]]
    assert option_values(start_argv, "--warmup-epochs") == [arm["warmup"]] * 2  # common.sh's ARK_WARMUP_EPOCHS and the pair's own: the same value
    assert option_values(start_argv, "--input_size") == [arm["input_size"]] and option_values(start_argv, "--img_resize") == [IMG_RESIZE[arm["input_size"]]]  # once, from common.sh
    assert option_values(start_argv, "--data_set") == ["VinDrCXR", "ChestXray14", "MIMIC", "CheXpert"]
    for absent in ("--momentum", "--freeze_backbone", "--epochs"):  # the momentum default is the SGD arms' 0.9; nothing here freezes the backbone
        assert absent not in start_argv, (sweep_id, absent)
    start_options, resume_options = parsed(start_argv), parsed(resume_argv)
    if start_options is not None:
        assert (start_options.opt, start_options.lr, start_options.weight_decay, start_options.layer_decay, start_options.patch_embed_lr_mult) == (
            arm["opt"], float(arm["lr"]), float(arm["wd"]), float(arm["layer_decay"]), float(arm["patch_embed_mult"])), sweep_id
        assert (start_options.warmup_epochs, start_options.pretrain_epochs, start_options.eval_every, start_options.batch_size) == (
            int(arm["warmup"]), int(arm["cycles"]), int(arm["eval_every"]), 200), sweep_id
        assert start_options.eval_cycle0 is (arm["eval_cycle0"] == "true") and resume_options.eval_cycle0 is start_options.eval_cycle0, sweep_id
        assert (start_options.clip_grad, start_options.warmup_lr, start_options.min_lr, start_options.test_augment, start_options.momentum) == (
            3.0, 1e-6, 1e-6, False, 0.9), sweep_id
        assert (start_options.model_name, start_options.init, start_options.pretrained_weights, start_options.normalization, start_options.workers) == (
            arm["model"], arm["init"], resolved_weights(arm), arm["normalization"], 16), sweep_id
        assert (start_options.exp_name, start_options.seed, start_options.training_strategy, start_options.device) == (
            "all4_cyclic_dinov3_sweep_" + sweep_id, int(arm["seed"]), "cyclic", "hpu"), sweep_id
        assert start_options.dataset_list == ["VinDrCXR", "ChestXray14", "MIMIC", "CheXpert"] and start_options.freeze_backbone is False, sweep_id
        assert (start_options.crop_size, start_options.resize) == (int(arm["input_size"]), int(IMG_RESIZE[arm["input_size"]])), sweep_id
        assert start_options.resume is False and resume_options.resume is True and resume_options.resume_from is None, sweep_id
        assert {k for k in vars(start_options) if vars(start_options)[k] != vars(resume_options)[k]} == {"resume"}, sweep_id
    return resume_wrapper, start_options


ARMS_IDS_BEFORE_INPUT_SIZE = ("S2", "S4", "S10", "S11", "S12", "S1", "S3", "S5", "S6", "S7", "S13", "S16")
sweep_lines = {}
for arm in ARMS:
    resume_wrapper, start_options = check_sweep_arm(arm, TABLE_TEXT)
    sweep_lines[arm["id"]] = " ".join(resume_wrapper)
    if start_options is not None:
        assert (start_options.opt == "momentum") == (arm["id"] in ("S10", "S11", "S12")), arm["id"]  # the SGD arms; every other arm is AdamW
        assert (start_options.patch_embed_lr_mult == 0.0) == (arm["id"] == "S12"), arm["id"]  # only S12 freezes the patch embedding
# S2's interrupted-run argv as it was before the input_size column (frozen 2026-10-17): the same tokens, plus --input_size 224 --img_resize 256 after --eval_cycle0
S2_BEFORE_INPUT_SIZE = ("python -m torch.distributed.run --standalone --nproc-per-node 4 main_ark.py --opt sgd --warmup-epochs 2 --batch_size 200 --lr 7e-5 --test_augment false --eval_every 2 --eval_cycle0 false --workers 16 --model vit_base_dinov3 --init dinov3 --pretrain_epochs 200 --pretrained_weights /workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth --momentum_teacher 0.9 --projector_features 1376 --ema_mode epoch --device hpu --exp_name all4_cyclic_dinov3_sweep_S2 --seed 100 --resume true --data_set VinDrCXR --data_set ChestXray14 --data_set MIMIC --data_set CheXpert --opt adamw --weight-decay 0.01 --layer-decay 0.75 --patch-embed-lr-mult 1.0 --clip-grad 3.0 --warmup-lr 1e-6 --min-lr 1e-6 --pretrain_epochs 30 --warmup-epochs 2 --normalization imagenet").split()
s2_now = sweep_lines["S2"].split()
k = s2_now.index("--eval_cycle0") + 2
assert s2_now[k:k + 4] == ["--input_size", "224", "--img_resize", "256"] and s2_now[:k] + s2_now[k + 4:] == S2_BEFORE_INPUT_SIZE, s2_now
for sweep_id in ARMS_IDS_BEFORE_INPUT_SIZE:  # every arm of the table before the column: only those four tokens are new, the values 224 / 256
    now = sweep_lines[sweep_id].split()
    assert option_values(now, "--input_size") == ["224"] and option_values(now, "--img_resize") == ["256"], (sweep_id, now)
# the input-size arms: the center recipe's 30 cycles, every cycle evaluated, and their own crop / resize (S2 is the 224 px control)
for sweep_id, crop, resize in (("S18", "448", "512"), ("S19", "512", "585")):
    now = sweep_lines[sweep_id].split()
    assert [option_values(now, flag)[-1] for flag in ("--input_size", "--img_resize", "--pretrain_epochs", "--eval_every", "--eval_cycle0")] == [crop, resize, "30", "1", "false"], (sweep_id, now)
    assert " --input_size " + crop + " --img_resize " + resize + " " in sweep_lines[sweep_id], sweep_id
    assert sweep_lines[sweep_id].endswith("--pretrain_epochs 30 --warmup-epochs 2 --normalization imagenet"), sweep_id
    assert sweep_lines[sweep_id] == sweep_lines["S2"].replace(" --eval_every 2 ", " --eval_every 1 ").replace(" --input_size 224 --img_resize 256 ", " --input_size {} --img_resize {} ".format(crop, resize)).replace(
        "_sweep_S2 ", "_sweep_{} ".format(sweep_id)), sweep_id  # S2's line but for the id, eval_every 1 and the size
# a row that shares no value with wave 1: every column has to come from the row, none may be a constant of the scripts. The model and init names are
# ones the engine would refuse or only log; this checks the plumbing, nothing is trained on it
SYNTH = dict(id="X1", opt="momentum", lr="5e-4", wd="0.1", layer_decay="0.5", patch_embed_mult="0.3", warmup="3", cycles="12", model="vit_large_dinov3",
             init="dinov3_stage_a", weights="/workspace/weights/other_init.pth", normalization="imagenet", seed="31", eval_every="3", eval_cycle0="true",
             input_size="320")
SYNTH_TABLE = TABLE_TEXT + "\t".join(SYNTH[column] for column in COLUMNS) + "\n"
done, called = run("sweep_run.sh", [(TABLE, SYNTH_TABLE), host_path(SYNTH["weights"])], args=["X1"])
assert launched(done, called, SWEEP_START) and called["args"] == sweep_args(SYNTH) and called["env"] == sweep_env(SYNTH), (done, called)
check_sweep_arm(SYNTH, SYNTH_TABLE)
# the second seed of the center arm through the real scripts: --seed 7, the run dir ..._S2b_seed_7, --eval_every 1 and --eval_cycle0 true reach main_ark.py
s2b_resume, s2b_options = check_sweep_arm(S2B, table_s2b)
s2b_argv = main_argv(s2b_resume)
assert option_values(s2b_argv, "--seed") == ["7"] and option_values(s2b_argv, "--eval_every")[-1] == "1" and option_values(s2b_argv, "--eval_cycle0") == ["true"], s2b_argv
assert option_values(s2b_argv, "--exp_name")[-1] == "all4_cyclic_dinov3_sweep_S2b" and "--resume" in s2b_argv, s2b_argv
# every other launcher is unchanged by the new flag: common.sh passes --eval_cycle0 false right after --eval_every unless the caller exports ARK_EVAL_CYCLE0
for exported, wanted in ((None, "false"), ("true", "true")):
    saved = os.environ.pop("ARK_EVAL_CYCLE0", None)
    if exported:
        os.environ["ARK_EVAL_CYCLE0"] = exported
    try:
        for script, script_args in ((FROZEN_START, ["100", "gaudi"]), (FROZEN_LONG_RESUME, ["100", "gaudi"]),
                                    (PROBE_START, ["100", "gaudi", "all4_cyclic_dinov3_probe_a", export_a])):
            direct, direct_wrapper = real_run(script, script_args)
            direct_argv = main_argv(direct_wrapper)
            k = direct_argv.index("--eval_every")
            assert direct.returncode == 0 and direct_argv[k + 2:k + 4] == ["--eval_cycle0", wanted] and direct_argv.count("--eval_cycle0") == 1, (script, exported, direct_argv)
            assert parsed(direct_argv) is None or parsed(direct_argv).eval_cycle0 is (wanted == "true"), (script, exported)
            # and the input size: common.sh's defaults, today's 224 / 256, right after it (no ARK_INPUT_SIZE / ARK_IMG_RESIZE in their env)
            assert direct_argv[k + 4:k + 8] == ["--input_size", "224", "--img_resize", "256"] and direct_argv.count("--input_size") == 1, (script, direct_argv)
            assert parsed(direct_argv) is None or (parsed(direct_argv).crop_size, parsed(direct_argv).resize) == (224, 256), script
    finally:
        os.environ.pop("ARK_EVAL_CYCLE0", None)
        if saved is not None:
            os.environ["ARK_EVAL_CYCLE0"] = saved
# the pair itself stays a plain pass-through for --normalization (only the table restricts it, to imagenet): other presets reach main_ark.py's parser as given
for preset in ("chestx-ray", "none"):
    direct, direct_wrapper = real_run(SWEEP_START, sweep_args(dict(S2, normalization=preset)))
    assert direct.returncode == 0 and option_values(main_argv(direct_wrapper), "--normalization") == [preset], (preset, direct)
    assert parsed(main_argv(direct_wrapper)) is None or parsed(main_argv(direct_wrapper)).normalization == preset, preset
# the phase script's LR floor is the pair's fixed --warmup-lr and --min-lr; the pair itself is a one-line difference between start and resume
phase_text = open(os.path.join(REPO, "chain", "sweep_run.sh")).read()
floor = re.search(r"^LR_FLOOR=(\S+)", phase_text, re.M).group(1)
assert re.findall(r"^    --(?:warmup-lr|min-lr) (\S+) \\$", start_text, re.M) == [floor, floor], floor
# the start script takes exactly its ten arguments and the peak LR from ARK_LR (common.sh would silently use 0.01)
for count in (0, 1, 3, 9, 11):
    usage, wrapper_argv = real_run(SWEEP_START, (sweep_args(S2) + ["extra"])[:count])
    assert usage.returncode == 2 and wrapper_argv is None and "Usage:" in usage.stderr, (count, usage.returncode, usage.stderr)
usage, wrapper_argv = real_run(SWEEP_RESUME, sweep_args(S2)[:9])
assert usage.returncode == 2 and wrapper_argv is None and "Usage:" in usage.stderr
for exported in (None, "1e-3"):  # ARK_LR not exported in the caller's shell, and exported there: real_run(without=...) removes it either way
    saved = os.environ.pop("ARK_LR", None)
    if exported:
        os.environ["ARK_LR"] = exported
    try:
        no_lr, wrapper_argv = real_run(SWEEP_START, sweep_args(S2), without=("ARK_LR",))
    finally:
        os.environ.pop("ARK_LR", None)
        if saved is not None:
            os.environ["ARK_LR"] = saved
    assert no_lr.returncode == 2 and wrapper_argv is None and "ARK_LR" in no_lr.stderr, (exported, no_lr)
# the container argument is common.sh's: anything but nvidia / gaudi, and a non-integer seed, are refused before a launch
bad_container, wrapper_argv = real_run(SWEEP_START, [S2["id"], "100", "tpu"] + sweep_args(S2)[3:])
assert bad_container.returncode == 2 and wrapper_argv is None, bad_container
bad_seed, wrapper_argv = real_run(SWEEP_START, [S2["id"], "x", "gaudi"] + sweep_args(S2)[3:])
assert bad_seed.returncode == 2 and wrapper_argv is None, bad_seed
print("OK (g) sweep: wave1.tsv header (twelve positional columns + seed, eval_every, eval_cycle0, input_size read by name) + {} rows ({}) are the wave-1 definition; per arm a fresh run calls "
      "`{} <id> <seed> gaudi <opt> <wd> <layer_decay> <patch_embed_mult> <warmup> <cycles> <normalization>` and a run dir + checkpoint the resume script, with the ten ARK_* names "
      "(center == armD's recipe; ARK_EVAL_EVERY and ARK_EVAL_CYCLE0 from the row; a second seed gets its own ..._seed_<seed> dir for resume and done); done "
      "(the final cycle's student_cycle_00<cycles>_*.pth AND its cycle_00<cycles>_*.pth.tar, leading zeros read as decimal) skips first; missing/empty weights "
      "(S13: STAGEA_FINAL = teacher_backbone_epoch_<chain/main_epochs.txt> once chain/done/main exists) or a run dir with neither a *.pth.tar nor complete epoch_0000 "
      "initial weights (manifest + student + teacher) -> exit 1, while complete epoch_0000 without a checkpoint resumes from the initial weights; usage, "
      "unknown id, {} malformed table/row variants (schedules WarmupCosineScheduler would refuse, normalization other than imagenet included) -> exit 2; not in "
      "chain/phases.txt; the real scripts hand main_ark.py every value (last occurrence wins) and {}".format(
          len(ARMS), " ".join(a["id"] for a in ARMS), SWEEP_START, len(MALFORMED),
          "its option parser accepts every flag and reads opt / lr / wd / layer decay / patch-embed mult / warmup / cycles / seed / eval every N / eval_cycle0 / batch 200 back (S10-S12 momentum, S12 mult 0), "
          "also for a synthetic row that shares no value with wave 1"
          if main_ark is not None else "the parsed-options check is skipped"))
print("OK (g) input size: wave1.tsv's input_size column (by name) -> ARK_INPUT_SIZE / ARK_IMG_RESIZE = round(x * 8 / 7) -> --input_size / --img_resize: S2..S16 224 / 256, "
      "S18 448 / 512, S19 512 / 585; S2's argv is the frozen pre-column one plus `--input_size 224 --img_resize 256`; other launchers get the defaults 224 / 256; "
      "450 (not a multiple of 16), 96, 096 (below 112), abc, 448.0, a missing column -> exit 2")
for sweep_id in ("S2", "S12", "S18", "S19"):
    print("   chain/sweep_run.sh " + sweep_id + " (interrupted run) -> <repo>/gaudi.sh " + sweep_lines[sweep_id])

# (h) the sbatch wrappers run_d1_8sw_gaudi.sh (2 cards), run_d1_8sw4_gaudi.sh and run_d1_8sw8_gaudi.sh (4 / 8 cards, for arms whose per-card batch needs them) and
# the submitter sweep/submit_wave.sh (with a stub sbatch) -----------------------------------------
sweep_text = open(os.path.join(REPO, "chain", "sweep_run.sh")).read()
workers = int(re.search(r"^export ARK_WORKERS=(\d+)$", sweep_text, re.M).group(1))
sw_code_of = {}
for wrapper, want_cards, want_cores, want_mem in (("run_d1_8sw_gaudi.sh", 2, 36, "128G"), ("run_d1_8sw4_gaudi.sh", 4, 72, "256G"), ("run_d1_8sw8_gaudi.sh", 8, 144, "480G")):
    sw_path = os.path.join(REPO, wrapper)
    subprocess.run(["bash", "-n", sw_path], check=True)
    assert os.access(sw_path, os.X_OK), wrapper
    sw_text = open(sw_path).read()
    assert "sbatch --job-name=d1-8sw-<id> --dependency=singleton " + wrapper in sw_text, wrapper  # the header comment is the only guard for a hand-typed sbatch
    assert "A manual resubmit must use the same --job-name=d1-8sw-<id> --dependency=singleton" in sw_text and "is only the fallback for a bare sbatch" in sw_text, wrapper
    assert re.findall(r"^#SBATCH (.+)$", sw_text, re.M) == ["--job-name=d1-8sw", "-A grp_jliang12", "-p gaudi", "-q public", "-t 1-00:00:00",
                                                             "-N 1", "--gres=gpu:hl225:{}".format(want_cards), "-c {}".format(want_cores), "--mem=" + want_mem], wrapper
    sw_code = [line for line in sw_text.splitlines() if line and not line.startswith("#")]
    assert sw_code[-1] == 'exec ./chain/sweep_run.sh "$1"' and "ARK_" not in "\n".join(sw_code), (wrapper, sw_code)  # the phase script owns the ARK_* recipe
    order = [next(k for k, line in enumerate(sw_code) if line.startswith(start)) for start in
             ("[[ $# -eq 1 ]]", "export PT_HPU_LAZY_MODE=1", "cd /scratch/$USER/2026Fall-CSE507-Ark-Container", "./gaudi_check_cards.sh", "./gaudi_verify_busy.sh 900 &", "exec ")]
    assert order == sorted(order), (wrapper, order)  # the id check comes first, then the lazy mode / cd / card check / busy verifier, then the exec
    for count in (0, 2):  # no id or two: refused at once (exit 2), before the cd and the card check
        usage = subprocess.run([sw_path] + ["S2"] * count, capture_output=True, text=True)
        assert usage.returncode == 2 and "usage: sbatch " + wrapper in usage.stderr, (wrapper, count, usage)
    # the cards x (16 data workers + the rank itself) must fit the cores requested (18 per card)
    cards, cores = int(re.search(r"--gres=gpu:hl225:(\d+)", sw_text).group(1)), int(re.search(r"^#SBATCH -c (\d+)$", sw_text, re.M).group(1))
    assert (cards, workers, cores) == (want_cards, 16, want_cores) and cards * (workers + 1) <= cores == 18 * cards and 200 % cards == 0, (wrapper, cards, workers, cores)
    sw_code_of[wrapper] = sw_code
for n in (4, 8):  # the same code as the 2-card wrapper; only the usage line names its own file
    assert [line.replace("run_d1_8sw{}_gaudi.sh".format(n), "run_d1_8sw_gaudi.sh") for line in sw_code_of["run_d1_8sw{}_gaudi.sh".format(n)]] == sw_code_of["run_d1_8sw_gaudi.sh"], n
# gaudi_check_cards.sh counts the allocation's cards (SLURM_JOB_GPUS / SLURM_GPUS_ON_NODE), so 4 and 8 cards pass it as 2 do
check_text = open(os.path.join(REPO, "gaudi_check_cards.sh")).read()
assert re.findall(r"^\s*expected=(.*)$", check_text, re.M) == [
    """$(printf '%s\\n' "${SLURM_JOB_GPUS:-}" | awk -F, '{for (i = 1; i <= NF; i++) if ($i ~ /[^[:space:]]/) n++} END {print n + 0}')""", "$SLURM_GPUS_ON_NODE"], "gaudi_check_cards.sh no longer counts the allocation's cards"

SUBMIT = "sweep/submit_wave.sh"
subprocess.run(["bash", "-n", os.path.join(REPO, SUBMIT)], check=True)
assert os.access(os.path.join(REPO, SUBMIT), os.X_OK)
submit_text = open(os.path.join(REPO, SUBMIT)).read()
assert "run this script again" in submit_text and "queues the next link of each arm behind the one that is queued or running" in submit_text  # how to continue
assert "do not edit the table while sweep jobs are queued" in submit_text  # the known limit: a queued job reads sweep/wave1.tsv when it starts
assert "seed" not in submit_text  # a second seed is its own row and id: the job name d1-8sw-<id> stays one per arm, nothing in the submitter knows a seed
STUB_SBATCH = """#!/usr/bin/env bash
echo "$*" >> "$SBATCH_LOG"
n=$(wc -l < "$SBATCH_LOG" | tr -d ' ')
if [[ "$n" == "${SBATCH_FAIL_ON:-}" ]]; then echo "sbatch: error: boom" >&2; exit 1; fi
echo "$((1000 + n));sol"
"""


def submit(args, cwd="", fail_on="", extra=(), links=()):
    """Run sweep/submit_wave.sh `args` from <fake repo>/<cwd> with a stub sbatch first in PATH: it logs its arguments, prints "<job id>;sol" (1001, 1002,
    ...) and fails on its n-th call when fail_on=n. Returns (completed process, the stub's log: one line per sbatch call). The fake repo must come out unchanged.
    `extra`: more files in the fake repo (populate entries); `links`: (path, target) symbolic links in it; "{root}" in `args` stands for the fake repo's root."""
    with tempfile.TemporaryDirectory() as base:
        root = os.path.join(base, "repo")
        populate(root, [(TABLE, TABLE_TEXT), "run_d1_8sw_gaudi.sh", SUBMIT] + list(extra))
        for path, target in links:
            os.symlink(target, os.path.join(root, path))
        shutil.copy(os.path.join(REPO, SUBMIT), os.path.join(root, SUBMIT))  # keeps the executable bit
        populate(base, [("tools/sbatch", STUB_SBATCH)])
        os.chmod(os.path.join(base, "tools", "sbatch"), 0o755)
        log = os.path.join(base, "sbatch.log")
        open(log, "w").close()
        before = snapshot(root)
        env = dict(os.environ, PATH=os.path.join(base, "tools") + os.pathsep + os.environ["PATH"], SBATCH_LOG=log, SBATCH_FAIL_ON=str(fail_on))
        done = subprocess.run([os.path.join(root, SUBMIT)] + [a.replace("{root}", root) for a in args], cwd=os.path.join(root, cwd), env=env,
                              capture_output=True, text=True)
        assert snapshot(root) == before, "submit_wave.sh changed the tree"
        calls = open(log).read().splitlines()
    return done, calls


# one job name per arm plus --dependency=singleton: Slurm runs at most one job of a name per user at a time, so a resubmitted arm queues behind its running link
SBATCH_LINES = ["sbatch --parsable --job-name=d1-8sw-{0} --dependency=singleton run_d1_8sw_gaudi.sh {0}".format(arm["id"]) for arm in ARMS]
S13_AT = [arm["id"] for arm in ARMS].index("S13")
assert len(SBATCH_LINES) == 14 and all(" --dependency=singleton run_d1_8sw_gaudi.sh " in line for line in SBATCH_LINES)
done, calls = submit(["--dry-run", TABLE])  # --dry-run: the 14 sbatch lines in file order, nothing run
assert done.returncode == 0 and done.stdout.splitlines() == SBATCH_LINES and len(SBATCH_LINES) == 14 and done.stderr == "" and calls == [], done
done, calls = submit([TABLE, "--dry-run"])  # the flag may follow the table
assert done.returncode == 0 and done.stdout.splitlines() == SBATCH_LINES and calls == [], done
done, calls = submit(["--dry-run", TABLE, "S13", "S2"])  # ids select and order the arms
assert done.returncode == 0 and done.stdout.splitlines() == [SBATCH_LINES[S13_AT], SBATCH_LINES[0]] and calls == [], done
done, calls = submit([TABLE, "S4", "S2"])  # for real: sbatch gets those arguments, stdout gets "<id> <job id>" (the cluster after ';' dropped), stderr the commands
assert done.returncode == 0 and done.stdout.splitlines() == ["S4 1001", "S2 1002"], done
assert calls == [SBATCH_LINES[1][len("sbatch "):], SBATCH_LINES[0][len("sbatch "):]], calls
assert done.stderr.splitlines() == ["+ " + SBATCH_LINES[1], "+ " + SBATCH_LINES[0]], done.stderr
done, calls = submit([TABLE])  # the default is every row, in file order
assert done.returncode == 0 and done.stdout.splitlines() == ["{} {}".format(arm["id"], 1001 + k) for k, arm in enumerate(ARMS)] and len(calls) == 14, done
done, calls = submit([TABLE, "S2", "S4", "S10"], fail_on=2)  # the first failing sbatch stops the wave and names what was not submitted
assert done.returncode == 1 and done.stdout.splitlines() == ["S2 1001"] and len(calls) == 2, (done, calls)
assert "ERROR: sbatch failed for S4; stopping. Not submitted: S4 S10" in done.stderr, done.stderr
for cwd in ("sweep", ".."):  # not the repo root: refused before anything is printed or submitted, dry run or not
    for flags in (["--dry-run"], []):
        done, calls = submit(flags + [TABLE], cwd=cwd)
        assert done.returncode == 2 and done.stdout == "" and calls == [] and "run this from the repo root" in done.stderr, (cwd, flags, done)
for args, message in ((["--dry-run", TABLE, "S2", "S99"], "sweep id 'S99' is not a row of sweep/wave1.tsv"),
                      (["--dry-run", TABLE, "S2", "S4", "S2"], "repeated sweep id(s): S2"), (["--dry-run", TABLE, "*"], "'*' is not a valid sweep id"),
                      (["--dry-run", TABLE, TABLE], "is not a valid sweep id"), (["--dry-run", "no_such.tsv"], "no_such.tsv is missing or unreadable"),
                      (["--dry-run"], "usage: "), ([], "usage: "), (["-n", TABLE], "unknown option '-n'")):  # nothing is printed or submitted
    done, calls = submit(args)
    assert done.returncode == 2 and done.stdout == "" and calls == [] and message in done.stderr, (args, done)
for args in ([], ["-n", TABLE]):
    done, calls = submit(args)
    assert "[--dry-run] sweep/wave1.tsv [sweep id ...]" in done.stderr, done.stderr  # the usage line names the one table that is accepted
# The jobs always read sweep/wave1.tsv (chain/sweep_run.sh), so the table argument has to BE that file: the same file under any spelling passes (a link to it
# included), anything else is refused (exit 2) before a single sbatch -- a byte-identical copy too, which would silently run wave-1 parameters under its own ids
LINKS = [("sweep/alias.tsv", "wave1.tsv"), ("sweepdir", "sweep"), ("sweep/to_copy.tsv", "wave2.tsv")]  # a link to wave 1, to its directory, and to a copy
COPY = [("sweep/wave2.tsv", TABLE_TEXT)]
for spelling in ("sweep/wave1.tsv", "./sweep/wave1.tsv", "sweep/../sweep/wave1.tsv", "./sweep//wave1.tsv", "{root}/sweep/wave1.tsv", "sweep/alias.tsv", "sweepdir/wave1.tsv"):
    done, calls = submit(["--dry-run", spelling, "S13", "S2"], extra=COPY, links=LINKS)
    assert done.returncode == 0 and done.stdout.splitlines() == [SBATCH_LINES[S13_AT], SBATCH_LINES[0]] and done.stderr == "" and calls == [], (spelling, done)
    done, calls = submit([spelling, "S4"], extra=COPY, links=LINKS)  # and for real
    assert done.returncode == 0 and done.stdout.splitlines() == ["S4 1001"] and len(calls) == 1, (spelling, done, calls)
REFUSED_MESSAGE = "a wave 2 will need the table passed through to chain/sweep_run.sh first"
for path, content in (("sweep/wave2.tsv", TABLE_TEXT),                     # a byte-identical copy
                      ("sweep/wave1.tsv.bak", TABLE_TEXT),                 # one named like it
                      ("wave1.tsv", TABLE_TEXT), ("sweep/old/wave1.tsv", TABLE_TEXT),  # the same name in the repo root, in another directory
                      ("sweep/wave2.tsv", edited(with_field("lr", "1e-4")))):  # another table
    for flags in (["--dry-run"], []):
        done, calls = submit(flags + [path, "S2"], extra=[(path, content)])
        assert done.returncode == 2 and done.stdout == "" and calls == [], (path, flags, done)
        assert path + " is not sweep/wave1.tsv" in done.stderr and REFUSED_MESSAGE in done.stderr, (path, done.stderr)
done, calls = submit(["--dry-run", "sweep/to_copy.tsv"], extra=COPY, links=LINKS)  # a link to a copy is a copy
assert done.returncode == 2 and done.stdout == "" and calls == [] and REFUSED_MESSAGE in done.stderr, done
print("OK (h) run_d1_8sw_gaudi.sh: 2 cards, -c 36 (2 x (16 workers + 1) <= 36), 1 day, no ARK_* exports, id check first, ends in exec ./chain/sweep_run.sh \"$1\"; "
      "run_d1_8sw4_gaudi.sh / run_d1_8sw8_gaudi.sh: the same code on 4 cards -c 72 --mem=256G / 8 cards -c 144 --mem=480G; gaudi_check_cards.sh takes the allocation's card count; "
      "sweep/submit_wave.sh: --dry-run prints the 14 `sbatch --parsable --job-name=d1-8sw-<id> --dependency=singleton run_d1_8sw_gaudi.sh <id>` lines in file order, a real run prints "
      "`<id> <job id>`, the first failing sbatch stops the wave, and anything but the repo root, an unknown / repeated / malformed id or a bad table is refused (exit 2) "
      "before a single sbatch; the table has to be sweep/wave1.tsv itself (any spelling, links followed; a byte-identical copy is refused, since the jobs always read wave1)")
print("   " + SBATCH_LINES[0] + "   (sweep/submit_wave.sh --dry-run sweep/wave1.tsv, from the repo root)")

print("ALL OK")
