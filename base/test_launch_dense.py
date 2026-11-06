"""Checks for launch_dense.sh, the Slurm launcher of the dense segmentation probe (base/dense/seg_probe.py): the --check form, its refusals, and the real submission behind a stub sbatch.
Run from base/: python test_launch_dense.py (standard library only, no Slurm). Every call sets ARK_ROOT to a directory of the test's own and drops the caller's SLURM_* and DENSE_*
variables, so nothing depends on /scratch or on where the test runs."""
import os
import shutil
import subprocess
import tempfile

LAUNCHER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "launch_dense.sh")
STOCK = "/workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
BASE = ["candidptx", "stock_dinov3", "vit_base_dinov3", STOCK, "imagenet"]
OPTIONS = ["-A", "class_cse49478170fall2026", "-p", "public,htc", "-q", "class", "-N", "1", "--gres=gpu:1", "--constraint=a100_80", "-c", "12", "--mem=64G", "-t", "02:00:00",
           "-J", "dense-stock_dinov3-candidptx"]
STUB_SBATCH = '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$STUB_DIR/argv"\ncat > "$STUB_DIR/stdin"\necho "Submitted batch job 4242"\nexit "${STUB_EXIT:-0}"\n'


def launch(args, root, **env):
    """launch_dense.sh `args` with ARK_ROOT = root, the caller's SLURM_* and DENSE_* variables removed and `env` added."""
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "DENSE_"))}
    environment.update(ARK_ROOT=root, **env)
    return subprocess.run([LAUNCHER] + list(args), capture_output=True, text=True, env=environment, stdin=subprocess.DEVNULL, timeout=60)  # no stdin: a stub that waits for a script gets EOF


def job_script(root, *extra):
    """What the job script must be: the repo copy, then seg_probe on BASE's arguments (and `extra` after --out)."""
    return ("#!/usr/bin/env bash\ncd " + root + " && ./nvidia.sh python -m dense.seg_probe --model vit_base_dinov3 --weights " + STOCK + " --normalization imagenet --dataset candidptx "
            "--tag stock_dinov3 --workers 12 --out /workspace/results/dense" + "".join(" " + word for word in extra) + "\n")


subprocess.run(["bash", "-n", LAUNCHER], check=True)
assert os.access(LAUNCHER, os.X_OK), "launch_dense.sh is not executable"

with tempfile.TemporaryDirectory() as tmp:
    absent = os.path.join(tmp, "absent")  # no ARK_ROOT, as on a laptop: the weights and results checks are skipped, with a note
    root = os.path.join(tmp, "root")  # a repo copy that holds the weights file
    os.makedirs(os.path.join(root, "weights"))
    with open(os.path.join(root, "weights", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"), "w") as f:
        f.write("x")

    # --check prints exactly the sbatch line and the job script (as a heredoc), and nothing else on stdout
    done = launch(["--check"] + BASE, absent)
    assert done.returncode == 0, done.stderr
    assert done.stdout == "sbatch " + " ".join(OPTIONS) + " <<'JOB'\n" + job_script(absent) + "JOB\n", done.stdout
    assert "-J dense-stock_dinov3-candidptx" in done.stdout and "--dataset candidptx" in done.stdout
    assert "note:" in done.stderr and "not checked" in done.stderr, done.stderr
    done = launch(["--check"] + BASE + ["--iters", "10", "--seed", "3"], absent)  # the two options follow --out, in this order, whichever order they were given in
    assert done.returncode == 0 and job_script(absent, "--iters", "10", "--seed", "3") in done.stdout, done.stdout
    assert launch(["--check"] + BASE + ["--seed", "3", "--iters", "10"], absent).stdout == done.stdout
    assert job_script(absent, "--seed", "0") in launch(["--check"] + BASE + ["--seed", "0"], absent).stdout  # an explicit seed 0 is passed on
    done = launch(["--check"] + BASE + ["--seed", "1", "--runs", "3"], absent)  # --runs follows --seed, and the job script carries both
    assert done.returncode == 0 and job_script(absent, "--seed", "1", "--runs", "3") in done.stdout, done.stdout
    done = launch(["--check", "jsrt_lung", "raydino", "vit_large_raydino", "/workspace/weights/xray_dino_vitl16_pretrained-ad31c2b0.pth", "imagenet"], absent, DENSE_TIME="00:45:00", DENSE_ACCOUNT="class_x")
    assert done.returncode == 0 and "-A class_x -p" in done.stdout and "-t 00:45:00 -J dense-raydino-jsrt_lung" in done.stdout, done.stdout

    # refusals: the exit status and a one-line reason, with and without --check, and nothing on stdout (no job line)
    refused = [(["candidptx", "bad tag"] + BASE[2:], "tag 'bad tag'"),
               (BASE[:3] + ["/home/x/w.pth", "imagenet"], "is not an in-container path under /workspace"),
               (["chestxdet"] + BASE[1:], "dataset 'chestxdet'"),
               (BASE[:4] + ["foo"], "normalization 'foo'"),
               (BASE[:2] + ["vit_base"] + BASE[3:], "model 'vit_base'"),
               (BASE[:4], "Usage: ./launch_dense.sh"),
               (BASE + ["--tag", "other"], "--tag: not accepted after <normalization>"),
               (BASE + ["--iters"], "--iters takes"), (BASE + ["--iters", "0"], "--iters takes"), (BASE + ["--iters", "5", "--iters", "6"], "--iters takes"),
               (BASE + ["--runs", "0"], "--runs takes"), (BASE + ["--runs", "1", "--runs", "2"], "--runs takes"),
               (BASE + ["--seed", "-1"], "--seed takes"), (BASE + ["--seed", "1", "--seed", "2"], "--seed takes"),
               (BASE + ["--seed", "007"], "--seed takes"), (BASE + ["--seed", "1000000000"], "--seed takes")]  # seg_probe names the folder seed<int>: 007 and 7 would be one run
    for args, reason in refused:
        for form in ([], ["--check"]):
            done = launch(form + args, absent)
            assert done.returncode == 2 and done.stdout == "" and reason in done.stderr and len(done.stderr.splitlines()) == 1, (form, args, done)
    for variable, value in (("DENSE_TIME", "soon"), ("DENSE_ACCOUNT", "a b")):
        done = launch(["--check"] + BASE, absent, **{variable: value})
        assert done.returncode == 2 and done.stdout == "" and variable + "='" + value + "'" in done.stderr, (variable, done)

    # with an ARK_ROOT that exists: the weights file and an existing result are checked, in --check as in the real submission
    done = launch(["--check"] + BASE, root)
    assert done.returncode == 0 and done.stdout.count("cd " + root + " && ") == 1 and done.stderr == "", done
    for name, make in (("missing", lambda path: None), ("empty", lambda path: open(path, "w").close()), ("a directory", os.mkdir)):
        bad = os.path.join(root, "weights", name.replace(" ", "_") + ".pth")
        make(bad)
        done = launch(["--check"] + BASE[:3] + ["/workspace/weights/" + os.path.basename(bad), "imagenet"], root)
        assert done.returncode == 1 and done.stdout == "" and "ERROR: weights not found at " + bad in done.stderr, (name, done)
    runs = os.path.join(root, "results", "dense", "stock_dinov3", "candidptx")  # the seed folders of one (tag, dataset)
    os.makedirs(os.path.join(runs, "seed0"))
    with open(os.path.join(runs, "seed0", "head_best.pt"), "w") as f:
        f.write("x")
    done = launch(["--check"] + BASE, root)  # a seed folder without a result.json (a run that died) blocks too: seg_probe refuses any seed<S> that exists
    assert done.returncode == 1 and done.stdout == "" and "results/dense/stock_dinov3/candidptx/seed0 exists" in done.stderr, done
    os.makedirs(os.path.join(runs, "seed4"))  # an empty seed folder blocks as well
    done = launch(["--check"] + BASE + ["--seed", "4"], root)
    assert done.returncode == 1 and done.stdout == "" and "seed4 exists" in done.stderr, done
    os.symlink(os.path.join(runs, "nowhere"), os.path.join(runs, "seed5"))  # a dangling symlink at that name blocks too (-e is false for it, -L is true)
    done = launch(["--check"] + BASE + ["--seed", "5"], root)
    assert done.returncode == 1 and done.stdout == "" and "seed5 exists" in done.stderr, done
    os.makedirs(os.path.join(runs, "seed6.partial-123-abcd"))  # a staging leftover (a run killed mid-run) does not block a run of its seed
    assert launch(["--check"] + BASE + ["--seed", "6"], root).returncode == 0

    def finish(seed):  # the run of that seed has written its result
        os.makedirs(os.path.join(runs, "seed" + seed), exist_ok=True)
        with open(os.path.join(runs, "seed" + seed, "result.json"), "w") as f:
            f.write("{}")

    finish("0")
    for seed_args in ([], ["--seed", "0"]):  # seed 0 is the default
        done = launch(["--check"] + BASE + seed_args, root)
        assert done.returncode == 1 and done.stdout == "" and "results/dense/stock_dinov3/candidptx/seed0 exists" in done.stderr, (seed_args, done)
    assert launch(["--check"] + BASE + ["--seed", "1"], root).returncode == 0  # another seed of the same (tag, dataset) is another folder
    os.makedirs(os.path.join(runs, "seed3"))  # --runs checks every seed of the job: seeds 1-3 meet seed3's folder, seeds 1-2 do not
    done = launch(["--check"] + BASE + ["--seed", "1", "--runs", "3"], root)
    assert done.returncode == 1 and done.stdout == "" and "seed3 exists" in done.stderr, done
    assert launch(["--check"] + BASE + ["--seed", "1", "--runs", "2"], root).returncode == 0
    finish("1")
    done = launch(["--check"] + BASE + ["--seed", "1"], root)
    assert done.returncode == 1 and done.stdout == "" and "seed1 exists" in done.stderr, done
    assert launch(["--check"] + BASE + ["--seed", "2"], root).returncode == 0
    assert launch(["--check", "jsrt_lung"] + BASE[1:], root).returncode == 0  # the same tag on the other dataset is another result folder

    # the real submission, behind a stub sbatch: the options, then the job script on stdin, which is what --check printed
    stub_dir = os.path.join(tmp, "stub")
    os.makedirs(os.path.join(stub_dir, "bin"))
    with open(os.path.join(stub_dir, "bin", "sbatch"), "w") as f:
        f.write(STUB_SBATCH)
    os.chmod(os.path.join(stub_dir, "bin", "sbatch"), 0o755)
    stubbed = dict(STUB_DIR=stub_dir, PATH=os.path.join(stub_dir, "bin") + os.pathsep + os.environ["PATH"])

    def submitted():
        with open(os.path.join(stub_dir, "argv")) as argv, open(os.path.join(stub_dir, "stdin")) as stdin:
            return argv.read().splitlines(), stdin.read()

    done = launch(BASE, root, **stubbed)  # an existing result blocks the real submission too, before sbatch is called
    assert done.returncode == 1 and "seed0 exists" in done.stderr and not os.path.exists(os.path.join(stub_dir, "argv")), done
    shutil.rmtree(os.path.join(root, "results"))
    done = launch(BASE + ["--iters", "10", "--seed", "3"], root, **stubbed)
    assert done.returncode == 0 and done.stdout == "Submitted batch job 4242\n" and done.stderr == "", done
    assert submitted() == (OPTIONS, job_script(root, "--iters", "10", "--seed", "3")), submitted()
    assert job_script(root, "--iters", "10", "--seed", "3") in launch(["--check"] + BASE + ["--iters", "10", "--seed", "3"], root).stdout
    os.remove(os.path.join(stub_dir, "argv"))
    done = launch(BASE, root, SLURM_JOB_ID="7", **stubbed)  # inside a job: refused before sbatch is called; --check still works there
    assert done.returncode == 1 and "not inside Slurm job 7" in done.stderr and not os.path.exists(os.path.join(stub_dir, "argv")), done
    assert launch(["--check"] + BASE, root, SLURM_JOB_ID="7", **stubbed).returncode == 0
    done = launch(BASE, absent, **stubbed)  # a real submission needs the repo copy
    assert done.returncode == 1 and "is not a directory" in done.stderr and not os.path.exists(os.path.join(stub_dir, "argv")), done
    done = launch(BASE, root, STUB_EXIT="1", **stubbed)  # sbatch refusing the job is the launcher's failure
    assert done.returncode == 1 and "sbatch refused the job" in done.stderr, done
print("OK (launcher)")
print("ALL OK")
