"""CPU check of ssl_dinov3/export_student.py and the repo-root export_student.sh on a tiny fake run (no Slurm, no cluster data). Run from base/:  python -m ssl_dinov3.test_export_student   (needs torch, timm)
The fake checkpoint holds the state dicts of a tiny student and teacher built by build_models (untied local_cls_norm, per-dataset heads), so its keys are the ones save_state writes, and the teacher export is
export_backbone(teacher.backbone). The student is the teacher plus noise.
(a) the student's backbone is written: the check file's keys, the student's own values (not the teacher's), no local_cls_norm, and one line with the key count, the mean absolute difference and the cosine
(b) a check file with one value changed or one key missing, and a student with one tensor of another shape: exit 1 naming the cause, nothing written (no .tmp either)
(c) an existing --out is refused and left as it was
(d) export_student.sh against a stub sbatch: the sbatch line for epoch 020 (padded, not read as octal), and bad arguments, missing inputs and an existing output submit nothing"""
import atexit
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
REPO = os.path.dirname(BASE)
sys.path[:] = [BASE] + [p for p in sys.path if os.path.abspath(p or ".") not in (HERE, BASE)]  # base/ first; this folder's model.py, data.py ... never as top-level modules

import torch

from models import DINOV3_VITB16_KWARGS
from ssl_dinov3.model import build_models, export_backbone

TMP = tempfile.mkdtemp()
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
TINY_VIT = dict(DINOV3_VITB16_KWARGS, embed_dim=64, depth=2, num_heads=2, ffn_ratio=2, untie_global_and_local_cls_norm=True)
TINY_HEADS = dict(out_dim=128, hidden_dim=64, bottleneck_dim=32, nlayers=3)


def fake(name, check=lambda c: c, student=lambda s: s):
    """<TMP>/<name>/ckpt/epoch_0001.pth (the state dicts of a tiny student and teacher; `student` edits the student's) and export/teacher_backbone_epoch_0001.pth (export_backbone(teacher.backbone);
    `check` edits it). Returns (student, teacher, export_student argv, its --out path)."""
    ckpt, check_file, out = (os.path.join(TMP, name, p) for p in ("ckpt/epoch_0001.pth", "export/teacher_backbone_epoch_0001.pth", "export/student_backbone_epoch_0001.pth"))
    for f in (ckpt, check_file):
        os.makedirs(os.path.dirname(f))
    s, t, _ = build_models(TINY_VIT, None, TINY_HEADS, gram=False, heads=["MIMIC", "CheXpert"])
    with torch.no_grad():
        for p in s.parameters():
            p.add_(torch.randn_like(p))
    torch.save(dict(student=student(s.state_dict()), teacher=t.state_dict(), step=1), ckpt)
    torch.save(check(export_backbone(t.backbone)), check_file)
    return s, t, ["--ckpt", ckpt, "--check", check_file, "--out", out], out


def export(argv):
    return subprocess.run([sys.executable, "-m", "ssl_dinov3.export_student", *argv], cwd=BASE, capture_output=True, text=True)


def test_writes_the_students_backbone():
    s, t, argv, out = fake("ok")
    done = export(argv)
    assert done.returncode == 0, done.stderr
    saved, student, teacher = torch.load(out, weights_only=True), export_backbone(s.backbone), export_backbone(t.backbone)  # weights_only: a plain dict of tensors, as the teacher's is
    assert any(k.startswith("backbone.local_cls_norm.") for k in s.state_dict()) and not any(k.startswith("local_cls_norm.") for k in saved)
    assert list(saved) == list(student) == list(teacher) and not os.path.exists(out + ".tmp")
    assert all(torch.equal(saved[k], student[k]) for k in saved) and not all(torch.equal(saved[k], teacher[k]) for k in saved)
    d = lambda f: sum(f(student[k].double(), teacher[k].double()).sum().item() for k in saved if saved[k].is_floating_point())
    n = sum(v.numel() for v in saved.values() if v.is_floating_point())
    mad, cosine = d(lambda a, b: (a - b).abs()) / n, d(lambda a, b: a * b) / (d(lambda a, b: a * a) * d(lambda a, b: b * b)) ** 0.5
    line = re.fullmatch(re.escape(f"{out}: {len(saved)} keys, mean abs diff ") + r"(\S+), cosine (\S+)\n", done.stdout)
    assert line and abs(float(line[1]) - mad) < 1e-5 * mad and abs(float(line[2]) - cosine) < 1e-6, (done.stdout, mad, cosine)


def test_mismatch_writes_nothing():
    for name, edits, cause in (
            ("value", dict(check=lambda c: {**c, "cls_token": c["cls_token"] + 1}), "teacher: the mapping does not reproduce"),
            ("key", dict(check=lambda c: {k: v for k, v in c.items() if k != "norm.bias"}), "keys differ, only in the checkpoint's: ['norm.bias']"),
            ("shape", dict(student=lambda s: {**s, "backbone.norm.weight": torch.zeros(3)}), "student: not laid out like")):
        _, _, argv, out = fake("bad_" + name, **edits)
        done = export(argv)
        assert done.returncode == 1 and cause in done.stderr and "nothing written" in done.stderr, (name, done)
        assert not os.path.exists(out) and not os.path.exists(out + ".tmp"), name


def test_existing_out_is_refused():
    _, _, argv, out = fake("exists")
    with open(out, "w") as f:
        f.write("keep")
    done = export(argv)
    assert done.returncode == 1 and "exists: not overwriting" in done.stderr and open(out).read() == "keep", done


def test_wrapper():
    repo, bin_dir, log = (os.path.join(TMP, p) for p in ("repo", "bin", "sbatch.log"))
    for f in ("run_x/ckpt/epoch_0020.pth", "run_x/export/teacher_backbone_epoch_0020.pth"):
        os.makedirs(os.path.dirname(os.path.join(repo, f)), exist_ok=True)
        open(os.path.join(repo, f), "w").close()
    os.makedirs(bin_dir)
    with open(os.path.join(bin_dir, "sbatch"), "w") as f:  # a stub that records its arguments, one per line, and prints a job id
        f.write('#!/bin/sh\nprintf "%s\\n" "$@" > "$SBATCH_LOG"\necho 4242\n')
    os.chmod(os.path.join(bin_dir, "sbatch"), 0o755)
    shutil.copy(os.path.join(REPO, "export_student.sh"), repo)
    env = dict(os.environ, PATH=bin_dir + os.pathsep + os.environ["PATH"], SBATCH_LOG=log)

    def submit(*argv):
        if os.path.exists(log):
            os.remove(log)
        return subprocess.run(["bash", "export_student.sh", *argv], cwd=repo, env=env, capture_output=True, text=True)

    done, w = submit("run_x", "020"), "/workspace/run_x"  # 020 is twenty, not octal sixteen
    assert done.returncode == 0 and "job 4242" in done.stdout and "run_x/export/student_backbone_epoch_0020.pth" in done.stdout, done
    assert open(log).read().splitlines() == [
        "--parsable", "-J", "student-export-0020", "-A", "class_cse49478170fall2026", "-p", "public,htc", "-q", "class", "-N", "1", "-c", "2", "--mem=16G", "-t", "00:10:00",
        "--wrap", f"cd {os.path.realpath(repo)} && ./nvidia.sh python -m ssl_dinov3.export_student --ckpt {w}/ckpt/epoch_0020.pth "
                  f"--check {w}/export/teacher_backbone_epoch_0020.pth --out {w}/export/student_backbone_epoch_0020.pth"]
    for argv, code in ((("run_x",), 2), (("run_x", "abc"), 2), (("run_x", "2x"), 2), (("run_x", ""), 2), (("run_x", "12345"), 2), (("../run_x", "20"), 2), (("run_x", "21"), 1), (("nope", "20"), 1)):
        done = submit(*argv)
        assert done.returncode == code and not os.path.exists(log), (argv, done)
    open(os.path.join(repo, "run_x/export/student_backbone_epoch_0020.pth"), "w").close()
    done = submit("run_x", "20")
    assert done.returncode == 1 and "exists" in done.stderr and not os.path.exists(log), done


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("OK", name)
    print("ALL OK")
