"""CPU checks for the Stage A SSL trainer: ssl_pretrain.py, ssl_dinov3/ (data, meta, model, optim) and the launch scripts
(chain/*.sh, run_d1_8s_timing_gaudi.sh). Run from base/: python test_ssl_pretrain.py (needs torch, timm, torchvision, PIL)."""
import atexit
import contextlib
import csv
import functools
import io
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

import ssl_pretrain
from accelerator import Accelerator
from models import DINOV3_VITB16_KWARGS, build_omni_model
from ssl_dinov3 import data as ssl_data
from ssl_dinov3.data import DATASET_NAMES, DataAugmentationCXR, build_dataset, make_collate_fn
from ssl_dinov3.meta import SSLMetaArch
from ssl_dinov3.model import build_models, export_backbone
from ssl_dinov3.optim import build_schedules, lr_scaling_factor

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
os.chdir(BASE)  # datasets_config.yaml and dataset/ lists are relative to base/
WEIGHT_NAME = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
# the stock file: <repo>/weights/ (as on Sol), or wherever DINOV3_VITB16_WEIGHTS points; check (f) is skipped without it
WEIGHTS = next((p for p in (os.environ.get("DINOV3_VITB16_WEIGHTS", ""), os.path.join(REPO, "weights", WEIGHT_NAME))
                if os.path.isfile(p)), None)
TINY_VIT = dict(DINOV3_VITB16_KWARGS, embed_dim=64, depth=2, num_heads=2, ffn_ratio=2)
TINY_HEADS = dict(out_dim=128, hidden_dim=64, bottleneck_dim=32, nlayers=3)
LOSS_KEYS = {"dino_local_crops_loss", "dino_global_crops_loss", "koleo_loss", "ibot_loss", "gram_loss"}
CSV_HEADER = "step,epoch,dino_global,dino_local,koleo,ibot,gram,total,lr,wd,momentum,teacher_temp,gram_weight,step_time,data_time,dataset"

TMP = tempfile.mkdtemp()
atexit.register(shutil.rmtree, TMP, ignore_errors=True)


def make_synthetic_data(root, counts=None):
    """12 grayscale images of a few non-square sizes (JPEG and PNG), 3 per dataset, list files in the 4 real formats (counts: {dataset: images} instead)."""
    os.makedirs(root, exist_ok=True)
    rng, sizes, count = np.random.RandomState(0), [(300, 400), (512, 384), (257, 331)], [0]

    def save(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        h, w = sizes[count[0] % 3]
        count[0] += 1
        Image.fromarray(rng.randint(0, 255, (h, w), dtype=np.uint8), "L").save(path)

    config = {}
    for name, ext, header in (("ChestXray14", "png", None), ("VinDrCXR", "jpeg", None),
                              ("CheXpert", "jpg", "Path,Sex,Age"), ("MIMIC", "jpg", "path,subject_id")):
        data_dir, list_file = os.path.join(root, name), os.path.join(root, name + ".list")
        with open(list_file, "w") as f:
            if header:
                f.write(header + "\n")
            for i in range(3 if counts is None else counts[name]):
                if name == "ChestXray14":    # relpath label...
                    save(os.path.join(data_dir, "x%d.png" % i))
                    f.write("x%d.png %s\n" % (i, " ".join("01" * 7)))
                elif name == "VinDrCXR":     # id label...  (path = data_dir/id.jpeg)
                    save(os.path.join(data_dir, "v%d.jpeg" % i))
                    f.write("v%d 0 0 1 0 0 1\n" % i)
                else:                        # csv, path in column 0
                    rel = ("CheXpert-v1.0/train/p%d/s1/view1.jpg" if name == "CheXpert" else "files/p10/p%d/s1/a.jpg") % i
                    save(os.path.join(data_dir, rel))
                    f.write("%s,F\n" % rel)
        config[name] = dict(data_dir=data_dir, train_list=list_file)
    return config


CFG = make_synthetic_data(os.path.join(TMP, "data"))
ssl_pretrain.ensure_process_group(Accelerator("cpu"))  # the vendored iBOT Sinkhorn-Knopp all_reduces, so a group must exist


def fresh_meta():
    torch.manual_seed(0)
    random.seed(0)
    student, teacher, gram_teacher = build_models(TINY_VIT, None, TINY_HEADS)
    return SSLMetaArch(student, teacher, gram_teacher, n_local_crops=8).train()


# (a) static shapes -------------------------------------------------------------------------------------------------
dataset = build_dataset(list(DATASET_NAMES), CFG, DataAugmentationCXR(), check_exists=True)  # gram teacher crops 448
assert len(dataset) == 12
collate = make_collate_fn()
batches = [collate([dataset[i] for i in range(4 * j, 4 * j + 4)]) for j in range(3)]
shape_of = lambda batch: {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)}
assert shape_of(batches[0]) == shape_of(batches[1]) == shape_of(batches[2])
assert shape_of(batches[0])["collated_global_crops"] == (8, 3, 224, 224)
assert shape_of(batches[0])["collated_local_crops"] == (32, 3, 96, 96)
assert shape_of(batches[0])["collated_gram_teacher_crops"] == (8, 3, 448, 448)
for batch in batches:
    assert len(batch["mask_indices_list"]) == batch["upperbound"] > 0
assert len({b["upperbound"] for b in batches}) == 1
assert all(not torch.equal(batches[i]["collated_masks"], batches[j]["collated_masks"]) for i, j in ((0, 1), (0, 2), (1, 2)))
print("OK (a) static shapes: 3 collates with different mask draws share every tensor shape; %d masked patches each = upperbound"
      % batches[0]["upperbound"])

# (b) one meta step through the real augmentation pipeline (Gram teacher crops 448) -----------------------------------------
meta = fresh_meta()
student, teacher, gram_teacher = meta.student, meta.teacher, meta.gram_teacher
loss, loss_dict = meta.forward_backward(batches[0], teacher_temp=0.04)
assert torch.isfinite(loss) and set(loss_dict) == LOSS_KEYS and all(torch.isfinite(v) for v in loss_dict.values())
gram_448 = loss_dict["gram_loss"].item()
assert 0 <= gram_448 < 1.0
print("OK (b) meta step: total %.4f, %s; gram (448 teacher crops, step 0) = %.5f" % (
    loss.item(), ", ".join("%s %.4f" % (k.replace("_loss", ""), v.item()) for k, v in sorted(loss_dict.items())), gram_448))

# (c) Gram is exactly 0 when nothing differs between student and Gram teacher -------------------------------------------
meta_c = fresh_meta()  # Gram teacher == student backbone at init
dataset_224 = build_dataset(list(DATASET_NAMES), CFG, DataAugmentationCXR(gram_teacher_crops_size=224), check_exists=False)
samples = [dataset_224[i] for i in range(4)]
for crops, _ in samples:
    crops["gram_teacher_crops"] = crops["global_crops"]  # the very same tensors the student sees
batch_c = make_collate_fn(mask_sample_probability=0.0)(samples)  # no iBOT masks
assert batch_c["mask_indices_list"].numel() == 0 and not batch_c["collated_masks"].any()
_, ld = meta_c.forward_backward(batch_c, teacher_temp=0.04)
assert ld["gram_loss"].item() == 0.0, ld["gram_loss"].item()
_, ld_real = fresh_meta().forward_backward(make_collate_fn()([dataset_224[i] for i in range(4)]), teacher_temp=0.04)
gram_224 = ld_real["gram_loss"].item()
assert gram_224 > 0.0
print("OK (c) Gram == 0.0 exactly at 224 with gram crops == global crops, no masks, Gram teacher == student at init; "
      "with the real pipeline at 224 (distorted + masked student crops) it is %.5f, not 0" % gram_224)

# (d) EMA ----------------------------------------------------------------------------------------------------------------
torch.optim.SGD(student.parameters(), lr=1e-3).step()  # grads of (b); the student now differs from the teacher
old_teacher = {n: p.detach().clone() for n, p in teacher.named_parameters()}
new_student = {n: p.detach().clone() for n, p in student.named_parameters()}
assert any(not torch.equal(old_teacher[n], new_student[n]) for n in old_teacher)
meta.update_ema(0.9)
worst, moved = 0.0, 0.0
for n, p in teacher.named_parameters():
    expected = 0.9 * old_teacher[n].double() + 0.1 * new_student[n].double()
    assert torch.allclose(p.detach().double(), expected, rtol=1e-6, atol=1e-7), n
    worst = max(worst, ((p.detach().double() - expected).abs() / (expected.abs() + 1e-7)).max().item())
    moved = max(moved, (p.detach() - old_teacher[n]).abs().max().item())
assert moved > 1e-3  # the check is not vacuous: the update really moved the teacher
print("OK (d) EMA: every teacher param == 0.9 * old teacher + 0.1 * new student (max relative error %.1e, teacher moved by up to %.2g)"
      % (worst, moved))

# (e) Gram refresh -------------------------------------------------------------------------------------------------------
pairs = list(zip(gram_teacher.backbone.parameters(), teacher.backbone.parameters()))
assert any(not torch.equal(g, t) for g, t in pairs)  # the EMA step moved the teacher away from the Gram teacher
meta.update_gram()
assert all(torch.equal(g, t) for g, t in pairs) and not any(p.requires_grad for p in gram_teacher.parameters())
print("OK (e) refresh: update_gram() makes all %d Gram teacher backbone params bitwise equal to the EMA teacher's" % len(pairs))

# (f) export round trip into Ark's strict vit_base_dinov3 guard -------------------------------------------------------------
if WEIGHTS is None:
    print("SKIP (f) export round trip: %s not found" % WEIGHT_NAME)
else:
    _, export_teacher, _ = build_models(DINOV3_VITB16_KWARGS, WEIGHTS, TINY_HEADS)
    stock = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for p in export_teacher.backbone.parameters():
            if p.is_floating_point():
                p.add_(1e-3)  # no longer the stock weights
    exported = export_backbone(export_teacher.backbone)
    assert len(exported) == 188 == len(stock) and set(exported) == set(stock)
    assert any(not torch.equal(exported[k], stock[k]) for k in exported)
    path = os.path.join(TMP, "export.pth")
    torch.save(exported, path)
    ark_args = lambda p: SimpleNamespace(model_name="vit_base_dinov3", pretrained_weights=p, projector_features=1376, use_mlp=False)
    ark_vit = build_omni_model(ark_args(path), [6, 14, 14, 14])  # raises RuntimeError unless backbone keys match strictly
    ark_state = ark_vit.state_dict()
    assert all(torch.equal(ark_state[k], v) for k, v in exported.items())
    broken_path = os.path.join(TMP, "export_broken.pth")
    torch.save({k: v for k, v in list(exported.items())[1:]}, broken_path)  # one key removed
    try:
        build_omni_model(ark_args(broken_path), [6, 14, 14, 14])
    except RuntimeError as error:
        assert "does not match the backbone" in str(error)
    else:
        raise AssertionError("an export with a missing key passed the strict guard")
    del export_teacher, ark_vit, ark_state, stock, exported
    print("OK (f) export: 188 stock keys, loads into build_omni_model(vit_base_dinov3) with every backbone tensor bitwise equal; "
          "a missing key raises RuntimeError")

# (g)-(j), (m)-(o): the whole trainer, CPU, tiny ViT, 12 synthetic images ---------------------------------------------------------
dist.destroy_process_group()  # let run() set up its own world-size-1 group, as it does in a plain `python ssl_pretrain.py`


def make_args(out, *extra):
    argv = ["--data_set", "VinDrCXR", "--data_set", "ChestXray14", "--data_set", "MIMIC", "--data_set", "CheXpert",
            "--pretrained_weights", "unused", "--output_dir", out, "--device", "cpu", "--workers", "0",
            "--batch_size_per_gpu", "2", "--epochs", "3", "--warmup_epochs", "1", "--teacher_temp_warmup_epochs", "1",
            "--gram_refresh_steps", "4,8", "--freeze_backbone_epochs", "1", "--print_freq", "2"]
    if "--no_gram" in extra:  # parse_args refuses --gram_refresh_steps under --no_gram: drop the base pair
        i = argv.index("--gram_refresh_steps")
        del argv[i:i + 2]
    args = ssl_pretrain.parse_args(argv + list(extra))
    args.pretrained_weights = None  # random init (tests have no stock weights to load into the tiny ViT)
    return args


def go(out, *extra):
    """run() with stdout captured; returns (summary, printed text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary = ssl_pretrain.run(make_args(out, *extra), TINY_VIT, TINY_HEADS, CFG)
    return summary, buf.getvalue()


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as error:
        return str(error)
    raise AssertionError("expected %s" % exc.__name__)


def read_csv(out):
    with open(os.path.join(out, "ssl_losses.csv"), newline="") as f:
        return list(csv.reader(f))


def load_ckpt(out, name):
    return torch.load(os.path.join(out, "ckpt", name), map_location="cpu", weights_only=False)


DIR_A, DIR_B = os.path.join(TMP, "A"), os.path.join(TMP, "B")
assert not dist.is_initialized()
summary_a1, text_a1 = go(DIR_A, "--stop_after_epochs", "2", "--resume")  # chain style: --resume with nothing to resume
assert dist.is_initialized() and dist.get_world_size() == 1  # run() created the group itself
assert "no checkpoint in" in text_a1 and "starting from scratch" in text_a1
assert summary_a1 == dict(step=12, epoch=2, gram_refreshes=[4, 8], peak_lr=summary_a1["peak_lr"], epoch_length=6)
csv_a1 = open(os.path.join(DIR_A, "ssl_losses.csv")).read()
for name in ("epoch_0001.pth", "epoch_0002.pth"):
    assert os.path.isfile(os.path.join(DIR_A, "ckpt", name))
assert not os.path.exists(os.path.join(DIR_A, "ckpt", "epoch_0003.pth"))
assert "Gram refresh 1/2 after step 4: Gram teacher <- EMA teacher backbone; gram loss before (step 3 row)" in text_a1
assert "Gram refresh 2/2 after step 8:" in text_a1 and "after (step 8 row)" in text_a1

with open(os.path.join(DIR_A, "ssl_losses.csv"), "a") as f:  # rows of an epoch that crashed before its checkpoint
    f.write(",".join(["12", "2"] + ["999.0"] * 13) + "\n" + ",".join(["13", "2"] + ["999.0"] * 13) + "\n")
summary_a2, text_a2 = go(DIR_A, "--resume", "--stop_after_epochs", "3")  # the same run again: continues to epoch 3
assert "resumed from" in text_a2 and "2 epochs / 12 steps done, Gram refreshes [4, 8]" in text_a2
ck2 = load_ckpt(DIR_A, "epoch_0002.pth")
assert (ck2["step"], ck2["epoch"], ck2["gram_refreshes"], ck2["world_size"]) == (12, 2, [4, 8], 1)
assert set(ck2) == {"student", "teacher", "gram_teacher", "optimizer", "step", "epoch", "gram_refreshes", "args", "world_size", "rng", "data_fingerprint"}
assert set(ck2["rng"]) == {"python", "numpy", "torch"} and ck2["args"]["seed"] == 100
assert all(t.device.type == "cpu" for sd in ("student", "teacher", "gram_teacher") for t in ck2[sd].values())
assert summary_a2["step"] == 18 and summary_a2["epoch"] == 3 and summary_a2["gram_refreshes"] == [4, 8]
summary_b, text_b = go(DIR_B)  # uninterrupted 3 epochs, no --resume
assert summary_b == summary_a2
for out in (DIR_A, DIR_B):
    for epoch in (1, 2, 3):
        assert os.path.isfile(os.path.join(out, "ckpt", "epoch_%04d.pth" % epoch))
        assert os.path.isfile(os.path.join(out, "export", "teacher_backbone_epoch_%04d.pth" % epoch))

export_2 = torch.load(os.path.join(DIR_A, "export", "teacher_backbone_epoch_0002.pth"), weights_only=True)
assert len(export_2) == len(ck2["teacher"]) - sum(k.split(".")[0] != "backbone" for k in ck2["teacher"])  # backbone keys only
assert all(torch.equal(export_2[k], ck2["teacher"]["backbone." + k]) for k in export_2)  # the export IS the EMA teacher's backbone
assert any(not torch.equal(export_2[k], ck2["student"]["backbone." + k]) for k in export_2)  # ... not the student's
export_a = torch.load(os.path.join(DIR_A, "export", "teacher_backbone_epoch_0003.pth"), weights_only=True)
export_b = torch.load(os.path.join(DIR_B, "export", "teacher_backbone_epoch_0003.pth"), weights_only=True)
bitwise = all(torch.equal(export_a[k], export_b[k]) for k in export_b)
if not bitwise:
    assert all(torch.allclose(export_a[k], export_b[k], rtol=1e-5, atol=1e-6) for k in export_b)
    print("NOTE (g) resumed teacher backbone matches the uninterrupted run only to allclose, not bitwise")

rows_a, rows_b = read_csv(DIR_A), read_csv(DIR_B)
for rows in (rows_a, rows_b):
    assert ",".join(rows[0]) == CSV_HEADER
    steps = [int(r[0]) for r in rows[1:]]
    assert steps == sorted(set(steps)) and steps[0] == 0 and steps[-1] == 17 and len(rows[1]) == 16  # 0-based, no duplicates
    assert all(int(r[1]) == int(r[0]) // 6 for r in rows[1:])  # epoch = step // epoch_length, 0-based
    assert {3, 4, 7, 8} <= set(steps)  # the rows either side of each Gram refresh
assert open(os.path.join(DIR_A, "ssl_losses.csv")).read().startswith(csv_a1)  # the resume kept the finished rows as written
assert not any("999.0" in line for line in open(os.path.join(DIR_A, "ssl_losses.csv")))  # and dropped the crashed epoch's rows
by_step_b = {r[0]: r[1:13] for r in rows_b[1:]}  # losses, lr, wd, momentum, temp, gram weight (not the two timings)
common = [r for r in rows_a[1:] if r[0] in by_step_b]
assert len(common) >= 8
if bitwise:
    assert all(r[1:13] == by_step_b[r[0]] for r in common)  # same losses, step for step, as the uninterrupted run
print("OK (g) end to end: A (2 epochs, then --resume) and B (3 epochs) both have epoch_0001..0003 + exports; resume restored "
      "12 steps / 2 epochs / refreshes [4, 8]; final teacher backbone %s; CSV header exact, steps 0-based without duplicates"
      % ("bitwise equal" if bitwise else "allclose"))

info = json.load(open(os.path.join(DIR_A, "args.json")))
want = set(vars(make_args(DIR_A))) | {"effective_peak_lr", "effective_end_lr", "lr_scaling_factor", "world_size",
                                      "global_batch_size", "epoch_length", "total_steps"}
assert want <= set(info), want - set(info)
peak = 1e-3 * 4 * math.sqrt(2 / 1024)
assert info["effective_peak_lr"] == peak and info["effective_end_lr"] == 1e-6 * 4 * math.sqrt(2 / 1024)
assert info["lr_scaling_factor"] == 4 * math.sqrt(2 / 1024) and info["world_size"] == 1 and info["global_batch_size"] == 2
assert (info["epoch_length"], info["total_steps"], info["gram_refresh_steps"], info["stop_after_epochs"]) == (6, 18, [4, 8], 3)
assert info["data_set"] == ["VinDrCXR", "ChestXray14", "MIMIC", "CheXpert"] and info["lr"] == 1e-3
print("OK (g) args.json: %d keys incl. effective_peak_lr = %g = 1e-3 * 4*sqrt(2/1024)" % (len(info), peak))

# (h) freeze: backbone LR 0 and last_layer LR 0 in epoch 1 ----------------------------------------------------------------------
torch.manual_seed(100)
random.seed(100)
np.random.seed(100)
init_student = build_models(TINY_VIT, None, TINY_HEADS)[0].state_dict()  # what run() built before epoch 1 (same seed)
ck1 = load_ckpt(DIR_A, "epoch_0001.pth")
select = lambda sd, prefix: [k for k in sd if k.startswith(prefix)]
assert all(torch.equal(ck1["student"][k], init_student[k]) for k in select(init_student, "backbone."))  # bitwise unchanged
drift = max((ck1["teacher"][k] - init_student[k]).abs().max().item() for k in select(init_student, "backbone."))
assert drift < 1e-6  # not bitwise: 0.999 * w + 0.001 * w rounds to w only up to an ulp or so in fp32
assert any(not torch.equal(ck2["student"][k], init_student[k]) for k in select(init_student, "backbone."))
backbone_drift = lambda sd: max((sd[k] - init_student[k]).abs().max().item() for k in select(init_student, "backbone."))
student_2, teacher_2 = backbone_drift(ck2["student"]), backbone_drift(ck2["teacher"])
gram_drift_2 = max((ck2["gram_teacher"][k] - init_student[k]).abs().max().item() for k in select(init_student, "backbone."))
assert 1e-6 < teacher_2 < student_2 / 10  # EMA: moved with the student, slowly (momentum 0.999)
assert 0 < gram_drift_2 <= teacher_2  # refreshed at step 8 from the teacher of that moment; never refreshed it would still be the init
for head in ("dino_head", "ibot_head"):
    assert any(not torch.equal(ck1["student"][k], init_student[k]) for k in select(init_student, head + ".mlp."))
    assert torch.equal(ck1["student"][head + ".last_layer.weight"], init_student[head + ".last_layer.weight"])
    assert not torch.equal(ck2["student"][head + ".last_layer.weight"], init_student[head + ".last_layer.weight"])
print("OK (h) freeze: after epoch 1 the student backbone is bitwise the initial one (teacher within %.1e: EMA rounding), the head MLPs "
      "moved, the heads' last_layer did not; by epoch 2 the backbone and last_layer moved (backbone drift: student %.1e, teacher %.1e, "
      "Gram teacher %.1e)" % (drift, student_2, teacher_2, gram_drift_2))

# (i) LR rule -----------------------------------------------------------------------------------------------------------------------
assert lr_scaling_factor(256) == 2.0
sch = build_schedules(lr=1e-3, min_lr=1e-6, epochs=40, warmup_epochs=4, epoch_length=100, global_batch_size=256,
                      teacher_temp_warmup_epochs=12)
assert sch["peak_lr"] == 2e-3 and sch["end_lr"] == 2e-6 and sch["factor"] == 2.0 and max(sch["lr"]) == 2e-3
lr_lines = [line for line in text_a1.splitlines() if line.startswith("LR:")]
assert len(lr_lines) == 1 and "peak %g" % peak in lr_lines[0] and "4*sqrt(2/1024)" in lr_lines[0]
print("OK (i) LR rule: factor(256) = 2.0, peak 1e-3 -> 2e-3; run() printed one line: " + lr_lines[0])

# (j) guards ------------------------------------------------------------------------------------------------------------------------
raises(FileExistsError, go, DIR_B)  # a finished run is not overwritten without --resume
before = open(os.path.join(DIR_B, "args.json")).read()
message = raises(ValueError, go, DIR_B, "--resume", "--lr", "2e-3")
assert "cannot --resume" in message and "lr: checkpoint 0.001, now 0.002" in message and "seed" not in message
assert open(os.path.join(DIR_B, "args.json")).read() == before  # the rejected run left the record alone
assert "data_set" in raises(ValueError, go, DIR_B, "--resume", "--data_set", "MIMIC")
ck_b = load_ckpt(DIR_B, "epoch_0003.pth")
assert "world_size: checkpoint 1, now 4" in raises(ValueError, ssl_pretrain.check_resume, ck_b, make_args(DIR_B), 4, 6, ck_b["data_fingerprint"])
assert "epoch_length" in raises(ValueError, ssl_pretrain.check_resume, ck_b, make_args(DIR_B), 1, 5, ck_b["data_fingerprint"])
# same dataset names and epoch_length, different image list: rejected
assert "training image list" in raises(ValueError, ssl_pretrain.check_resume, ck_b, make_args(DIR_B), 1, 6, "0" * 64)
ssl_pretrain.check_resume(ck_b, make_args(DIR_B), 1, 6, ck_b["data_fingerprint"])  # the real one passes
# --resume with --max_steps already reached at an epoch boundary: no extra step, no maxsteps_ file
DIR_C = os.path.join(TMP, "C")
go(DIR_C, "--stop_after_epochs", "1")
with open(os.path.join(DIR_C, "ssl_losses.csv"), "a", newline="") as f:  # a row of an interrupted, uncheckpointed epoch
    csv.writer(f).writerow(["6", "1"] + ["0"] * 13)
_, text_c = go(DIR_C, "--resume", "--max_steps", "6")
assert "nothing to do: 6 steps done" in text_c and not os.path.exists(os.path.join(DIR_C, "ckpt", "maxsteps_0000007.pth"))
assert max(int(r[0]) for r in read_csv(DIR_C)[1:]) == 5  # the stale row is trimmed on the no-op path too
csv_before = open(os.path.join(DIR_B, "ssl_losses.csv")).read()
summary_done, text_done = go(DIR_B, "--resume")  # finished run + --resume: nothing to do, exit 0
assert "nothing to do" in text_done and summary_done == summary_b
assert open(os.path.join(DIR_B, "ssl_losses.csv")).read() == csv_before
raises(ValueError, go, os.path.join(TMP, "x1"), "--warmup_epochs", "4")
raises(ValueError, go, os.path.join(TMP, "x2"), "--teacher_temp_warmup_epochs", "4")
raises(ValueError, go, os.path.join(TMP, "x3"), "--stop_after_epochs", "4")
raises(ValueError, go, os.path.join(TMP, "x4"), "--stop_after_epochs", "0")
assert not any(os.path.exists(os.path.join(TMP, "x%d" % i)) for i in (1, 2, 3, 4))  # validated before anything is created
assert ssl_pretrain.parse_args(["--data_set", "MIMIC", "--pretrained_weights", "w", "--output_dir", "o",
                                "--gram_refresh_steps", "30000,10000,10000"]).gram_refresh_steps == [10000, 30000]
assert ssl_pretrain.parse_args(["--data_set", "MIMIC", "--pretrained_weights", "w", "--output_dir", "o",
                                "--gram_refresh_steps", ""]).gram_refresh_steps == []
assert ssl_pretrain.parse_args(["--data_set", "MIMIC", "--pretrained_weights", "w", "--output_dir", "o"]).gram_refresh_steps \
    == [10000, 20000, 30000]
print("OK (j) guards: no --resume into a finished run raises; --resume with other --lr / --data_set / world size / epoch_length "
      "raises ValueError naming the difference and leaves args.json alone; a changed image list of the same length raises; --resume past --max_steps takes no step; finished run + --resume is a no-op; bad epoch args raise")

# (m) --max_steps (the timing run): saves maxsteps_* and ignores them on resume; --no_lr_scaling -----------------------------------------
DIR_M = os.path.join(TMP, "M")
summary_m, text_m = go(DIR_M, "--no_lr_scaling", "--lr", "3e-4", "--max_steps", "3")
assert summary_m["step"] == 3 and summary_m["epoch"] == 0
assert os.path.isfile(os.path.join(DIR_M, "ckpt", "maxsteps_0000003.pth"))
assert os.path.isfile(os.path.join(DIR_M, "export", "teacher_backbone_maxsteps_0000003.pth"))
assert ssl_pretrain.epoch_checkpoints(os.path.join(DIR_M, "ckpt")) == []
assert load_ckpt(DIR_M, "maxsteps_0000003.pth")["step"] == 3 and "--max_steps 3 reached at step 3" in text_m
assert [r[0] for r in read_csv(DIR_M)[1:]] == ["0", "1", "2"]  # first step, then print_freq 2 -> step 2, then the last step
assert "LR: --no_lr_scaling, peak = --lr = 0.0003, end 1e-06" in text_m.splitlines()
info_m = json.load(open(os.path.join(DIR_M, "args.json")))
assert info_m["effective_peak_lr"] == 3e-4 and info_m["effective_end_lr"] == 1e-6 and info_m["lr_scaling_factor"] == 1.0
_, text_m2 = go(DIR_M, "--no_lr_scaling", "--lr", "3e-4", "--max_steps", "3", "--resume")
assert "starting from scratch" in text_m2  # maxsteps_* are not epoch checkpoints
print("OK (m) --max_steps 3: stops mid-epoch, writes ckpt/maxsteps_0000003.pth + export/teacher_backbone_maxsteps_0000003.pth, "
      "--resume ignores them; --no_lr_scaling: peak = --lr")

# (n) checkpoint retention: last 3 + every 10th --------------------------------------------------------------------------------------
prune_dir = os.path.join(TMP, "prune")
os.makedirs(prune_dir)
for epoch in range(1, 26):
    open(os.path.join(prune_dir, "epoch_%04d.pth" % epoch), "w").close()
open(os.path.join(prune_dir, "maxsteps_0000200.pth"), "w").close()
ssl_pretrain.prune_checkpoints(prune_dir)
assert [e for e, _ in ssl_pretrain.epoch_checkpoints(prune_dir)] == [10, 20, 23, 24, 25]
assert os.path.exists(os.path.join(prune_dir, "maxsteps_0000200.pth"))
print("OK (n) retention: of epochs 1..25 only 10, 20, 23, 24, 25 remain")

# (o) a NaN loss at a log step: the row is written, then the run aborts ------------------------------------------------------------------
DIR_N = os.path.join(TMP, "N")
real_forward_backward = SSLMetaArch.forward_backward
SSLMetaArch.forward_backward = lambda self, data, *, teacher_temp: (
    lambda out: (out[0] * float("nan"), out[1]))(real_forward_backward(self, data, teacher_temp=teacher_temp))
try:
    message = raises(RuntimeError, go, DIR_N, "--max_steps", "4")
    SSLMetaArch.forward_backward = lambda self, data, *, teacher_temp: (
        lambda out: (out[0] * float("inf"), out[1]))(real_forward_backward(self, data, teacher_temp=teacher_temp))
    message_inf = raises(RuntimeError, go, os.path.join(TMP, "N_inf"), "--max_steps", "4")
finally:
    SSLMetaArch.forward_backward = real_forward_backward
rows_n = read_csv(DIR_N)
assert "non-finite loss nan at step 0" in message and len(rows_n) == 2 and rows_n[1][0] == "0" and rows_n[1][7] == "nan"
assert "non-finite loss nan at step 0 on rank 0 (all-reduced over 1 ranks)" in message
assert "non-finite loss inf at step 0" in message_inf
print("OK (o) NaN / inf: the first log step wrote its row and raised RuntimeError")

# (p) per-step order (dinov3 do_train) and Ark's mark_step placement ------------------------------------------------------------------
events = []


def spied(owner, name, tag, check=None):
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        if check:
            check(kwargs)
        events.append(tag)
        return original(*args, **kwargs)

    setattr(owner, name, wrapper)
    return lambda: setattr(owner, name, original)


def check_clip(kwargs):
    assert kwargs["max_norm"] == 3.0


def check_zero_grad(kwargs):
    assert kwargs["set_to_none"] is True


restore = [spied(SSLMetaArch, "forward_backward", "fb"), spied(SSLMetaArch, "update_ema", "ema"),
           spied(SSLMetaArch, "update_gram", "gram"), spied(Accelerator, "mark_step", "mark"),
           spied(torch.nn.utils, "clip_grad_norm_", "clip", check_clip), spied(torch.optim.AdamW, "step", "opt.step"),
           spied(torch.optim.AdamW, "zero_grad", "zero_grad", check_zero_grad)]
try:
    go(os.path.join(TMP, "P"), "--max_steps", "5", "--gram_refresh_steps", "3", "--print_freq", "100")
finally:
    for undo in restore:
        undo()
one_step = ["zero_grad", "fb", "mark", "clip", "clip", "clip", "opt.step", "mark", "ema", "mark"]  # clip: backbone, dino_head, ibot_head
assert events == one_step * 3 + ["gram", "mark"] + one_step * 2, events
print("OK (p) step order: zero_grad(set_to_none) -> forward_backward -> mark_step -> clip_grad_norm_(3.0) x3 (one per student submodule) "
      "-> optimizer.step -> mark_step -> update_ema -> mark_step; a Gram refresh adds update_gram -> mark_step after its step")

# (q) a log step reads each loss on the host (.item()), builds a tensor from those floats and all-reduces that, as Ark's trainer.py does ---
# (never a torch.stack of the lazy loss tensors: an all_reduce of that died on Gaudi in lazy mode, concat); the CSV loss columns are the
# global means. The world is faked as test_freeze_backbone.py (d2) does: the one-rank gloo group is still up from the runs above and
# WORLD_SIZE=2 is in the env, so Accelerator is distributed (its all_reduce branch runs; the sum over the one real rank is that rank's
# own value) and world_size is 2: the logged values must be half of the per-rank reads.
LOSS_ORDER = ("dino_global_crops_loss", "dino_local_crops_loss", "koleo_loss", "ibot_loss", "gram_loss")  # CSV columns 2..6, then total
step_values, reduced, builds, reduce_callers, stack_callers = [], [], [], set(), set()
real_dist_all_reduce, real_accelerator_all_reduce = dist.all_reduce, Accelerator.all_reduce
real_torch_tensor, real_torch_stack = torch.tensor, torch.stack
caller = lambda: os.path.basename(sys._getframe(2).f_code.co_filename)  # the file that called the spy that called this


def recording_forward_backward(self, data, *, teacher_temp):
    loss, loss_dict = real_forward_backward(self, data, teacher_temp=teacher_temp)
    step_values.append([loss_dict[k].item() for k in LOSS_ORDER] + [loss.item()])  # this rank's own reads, as the log step makes them
    return loss, loss_dict


def recording_accelerator_all_reduce(self, tensor):  # what the log step hands the collective
    reduced.append((tensor.dtype, tuple(tensor.shape), tensor.is_contiguous(), tensor.requires_grad, tensor.device.type, tensor.tolist()))
    return real_accelerator_all_reduce(self, tensor)


def recording_dist_all_reduce(*args, **kwargs):
    reduce_callers.add(caller())
    return real_dist_all_reduce(*args, **kwargs)


def recording_torch_tensor(data, **kwargs):
    if caller() == "ssl_pretrain.py":
        builds.append(isinstance(data, list) and all(type(v) is float for v in data))  # built from Python floats?
    return real_torch_tensor(data, **kwargs)


def recording_torch_stack(*args, **kwargs):
    stack_callers.add(caller())
    return real_torch_stack(*args, **kwargs)


assert dist.is_initialized() and dist.get_world_size() == 1  # run() reuses it (else it would wait on env:// for a second rank)
saved_env = {key: os.environ.get(key) for key in ("RANK", "WORLD_SIZE")}
os.environ.update(RANK="0", WORLD_SIZE="2")
SSLMetaArch.forward_backward = recording_forward_backward
Accelerator.all_reduce, dist.all_reduce = recording_accelerator_all_reduce, recording_dist_all_reduce
torch.tensor, torch.stack = recording_torch_tensor, recording_torch_stack
try:
    go(os.path.join(TMP, "Q"), "--max_steps", "5", "--print_freq", "100")  # epoch_length 3 (12 images / 2 ranks / 2 each): rows 0, 2, 3, 4
finally:
    SSLMetaArch.forward_backward = real_forward_backward
    Accelerator.all_reduce, dist.all_reduce = real_accelerator_all_reduce, real_dist_all_reduce
    torch.tensor, torch.stack = real_torch_tensor, real_torch_stack
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
rows_q = read_csv(os.path.join(TMP, "Q"))
log_steps = [int(r[0]) for r in rows_q[1:]]
assert log_steps == [0, 2, 3, 4] and len(step_values) == 5  # step 1 is not a log step: no collective there
assert [r[:5] for r in reduced] == [(torch.float32, (6,), True, False, "cpu")] * 4  # one per log step, a plain float32 leaf
assert [r[5] for r in reduced] == [step_values[i] for i in log_steps]  # holding that step's own .item() reads
assert all([float(x) for x in r[2:8]] == [v / 2 for v in step_values[int(r[0])]] for r in rows_q[1:])  # CSV = sum over ranks / world_size
assert builds == [True] * 4 and "ssl_pretrain.py" not in stack_callers  # built by torch.tensor from Python floats, no torch.stack
assert reduce_callers == {"accelerator.py", "ibot_patch_loss.py", "dino_clstoken_loss.py"}, reduce_callers  # the log path + Sinkhorn-Knopp
print("OK (q) log path: per log step one Accelerator.all_reduce of a float32 tensor built by torch.tensor from the step's own .item() "
      "reads (no torch.stack in ssl_pretrain.py); the CSV loss columns are that sum / world_size (2); dist.all_reduce is called from "
      + ", ".join(sorted(reduce_callers)))

# (r) crop sizes (validated, recorded, locked on --resume) and --init_from_checkpoint ---------------------------------------------------------
P = lambda *extra: ssl_pretrain.parse_args(["--data_set", "MIMIC", "--pretrained_weights", "w", "--output_dir", "o"] + list(extra))
assert (P().global_crops_size, P().local_crops_size, P().gram_teacher_size, P().init_from_checkpoint) == (224, 96, 448, None)
assert P("--global_crops_size", "512").gram_teacher_size == 1024  # the Gram teacher follows the global size (2x) unless it is given
assert P("--global_crops_size", "512", "--gram_teacher_size", "512").gram_teacher_size == 512
assert P("--gram_refresh_steps", "none").gram_refresh_steps == [] == P("--gram_refresh_steps", " None ").gram_refresh_steps
with contextlib.redirect_stderr(io.StringIO()) as err:
    message = raises(SystemExit, ssl_pretrain.parse_args, ["--data_set", "MIMIC", "--output_dir", "o"])
assert "--pretrained_weights is required unless --init_from_checkpoint is given" in err.getvalue(), err.getvalue()
assert ssl_pretrain.parse_args(["--data_set", "MIMIC", "--output_dir", "o", "--init_from_checkpoint", "c.pth"]).pretrained_weights is None
for extra, words in ((["--global_crops_size", "200"], "--global_crops_size 200 is not a multiple of the patch size 16"),
                     (["--local_crops_size", "100"], "--local_crops_size 100 is not a multiple of the patch size 16"),
                     (["--gram_teacher_size", "450"], "--gram_teacher_size 450 is not a multiple of the patch size 16"),
                     (["--local_crops_size", "224"], "got local 224, global 224, Gram teacher 448"),  # local == global
                     (["--local_crops_size", "256"], "got local 256, global 224, Gram teacher 448"),  # local > global
                     (["--gram_teacher_size", "464"], "got local 96, global 224, Gram teacher 464"),  # Gram > 2 x global
                     (["--global_crops_size", "512", "--gram_teacher_size", "448"], "got local 96, global 512, Gram teacher 448"),  # Gram < global
                     (["--local_crops_size", "0"], "got local 0, global 224")):
    message = raises(ValueError, go, os.path.join(TMP, "r_bad"), *extra)
    assert words in message and ("0 < local < global <= Gram teacher <= 2 x global" in message or "patch size" in message), (extra, message)
assert not os.path.exists(os.path.join(TMP, "r_bad"))  # validated before anything is created
assert ssl_pretrain.check_args(P("--global_crops_size", "512", "--local_crops_size", "224")) == 40  # 512 / 224 / 1024 is valid
assert ssl_pretrain.check_args(P("--global_crops_size", "512", "--local_crops_size", "224", "--gram_teacher_size", "512")) == 40
# recorded in args.json and the checkpoint; locked on --resume, an old checkpoint without the keys reading as 224 / 96 / 448
info_b = json.load(open(os.path.join(DIR_B, "args.json")))
assert (info_b["global_crops_size"], info_b["local_crops_size"], info_b["gram_teacher_size"], info_b["init_from_checkpoint"]) == (224, 96, 448, None)
assert (ck_b["args"]["global_crops_size"], ck_b["args"]["local_crops_size"], ck_b["args"]["gram_teacher_size"]) == (224, 96, 448)
assert {"global_crops_size", "local_crops_size", "gram_teacher_size", "init_from_checkpoint"} <= set(ssl_pretrain.RESUME_MATCH)
before = open(os.path.join(DIR_B, "args.json")).read()
message = raises(ValueError, go, DIR_B, "--resume", "--global_crops_size", "256")
assert "cannot --resume" in message and "global_crops_size: checkpoint 224, now 256" in message and "gram_teacher_size: checkpoint 448, now 512" in message, message
message = raises(ValueError, go, DIR_B, "--resume", "--local_crops_size", "64")
assert "local_crops_size: checkpoint 96, now 64" in message and "global_crops_size" not in message, message
assert open(os.path.join(DIR_B, "args.json")).read() == before
old_ckpt = dict(ck_b, args={k: v for k, v in ck_b["args"].items() if k not in ("global_crops_size", "local_crops_size", "init_from_checkpoint")})
ssl_pretrain.check_resume(old_ckpt, make_args(DIR_B), 1, 6, ck_b["data_fingerprint"])  # a pre-flag checkpoint resumes at the defaults
assert "global_crops_size: checkpoint 224, now 256" in raises(ValueError, ssl_pretrain.check_resume, old_ckpt, make_args(DIR_B, "--global_crops_size", "256"), 1, 6,
                                                              ck_b["data_fingerprint"])
print("OK (r) crops: defaults 224 / 96 / 448, Gram = 2 x global unless given, --gram_refresh_steps none = []; not x16, local >= global, Gram > 2 x global, "
      "Gram < global raise ValueError before anything is created; recorded in args.json + checkpoint; a --resume with another global or local size is "
      "refused naming both values; a checkpoint without the keys resumes as 224 / 96 / 448")

# the JPEG decode side follows the crops (Codex r2 P1, r3 P2): the short side of the smallest crop region (global / Gram at area 0.25, local at 0.04,
# aspect ratio down to 3/4) must not be upsampled. 224 / 96 / 448 keeps the old 1024 (Stage A's runs resume across the change; the rule alone gives
# 1088); 512 / 224 / 1024 needs 2368; with a 512 Gram the local crops' 1344 wins
R = (3 / 4, 4 / 3)  # torchvision's RandomResizedCrop default, which both of DataAugmentationCXR's crops use
assert [ssl_data.decode_min_side(c, s_, R) for c, s_ in ((1024, 0.25), (512, 0.25), (224, 0.04), (96, 0.04), (448, 0.25))] == [2368, 1216, 1344, 1024, 1088]
assert ssl_data.decode_min_side(1024, 0.25, (1, 1)) == 2048 and ssl_data.decode_min_side(1024, 0.25, (4 / 3, 3 / 4)) == 2368  # either way round
for sizes, side in (((224, 96, 448), 1024), ((512, 224, 1024), 2368), ((512, 224, 512), 1344), ((256, 112, 512), 1216)):
    augment = DataAugmentationCXR(global_crops_size=sizes[0], local_crops_size=sizes[1], gram_teacher_crops_size=sizes[2])
    assert augment.decode_min_side == side, (sizes, augment.decode_min_side)
    assert augment.geometric_augmentation_global.transforms[0].ratio == R == augment.geometric_augmentation_local.transforms[0].ratio
big = os.path.join(TMP, "big", "x.jpg")  # a MIMIC-sized JPEG: 2600 wide, 3000 high
os.makedirs(os.path.dirname(big))
Image.fromarray(np.random.RandomState(1).randint(0, 255, (3000, 2600), dtype=np.uint8)).save(big, quality=90)
for sizes, side, want in (((224, 96, 448), 1024, (1300, 1500)), ((512, 224, 1024), 2368, (2600, 3000))):
    augment, decoded = DataAugmentationCXR(global_crops_size=sizes[0], local_crops_size=sizes[1], gram_teacher_crops_size=sizes[2]), []
    big_set = ssl_data.SSLImageDataset([big], lambda img: (decoded.append(img.size), augment(img))[1], augment.decode_min_side)
    crops = big_set[0][0]
    assert decoded == [want] and min(want) >= side, (sizes, decoded)  # draft: halves while both sides stay >= the target, never below it
    assert tuple(crops["gram_teacher_crops"][0].shape) == (3, sizes[2], sizes[2]) and tuple(crops["global_crops"][0].shape) == (3, sizes[0], sizes[0])
with Image.open(big) as img:
    img.draft("L", (1024, 1024))
    assert img.size == (1300, 1500)  # the 224 runs: exactly what the fixed DECODE_MIN_SIDE 1024 decoded before
# ... and run() uses, prints and records it: the default run above at 1024, a 512 / 224 run (Gram 2 x 512 = 1024) at 2368. The 512 run stops at
# its first forward (nothing to train at that size here): the line, args.json and the dataset are all in place by then
info_a = json.load(open(os.path.join(DIR_A, "args.json")))
assert info_a["decode_min_side"] == 1024 and "Gram teacher 448; JPEG decode min side 1024" in text_a1, info_a["decode_min_side"]


class StopAtForward(Exception):
    pass


def stop_at_forward(self, data, *, teacher_temp):
    raise StopAtForward()


real_build_dataset, decode_sides = ssl_pretrain.build_dataset, []


def recording_build_dataset(*args, **kwargs):
    built_set = real_build_dataset(*args, **kwargs)
    decode_sides.append(built_set.decode_side)
    return built_set


DIR_D = os.path.join(TMP, "D512")
SSLMetaArch.forward_backward, ssl_pretrain.build_dataset = stop_at_forward, recording_build_dataset
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        raises(StopAtForward, ssl_pretrain.run, make_args(DIR_D, "--global_crops_size", "512", "--local_crops_size", "224"), TINY_VIT, TINY_HEADS, CFG)
finally:
    SSLMetaArch.forward_backward, ssl_pretrain.build_dataset = real_forward_backward, real_build_dataset
assert "crops: global 512, local 224 (x8), Gram teacher 1024; JPEG decode min side 2368" in buf.getvalue(), buf.getvalue()
assert json.load(open(os.path.join(DIR_D, "args.json")))["decode_min_side"] == 2368 and decode_sides == [2368]
print("OK (r) decode side: decode_min_side = max(1024, ceil(crop / (sqrt(min area) x sqrt(min ratio 3/4)) / 64) x 64), ratio read from the crops: "
      "224 / 96 / 448 -> 1024 (Stage A, kept; a 2600 x 3000 JPEG decodes to 1300 x 1500, as with the fixed 1024 before), 512 / 224 / 1024 -> 2368 "
      "(2600 x 3000, no draft shrink), 512 / 224 / 512 -> 1344 (the local crops), 256 / 112 / 512 -> 1216; run() hands it to the dataset, prints it "
      "on the crops: line and records decode_min_side in args.json (1024 / 2368)")

# --init_from_checkpoint: DIR_B's last full checkpoint (18 steps, 3 epochs, Gram refreshed at steps 4 and 8, so its Gram teacher != its EMA teacher)
INIT, OTHER = os.path.join(DIR_B, "ckpt", "epoch_0003.pth"), os.path.join(DIR_A, "ckpt", "epoch_0003.pth")
src = load_ckpt(DIR_B, "epoch_0003.pth")
backbone_of = lambda sd: {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
assert any(not torch.equal(src["gram_teacher"]["backbone." + k], v) for k, v in backbone_of(src["teacher"]).items())  # the re-anchor check is not vacuous
HIRES = ["--init_from_checkpoint", INIT, "--global_crops_size", "64", "--local_crops_size", "32", "--gram_refresh_steps", "none",
         "--warmup_epochs", "0", "--freeze_backbone_epochs", "0", "--epochs", "2"]
DIR_H, built, batch_shapes, real_meta_init = os.path.join(TMP, "H"), [], [], SSLMetaArch.__init__


def recording_forward_backward(self, data, *, teacher_temp):  # the shapes of every batch the hi-res run trains on
    batch_shapes.append({k: tuple(v.shape) for k, v in data.items() if k.startswith("collated_") and k != "collated_masks"} | {"masks": tuple(data["collated_masks"].shape)})
    return real_forward_backward(self, data, teacher_temp=teacher_temp)


def recording_meta_init(self, student, teacher, gram_teacher, **kwargs):  # the models as run() hands them over, before any step
    built.append({name: {k: v.detach().clone() for k, v in m.state_dict().items()}
                  for name, m in (("student", student), ("teacher", teacher), ("gram_teacher", gram_teacher))})
    return real_meta_init(self, student, teacher, gram_teacher, **kwargs)


args_h = make_args(DIR_H, *HIRES, "--stop_after_epochs", "1")
args_h.pretrained_weights = "unused"  # both given: the checkpoint wins, with a warning (and the weights path is never read)
SSLMetaArch.__init__, SSLMetaArch.forward_backward = recording_meta_init, recording_forward_backward
try:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary_h1 = ssl_pretrain.run(args_h, TINY_VIT, TINY_HEADS, CFG)
    text_h1 = buf.getvalue()
finally:
    SSLMetaArch.__init__, SSLMetaArch.forward_backward = real_meta_init, real_forward_backward
assert len(built) == 1 and len(batch_shapes) == 6
assert all(b == {"collated_global_crops": (4, 3, 64, 64), "collated_local_crops": (16, 3, 32, 32), "collated_gram_teacher_crops": (4, 3, 128, 128),
                 "masks": (4, 16)} for b in batch_shapes), batch_shapes  # 2 x 2 global, 8 x 2 local, 2 x 2 Gram crops; masks on the 4 x 4 patch grid
assert set(built[0]["student"]) == set(src["student"]) and all(torch.equal(built[0]["student"][k], v) for k, v in src["student"].items())
assert set(built[0]["teacher"]) == set(src["teacher"]) and all(torch.equal(built[0]["teacher"][k], v) for k, v in src["teacher"].items())  # heads included
gram_h = backbone_of(built[0]["gram_teacher"])
assert set(gram_h) == set(backbone_of(src["teacher"])) and all(torch.equal(gram_h[k], v) for k, v in backbone_of(src["teacher"]).items())  # re-anchored
assert "WARNING: --pretrained_weights unused ignored: --init_from_checkpoint supplies every weight" in text_h1, text_h1
assert "crops: global 64, local 32 (x8), Gram teacher 128" in text_h1
assert ("initialized from %s: its step 18, epoch 3, crops global 224 / local 96 / Gram teacher 448;" % INIT) in text_h1, text_h1
assert "Gram refresh" not in text_h1 and summary_h1["step"] == 6 and summary_h1["epoch"] == 1 and summary_h1["gram_refreshes"] == []
ck_h1 = load_ckpt(DIR_H, "epoch_0001.pth")
assert (ck_h1["step"], ck_h1["epoch"], ck_h1["gram_refreshes"]) == (6, 1, [])  # a NEW run: counted from step 0
assert {int(state["step"]) for state in ck_h1["optimizer"]["state"].values()} == {6}  # a fresh optimizer: 6 AdamW steps, not 18 + 6
assert all(torch.equal(ck_h1["gram_teacher"]["backbone." + k], v) for k, v in backbone_of(src["teacher"]).items())  # no refresh: still the init teacher
assert any(not torch.equal(ck_h1["student"][k], v) for k, v in src["student"].items())  # and the student trained
info_h = json.load(open(os.path.join(DIR_H, "args.json")))
assert (info_h["init_from_checkpoint"], info_h["global_crops_size"], info_h["local_crops_size"], info_h["gram_teacher_size"]) == (INIT, 64, 32, 128)
assert ck_h1["args"]["init_from_checkpoint"] == INIT and os.path.isfile(os.path.join(DIR_H, "export", "teacher_backbone_epoch_0001.pth"))
summary_h2, text_h2 = go(DIR_H, *HIRES, "--resume", "--stop_after_epochs", "2")  # the same run again, as a chain re-submit sends it: it resumes itself
assert "resumed from" in text_h2 and "1 epochs / 6 steps done" in text_h2 and "initialized from" not in text_h2, text_h2
assert summary_h2["step"] == 12 and summary_h2["epoch"] == 2 and os.path.isfile(os.path.join(DIR_H, "export", "teacher_backbone_epoch_0002.pth"))
# the init file is read only when the run starts: a re-submit of a run that has its own epoch checkpoint resumes with the source gone (Codex r1, P2)
SRC_COPY, DIR_H2 = os.path.join(TMP, "init_copy.pth"), os.path.join(TMP, "H2")
shutil.copy(INIT, SRC_COPY)
HIRES2 = [SRC_COPY if a == INIT else a for a in HIRES]
assert go(DIR_H2, *HIRES2, "--stop_after_epochs", "1")[0]["epoch"] == 1
os.remove(SRC_COPY)
summary_h3, text_h3 = go(DIR_H2, *HIRES2, "--resume", "--stop_after_epochs", "2")
assert "resumed from" in text_h3 and "initialized from" not in text_h3 and summary_h3["epoch"] == 2, text_h3
assert json.load(open(os.path.join(DIR_H2, "args.json")))["init_from_checkpoint"] == SRC_COPY  # still recorded
gone = os.path.join(TMP, "elsewhere.pth")  # a resume naming another init is refused by the comparison even when that path does not exist either
message = raises(ValueError, go, DIR_H2, *[gone if a == SRC_COPY else a for a in HIRES2], "--resume", "--stop_after_epochs", "2")
assert ("init_from_checkpoint: checkpoint %r, now %r" % (SRC_COPY, gone)) in message, message
assert "is not a file" in raises(FileNotFoundError, go, os.path.join(TMP, "r_missing2"), *HIRES2, "--resume")  # a start (no own checkpoint): refused
assert not os.path.exists(os.path.join(TMP, "r_missing2"))
print("OK (r) init file needed only at the start: init -> 1 epoch -> source deleted -> --resume continues to epoch 2 (path still recorded); a resume "
      "naming another (missing) init is refused by the comparison; a fresh start with the source missing is refused before anything is created, "
      "--resume or not")
# refusals: another init into this run, an init into a resumable run that had none, no --resume, a missing file, shape / key mismatches, an export
before_h = open(os.path.join(DIR_H, "args.json")).read()
message = raises(ValueError, go, DIR_H, *HIRES, "--init_from_checkpoint", OTHER, "--resume", "--stop_after_epochs", "2")
assert "cannot --resume" in message and ("init_from_checkpoint: checkpoint %r, now %r" % (INIT, OTHER)) in message, message
message = raises(ValueError, go, DIR_B, "--resume", "--init_from_checkpoint", INIT)
assert "cannot --resume" in message and ("init_from_checkpoint: checkpoint None, now %r" % INIT) in message, message
assert open(os.path.join(DIR_H, "args.json")).read() == before_h and open(os.path.join(DIR_B, "args.json")).read() == before
assert "already holds epoch checkpoints" in raises(FileExistsError, go, DIR_H, *HIRES)
assert "is not a file" in raises(FileNotFoundError, go, os.path.join(TMP, "r_missing"), "--init_from_checkpoint", os.path.join(TMP, "nope.pth"))
assert not os.path.exists(os.path.join(TMP, "r_missing"))
for name, broken, words in (("shape", dict(src, student=dict(src["student"], **{"backbone.cls_token": torch.zeros(1, 1, 32)})), "size mismatch"),
                            ("missing", dict(src, student={k: v for k, v in src["student"].items() if k != "dino_head.last_layer.weight"}), "Missing key(s)"),
                            ("unexpected", dict(src, teacher=dict(src["teacher"], extra=torch.zeros(1))), "Unexpected key(s)")):
    path = os.path.join(TMP, "init_%s.pth" % name)
    torch.save(broken, path)
    assert words in raises(RuntimeError, go, os.path.join(TMP, "r_" + name), "--init_from_checkpoint", path), name
assert "not a full checkpoint" in raises(ValueError, go, os.path.join(TMP, "r_export"), "--init_from_checkpoint",
                                         os.path.join(DIR_B, "export", "teacher_backbone_epoch_0003.pth"))
print("OK (r) --init_from_checkpoint: a NEW run whose student and teacher (heads included) are bitwise the checkpoint's and whose Gram teacher is its "
      "EMA teacher's backbone (not its own Gram teacher); step 0, a fresh optimizer, no refresh with 'none', exports every epoch; --pretrained_weights "
      "too -> WARNING, the checkpoint wins; a re-submit resumes the run itself; refused: another init, an init into a resumable run that had none, "
      "no --resume, a missing file, a shape mismatch, a missing or unexpected key (strict), a backbone-only export")

# (s) --no_gram: no Gram teacher built, loaded or run; no Gram loss; no refreshes ---------------------------------------------------------------
from dinov3.loss import GramLoss

# parse: the Gram knobs resolve per mode, and an explicit one under --no_gram is refused (the text after "error: " names it; the usage text above it names all)
assert (P().no_gram, P().gram_weight, P().gram_refresh_steps) == (False, 2.0, [10000, 20000, 30000])
assert ssl_pretrain.DEFAULT_GRAM_REFRESH_STEPS == "10000,20000,30000" and P("--gram_weight", "3").gram_weight == 3.0
ng_parsed = P("--no_gram")
assert (ng_parsed.no_gram, ng_parsed.gram_weight, ng_parsed.gram_refresh_steps, ng_parsed.gram_teacher_size) == (True, 0.0, [], 448)
assert P("--no_gram", "--global_crops_size", "512").gram_teacher_size == 1024  # still 2 x global: the size the global crops are cut at
for extra, flag in ((["--gram_weight", "2"], "--gram_weight"), (["--gram_weight", "0"], "--gram_weight"),
                    (["--gram_refresh_steps", "10000"], "--gram_refresh_steps"), (["--gram_refresh_steps", "none"], "--gram_refresh_steps"),
                    (["--gram_refresh_steps", ""], "--gram_refresh_steps"), (["--gram_teacher_size", "448"], "--gram_teacher_size")):
    with contextlib.redirect_stderr(io.StringIO()) as err:
        raises(SystemExit, P, "--no_gram", *extra)
    complaint = err.getvalue().split("error: ", 1)[1]
    assert flag in complaint and "--no_gram" in complaint, (extra, err.getvalue())
print("OK (s) parse: no flag -> no_gram False, gram_weight 2.0, refreshes 10000,20000,30000; --no_gram -> gram_weight 0.0, refreshes [], Gram size 2 x global "
      "(448); an explicit --gram_weight (2 or 0), --gram_refresh_steps (10000, none, empty) or --gram_teacher_size with --no_gram exits 2 naming the flag")


def ng_seeded(fn, *args, **kwargs):  # run() seeds random, numpy and torch before it builds anything
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    return fn(*args, **kwargs)


# models: no Gram teacher is built, and student / teacher come out bitwise as with one (deepcopy draws no RNG)
models_on = ng_seeded(build_models, TINY_VIT, None, TINY_HEADS)
models_off = ng_seeded(build_models, TINY_VIT, None, TINY_HEADS, gram=False)
assert models_on[2] is not None and models_off[2] is None
for with_gram, without_gram in zip(models_on[:2], models_off[:2]):
    assert list(with_gram.state_dict()) == list(without_gram.state_dict())
    assert all(torch.equal(v, without_gram.state_dict()[k]) for k, v in with_gram.state_dict().items())
# data: emit_gram_teacher_crops=False drops the Gram crops and changes nothing else (same seeds: bitwise the same student / teacher crops, same decode side)
ng_image = Image.fromarray(np.random.RandomState(3).randint(0, 255, (300, 400), dtype=np.uint8), "L")
crops_on = ng_seeded(DataAugmentationCXR(), ng_image)
crops_off = ng_seeded(DataAugmentationCXR(emit_gram_teacher_crops=False), ng_image)
assert tuple(crops_on["gram_teacher_crops"][0].shape) == (3, 448, 448) and tuple(crops_off["global_crops"][0].shape) == (3, 224, 224)
assert "gram_teacher_crops" not in crops_off and set(crops_off) == set(crops_on) - {"gram_teacher_crops"}
for key in ("global_crops", "global_crops_teacher", "local_crops"):
    assert len(crops_off[key]) == len(crops_on[key]) and all(torch.equal(a, b) for a, b in zip(crops_off[key], crops_on[key])), key
assert DataAugmentationCXR().decode_min_side == DataAugmentationCXR(emit_gram_teacher_crops=False).decode_min_side == 1024
print("OK (s) models + data: build_models(gram=False) -> no Gram teacher, student and teacher bitwise those of gram=True; the transform without Gram crops "
      "gives bitwise the same global / teacher / local crops and decode side (1024) from the same seeds")


def ng_refuse(*args, **kwargs):
    raise AssertionError("a Gram code path ran in --no_gram mode")


@contextlib.contextmanager
def gram_code_forbidden():  # SSLMetaArch's Gram forward and refresh, and GramLoss.forward, raise while this is active
    saved = SSLMetaArch.get_gram_teacher_output, SSLMetaArch.update_gram, GramLoss.forward
    SSLMetaArch.get_gram_teacher_output = SSLMetaArch.update_gram = GramLoss.forward = ng_refuse
    try:
        yield
    finally:
        SSLMetaArch.get_gram_teacher_output, SSLMetaArch.update_gram, GramLoss.forward = saved


# meta step: SSLMetaArch with gram_teacher=None on a batch without Gram crops runs, and the total has no Gram term
assert dist.is_initialized()  # the trainer runs above leave their world-size-1 group up, which SSLMetaArch's Sinkhorn-Knopp needs
ng_student, ng_teacher, ng_gram = ng_seeded(build_models, TINY_VIT, None, TINY_HEADS, gram=False)
meta_ng = SSLMetaArch(ng_student, ng_teacher, None, n_local_crops=8).train()
ng_dataset = build_dataset(list(DATASET_NAMES), CFG, DataAugmentationCXR(emit_gram_teacher_crops=False), check_exists=False)
ng_batch = make_collate_fn()([ng_dataset[i] for i in range(4)])
assert ng_gram is None and meta_ng.gram_teacher is None and meta_ng.gram_loss is None
assert "collated_gram_teacher_crops" not in ng_batch and tuple(ng_batch["collated_global_crops"].shape) == (8, 3, 224, 224)
with gram_code_forbidden():
    ng_loss, ng_dict = meta_ng.forward_backward(ng_batch, teacher_temp=0.04)
assert set(ng_dict) == LOSS_KEYS - {"gram_loss"} and torch.isfinite(ng_loss) and all(torch.isfinite(v) for v in ng_dict.values())
ng_want = (16 / 18 * ng_dict["dino_local_crops_loss"].item() + 2 / 18 * ng_dict["dino_global_crops_loss"].item()
           + 0.1 * 2 * ng_dict["koleo_loss"].item() + 1.0 * ng_dict["ibot_loss"].item())  # 2 global x 8 local crops: 16 local terms, 2 global, scale 2 for KoLeo
assert math.isclose(ng_loss.item(), ng_want, rel_tol=1e-6), (ng_loss.item(), ng_want)
print("OK (s) meta step: SSLMetaArch(gram_teacher=None) on a batch without collated_gram_teacher_crops runs with the Gram forward and GramLoss.forward "
      "set to raise; loss keys = the four others; total %.5f = 16/18 dino_local + 2/18 dino_global + 0.1 x 2 koleo + ibot (no Gram term)" % ng_loss.item())

# end to end: A = 1 epoch then --resume to 2, B = 2 epochs straight, both --no_gram; the spies see what run() hands SSLMetaArch and what it trains on
ng_inits, ng_batches = [], []


def ng_meta_init(self, student, teacher, gram_teacher, **kwargs):  # the models as run() hands them over, before any step
    ng_inits.append(dict(gram_teacher=gram_teacher, **{name: {k: v.detach().clone() for k, v in m.state_dict().items()}
                                                     for name, m in (("student", student), ("teacher", teacher))}))
    return real_meta_init(self, student, teacher, gram_teacher, **kwargs)


def ng_forward_backward(self, data, *, teacher_temp):
    ng_batches.append(set(data))
    return real_forward_backward(self, data, teacher_temp=teacher_temp)


@contextlib.contextmanager
def spying_on_meta():  # SSLMetaArch.__init__ / forward_backward record into ng_inits / ng_batches (cleared on entry) while this is active
    ng_inits.clear()
    ng_batches.clear()
    SSLMetaArch.__init__, SSLMetaArch.forward_backward = ng_meta_init, ng_forward_backward
    try:
        yield
    finally:
        SSLMetaArch.__init__, SSLMetaArch.forward_backward = real_meta_init, real_forward_backward


OFF_LINE = "Gram: OFF (--no_gram): no Gram teacher built or loaded, no Gram loss, no Gram refreshes; ssl_losses.csv gram and gram_weight columns are 0.0"
NG_A, NG_B = os.path.join(TMP, "NG_A"), os.path.join(TMP, "NG_B")
with spying_on_meta(), gram_code_forbidden():
    summary_nga1, text_nga1 = go(NG_A, "--no_gram", "--stop_after_epochs", "1")
    summary_nga2, text_nga2 = go(NG_A, "--no_gram", "--resume", "--stop_after_epochs", "2")
    summary_ngb, text_ngb = go(NG_B, "--no_gram", "--stop_after_epochs", "2")
assert [m["gram_teacher"] for m in ng_inits] == [None] * 3  # run() never built a Gram teacher: SSLMetaArch got None in all three runs
assert len(ng_batches) == 6 + 6 + 12 and all("collated_global_crops" in keys and "collated_gram_teacher_crops" not in keys for keys in ng_batches)
for text in (text_nga1, text_nga2, text_ngb):
    lines = text.splitlines()
    # no "Gram refresh scheduled" / "Gram refresh k/n" line, or any other Gram refresh text: only the OFF line ("no Gram refreshes") and the resumed line
    # ("Gram refreshes []") may name it
    assert OFF_LINE in lines and not any("Gram refresh" in line for line in lines if line != OFF_LINE and not line.startswith("resumed from"))
    assert ("crops: global 224, local 96 (x8), no Gram teacher (global crops cut at 448 and resized to 224, as in the Gram runs); "
            "JPEG decode min side 1024") in lines, text
resumed = [line for line in text_nga2.splitlines() if line.startswith("resumed from")]
assert len(resumed) == 1 and resumed[0].endswith("1 epochs / 6 steps done, Gram refreshes [], student/teacher/optimizer/RNG restored"), resumed
assert summary_nga2 == summary_ngb and (summary_ngb["step"], summary_ngb["epoch"], summary_ngb["gram_refreshes"]) == (12, 2, [])
export_nga = torch.load(os.path.join(NG_A, "export", "teacher_backbone_epoch_0002.pth"), weights_only=True)
export_ngb = torch.load(os.path.join(NG_B, "export", "teacher_backbone_epoch_0002.pth"), weights_only=True)
assert set(export_nga) == set(export_ngb)
ng_bitwise = all(torch.equal(export_nga[k], export_ngb[k]) for k in export_ngb)
if not ng_bitwise:
    assert all(torch.allclose(export_nga[k], export_ngb[k], rtol=1e-5, atol=1e-6) for k in export_ngb)
    print("NOTE (s) the resumed --no_gram teacher backbone matches the uninterrupted run only to allclose, not bitwise")
ck_ng = load_ckpt(NG_A, "epoch_0002.pth")
assert set(ck_ng) == set(ck2) and ck_ng["gram_teacher"] is None and load_ckpt(NG_B, "epoch_0002.pth")["gram_teacher"] is None  # the key stays, holding None
assert (ck_ng["step"], ck_ng["epoch"], ck_ng["gram_refreshes"], ck_ng["world_size"]) == (12, 2, [], 1)
assert ck_ng["args"]["no_gram"] is True and ck2["args"]["no_gram"] is False  # the Gram run's checkpoint records the flag too
info_ng = json.load(open(os.path.join(NG_A, "args.json")))
assert (info_ng["no_gram"], info_ng["gram_weight"], info_ng["gram_refresh_steps"]) == (True, 0.0, [])
assert json.load(open(os.path.join(DIR_A, "args.json")))["no_gram"] is False
ng_backbone = backbone_of(ck_ng["teacher"])
assert set(export_nga) == set(ng_backbone) and all(torch.equal(export_nga[k], v) for k, v in ng_backbone.items())  # the export IS the EMA teacher's backbone
ng_cols = CSV_HEADER.split(",")
ng_gram, ng_gram_weight = ng_cols.index("gram"), ng_cols.index("gram_weight")
for out in (NG_A, NG_B):
    rows = read_csv(out)
    assert ",".join(rows[0]) == CSV_HEADER and len(rows) > 3
    for row in rows[1:]:
        assert len(row) == 16 and row[ng_gram] == "0.0" and row[ng_gram_weight] == "0.0", row  # both columns present and numeric, never blank
        dg, dl, ko, ib, total = (float(row[ng_cols.index(c)]) for c in ("dino_global", "dino_local", "koleo", "ibot", "total"))
        assert all(map(math.isfinite, (dg, dl, ko, ib, total))) and math.isclose(total, 16 / 18 * dl + 2 / 18 * dg + 0.2 * ko + ib, rel_tol=1e-5), row
print("OK (s) end to end: A (1 epoch, then --resume to 2) and B (2 epochs) with --no_gram: SSLMetaArch got gram_teacher None in all 3 runs, none of the %d "
      "batches had Gram crops, the Gram forward / GramLoss / refresh were never called; printed the OFF line and no 'Gram refresh' lines; the resume restored "
      "student/teacher/optimizer/RNG; final teacher backbone %s; checkpoint keys as a Gram run's with gram_teacher None; args.json no_gram True, gram_weight "
      "0.0, refreshes []; CSV: 15 columns, gram and gram_weight exactly 0.0 in every row, total = the four non-Gram terms"
      % (len(ng_batches), "bitwise equal" if ng_bitwise else "allclose"))

# --no_gram is locked on --resume both ways, and a checkpoint written before the flag reads as no_gram False
ng_args_b, ng_args_dir_b = open(os.path.join(NG_B, "args.json")).read(), open(os.path.join(DIR_B, "args.json")).read()
message = raises(ValueError, go, NG_B, "--resume")
assert "cannot --resume" in message and "no_gram: checkpoint True, now False" in message, message
message = raises(ValueError, go, DIR_B, "--resume", "--no_gram")
assert "cannot --resume" in message and "no_gram: checkpoint False, now True" in message, message
assert open(os.path.join(NG_B, "args.json")).read() == ng_args_b and open(os.path.join(DIR_B, "args.json")).read() == ng_args_dir_b  # the record is left alone
assert "no_gram" in ck_b["args"] and "no_gram" in ssl_pretrain.RESUME_MATCH
ck_b_old = dict(ck_b, args={k: v for k, v in ck_b["args"].items() if k != "no_gram"})  # as written before the flag existed
assert "no_gram" not in ck_b_old["args"]
ssl_pretrain.check_resume(ck_b_old, make_args(DIR_B), 1, 6, ck_b["data_fingerprint"])  # reads as no_gram False: a Gram resume passes
ssl_pretrain.check_resume(ck_b, make_args(DIR_B), 1, 6, ck_b["data_fingerprint"])  # a checkpoint that has the key passes too
ssl_pretrain.check_resume(ck_ng, make_args(NG_A, "--no_gram"), 1, 6, ck_ng["data_fingerprint"])  # ... a no-Gram one with --no_gram
assert "no_gram: checkpoint False, now True" in raises(ValueError, ssl_pretrain.check_resume, ck_b_old, make_args(DIR_B, "--no_gram"), 1, 6,
                                                       ck_b["data_fingerprint"])
print("OK (s) refusals: --resume without --no_gram into a no-Gram run (checkpoint True, now False) and with it into a Gram run (checkpoint False, now True) "
      "raise ValueError and leave args.json alone; a checkpoint without the no_gram key reads as False (passes a Gram resume, refuses --no_gram), one with it passes")

# --init_from_checkpoint with --no_gram: student and teacher load, the Gram-teacher load is skipped (--gram_teacher_size is not passed: global 64 -> 128)
DIR_NGH = os.path.join(TMP, "NGH")
with spying_on_meta(), gram_code_forbidden():
    _, text_ngh = go(DIR_NGH, "--no_gram", "--init_from_checkpoint", INIT, "--global_crops_size", "64", "--local_crops_size", "32",
                     "--warmup_epochs", "0", "--freeze_backbone_epochs", "0", "--epochs", "2", "--stop_after_epochs", "1")
init_line = next(line for line in text_ngh.splitlines() if line.startswith("initialized from"))
assert init_line.endswith("teacher (heads included) loaded strictly, no Gram teacher (--no_gram); fresh optimizer, step 0"), init_line
assert "Gram teacher <- its EMA teacher backbone" not in text_ngh and OFF_LINE in text_ngh.splitlines()
assert "crops: global 64, local 32 (x8), no Gram teacher (global crops cut at 128 and resized to 64, as in the Gram runs);" in text_ngh
assert len(ng_inits) == 1 and ng_inits[0]["gram_teacher"] is None and len(ng_batches) == 6
assert all(torch.equal(ng_inits[0]["student"][k], v) for k, v in src["student"].items()) and set(ng_inits[0]["student"]) == set(src["student"])
assert all(torch.equal(ng_inits[0]["teacher"][k], v) for k, v in src["teacher"].items()) and set(ng_inits[0]["teacher"]) == set(src["teacher"])  # heads included
ck_ngh = load_ckpt(DIR_NGH, "epoch_0001.pth")
assert ck_ngh["gram_teacher"] is None and (ck_ngh["step"], ck_ngh["epoch"], ck_ngh["gram_refreshes"]) == (6, 1, [])
assert (ck_ngh["args"]["no_gram"], ck_ngh["args"]["init_from_checkpoint"], ck_ngh["args"]["gram_teacher_size"]) == (True, INIT, 128)
assert any(not torch.equal(ck_ngh["student"][k], v) for k, v in src["student"].items())  # and the student trained
# the planned use: a Gram run branches from that no-Gram checkpoint; its Gram teacher is the trunk's EMA teacher backbone, as for any init
DIR_NGG = os.path.join(TMP, "NGG")
_, text_ngg = go(DIR_NGG, *[os.path.join(DIR_NGH, "ckpt", "epoch_0001.pth") if a == INIT else a for a in HIRES], "--stop_after_epochs", "1")
assert "teacher (heads included) loaded strictly, Gram teacher <- its EMA teacher backbone; fresh optimizer, step 0" in text_ngg and OFF_LINE not in text_ngg.splitlines()
ck_ngg = load_ckpt(DIR_NGG, "epoch_0001.pth")
assert ck_ngg["args"]["no_gram"] is False and ck_ngg["gram_teacher"] is not None
assert all(torch.equal(ck_ngg["gram_teacher"]["backbone." + k], v) for k, v in backbone_of(ck_ngh["teacher"]).items())  # no refresh: still the trunk's teacher
assert all(float(row[ng_gram]) > 0 and row[ng_gram_weight] == "2.0" for row in read_csv(DIR_NGG)[1:])  # and the Gram loss is on
print("OK (s) --no_gram --init_from_checkpoint: a Gram run's full checkpoint loads (student and teacher bitwise, heads included), no Gram teacher is loaded "
      "('no Gram teacher (--no_gram)' in the initialized line), the new checkpoint's gram_teacher is None; a Gram run initialized from that no-Gram "
      "checkpoint anchors its Gram teacher to the trunk's EMA teacher and trains with the Gram loss")

# (t) --branch_from: a no-Gram trunk continued exactly, with the Gram teacher switched on -------------------------------------------------------

# parse: --branch_from stands in for --pretrained_weights, needs --gram_refresh_steps (none allowed: an anchor that never refreshes) and refuses --no_gram / --init_from_checkpoint
PB = lambda *extra: ssl_pretrain.parse_args(["--data_set", "MIMIC", "--output_dir", "o", "--branch_from", "x.pth"] + list(extra))
assert P().branch_from is None and ssl_pretrain.ARG_DEFAULTS["branch_from"] is None and "branch_from" in ssl_pretrain.RESUME_MATCH
assert set(ssl_pretrain.BRANCH_FREE) == {"no_gram", "gram_weight", "gram_refresh_steps", "gram_teacher_size", "branch_from", "init_from_checkpoint", "gram_teacher_from"}
branch_parsed = PB("--gram_refresh_steps", "5")  # no --pretrained_weights
assert (branch_parsed.branch_from, branch_parsed.pretrained_weights, branch_parsed.no_gram, branch_parsed.gram_weight, branch_parsed.gram_teacher_size,
        branch_parsed.gram_refresh_steps) == ("x.pth", None, False, 2.0, 448, [5])
assert PB("--gram_refresh_steps", "none").gram_refresh_steps == [] and PB("--gram_refresh_steps", "5", "--pretrained_weights", "w").pretrained_weights == "w"
with contextlib.redirect_stderr(io.StringIO()) as err:
    raises(SystemExit, ssl_pretrain.parse_args, ["--data_set", "MIMIC", "--output_dir", "o"])
assert "--pretrained_weights is required unless --init_from_checkpoint is given (or --branch_from)" in err.getvalue(), err.getvalue()
for extra, flags in ((["--no_gram"], ("--branch_from", "--no_gram")),
                     (["--gram_refresh_steps", "5", "--init_from_checkpoint", "c.pth"], ("--branch_from", "--init_from_checkpoint")),
                     ([], ("--branch_from", "--gram_refresh_steps"))):
    with contextlib.redirect_stderr(io.StringIO()) as err:
        raises(SystemExit, PB, *extra)
    complaint = err.getvalue().split("error: ", 1)[1]
    assert all(flag in complaint for flag in flags), (extra, err.getvalue())
for flag, extra in (("--branch_from", ["--gram_refresh_steps", "5"]), ("--init_from_checkpoint", [])):  # an empty source path is never a silent no-op, with or without --pretrained_weights
    for weights in ([], ["--pretrained_weights", "w"]):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            assert raises(SystemExit, ssl_pretrain.parse_args, ["--data_set", "MIMIC", "--output_dir", "o", flag, ""] + extra + weights) == "2"
        assert f"{flag} is empty" in err.getvalue().split("error: ", 1)[1], (flag, weights, err.getvalue())
print("OK (t) parse: no --branch_from -> None (read as None in an old checkpoint, locked on --resume); --branch_from alone (with --gram_refresh_steps 5 or none) needs no "
      "--pretrained_weights and keeps gram_weight 2.0, Gram size 448; --no_gram, --init_from_checkpoint, or no --gram_refresh_steps with it exits 2 naming both flags; "
      "an empty --branch_from / --init_from_checkpoint exits 2")


def same_state(a, b):  # bitwise: tensors torch.equal (and the same dtype), arrays array_equal, containers recursively, the rest ==
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return a == b


# the key claim: a branch is the trunk continued exactly. The no-Gram trunk (1 epoch, then --resume to 2) against a --branch_from of its epoch 1 with --gram_weight 0
# (the Gram teacher is built, fed Gram crops and run, its loss computed and weighted 0; no refresh before step 100). Gram crops draw no RNG, the Gram teacher runs in eval
# mode under no_grad and the 0-weighted term adds exact zeros to the loss and the gradients, so epoch 2 must come out bitwise the same, not merely close
BR_T, BR_II = os.path.join(TMP, "BR_T"), os.path.join(TMP, "BR_II")
go(BR_T, "--no_gram", "--stop_after_epochs", "1")
SRC1 = os.path.join(BR_T, "ckpt", "epoch_0001.pth")
src1 = torch.load(SRC1, map_location="cpu", weights_only=False)
assert src1["args"]["no_gram"] is True and (src1["step"], src1["epoch"]) == (6, 1) and src1["gram_teacher"] is None
summary_ii, text_ii = go(BR_II, "--branch_from", SRC1, "--gram_weight", "0", "--gram_refresh_steps", "100", "--resume", "--stop_after_epochs", "2")
summary_t, text_t = go(BR_T, "--no_gram", "--resume", "--stop_after_epochs", "2")
assert "branched from %s: its step 6, epoch 1;" % SRC1 in text_ii and "starting from scratch" not in text_ii and "resumed from" in text_t, (text_ii, text_t)
assert summary_ii == summary_t and (summary_t["step"], summary_t["epoch"], summary_t["gram_refreshes"]) == (12, 2, [])
exp_t = torch.load(os.path.join(BR_T, "export", "teacher_backbone_epoch_0002.pth"), weights_only=True)
exp_ii = torch.load(os.path.join(BR_II, "export", "teacher_backbone_epoch_0002.pth"), weights_only=True)
ck_t, ck_ii = load_ckpt(BR_T, "epoch_0002.pth"), load_ckpt(BR_II, "epoch_0002.pth")
assert (ck_t["step"], ck_t["epoch"], ck_ii["step"], ck_ii["epoch"]) == (12, 2, 12, 2)
assert not same_state(ck_t["student"], src1["student"]) and not same_state(ck_t["rng"], src1["rng"])  # the comparisons below are not vacuous: epoch 2 did train
assert same_state(exp_ii, exp_t), "export"
for part in ("student", "teacher", "optimizer", "rng"):  # optimizer: the whole state dict, i.e. the state tensors (exp_avg, exp_avg_sq, step) and param_groups
    assert same_state(ck_ii[part], ck_t[part]), part
assert ck_ii["gram_teacher"] is not None and ck_t["gram_teacher"] is None and ck_ii["args"]["branch_from"] == SRC1 and ck_t["args"]["branch_from"] is None
rows_t, rows_ii = read_csv(BR_T), read_csv(BR_II)
by_step_t = {r[0]: r for r in rows_t[1:]}
common = [r for r in rows_ii[1:] if r[0] in by_step_t]
assert rows_ii[1][0] == "6" and all(int(r[0]) >= 6 for r in rows_ii[1:]) and len(common) >= 3, rows_ii  # the branch's rows start at its step; the trunk's earlier ones are not copied
for r in common:
    for column in ("dino_global", "dino_local", "koleo", "ibot", "total", "lr", "wd", "momentum", "teacher_temp"):
        assert r[ng_cols.index(column)] == by_step_t[r[0]][ng_cols.index(column)], (column, r, by_step_t[r[0]])
assert all(r[ng_gram_weight] == "0.0" and float(r[ng_gram]) > 0 for r in rows_ii[1:])  # the Gram loss is computed (positive), its weight 0
print("OK (t) bitwise: a --no_gram trunk (1 epoch, then --resume to 2) and a --branch_from of its epoch 1 (Gram teacher built and run, --gram_weight 0, no refresh) have the "
      "same epoch-2 export, student, optimizer (state and param_groups), teacher and RNG states, bitwise equal; step 12 / epoch 2 both; the %d common CSV rows (steps >= 6) agree "
      "as strings on dino_global, dino_local, koleo, ibot, total, lr, wd, momentum, teacher_temp; the branch's rows start at step 6 with gram > 0 and gram_weight 0.0" % len(common))

# default weight: the Gram teacher is the source's EMA teacher backbone before the first step, the models are the source's, and the run is a normal Gram run from there
BR_G = os.path.join(TMP, "BR_G")
args_g = make_args(BR_G, "--branch_from", SRC1, "--gram_refresh_steps", "8,10", "--resume", "--stop_after_epochs", "2")
args_g.pretrained_weights = "unused"  # both given: the checkpoint wins, with a warning (as args_h above)
br_first = []


def br_forward_backward(self, data, *, teacher_temp):  # the models as the first step sees them (cloned: the step then updates them); later calls only pass through
    if not br_first:
        br_first.append({name: {k: v.detach().clone() for k, v in module.state_dict().items()}
                         for name, module in (("gram_teacher", self.gram_teacher), ("teacher", self.teacher), ("student", self.student))})
    return real_forward_backward(self, data, teacher_temp=teacher_temp)


SSLMetaArch.forward_backward = br_forward_backward
try:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary_g = ssl_pretrain.run(args_g, TINY_VIT, TINY_HEADS, CFG)
    text_g = buf.getvalue()
finally:
    SSLMetaArch.forward_backward = real_forward_backward
assert same_state(br_first[0]["gram_teacher"], {k: v for k, v in src1["teacher"].items() if k.startswith("backbone.")})  # anchored to the source's EMA teacher backbone
assert same_state(br_first[0]["teacher"], src1["teacher"]) and same_state(br_first[0]["student"], src1["student"])  # restored before the first step
lines_g = text_g.splitlines()
assert "WARNING: --pretrained_weights unused ignored: --branch_from supplies every weight" in lines_g, text_g
assert (f"branched from {SRC1}: its step 6, epoch 1; student/teacher/optimizer/RNG restored; Gram teacher <- its EMA teacher backbone; Gram ON, weight 2, "
        "refreshes after steps [8, 10]") in lines_g, text_g
assert "Gram refresh scheduled after step 8 (epoch 1.33 of 3)" in lines_g and "Gram refresh scheduled after step 10 (epoch 1.67 of 3)" in lines_g, text_g
assert "Gram refresh 1/2 after step 8:" in text_g and "Gram refresh 2/2 after step 10:" in text_g
assert not any(words in text_g for words in ("starting from scratch", "resumed from", "Gram: OFF")), text_g
assert summary_g["step"] == 12 and summary_g["epoch"] == 2 and summary_g["gram_refreshes"] == [8, 10]
assert [e for e, _ in ssl_pretrain.epoch_checkpoints(os.path.join(BR_G, "ckpt"))] == [2]  # the trunk's numbering continues: no epoch_0001 here
assert os.listdir(os.path.join(BR_G, "export")) == ["teacher_backbone_epoch_0002.pth"]
ck_g = load_ckpt(BR_G, "epoch_0002.pth")
assert ck_g["args"]["branch_from"] == SRC1 and ck_g["args"]["no_gram"] is False and ck_g["gram_refreshes"] == [8, 10] and ck_g["gram_teacher"] is not None
assert (ck_g["step"], ck_g["epoch"]) == (12, 2)
info_g = json.load(open(os.path.join(BR_G, "args.json")))
assert (info_g["branch_from"], info_g["no_gram"], info_g["gram_weight"], info_g["gram_refresh_steps"]) == (SRC1, False, 2.0, [8, 10])
rows_g = read_csv(BR_G)
assert ",".join(rows_g[0]) == CSV_HEADER and rows_g[1][0] == "6" and all(int(r[0]) >= 6 for r in rows_g[1:])
assert all(float(r[ng_gram]) > 0 and r[ng_gram_weight] == "2.0" for r in rows_g[1:])
print("OK (t) default-weight branch: before its first step the models are the source's, bitwise (student, teacher), and the Gram teacher is the source's EMA teacher backbone; "
      "WARNING for --pretrained_weights, the 'branched from' line, no 'starting from scratch' / 'resumed from' / 'Gram: OFF'; refreshes [8, 10] run and are reported; "
      "step 12 / epoch 2, only epoch_0002 + its export written (numbering continues the trunk's); the checkpoint and args.json record branch_from, no_gram False, "
      "weight 2.0, refreshes [8, 10]; CSV from step 6 with gram > 0 and gram_weight 2.0")

# the source is read only at the start: a re-submit resumes the branch's own checkpoint with the source gone, a different source path is a different run, and a trunk is never a branch
BR_R, COPY = os.path.join(TMP, "BR_R"), os.path.join(TMP, "branch_src_copy.pth")
shutil.copy(SRC1, COPY)
BR_R_ARGS = ["--branch_from", COPY, "--gram_refresh_steps", "8,10", "--resume"]
assert go(BR_R, *BR_R_ARGS, "--stop_after_epochs", "2")[0]["epoch"] == 2
os.remove(COPY)
summary_r, text_r = go(BR_R, *BR_R_ARGS, "--stop_after_epochs", "3")
assert "resumed from" in text_r and "2 epochs / 12 steps done, Gram refreshes [8, 10]" in text_r and "branched from" not in text_r, text_r
assert (summary_r["step"], summary_r["epoch"], summary_r["gram_refreshes"]) == (18, 3, [8, 10])
assert os.path.isfile(os.path.join(BR_R, "export", "teacher_backbone_epoch_0003.pth"))
assert json.load(open(os.path.join(BR_R, "args.json")))["branch_from"] == COPY  # still recorded
other = os.path.join(TMP, "elsewhere_branch_src.pth")  # a resume naming another source is refused by the comparison even when that path does not exist either
before_r = open(os.path.join(BR_R, "args.json")).read()
message = raises(ValueError, go, BR_R, "--branch_from", other, "--gram_refresh_steps", "8,10", "--resume", "--stop_after_epochs", "3")
assert "cannot --resume" in message and ("branch_from: checkpoint %r, now %r" % (COPY, other)) in message, message
assert open(os.path.join(BR_R, "args.json")).read() == before_r
before_t = open(os.path.join(BR_T, "args.json")).read()  # the no-Gram trunk resumed as a branch: it was not one
message = raises(ValueError, go, BR_T, "--branch_from", SRC1, "--gram_refresh_steps", "100", "--resume", "--stop_after_epochs", "3")
assert "cannot --resume" in message and ("branch_from: checkpoint None, now %r" % SRC1) in message, message
assert open(os.path.join(BR_T, "args.json")).read() == before_t
# a trunk checkpoint written before the key existed (0e8d453) reads as no branch and resumes unchanged
assert "branch_from" in ck_t["args"] and ck_t["args"]["branch_from"] is None
ck_old = dict(ck_t, args={k: v for k, v in ck_t["args"].items() if k != "branch_from"})
ssl_pretrain.check_resume(ck_old, make_args(BR_T, "--no_gram"), 1, 6, ck_t["data_fingerprint"])
print("OK (t) branch resume: branch -> epoch 2 -> source deleted -> --resume continues to epoch 3 ('resumed from', Gram refreshes [8, 10] restored, no 'branched from'; "
      "path still recorded); a resume naming another (missing) source is refused by the comparison, and so is --branch_from on a trunk's resume (checkpoint None), args.json "
      "untouched both times; a trunk checkpoint without the branch_from key still resumes as no branch")

# a source that was itself started with --init_from_checkpoint (the (s) DIR_NGH run: INIT, --no_gram, global 64 / local 32 / Gram size 128) still branches: its full state is
# restored, and a branch cannot pass --init_from_checkpoint, so the recorded init would always differ (Codex r1, P2)
BR_INIT, SRC_NGH = os.path.join(TMP, "BR_INIT"), os.path.join(DIR_NGH, "ckpt", "epoch_0001.pth")
assert ck_ngh["args"]["init_from_checkpoint"] == INIT and (ck_ngh["step"], ck_ngh["epoch"]) == (6, 1)
summary_bi, text_bi = go(BR_INIT, "--branch_from", SRC_NGH, "--global_crops_size", "64", "--local_crops_size", "32", "--warmup_epochs", "0", "--freeze_backbone_epochs", "0",
                         "--epochs", "2", "--gram_refresh_steps", "100", "--stop_after_epochs", "2")
assert (summary_bi["step"], summary_bi["epoch"]) == (12, 2) and "branched from %s: its step 6, epoch 1;" % SRC_NGH in text_bi, (summary_bi, text_bi)
ck_bi = load_ckpt(BR_INIT, "epoch_0002.pth")
assert ck_bi["args"]["init_from_checkpoint"] is None and ck_bi["args"]["branch_from"] == SRC_NGH and ck_bi["gram_teacher"] is not None
print("OK (t) init-started source: a --no_gram run started with --init_from_checkpoint branches (its recorded init is skipped): step 12 / epoch 2, the 'branched from' line, "
      "and the branch's checkpoint records init_from_checkpoint None and branch_from the source")

# refusals: each in a fresh dir, naming the cause, with nothing created on disk
BRX = ["--branch_from", SRC1, "--gram_refresh_steps", "100", "--stop_after_epochs", "2"]  # the base args of every run below: valid, each adds one fault (a later flag wins)


def br_refused(name, exc, *extra):  # a --branch_from start in a fresh dir TMP/brx_<name> that must raise exc: its message, after checking that nothing was created
    out = os.path.join(TMP, "brx_" + name)
    message = raises(exc, go, out, *BRX, *extra)
    assert not os.path.exists(out), name
    return message


no_such = os.path.join(TMP, "no_such_branch_src.pth")
assert "--branch_from %s is not a file" % no_such in br_refused("missing", FileNotFoundError, "--branch_from", no_such)
assert "not a full epoch checkpoint" in br_refused("export", ValueError, "--branch_from", os.path.join(BR_T, "export", "teacher_backbone_epoch_0001.pth"))
message = br_refused("gram_run", ValueError, "--branch_from", os.path.join(DIR_B, "ckpt", "epoch_0001.pth"))  # a Gram run's full checkpoint
assert "not a --no_gram run's checkpoint" in message and "its no_gram is False" in message, message
for flag, value, diff in (("--lr", "2e-3", "lr: checkpoint 0.001, now 0.002"), ("--seed", "7", "seed: checkpoint 100, now 7")):  # any other locked setting
    message = br_refused(flag[2:], ValueError, flag, value)
    listed = message.split("differs from the checkpoint in ", 1)[1]  # the differences themselves ("cannot --branch_from:" ahead of them is only the lead-in)
    assert "cannot --branch_from" in message and diff in message, message
    assert not any(free in listed for free in ("no_gram", "gram_weight", "gram_refresh_steps", "gram_teacher_size", "branch_from:")), message  # the free keys are skipped
rev_list = os.path.join(TMP, "ChestXray14_reversed.list")  # the same 3 lines in reverse order: same names and length, a different image list
with open(CFG["ChestXray14"]["train_list"]) as f:
    rev_lines = f.read().splitlines()
with open(rev_list, "w") as f:
    f.write("\n".join(reversed(rev_lines)) + "\n")
cfg_rev, out_rev = dict(CFG, ChestXray14=dict(CFG["ChestXray14"], train_list=rev_list)), os.path.join(TMP, "brx_list")
with contextlib.redirect_stdout(io.StringIO()):
    message = raises(ValueError, ssl_pretrain.run, make_args(out_rev, *BRX), TINY_VIT, TINY_HEADS, cfg_rev)
assert "cannot --branch_from" in message and "training image list" in message and not os.path.exists(out_rev), message
helper_args = make_args(os.path.join(TMP, "brx_helper"), *BRX)  # world size and epoch length, through the helper (the run is one rank)
assert ssl_pretrain.load_branch_source(helper_args, 2, 1, 6, src1["data_fingerprint"])["step"] == 6  # the real values pass
assert "world_size: checkpoint 1, now 4" in raises(ValueError, ssl_pretrain.load_branch_source, helper_args, 2, 4, 6, src1["data_fingerprint"])
assert "epoch_length 5" in raises(ValueError, ssl_pretrain.load_branch_source, helper_args, 2, 1, 5, src1["data_fingerprint"])
assert not os.path.exists(os.path.join(TMP, "brx_helper"))
assert "[6] not after the branch step 6" in br_refused("refresh_at", ValueError, "--gram_refresh_steps", "6,20")  # the refresh at the branch step itself
assert "[3] not after" in br_refused("refresh_before", ValueError, "--gram_refresh_steps", "3")
assert "--stop_after_epochs 1 is not after the source's epoch 1" in br_refused("stop", ValueError, "--stop_after_epochs", "1")
print("OK (t) refusals: a missing source (FileNotFoundError), an export/teacher_backbone_*.pth ('not a full epoch checkpoint'), a Gram run's checkpoint (its no_gram is False), "
      "another --lr / --seed (naming the difference, the Gram keys and branch_from skipped), a different image list, world size 4 and epoch length 5 (through load_branch_source), "
      "a refresh step at (6) or before (3) the branch step, and a stop epoch not after the source's: each ValueError (or FileNotFoundError) names its cause and leaves the "
      "output dir uncreated")

# --gram_teacher_from (DINOv3's early-teacher rule, Sec. 4.2; App. C p. 58): a branch whose Gram teacher is an EARLIER epoch checkpoint's EMA teacher backbone, of the source's own run; the student,
# EMA teacher, optimizer, step and RNG are still the source's. parse: --branch_from only, never with --no_gram / --init_from_checkpoint (each refused with its own text, before anything is read), an
# empty path refused; None by default and in an old checkpoint (ARG_DEFAULTS); locked on --resume; branch-free against the source (a no-Gram trunk never has one)
assert P().gram_teacher_from is None and ssl_pretrain.ARG_DEFAULTS["gram_teacher_from"] is None and "gram_teacher_from" in ssl_pretrain.RESUME_MATCH
assert PB("--gram_refresh_steps", "5", "--gram_teacher_from", "g.pth").gram_teacher_from == "g.pth" and PB("--gram_refresh_steps", "5").gram_teacher_from is None
GT_NEEDS_BRANCH = "--gram_teacher_from needs --branch_from: it replaces the Gram teacher a branch takes from its source with an earlier checkpoint's"
GT_NO_NOGRAM = "--gram_teacher_from cannot be combined with --no_gram (there is no Gram teacher to anchor)"
GT_NO_INIT = "--gram_teacher_from cannot be combined with --init_from_checkpoint (an --init_from_checkpoint run anchors its Gram teacher to that checkpoint's EMA teacher)"
for argv, complaint in (([], GT_NEEDS_BRANCH), (["--no_gram"], GT_NO_NOGRAM), (["--init_from_checkpoint", "c.pth"], GT_NO_INIT),
                        (["--branch_from", "x.pth", "--no_gram"], GT_NO_NOGRAM), (["--branch_from", "x.pth", "--gram_refresh_steps", "5", "--init_from_checkpoint", "c.pth"], GT_NO_INIT)):
    with contextlib.redirect_stderr(io.StringIO()) as err:  # P passes --pretrained_weights, so the only fault is the Gram teacher's (with --branch_from, its refusal comes first)
        assert raises(SystemExit, P, "--gram_teacher_from", "g.pth", *argv) == "2"
    assert err.getvalue().split("error: ", 1)[1] == complaint + "\n", (argv, err.getvalue())
with contextlib.redirect_stderr(io.StringIO()) as err:
    assert raises(SystemExit, PB, "--gram_refresh_steps", "5", "--gram_teacher_from", "") == "2"
assert err.getvalue().split("error: ", 1)[1] == "--gram_teacher_from is empty: pass a checkpoint path, or leave the flag out\n", err.getvalue()
print("OK (t) --gram_teacher_from parse: None by default (= ARG_DEFAULTS, locked on --resume, branch-free); with --branch_from it is recorded as given; without --branch_from, with --no_gram or "
      "--init_from_checkpoint (with --branch_from as well) or empty it exits 2 with its own text")

# a real branch on CPU: GT_T, a tiny no-Gram trunk saving epochs 1 and 2, then a branch of its epoch 2 with --gram_teacher_from its epoch 1. Before the first step the Gram teacher is epoch 1's EMA teacher
# backbone, tensor-equal (and not epoch 2's, which the default would load: epoch 2 trained the backbone), while the student, the EMA teacher and the step are epoch 2's
GT_T, GT_BR = os.path.join(TMP, "GT_T"), os.path.join(TMP, "GT_BR")
go(GT_T, "--no_gram", "--stop_after_epochs", "2")
GT_E1, GT_E2 = (os.path.join(GT_T, "ckpt", "epoch_%04d.pth" % epoch) for epoch in (1, 2))
gt_e1, gt_e2 = (torch.load(path, map_location="cpu", weights_only=False) for path in (GT_E1, GT_E2))
assert (gt_e1["epoch"], gt_e2["epoch"], gt_e2["step"]) == (1, 2, 12) and gt_e1["args"]["output_dir"] == gt_e2["args"]["output_dir"] == GT_T
gt_anchor = {k: v for k, v in gt_e1["teacher"].items() if k.startswith("backbone.")}
assert not same_state(gt_anchor, {k: v for k, v in gt_e2["teacher"].items() if k.startswith("backbone.")})  # the check below tells the two apart
GT_ARGS = ["--branch_from", GT_E2, "--gram_refresh_steps", "100", "--resume", "--stop_after_epochs", "3"]  # no refresh in the run: the anchor stays to the end
br_first.clear()
SSLMetaArch.forward_backward = br_forward_backward  # (t)'s first-step spy: the models as the first step sees them
try:
    summary_gt, text_gt = go(GT_BR, *GT_ARGS, "--gram_teacher_from", GT_E1)
finally:
    SSLMetaArch.forward_backward = real_forward_backward
assert same_state(br_first[0]["gram_teacher"], gt_anchor), "the Gram teacher is not epoch 1's EMA teacher backbone"
assert same_state(br_first[0]["teacher"], gt_e2["teacher"]) and same_state(br_first[0]["student"], gt_e2["student"])  # the rest is the branch point's, as without the flag
assert (f"branched from {GT_E2}: its step 12, epoch 2; student/teacher/optimizer/RNG restored; Gram teacher <- EMA teacher backbone of {GT_E1} (its epoch 1); Gram ON, weight 2, "
        "refreshes after steps [100]") in text_gt.splitlines(), text_gt
assert (summary_gt["step"], summary_gt["epoch"], summary_gt["gram_refreshes"]) == (18, 3, []) and read_csv(GT_BR)[1][0] == "12"  # the branch continues at epoch 2's step
ck_gt = load_ckpt(GT_BR, "epoch_0003.pth")
assert same_state(ck_gt["gram_teacher"], gt_anchor) and (ck_gt["args"]["gram_teacher_from"], ck_gt["args"]["branch_from"]) == (GT_E1, GT_E2)  # the checkpoint carries it
info_gt = json.load(open(os.path.join(GT_BR, "args.json")))
assert (info_gt["gram_teacher_from"], info_gt["branch_from"]) == (GT_E1, GT_E2) and json.load(open(os.path.join(GT_T, "args.json")))["gram_teacher_from"] is None
# the refreshes still copy the EMA teacher, not the earlier checkpoint: the same branch refreshed after its last step (18) ends with the Gram teacher = its EMA teacher backbone
summary_gt18, _ = go(os.path.join(TMP, "GT_BR18"), *GT_ARGS, "--gram_teacher_from", GT_E1, "--gram_refresh_steps", "18")  # the later flag wins
ck_gt18 = load_ckpt(os.path.join(TMP, "GT_BR18"), "epoch_0003.pth")
assert summary_gt18["gram_refreshes"] == [18] and same_state(ck_gt18["gram_teacher"], {k: v for k, v in ck_gt18["teacher"].items() if k.startswith("backbone.")})
assert not same_state(ck_gt18["gram_teacher"], gt_anchor)
# --resume of the branch: the flag is locked (omitted, another checkpoint of the run, or a path that does not exist: refused naming it, args.json untouched); the same one resumes, its file not read
gt_record = open(os.path.join(GT_BR, "args.json")).read()
for extra, now in (([], None), (["--gram_teacher_from", GT_E2], GT_E2), (["--gram_teacher_from", os.path.join(TMP, "elsewhere_gt.pth")], os.path.join(TMP, "elsewhere_gt.pth"))):
    message = raises(ValueError, go, GT_BR, *GT_ARGS, *extra)
    assert "cannot --resume" in message and ("gram_teacher_from: checkpoint %r, now %r" % (GT_E1, now)) in message, message
    assert open(os.path.join(GT_BR, "args.json")).read() == gt_record
os.rename(GT_E1, GT_E1 + ".away")  # read only at the branch's start, as the source
try:
    summary_gtr, text_gtr = go(GT_BR, *GT_ARGS, "--gram_teacher_from", GT_E1)
finally:
    os.rename(GT_E1 + ".away", GT_E1)
assert "resumed from" in text_gtr and "nothing to do: 3 epochs done" in text_gtr and summary_gtr["epoch"] == 3, text_gtr
# refusals at the branch's start, each in a fresh dir (br_refused: nothing created), the source checks passed: a missing file, an export, the same epoch, a later one, another run's checkpoint
# (BR_T's epoch 1: the same settings, another run dir), each naming both files
GTX = ["--branch_from", GT_E2, "--stop_after_epochs", "3", "--gram_teacher_from"]
gt_missing = os.path.join(TMP, "no_such_gt.pth")
assert br_refused("gt_missing", FileNotFoundError, *GTX, gt_missing) == "--gram_teacher_from %s is not a file" % gt_missing
gt_export = os.path.join(GT_T, "export", "teacher_backbone_epoch_0001.pth")
assert br_refused("gt_export", ValueError, *GTX, gt_export) == ("--gram_teacher_from %s is not a full epoch checkpoint (student, teacher, optimizer, step, epoch, rng, args, world_size); an "
                                                                 "export/teacher_backbone_*.pth holds the backbone only" % gt_export)
GT_LATE = "--gram_teacher_from %s is from epoch %d, not before the epoch %d of --branch_from %s: the Gram teacher is an earlier iteration of the teacher (DINOv3 Sec. 4.2)"
assert br_refused("gt_same", ValueError, *GTX, GT_E2) == GT_LATE % (GT_E2, 2, 2, GT_E2)
assert br_refused("gt_later", ValueError, *GTX, GT_E2, "--branch_from", GT_E1) == GT_LATE % (GT_E2, 2, 1, GT_E1)  # the later flag wins: a branch of epoch 1
assert br_refused("gt_other_run", ValueError, *GTX, SRC1) == ("--gram_teacher_from %s is not of the run of --branch_from %s: its output_dir is %r, the source's %r" % (SRC1, GT_E2, BR_T, GT_T))
# the run is the output_dir both checkpoints record, not the file's path: a copy of epoch 1 elsewhere is the same run (through the helper, as load_branch_source above)
GT_COPY = os.path.join(TMP, "gt_epoch_0001_copy.pth")
shutil.copy(GT_E1, GT_COPY)
copy_anchor, copy_epoch = ssl_pretrain.load_gram_teacher_source(make_args(os.path.join(TMP, "gtx_helper"), *GT_ARGS, "--gram_teacher_from", GT_COPY), gt_e2, 6)
assert copy_epoch == 1 and same_state(copy_anchor, gt_anchor) and not os.path.exists(os.path.join(TMP, "gtx_helper"))
# only an epoch-boundary state (Codex r1 P2 on 4c613a0): GT_M, a no-Gram trunk stopped by --max_steps 9 mid-epoch 2 (ckpt/maxsteps_0000009.pth: step 9, epoch 1, every BRANCH_SOURCE_KEYS key,
# its run's output_dir) and resumed from its epoch 1 to epoch 2. A branch of its epoch 2 refuses the maxsteps file as its Gram teacher (step 9, not epoch 1 x epoch_length 6 = 6; its epoch 1 is
# below 2 and it is of the same run, so only this check stands between them) with nothing created, and takes its epoch_0001.pth (step 6 = 1 x 6)
GT_M = os.path.join(TMP, "GT_M")
go(GT_M, "--no_gram", "--max_steps", "9", "--stop_after_epochs", "2")
go(GT_M, "--no_gram", "--resume", "--stop_after_epochs", "2")
GT_M_MAX, GT_M_E1, GT_M_E2 = (os.path.join(GT_M, "ckpt", name) for name in ("maxsteps_0000009.pth", "epoch_0001.pth", "epoch_0002.pth"))
gt_m_max, gt_m_e2 = (torch.load(path, map_location="cpu", weights_only=False) for path in (GT_M_MAX, GT_M_E2))
assert (gt_m_max["step"], gt_m_max["epoch"], gt_m_e2["step"], gt_m_e2["epoch"]) == (9, 1, 12, 2) and set(ssl_pretrain.BRANCH_SOURCE_KEYS) <= set(gt_m_max)
assert gt_m_max["args"]["output_dir"] == gt_m_e2["args"]["output_dir"] == GT_M
assert br_refused("gt_maxsteps", ValueError, "--branch_from", GT_M_E2, "--stop_after_epochs", "3", "--gram_teacher_from", GT_M_MAX) == (
    "--gram_teacher_from %s is not saved at an epoch boundary: its step 9, its epoch 1, so step 6 expected (epoch x epoch_length 6); a maxsteps_*.pth is saved mid-epoch" % GT_M_MAX)
gt_m_args = make_args(os.path.join(TMP, "gtx_helper"), "--branch_from", GT_M_E2, "--gram_refresh_steps", "100", "--gram_teacher_from", GT_M_E1)
assert ssl_pretrain.load_gram_teacher_source(gt_m_args, gt_m_e2, 6)[1] == 1  # the epoch-boundary checkpoint of the same run passes
assert "its step 6, its epoch 1, so step 7 expected" in raises(ValueError, ssl_pretrain.load_gram_teacher_source, gt_m_args, gt_m_e2, 7)  # the epoch_length is the one passed in
assert not os.path.exists(os.path.join(TMP, "gtx_helper"))
print("OK (t) --gram_teacher_from: a branch of the no-Gram trunk's epoch 2 with the Gram teacher from its epoch 1: before the first step the Gram teacher is epoch 1's EMA teacher backbone, tensor-equal "
      "(not epoch 2's), and the student / EMA teacher are epoch 2's, the step 12; the 'branched from' line names the file and its epoch; without a refresh it is still epoch 1's at epoch 3; the "
      "checkpoint and args.json record it (the trunk's None); a refresh (after step 18) copies the EMA teacher, not the earlier checkpoint; a --resume without it, with another checkpoint or a missing path is refused naming it (args.json untouched), the same one resumes "
      "with the file gone; at the start a missing file (FileNotFoundError), an export, the same epoch, a later epoch and another run dir's checkpoint are refused with nothing created, and the "
      "same-run test reads the recorded output_dir (a copy of epoch 1 elsewhere passes); a mid-epoch maxsteps_0000009.pth of the source's run (step 9, epoch 1) is refused naming its step, "
      "epoch and the expected step 6, its epoch_0001.pth passes")

# (u) DINOv3 recipe: --drop_path_rate, --rope_rescale_coords, --untie_local_cls_norm (the student's; train mode only) ------------------------------------------
from dinov3.models.vision_transformer import DinoVisionTransformer
from ssl_dinov3.model import build_backbone

RECIPE = ["--drop_path_rate", "0.3", "--rope_rescale_coords", "2", "--untie_local_cls_norm"]  # what chain/constlr.sh passes the no-Gram trunk and its branches
RECIPE_KEYS = ("drop_path_rate", "rope_rescale_coords", "untie_local_cls_norm")
RECIPE_ON = dict(drop_path_rate=0.3, pos_embed_rope_rescale_coords=2.0, untie_global_and_local_cls_norm=True)  # the kwargs run() adds to the student's for RECIPE
CLS_NORM_KEYS = {"local_cls_norm.weight", "local_cls_norm.bias"}

# parse: the defaults are the behaviour before the flags existed (and what an old checkpoint reads as); locked on --resume; a branch must match its source on them
assert tuple(getattr(P(), k) for k in RECIPE_KEYS) == (0.0, None, False) == tuple(ssl_pretrain.ARG_DEFAULTS[k] for k in RECIPE_KEYS)
assert set(RECIPE_KEYS) <= set(ssl_pretrain.RESUME_MATCH) and not set(RECIPE_KEYS) & set(ssl_pretrain.BRANCH_FREE)
assert tuple(getattr(P(*RECIPE), k) for k in RECIPE_KEYS) == (0.3, 2.0, True)
assert P("--rope_rescale_coords", " None ").rope_rescale_coords is None and P("--rope_rescale_coords", "1.5").rope_rescale_coords == 1.5
for extra in (["--drop_path_rate", "-0.1"], ["--drop_path_rate", "1"], ["--drop_path_rate", "nan"], ["--rope_rescale_coords", "1"], ["--rope_rescale_coords", "0.5"],
              ["--rope_rescale_coords", "-2"], ["--rope_rescale_coords", "inf"], ["--rope_rescale_coords", "nan"], ["--rope_rescale_coords", "two"]):
    with contextlib.redirect_stderr(io.StringIO()) as err:
        raises(SystemExit, P, *extra)
    assert extra[0] in err.getvalue().split("error: ", 1)[1], (extra, err.getvalue())
print("OK (u) parse: defaults drop path 0.0, rescale None, tied (= ARG_DEFAULTS, what a checkpoint without the keys reads as); RECIPE -> 0.3, 2.0, True; the three are "
      "--resume-locked and not branch-free; a drop path outside [0, 1) or NaN, a rescale <= 1, inf, NaN or a word exits 2 naming the flag")

# models: run() hands build_models the student's kwargs plus the recipe; at the defaults that builds bitwise what the kwargs alone built, drawing the same RNG
plain = ng_seeded(build_models, TINY_VIT, None, TINY_HEADS), torch.get_rng_state()
dflt = ng_seeded(build_models, dict(TINY_VIT, drop_path_rate=0.0, pos_embed_rope_rescale_coords=None, untie_global_and_local_cls_norm=False), None, TINY_HEADS), torch.get_rng_state()
for a, b in zip(plain[0], dflt[0]):
    assert list(a.state_dict()) == list(b.state_dict()) and all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
assert torch.equal(plain[1], dflt[1])
recipe_seen = []


def recipe_meta_init(self, student, teacher, gram_teacher, **kwargs):  # (local_cls_norm built?, drop path rates, RoPE rescale) of each backbone run() hands over
    recipe_seen.append([(m.backbone.local_cls_norm is not None, {blk.sample_drop_ratio for blk in m.backbone.blocks}, m.backbone.rope_embed.rescale_coords)
                        for m in (student, teacher, gram_teacher) if m is not None])
    return real_meta_init(self, student, teacher, gram_teacher, **kwargs)


SSLMetaArch.__init__ = recipe_meta_init
try:
    _, text_u0 = go(os.path.join(TMP, "U0"), "--max_steps", "1")
    _, text_u0r = go(os.path.join(TMP, "U0R"), *RECIPE, "--max_steps", "1")
finally:
    SSLMetaArch.__init__ = real_meta_init
assert recipe_seen == [[(False, {0.0}, None)] * 3, [(True, {0.3}, 2.0)] * 3], recipe_seen  # student, EMA teacher, Gram teacher: all off / all on
assert "recipe (" not in text_u0 and "local_cls_norm" not in text_u0  # the defaults print nothing new
assert "recipe (student, train mode only; the EMA / Gram teachers run in eval): drop path 0.3, RoPE coordinate rescale 2, local-crop CLS norm untied" in text_u0r.splitlines(), text_u0r
assert json.load(open(os.path.join(DIR_A, "args.json")))["drop_path_rate"] == 0.0 and json.load(open(os.path.join(DIR_A, "args.json")))["untie_local_cls_norm"] is False
info_u0r = json.load(open(os.path.join(TMP, "U0R", "args.json")))
assert tuple(info_u0r[k] for k in RECIPE_KEYS) == (0.3, 2.0, True) and ck2["args"]["rope_rescale_coords"] is None
print("OK (u) models: the default recipe kwargs build bitwise the models the plain kwargs build (same keys, values and RNG state after); through run() the defaults give "
      "student, EMA teacher and Gram teacher local_cls_norm None, drop path 0, rescale None, and RECIPE gives all three local_cls_norm, 0.3, 2.0 and prints one recipe line; "
      "args.json records the three (0.0 / None / False by default)")

# the stock load: with the CLS norm untied, every stock key loads strictly and local_cls_norm starts as a copy (not a share) of the stock norm; any other missing,
# unexpected or misshapen key is still refused. A tiny "stock" file whose norm is not a fresh LayerNorm's ones / zeros, so a copy is told apart from an init
tiny_stock = export_backbone(ng_seeded(build_backbone, TINY_VIT))
tiny_stock["norm.weight"], tiny_stock["norm.bias"] = torch.linspace(0.5, 1.5, 64), torch.linspace(-0.1, 0.1, 64)
TINY_STOCK = os.path.join(TMP, "tiny_stock.pth")
torch.save(tiny_stock, TINY_STOCK)
untied = build_backbone(dict(TINY_VIT, untie_global_and_local_cls_norm=True), TINY_STOCK)
untied_sd = untied.state_dict()
assert set(untied_sd) == set(tiny_stock) | CLS_NORM_KEYS and all(torch.equal(untied_sd[k], v) for k, v in tiny_stock.items())
assert torch.equal(untied_sd["local_cls_norm.weight"], tiny_stock["norm.weight"]) and torch.equal(untied_sd["local_cls_norm.bias"], tiny_stock["norm.bias"])
assert untied.local_cls_norm.weight.data_ptr() != untied.norm.weight.data_ptr() and untied.local_cls_norm.bias.data_ptr() != untied.norm.bias.data_ptr()
with torch.no_grad():
    untied.local_cls_norm.weight.add_(1.0)
assert torch.equal(untied.norm.weight, tiny_stock["norm.weight"])  # moving the copy leaves the norm alone
assert set(build_backbone(TINY_VIT, TINY_STOCK).state_dict()) == set(tiny_stock)  # tied: the stock load exactly as before
for name, broken, words in (("missing", {k: v for k, v in tiny_stock.items() if k != "blocks.0.attn.qkv.weight"}, "Missing key(s)"),
                            ("unexpected", dict(tiny_stock, extra=torch.zeros(1)), "Unexpected key(s)"),
                            ("shape", dict(tiny_stock, cls_token=torch.zeros(1, 1, 32)), "size mismatch")):
    path = os.path.join(TMP, "tiny_stock_%s.pth" % name)
    torch.save(broken, path)
    message = raises(RuntimeError, build_backbone, dict(TINY_VIT, untie_global_and_local_cls_norm=True), path)
    assert words in message and "local_cls_norm" not in message, (name, message)
print("OK (u) stock load, untied: every stock key bitwise, local_cls_norm.weight / .bias bitwise the stock norm's in their own storage (moving the copy leaves norm alone); "
      "a missing, unexpected or misshapen other key still raises (not naming local_cls_norm); tied, the stock key set exactly as before")

# one step through run() from that file (Gram run, backbone not frozen, no warmup, teacher momentum 0.9 so the EMA move is many ulps): before it the student's, the EMA
# teacher's and the Gram teacher's local_cls_norm are the stock norm; after it the student's has left its norm, the EMA teacher's follows it, and the export is the stock
# key set, loading strictly
U1 = os.path.join(TMP, "U1")
args_u1 = make_args(U1, *RECIPE, "--warmup_epochs", "0", "--freeze_backbone_epochs", "0", "--momentum_teacher", "0.9", "--max_steps", "1")
args_u1.pretrained_weights = TINY_STOCK
u_first = []


def u_forward_backward(self, data, *, teacher_temp):  # the backbones as the first step sees them (cloned)
    if not u_first:
        u_first.append({name: {k: v.detach().clone() for k, v in module.backbone.state_dict().items()}
                        for name, module in (("student", self.student), ("teacher", self.teacher), ("gram_teacher", self.gram_teacher))})
    return real_forward_backward(self, data, teacher_temp=teacher_temp)


SSLMetaArch.forward_backward = u_forward_backward
try:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ssl_pretrain.run(args_u1, TINY_VIT, TINY_HEADS, CFG)
    text_u1 = buf.getvalue()
finally:
    SSLMetaArch.forward_backward = real_forward_backward
assert ("local_cls_norm: initialized as a copy of norm.weight / norm.bias from %s (the stock file has no local_cls_norm)" % TINY_STOCK) in text_u1.splitlines(), text_u1
for name in ("student", "teacher", "gram_teacher"):
    assert all(torch.equal(u_first[0][name]["local_cls_norm." + p], tiny_stock["norm." + p]) for p in ("weight", "bias")), name
ck_u1 = load_ckpt(U1, "maxsteps_0000001.pth")
stu, tea = backbone_of(ck_u1["student"]), backbone_of(ck_u1["teacher"])
assert CLS_NORM_KEYS <= set(stu) and CLS_NORM_KEYS <= set(tea) and CLS_NORM_KEYS <= set(backbone_of(ck_u1["gram_teacher"]))  # the full checkpoint carries it, all three models
assert not torch.equal(stu["local_cls_norm.weight"], stu["norm.weight"]) and not torch.equal(stu["local_cls_norm.weight"], tiny_stock["norm.weight"])
for p in ("weight", "bias"):
    want = 0.9 * tiny_stock["norm." + p].double() + 0.1 * stu["local_cls_norm." + p].double()
    assert torch.allclose(tea["local_cls_norm." + p].double(), want, rtol=1e-6, atol=1e-7), p
assert not torch.equal(tea["local_cls_norm.weight"], tiny_stock["norm.weight"])  # the EMA really moved it
export_u1 = torch.load(os.path.join(U1, "export", "teacher_backbone_maxsteps_0000001.pth"), weights_only=True)
assert set(export_u1) == set(tiny_stock) and all(torch.equal(v, tea[k]) for k, v in export_u1.items())
DinoVisionTransformer(**TINY_VIT).load_state_dict(export_u1, strict=True)  # the tiny ViT's own kwargs, no recipe: as Ark builds it
print("OK (u) one step from a stock file: the student's, EMA teacher's and Gram teacher's local_cls_norm start as the stock norm (and run() says so); after the step the "
      "student's differs from its norm, the EMA teacher's is 0.9 x stock + 0.1 x the student's (momentum 0.9), the checkpoint carries it for all three models, and the export has "
      "exactly the stock keys (the EMA teacher's backbone without it), loading strictly into a backbone built without the recipe")

# the real ViT-B/16: the stock file with the recipe, the export into Ark's strict vit_base_dinov3 guard
if WEIGHTS is None:
    print("SKIP (u) ViT-B/16 stock load + export with the recipe: %s not found" % WEIGHT_NAME)
else:
    stock_b = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    models_b = build_models(dict(DINOV3_VITB16_KWARGS, **RECIPE_ON), WEIGHTS, TINY_HEADS)
    for model in models_b:
        sd_b = model.backbone.state_dict()
        assert set(sd_b) == set(stock_b) | CLS_NORM_KEYS and all(torch.equal(sd_b[k], v) for k, v in stock_b.items())
        assert torch.equal(sd_b["local_cls_norm.weight"], stock_b["norm.weight"]) and torch.equal(sd_b["local_cls_norm.bias"], stock_b["norm.bias"])
    teacher_b = models_b[1]
    with torch.no_grad():
        for p in teacher_b.backbone.parameters():
            p.add_(1e-3)  # no longer the stock weights
    exported_b = export_backbone(teacher_b.backbone)
    assert len(exported_b) == 188 == len(stock_b) and set(exported_b) == set(stock_b) and any(not torch.equal(exported_b[k], stock_b[k]) for k in stock_b)
    DinoVisionTransformer(**DINOV3_VITB16_KWARGS).load_state_dict(exported_b, strict=True)  # Ark's kwargs, unchanged
    path_b = os.path.join(TMP, "export_recipe.pth")
    torch.save(exported_b, path_b)
    ark_vit_b = build_omni_model(SimpleNamespace(model_name="vit_base_dinov3", pretrained_weights=path_b, projector_features=1376, use_mlp=False), [6, 14, 14, 14])
    assert all(torch.equal(ark_vit_b.state_dict()[k], v) for k, v in exported_b.items())
    del stock_b, models_b, teacher_b, exported_b, ark_vit_b
    print("OK (u) ViT-B/16 with the recipe: student, EMA teacher and Gram teacher load every stock key bitwise with local_cls_norm = the stock norm; the export of a moved "
          "teacher has the 188 stock keys, loads strictly into DinoVisionTransformer(**DINOV3_VITB16_KWARGS) and into build_omni_model(vit_base_dinov3) bitwise")

# the forward: the EMA and Gram teachers (eval) are deterministic with the recipe on; the student (train) is not, through drop path and the RoPE rescale each on its own;
# local_cls_norm normalizes entry 1 of the student's list forward, which SSLMetaArch fills with the local crops (8 x B rows, not the 2 x B global ones)
meta_u = SSLMetaArch(*ng_seeded(build_models, dict(TINY_VIT, **RECIPE_ON), None, TINY_HEADS), n_local_crops=8).train()
batch_u = batches[0]  # (a): 4 images, Gram crops 448
g_u, l_u, m_u = batch_u["collated_global_crops"], batch_u["collated_local_crops"], batch_u["collated_masks"]
assert meta_u.student.backbone.training and not meta_u.teacher.backbone.training and not meta_u.gram_teacher.backbone.training
same_out = lambda a, b: all(torch.equal(a[k], b[k]) for k in ("x_norm_clstoken", "x_storage_tokens", "x_norm_patchtokens"))
with torch.no_grad():
    for module, crops in ((meta_u.teacher.backbone, g_u), (meta_u.gram_teacher.backbone, batch_u["collated_gram_teacher_crops"])):
        assert same_out(module(crops, is_training=True), module(crops, is_training=True))
    s1, s2 = (meta_u.student.backbone([g_u, l_u], masks=[m_u, None], is_training=True) for _ in range(2))
    assert not same_out(s1[0], s2[0]) and not same_out(s1[1], s2[1])
    for kwargs, varies in ((dict(drop_path_rate=0.3), True), (dict(pos_embed_rope_rescale_coords=2.0), True),
                           (dict(untie_global_and_local_cls_norm=True), False), ({}, False)):
        alone = build_backbone(dict(TINY_VIT, **kwargs)).train()
        o1, o2 = alone([g_u, l_u], masks=[m_u, None], is_training=True), alone([g_u, l_u], masks=[m_u, None], is_training=True)
        assert (not (same_out(o1[0], o2[0]) and same_out(o1[1], o2[1]))) == varies, kwargs
cls_norm_calls = []
hooks = [meta_u.student.backbone.local_cls_norm.register_forward_hook(lambda mod, inp, out: cls_norm_calls.append(("student", tuple(inp[0].shape), out[:, 0].clone()))),
         meta_u.teacher.backbone.local_cls_norm.register_forward_hook(lambda mod, inp, out: cls_norm_calls.append(("teacher", tuple(inp[0].shape), None))),
         meta_u.gram_teacher.backbone.local_cls_norm.register_forward_hook(lambda mod, inp, out: cls_norm_calls.append(("gram", tuple(inp[0].shape), None)))]
try:
    with torch.no_grad():
        _, student_local_u = meta_u.get_student_output(global_crops=g_u.unflatten(0, (2, 4)), local_crops=l_u.unflatten(0, (8, 4)), upperbound=batch_u["upperbound"],
                                                       masks=m_u, mask_indices_list=batch_u["mask_indices_list"])
    calls_student = list(cls_norm_calls)
    cls_norm_calls.clear()
    _, loss_dict_u = meta_u.forward_backward(batch_u, teacher_temp=0.04)
finally:
    for hook in hooks:
        hook.remove()
assert [(who, shape) for who, shape, _ in calls_student] == [("student", (32, 5, 64))]  # 8 local crops x 4 images; CLS + 4 registers
assert torch.equal(calls_student[0][2], student_local_u["cls_pre_head"].flatten(0, 1))  # the local CLS the DINO local loss uses
assert [(who, shape) for who, shape, _ in cls_norm_calls] == [("student", (32, 5, 64))] and set(loss_dict_u) == LOSS_KEYS  # a whole step: once, never by a teacher
print("OK (u) forward: with the recipe on, two EMA-teacher and two Gram-teacher forwards of one batch are bitwise equal (eval); two student forwards differ, and so they do "
      "with drop path 0.3 alone or rescale 2 alone (not with the untied norm alone, nor with nothing); local_cls_norm runs once per step, on the student's entry 1 = the 32 "
      "local crops (8 x 4 images, CLS + 4 registers), and gives the local CLS the DINO loss uses; the teachers never call it. Crop order: get_student_output passes "
      "[global_crops, local_crops], so Meta's idx == 1 rule picks the local crops")

# resume lock: a recipe checkpoint refuses a --resume / --branch_from without the recipe and the reverse, naming the keys, before anything on disk is touched;
# a checkpoint written before the flags reads as the defaults
U_T = os.path.join(TMP, "U_T")
go(U_T, "--no_gram", *RECIPE, "--stop_after_epochs", "1")
U_SRC1 = os.path.join(U_T, "ckpt", "epoch_0001.pth")
before_ut = open(os.path.join(U_T, "args.json")).read()
RECIPE_DIFFS = ("drop_path_rate: checkpoint 0.3, now 0.0", "rope_rescale_coords: checkpoint 2.0, now None", "untie_local_cls_norm: checkpoint True, now False")
message = raises(ValueError, go, U_T, "--no_gram", "--resume", "--stop_after_epochs", "2")
assert "cannot --resume" in message and all(d in message for d in RECIPE_DIFFS), message
message = raises(ValueError, go, U_T, "--no_gram", "--resume", "--stop_after_epochs", "2", *RECIPE[:4])  # tied only
assert "untie_local_cls_norm: checkpoint True, now False" in message and "drop_path_rate" not in message and "rope_rescale_coords" not in message, message
assert open(os.path.join(U_T, "args.json")).read() == before_ut
before_b = open(os.path.join(DIR_B, "args.json")).read()
message = raises(ValueError, go, DIR_B, "--resume", *RECIPE)
assert all(d in message for d in ("drop_path_rate: checkpoint 0.0, now 0.3", "rope_rescale_coords: checkpoint None, now 2.0", "untie_local_cls_norm: checkpoint False, now True")), message
assert open(os.path.join(DIR_B, "args.json")).read() == before_b
for name, extra, diffs in (("u_branch_off", ["--branch_from", U_SRC1], RECIPE_DIFFS),  # a recipe trunk branched without it
                           ("u_branch_on", ["--branch_from", SRC1, *RECIPE], ("drop_path_rate: checkpoint 0.0, now 0.3", "untie_local_cls_norm: checkpoint False, now True"))):
    out = os.path.join(TMP, name)
    message = raises(ValueError, go, out, *extra, "--gram_refresh_steps", "100", "--stop_after_epochs", "2")
    assert "cannot --branch_from" in message and all(d in message for d in diffs) and not os.path.exists(out), (name, message)
ck_ut = load_ckpt(U_T, "epoch_0001.pth")
assert tuple(ck_ut["args"][k] for k in RECIPE_KEYS) == (0.3, 2.0, True)
ck_old_recipe = dict(ck_t, args={k: v for k, v in ck_t["args"].items() if k not in RECIPE_KEYS})  # the (t) trunk, as written before the flags existed
ssl_pretrain.check_resume(ck_old_recipe, make_args(BR_T, "--no_gram"), 1, 6, ck_t["data_fingerprint"])  # reads as the defaults: passes
message = raises(ValueError, ssl_pretrain.check_resume, ck_old_recipe, make_args(BR_T, "--no_gram", *RECIPE), 1, 6, ck_t["data_fingerprint"])
assert all(d in message for d in ("drop_path_rate: checkpoint 0.0, now 0.3", "rope_rescale_coords: checkpoint None, now 2.0", "untie_local_cls_norm: checkpoint False, now True")), message
print("OK (u) resume lock: a recipe trunk's checkpoint refuses --resume without the recipe (naming drop_path_rate 0.3 / rope_rescale_coords 2.0 / untie_local_cls_norm True "
      "against the defaults; a partial recipe names only the missing key) and a recipe-less run refuses --resume with it; --branch_from a recipe trunk without it, or a "
      "recipe-less trunk with it, is refused naming the keys; args.json untouched and no branch dir created; a checkpoint without the keys reads as the defaults")

# resume bitwise: a recipe trunk stopped after epoch 1 and resumed equals the uninterrupted one, with set_epoch acting as it does with --workers > 0 (the cluster's 16):
# the main process's random / numpy / torch RNGs are then not re-seeded per epoch, so epoch 2's draws (drop path, RoPE rescale, and here the in-process transforms) start
# from the checkpoint's saved RNG states. Control: the same resume from a copy whose saved torch RNG state was replaced differs. And a --gram_weight 0 branch of the recipe
# trunk is bitwise its resume: the Gram teacher (eval, local_cls_norm included) draws nothing and adds exact zeros


def set_epoch_main_rngs_untouched(loader, sampler, epoch, seed, rank):  # ssl_dinov3.data.set_epoch with num_workers > 0: sampler + worker base seed only
    sampler.set_epoch(epoch)
    loader.generator.manual_seed(seed + 1000 * epoch + rank)


U_A, U_B, U_C, U_BR = (os.path.join(TMP, n) for n in ("U_A", "U_B", "U_C", "U_BR"))
UA_SRC1 = os.path.join(U_A, "ckpt", "epoch_0001.pth")  # still there after the resume: the trainer keeps the last 3 epochs
real_set_epoch, ssl_pretrain.set_epoch = ssl_pretrain.set_epoch, set_epoch_main_rngs_untouched
try:
    go(U_A, "--no_gram", *RECIPE, "--stop_after_epochs", "1")
    shutil.copytree(U_A, U_C)
    ck_c = load_ckpt(U_C, "epoch_0001.pth")
    ck_c["rng"]["torch"] = torch.Generator().manual_seed(12345).get_state()
    torch.save(ck_c, os.path.join(U_C, "ckpt", "epoch_0001.pth"))
    go(U_A, "--no_gram", *RECIPE, "--resume", "--stop_after_epochs", "2")
    go(U_B, "--no_gram", *RECIPE, "--stop_after_epochs", "2")
    go(U_C, "--no_gram", *RECIPE, "--resume", "--stop_after_epochs", "2")
    go(U_BR, "--branch_from", UA_SRC1, *RECIPE, "--gram_weight", "0", "--gram_refresh_steps", "100", "--stop_after_epochs", "2")
finally:
    ssl_pretrain.set_epoch = real_set_epoch
ck_ua, ck_ub, ck_uc, ck_ubr = (load_ckpt(d, "epoch_0002.pth") for d in (U_A, U_B, U_C, U_BR))
src_ua = torch.load(UA_SRC1, map_location="cpu", weights_only=False)
assert not same_state(ck_ub["student"], src_ua["student"]) and not same_state(ck_ub["rng"], src_ua["rng"])  # epoch 2 trained and drew
exports_u = {d: torch.load(os.path.join(d, "export", "teacher_backbone_epoch_0002.pth"), weights_only=True) for d in (U_A, U_B, U_C, U_BR)}
assert same_state(exports_u[U_A], exports_u[U_B]) and same_state(exports_u[U_BR], exports_u[U_B]) and set(exports_u[U_B]) == set(tiny_stock)
for part in ("student", "teacher", "optimizer", "rng"):
    assert same_state(ck_ua[part], ck_ub[part]) and same_state(ck_ubr[part], ck_ub[part]), part
assert CLS_NORM_KEYS <= set(backbone_of(ck_ubr["gram_teacher"])) and ck_ubr["args"]["untie_local_cls_norm"] is True
assert not same_state(ck_uc["student"], ck_ub["student"]) and not same_state(exports_u[U_C], exports_u[U_B])  # the control: the saved RNG state is what makes it bitwise
by_step_ub = {r[0]: r[2:13] for r in read_csv(U_B)[1:]}  # dino_global .. gram_weight, as strings
common_ua = [r for r in read_csv(U_A)[1:] if r[0] in by_step_ub]  # a run's first step after a start or resume is logged off the print_freq grid
common_ubr = [r for r in read_csv(U_BR)[1:] if r[0] in by_step_ub]
assert len(common_ua) == 7 and all(r[2:13] == by_step_ub[r[0]] for r in common_ua)  # steps 0, 1, 3, 5, 7, 9, 11: every row the uninterrupted run logged
assert len(common_ubr) >= 3 and all(r[2:6] + r[7:12] == by_step_ub[r[0]][:4] + by_step_ub[r[0]][5:10] for r in common_ubr)  # the branch: all but its gram column
print("OK (u) resume bitwise, recipe on, set_epoch as with --workers > 0: 1 epoch + --resume to 2 == 2 epochs straight (export, student, EMA teacher, optimizer, RNG states, "
      "the CSV rows both logged), and a --gram_weight 0 branch of epoch 1 == both (its Gram teacher carries local_cls_norm); the same resume from a checkpoint whose torch RNG state "
      "was replaced differs (student, export)")

# (v) per-rank model-side RNG: seed + rank once the models exist; every rank's states in a multi-rank checkpoint -----------------------------------------------------------------------
# Once the models exist each rank re-seeds python / numpy / torch with seed + rank (so drop path and the RoPE rescale differ across ranks; rank 0 keeps seed). A multi-rank checkpoint also
# carries every rank's states (rng_per_rank; rng stays rank 0's, the whole story at one rank), and a --resume / --branch_from restores each rank its own.


def seeded_rng(seed):  # the (python, numpy, torch) RNG states right after seeding with `seed`
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


@contextlib.contextmanager
def as_rank(rank, world_size):  # the world faked as (q) does: the one-rank gloo group stays up, Accelerator reads RANK / WORLD_SIZE from the env
    saved = {key: os.environ.get(key) for key in ("RANK", "WORLD_SIZE")}
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size))
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


v_flat = lambda models: {f"{k}.{n}": t.detach().clone() for k, m in models.items() for n, t in Accelerator.unwrap_model(m).state_dict().items()}
v_handed = []


def v_meta_init(self, student, teacher, gram_teacher, **kwargs):  # the RNG states and the models as run() hands them over, before any step
    v_handed.append(dict(rng=(random.getstate(), np.random.get_state(), torch.get_rng_state()),
                         models={name: v_flat(m) for name, m in (("student", student), ("teacher", teacher), ("gram_teacher", gram_teacher)) if m is not None}))
    return real_meta_init(self, student, teacher, gram_teacher, **kwargs)


def v_start(out, *extra):  # run() up to its first step (models built, the checkpoint restored if there is one); returns what it printed
    buf = io.StringIO()
    SSLMetaArch.__init__, SSLMetaArch.forward_backward = v_meta_init, stop_at_forward
    try:
        with contextlib.redirect_stdout(buf):
            raises(StopAtForward, ssl_pretrain.run, make_args(out, *extra), TINY_VIT, TINY_HEADS, CFG)
    finally:
        SSLMetaArch.__init__, SSLMetaArch.forward_backward = real_meta_init, real_forward_backward
    return buf.getvalue()


# one process: the RNGs are re-seeded after the models are built, with seed + 0, and the models are those of a build without the re-seed
V_SEED = P().seed
v_start(os.path.join(TMP, "V_ONE"))
v_one = v_handed[-1]
fresh_state = seeded_rng(V_SEED)
assert same_state(v_one["rng"], fresh_state)  # python, numpy and torch: the fresh seed + 0 state
no_reseed = build_models(TINY_VIT, None, TINY_HEADS)  # seeded as above, built, carried on: what 28eacd8 handed over
assert not torch.equal(torch.get_rng_state(), fresh_state[2])  # the build drew torch's init, so the re-seed after it is not a no-op
assert same_state(v_one["models"], dict(zip(("student", "teacher", "gram_teacher"), map(v_flat, no_reseed))))  # student, EMA teacher, Gram teacher, heads included
print("OK (v) one process: when run() hands the models over, python / numpy / torch are in the fresh seed + 0 state (the build had moved torch's, so the re-seed came after it), and "
      "the student, EMA teacher and Gram teacher (heads included) are bitwise those of a build seeded once and not re-seeded")

# ranks 0 and 1 of 2 (the world faked as (q) does it): rank 1 is in the seed + 1 state, every stream differs from rank 0's, the models are bitwise rank 0's (and the one process')
with as_rank(0, 2):
    v_start(os.path.join(TMP, "V_RANK0"))
with as_rank(1, 2):
    v_start(os.path.join(TMP, "V_RANK1"))
v_rank0, v_rank1 = v_handed[-2:]
assert same_state(v_rank0["rng"], seeded_rng(V_SEED)) and same_state(v_rank1["rng"], seeded_rng(V_SEED + 1))
assert all(not same_state(a, b) for a, b in zip(v_rank0["rng"], v_rank1["rng"]))  # python, numpy and torch each
assert same_state(v_rank1["models"], v_rank0["models"]) and same_state(v_rank0["models"], v_one["models"]) and set(v_rank1["models"]) == {"student", "teacher", "gram_teacher"}
print("OK (v) two ranks, faked: rank 0 of 2 is in the seed + 0 state, rank 1 in the seed + 1 state (python, numpy and torch each differ from rank 0's), and the student, EMA teacher "
      "and Gram teacher are bitwise equal on both ranks and to the one-process run's")

# two real processes (gloo on this CPU, loopback rendezvous): the gather is a collective every rank must reach, which the faked world cannot show. set_epoch acts as with --workers > 0
# (the main process' RNGs are not re-seeded per epoch), the recipe is on, so epoch 2's draws (drop path, RoPE rescale, the in-process transforms) start from each rank's saved states
V_DRIVER, V_DRIVER_SOURCE = os.path.join(TMP, "v_driver.py"), '''# one rank of a real two-process (gloo, CPU) run() with the tiny ViT, written by test_ssl_pretrain.py (v); argv: base dir, spec json
import argparse, json, os, random, sys
import numpy as np
import torch
import torch.distributed as dist

base, spec = sys.argv[1], json.load(open(sys.argv[2]))
sys.path.insert(0, base)
os.chdir(base)
import ssl_pretrain
from ssl_dinov3.meta import SSLMetaArch

rank = int(os.environ["RANK"])
states = lambda: dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
real_init = SSLMetaArch.__init__


def init(self, *args, **kwargs):  # this rank's RNG states as run() hands the models over
    torch.save(states(), os.path.join(spec["dump"], "built_rank%d.pt" % rank))
    return real_init(self, *args, **kwargs)


def set_epoch(loader, sampler, epoch, seed, rank):  # as with --workers > 0: the sampler and the worker base seed, never this process' RNGs
    sampler.set_epoch(epoch)
    loader.generator.manual_seed(seed + 1000 * epoch + rank)


SSLMetaArch.__init__, ssl_pretrain.set_epoch = init, set_epoch
ssl_pretrain.run(argparse.Namespace(**spec["args"]), spec["vit"], spec["heads"], spec["cfg"])
torch.save(states(), os.path.join(spec["dump"], "end_rank%d.pt" % rank))
dist.destroy_process_group()
'''
with open(V_DRIVER, "w") as f:
    f.write(V_DRIVER_SOURCE)


def v_two_ranks(tag, out, *extra):  # run() on two real ranks; returns the dir holding each rank's RNG states (at the models, at the end) and both ranks' output
    dump = os.path.join(TMP, "v_dump_" + tag)
    os.makedirs(dump)
    spec = os.path.join(TMP, "v_spec_%s.json" % tag)
    with open(spec, "w") as f:
        json.dump(dict(args=vars(make_args(out, *extra)), vit=TINY_VIT, heads=TINY_HEADS, cfg=CFG, dump=dump), f)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        env = dict(os.environ, WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(s.getsockname()[1]))
    procs = [subprocess.Popen([sys.executable, V_DRIVER, BASE, spec], env=dict(env, RANK=str(r), LOCAL_RANK=str(r)), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) for r in range(2)]
    try:
        outputs = [p.communicate(timeout=120)[0] for p in procs]
    finally:
        for p in procs:
            p.kill()  # a rank stuck in a collective must not outlive the test
            p.wait()
    assert [p.returncode for p in procs] == [0, 0], outputs
    return dump, outputs


v_dumped = lambda dump, kind, rank: torch.load(os.path.join(dump, "%s_rank%d.pt" % (kind, rank)), weights_only=False)
V_A, V_B, V_C = (os.path.join(TMP, n) for n in ("V_A", "V_B", "V_C"))
V_TRUNK = ["--no_gram", *RECIPE]
v_two_ranks("a", V_A, *V_TRUNK, "--stop_after_epochs", "2")  # 2 epochs straight
dump_b1, _ = v_two_ranks("b1", V_B, *V_TRUNK, "--stop_after_epochs", "1")  # epoch 1, then the resume below
shutil.copytree(V_B, V_C)
ck_b1 = load_ckpt(V_B, "epoch_0001.pth")
ck_c1 = load_ckpt(V_C, "epoch_0001.pth")
ck_c1["rng_per_rank"][1] = ck_c1["rng_per_rank"][0]  # the control: rank 1 gets rank 0's states back, as every rank did before the per-rank states
torch.save(ck_c1, os.path.join(V_C, "ckpt", "epoch_0001.pth"))
v_two_ranks("b2", V_B, *V_TRUNK, "--resume", "--stop_after_epochs", "2")
v_two_ranks("c", V_C, *V_TRUNK, "--resume", "--stop_after_epochs", "2")
for rank in (0, 1):  # each real rank re-seeded with seed + rank once its models were built, and its saved entry is its own state at the save
    assert same_state(v_dumped(dump_b1, "built", rank), dict(zip(("python", "numpy", "torch"), seeded_rng(V_SEED + rank))))
    assert same_state(ck_b1["rng_per_rank"][rank], v_dumped(dump_b1, "end", rank))
assert ck_b1["world_size"] == 2 and len(ck_b1["rng_per_rank"]) == 2 and not same_state(*ck_b1["rng_per_rank"]) and same_state(ck_b1["rng"], ck_b1["rng_per_rank"][0])
ck_va, ck_vb, ck_vc = (load_ckpt(d, "epoch_0002.pth") for d in (V_A, V_B, V_C))
for part in ("student", "teacher", "optimizer", "rng", "rng_per_rank"):  # 1 epoch + --resume == 2 epochs straight, every rank's stream included
    assert same_state(ck_vb[part], ck_va[part]), part
assert same_state(torch.load(os.path.join(V_B, "export", "teacher_backbone_epoch_0002.pth"), weights_only=True),
                  torch.load(os.path.join(V_A, "export", "teacher_backbone_epoch_0002.pth"), weights_only=True))
assert not same_state(ck_va["student"], ck_b1["student"]) and not same_state(ck_vc["student"], ck_va["student"])  # epoch 2 trained; and the control, with rank 1 on rank 0's stream, differs
print("OK (v) two real ranks: each rank is in its seed + rank state when the models are handed over; the epoch-1 checkpoint has world_size 2, rng_per_rank of 2 distinct entries (each its "
      "own rank's state at the save) and rng = entry 0; a recipe run of 1 epoch + --resume equals 2 epochs straight (export, student, EMA teacher, optimizer, rng, both ranks' states), "
      "and the same resume with rank 1 handed rank 0's states differs (student)")

# which entry a rank restores (the epoch-1 checkpoint of the real run, the world faked): its own on --resume and on --branch_from; a checkpoint without rng_per_rank gives the one state to every
# rank and rank 0 says so; a list that does not match the world is refused
v_setters = random.setstate, np.random.set_state, torch.set_rng_state


@contextlib.contextmanager
def recording_rng_restores():  # yields the list of (name, state) the three setters are handed, in order
    restored = []
    spy = lambda name, real: lambda state: (restored.append((name, state)), real(state))[1]
    random.setstate, np.random.set_state, torch.set_rng_state = (spy(name, real) for name, real in zip(("python", "numpy", "torch"), v_setters))
    try:
        yield restored
    finally:
        random.setstate, np.random.set_state, torch.set_rng_state = v_setters


def v_restores(rank, out, *extra):  # run() as this rank of 2, up to its first step: (the (name, state) pairs the setters got, what it printed)
    with as_rank(rank, 2), recording_rng_restores() as restored:
        text = v_start(out, *extra)
    return restored, text


def v_one_checkpoint(name, ckpt):  # a run dir holding just this epoch-1 checkpoint (--resume reads the newest one)
    out = os.path.join(TMP, name)
    os.makedirs(os.path.join(out, "ckpt"))
    torch.save(ckpt, os.path.join(out, "ckpt", "epoch_0001.pth"))
    return out


V_NEW = v_one_checkpoint("V_NEW", ck_b1)
V_OLD = v_one_checkpoint("V_OLD", {k: v for k, v in ck_b1.items() if k != "rng_per_rank"})  # as written before the per-rank states
V_BAD = v_one_checkpoint("V_BAD", dict(ck_b1, rng_per_rank=ck_b1["rng_per_rank"] + ck_b1["rng_per_rank"][:1]))  # 3 entries, 2 ranks
OLD_LINE = "RNG: the checkpoint predates the per-rank RNG states and holds one: it is restored on every rank (2)"
for rank in (0, 1):
    for out, extra, said in ((V_NEW, (*V_TRUNK, "--resume", "--stop_after_epochs", "2"), "resumed from"),
                             (os.path.join(TMP, "V_BRANCH%d" % rank), (*RECIPE, "--branch_from", os.path.join(V_NEW, "ckpt", "epoch_0001.pth"),
                                                                      "--gram_refresh_steps", "100", "--stop_after_epochs", "2"), "branched from")):
        restored, text = v_restores(rank, out, *extra)
        assert [name for name, _ in restored] == ["python", "numpy", "torch"] and same_state(dict(restored), ck_b1["rng_per_rank"][rank]), (rank, said)
        assert "predates" not in text and (said in text) == (rank == 0), (rank, said)  # only rank 0 prints; nothing about old checkpoints
    restored, text = v_restores(rank, V_OLD, *V_TRUNK, "--resume", "--stop_after_epochs", "2")
    assert [name for name, _ in restored] == ["python", "numpy", "torch"] and same_state(dict(restored), ck_b1["rng"])  # the one state, on this rank too
    assert (OLD_LINE in text.splitlines()) == (rank == 0) and ("predates" in text) == (rank == 0)  # rank 0 says so, once
with as_rank(0, 2):
    message = raises(AssertionError, v_start, V_BAD, *V_TRUNK, "--resume", "--stop_after_epochs", "2")
assert "the checkpoint holds 3 per-rank RNG states, this run has 2 ranks" in message, message
print("OK (v) restore: a faked rank 0 / rank 1 of 2 takes entry 0 / entry 1 of rng_per_rank back on --resume and on --branch_from (python, numpy, torch, each bitwise the saved one) and says "
      "nothing; a checkpoint without rng_per_rank restores its one state on both ranks and rank 0 prints '%s' once; 3 entries for 2 ranks raise AssertionError naming both counts" % OLD_LINE)

# one process: no per-rank list in its checkpoint (the key set (g) asserts is unchanged), no extra collective; the bitwise resume tests (g), (t), (u) above run on exactly these checkpoints
gathers, real_all_gather_object = [], dist.all_gather_object
dist.all_gather_object = lambda *args, **kwargs: (gathers.append(args), real_all_gather_object(*args, **kwargs))[1]
try:
    go(os.path.join(TMP, "V_SINGLE"), "--no_gram", "--stop_after_epochs", "1")
finally:
    dist.all_gather_object = real_all_gather_object
assert gathers == [] and "rng_per_rank" not in load_ckpt(os.path.join(TMP, "V_SINGLE"), "epoch_0001.pth") and "rng_per_rank" not in ck2
print("OK (v) one process: its checkpoints carry no rng_per_rank and its run gathers nothing, so a one-process run saves, resumes and branches exactly as before (rng is the one state)")

# (w) DDP find_unused_parameters: False for the SSL student (all of its parameters get a gradient every step), True by default (Ark+ fine-tuning) -----------------------------------------------------
# run() wraps backbone, dino_head and ibot_head each in DDP with find_unused_parameters=False; Accelerator.wrap_model without the keyword keeps True. A one-process run builds no DDP (what (g), (s), (t),
# (u), (v) run on), so (a) checks the premise itself, a grad on every parameter after each backward, and (b), (c) the wrappers on two real ranks (gloo on this CPU, as (v)).
w_steps = []


def w_forward_backward(self, data, *, teacher_temp):  # per student parameter: (its grad was None before the backward, requires_grad, it has a grad after it); run() sets the grads to None before every step
    fresh = {name: p.grad is None for name, p in self.student.named_parameters()}
    result = real_forward_backward(self, data, teacher_temp=teacher_temp)
    w_steps.append({name: (fresh[name], p.requires_grad, p.grad is not None) for name, p in self.student.named_parameters()})
    return result


# (a) 2 epochs of 6 steps: in epoch 0 the backbone and the heads' last layer are at LR 0 ((h): they do not move, and stay requires_grad), in epoch 1 they train. The 4 settings are the 4 kinds of run the chain
# launches: the constlr phase and its Gram branches (Gram; recipe off / on), the no-Gram trunk (no Gram, recipe) and the plain trunk (no Gram, no recipe)
W_SETTINGS = (("recipe off, Gram", ()), ("recipe on, Gram", RECIPE), ("recipe off, no Gram", ("--no_gram",)), ("recipe on, no Gram", (*RECIPE, "--no_gram")))
w_counts = set()
for w_i, (w_setting, w_extra) in enumerate(W_SETTINGS):
    w_steps.clear()
    SSLMetaArch.forward_backward = w_forward_backward
    try:
        summary_w, _ = go(os.path.join(TMP, "W_ONE%d" % w_i), "--stop_after_epochs", "2", *w_extra)
    finally:
        SSLMetaArch.forward_backward = real_forward_backward
    w_names = set(w_steps[0])
    assert len(w_steps) == 2 * summary_w["epoch_length"] == 12 and all(set(step) == w_names for step in w_steps), w_setting
    assert all(fresh and trainable and has_grad for step in w_steps for fresh, trainable, has_grad in step.values()), w_setting  # grads start from None, none is frozen, none is missing afterwards
    assert {n.split(".")[0] for n in w_names} == {"backbone", "dino_head", "ibot_head"}, w_setting
    assert {"backbone.mask_token", "backbone.storage_tokens", "dino_head.last_layer.weight", "ibot_head.last_layer.weight"} <= w_names, w_setting
    assert ("backbone.local_cls_norm.weight" in w_names) == ("--untie_local_cls_norm" in w_extra), w_setting
    w_counts.add(len(w_names))
assert len(w_counts) == 2  # the untied CLS norm adds its weight and bias
print("OK (w) one process, 4 settings (recipe off / on x Gram / --no_gram): all %d student parameters (%d with --untie_local_cls_norm: backbone incl. mask_token, storage_tokens and local_cls_norm; both heads "
      "incl. last_layer) are trainable and have a grad after each of the 12 backwards, in epoch 0 (backbone and last layer at LR 0) and in epoch 1, every grad starting from None" % (min(w_counts), max(w_counts)))

# (b), (c) two real processes (gloo on this CPU, loopback rendezvous), the driver written like (v)'s: mode "run" is run() with the tiny ViT and notes the student's DDP modules, mode "wrap" calls
# Accelerator.wrap_model directly on a module with a parameter in no forward (as Ark+'s inactive dataset heads are in a step)
W_DRIVER, W_DRIVER_SOURCE = os.path.join(TMP, "w_driver.py"), '''# one rank of a real two-process (gloo, CPU) job, written by test_ssl_pretrain.py (w); argv: base dir, spec json
import argparse, json, os, sys
import torch
import torch.distributed as dist

base, spec = sys.argv[1], json.load(open(sys.argv[2]))
sys.path.insert(0, base)
os.chdir(base)
import ssl_pretrain
from accelerator import Accelerator
from ssl_dinov3.meta import SSLMetaArch

found = {}
flag = lambda ddp: int(ddp._get_ddp_logging_data()["find_unused_parameters"])  # held as 1 / 0


class Net(torch.nn.Module):  # `unused` is in no forward
    def __init__(self):
        super().__init__()
        self.used, self.unused = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.used(x)


if spec["mode"] == "run":
    real_init = SSLMetaArch.__init__

    def init(self, student, *args, **kwargs):  # the student's DDP modules as run() hands them over
        found["student"] = {name: flag(module) for name, module in student.items()}
        return real_init(self, student, *args, **kwargs)

    SSLMetaArch.__init__ = init
    ssl_pretrain.run(argparse.Namespace(**spec["args"]), spec["vit"], spec["heads"], spec["cfg"])
else:
    accelerator = Accelerator("cpu")
    accelerator.initialize_distributed()
    for name, kwargs in (("default", {}), ("true", dict(find_unused_parameters=True)), ("false", dict(find_unused_parameters=False))):  # the one that raises last
        ddp, error = accelerator.wrap_model(Net(), **kwargs), None
        try:
            for _ in range(3):
                ddp(torch.randn(2, 4)).sum().backward()
        except RuntimeError as exc:
            error = str(exc).split(".")[0]
        found[name] = [flag(ddp), error]
with open(os.path.join(spec["dump"], "found_rank%s.json" % os.environ["RANK"]), "w") as f:
    json.dump(found, f)
dist.destroy_process_group()
'''
with open(W_DRIVER, "w") as f:
    f.write(W_DRIVER_SOURCE)


def w_two_ranks(tag, **spec):  # the driver on two real ranks, spec = what it is told; returns what each rank found and both ranks' output
    dump = os.path.join(TMP, "w_dump_" + tag)
    os.makedirs(dump)
    spec_path = os.path.join(TMP, "w_spec_%s.json" % tag)
    with open(spec_path, "w") as f:
        json.dump(dict(spec, dump=dump), f)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        env = dict(os.environ, WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(s.getsockname()[1]))
    procs = [subprocess.Popen([sys.executable, W_DRIVER, BASE, spec_path], env=dict(env, RANK=str(r), LOCAL_RANK=str(r)), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) for r in range(2)]
    try:
        outputs = [p.communicate(timeout=120)[0] for p in procs]
    finally:
        for p in procs:
            p.kill()  # a rank stuck in a collective must not outlive the test
            p.wait()
    assert [p.returncode for p in procs] == [0, 0], outputs
    return [json.load(open(os.path.join(dump, "found_rank%d.json" % r))) for r in range(2)], outputs


# (b) the no-Gram trunk's run (recipe on): 2 epochs of 3 steps on 2 ranks, to completion; the student's three DDP modules report find_unused_parameters False on both ranks; the losses are finite
W_OUT = os.path.join(TMP, "W_TWO")
w_found, _ = w_two_ranks("run", mode="run", args=vars(make_args(W_OUT, "--no_gram", *RECIPE, "--stop_after_epochs", "2")), vit=TINY_VIT, heads=TINY_HEADS, cfg=CFG)
assert w_found == [dict(student=dict(backbone=0, dino_head=0, ibot_head=0))] * 2, w_found
ck_w, w_rows = load_ckpt(W_OUT, "epoch_0002.pth"), read_csv(W_OUT)
assert (ck_w["step"], ck_w["epoch"], ck_w["world_size"]) == (6, 2, 2)  # past the second step, where an unused parameter would have raised
assert len(w_rows) > 3 and all(math.isfinite(float(x)) for row in w_rows[1:] for x in row[2:8])  # dino_global .. total: global means over the 2 ranks
print("OK (w) two real ranks, --no_gram + recipe: the student's backbone, dino_head and ibot_head DDP modules report find_unused_parameters False on both ranks, and the run goes through its 6 steps (epoch 0 with the "
      "backbone at LR 0, epoch 1) to a step-6 epoch-2 checkpoint with finite losses (%d CSV rows)" % (len(w_rows) - 1))

# (c) Accelerator.wrap_model: without the keyword it is True (Ark+ fine-tuning, which never passes it: test_freeze_backbone.py (d2)), as with True; the module with a parameter in no forward trains 3 steps with
# those and raises at the second forward with False, so "False" is only for a model like the student
w_wrapped, _ = w_two_ranks("wrap", mode="wrap")
for w_rank in w_wrapped:
    assert w_rank["default"] == w_rank["true"] == [1, None], w_rank
    assert w_rank["false"][0] == 0 and "Expected to have finished reduction in the prior iteration" in w_rank["false"][1], w_rank
print("OK (w) two real ranks, Accelerator.wrap_model: called without the keyword it reports find_unused_parameters True (as with True) and trains a module with a parameter in no forward for 3 steps; with False it "
      "reports False and that module raises '%s' at its second forward, on both ranks" % w_wrapped[0]["false"][1])

# (x) --cyclic_ark_teacher: the Ark+ / FX teacher (cycles of one-dataset batches, largest first; no per-step EMA, a blend after every dataset's release pass on clean views;
# per-dataset heads; with --lock_half_pass Foundation X's lock pass over a random half before each release pass; every checkpoint kept) -----------------------------------------
# The data: 12 / 8 / 6 / 4 images (the real ranking: MIMIC 368960, CheXpert 223414, ChestXray14 75312, VinDrCXR 15000), listed in make_args's --data_set order VinDrCXR, ChestXray14,
# MIMIC, CheXpert, which the cycle order must not follow. Batch 2 on one rank: release passes of 6 / 4 / 3 / 2 steps, lock passes (halves of 6 / 4 / 3 / 2 images) of 3 / 2 / 1 / 1, so a
# cycle is 22 steps with the lock and 15 without
import hashlib

from torchvision.transforms import v2

X_COUNTS = dict(MIMIC=12, CheXpert=8, ChestXray14=6, VinDrCXR=4)
X_CFG = make_synthetic_data(os.path.join(TMP, "data_cyclic"), X_COUNTS)
X_ORDER = ["MIMIC", "CheXpert", "ChestXray14", "VinDrCXR"]
X_PASSES = [("MIMIC", "lock", 3), ("MIMIC", "release", 6), ("CheXpert", "lock", 2), ("CheXpert", "release", 4), ("ChestXray14", "lock", 1), ("ChestXray14", "release", 3),
            ("VinDrCXR", "lock", 1), ("VinDrCXR", "release", 2)]
X_L = sum(n for _, _, n in X_PASSES)  # 22
X_PATTERN = [(name, phase) for name, phase, n in X_PASSES for _ in range(n)]  # (dataset, phase) of each step of a cycle
X_ENDS = [9, 15, 19, 22]  # steps done at each release pass's end within a cycle: the blends
X_MODE = ["--cyclic_ark_teacher", "--heads", "per_dataset", "--lock_half_pass"]  # the lock arm's mode flags (the trunk's line with CONSTLR_ARK_LOCK=1)
ARK = ["--no_gram", *X_MODE]
ARK_OFF = ["--no_gram", "--cyclic_ark_teacher", "--heads", "per_dataset"]  # the arm without lock passes (the mode's default)
X_ARK_KEYS = ("cyclic_ark_teacher", "ark_teacher_momentum", "teacher_clean_view", "heads", "lock_half_pass", "keep_all_checkpoints")
X_DEFAULTS = (False, None, False, "shared", False, False)


def xgo(out, *extra, cfg=X_CFG):  # go() on the cyclic data
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary = ssl_pretrain.run(make_args(out, *extra), TINY_VIT, TINY_HEADS, cfg)
    return summary, buf.getvalue()


def x_images(name):  # the dataset's image paths, in list order (lock_subset indexes into this)
    return ssl_data.image_paths(name, X_CFG[name]["data_dir"], X_CFG[name]["train_list"])


def x_dataset_of(path):  # the dataset whose data_dir holds this image
    found = [name for name in X_ORDER if path.startswith(X_CFG[name]["data_dir"] + os.sep)]
    assert len(found) == 1, path
    return found[0]


@contextlib.contextmanager
def x_spying():
    """Per step (rec.steps): the batch's dataset and phase tags, the image paths fetched for it, the optimizer groups' (is_backbone, lr, is_last_layer) as the step set them, the teacher's /
    student's / Gram teacher's state before the step, which student parameters have a grad after its backward, and the global crops / the teacher's input;
    per update_ema call (rec.emas): the steps done, m, the teacher before and after, the student it blends in; prune_checkpoints calls (rec.prunes); per set_epoch call
    (rec.set_epochs): its epoch index and the loader's image count (the set_epoch in place on entry runs, so (g)'s --workers > 0 stand-in still acts); per AdamW step
    (rec.opt): the backbone's and the heads' optimizer state after it (per parameter its exp_avg / exp_avg_sq / step, None before its first update)."""
    rec = SimpleNamespace(steps=[], emas=[], prunes=[], fetched=[], lrs=None, teacher_in=[], set_epochs=[], opt=[])
    real_getitem, real_update_ema, real_apply = ssl_data.SSLImageDataset.__getitem__, SSLMetaArch.update_ema, ssl_pretrain.apply_optim_scheduler
    real_teacher_output, real_prune, entry_set_epoch = SSLMetaArch.get_teacher_output, ssl_pretrain.prune_checkpoints, ssl_pretrain.set_epoch
    real_opt_step = torch.optim.AdamW.step
    state = lambda module: {k: v.detach().clone() for k, v in module.state_dict().items()}

    def getitem(self, i):
        rec.fetched.append(self.paths[i])
        return real_getitem(self, i)

    def apply(optimizer, *args, **kwargs):
        real_apply(optimizer, *args, **kwargs)
        rec.lrs = [(g["is_backbone"], g["lr"], g["is_last_layer"]) for g in optimizer.param_groups]

    def forward_backward(self, data, *, teacher_temp):
        step = dict(dataset=data.get("dataset"), phase=data.get("phase"), paths=list(rec.fetched), lrs=rec.lrs, teacher=state(self.teacher), student=v_flat(self.student),
                    gram=None if self.gram_teacher is None else state(self.gram_teacher), data=data)
        rec.fetched.clear()
        result = real_forward_backward(self, data, teacher_temp=teacher_temp)
        step["grads"] = {name: p.grad is not None for name, p in self.student.named_parameters()}
        step["teacher_in"] = rec.teacher_in.pop()
        rec.steps.append(step)
        return result

    def update_ema(self, m):
        record = dict(step=len(rec.steps), m=m, before=state(self.teacher), student=v_flat(self.student), params={n for n, _ in self.teacher.named_parameters()})
        result = real_update_ema(self, m)
        rec.emas.append(dict(record, after=state(self.teacher)))
        return result

    def get_teacher_output(self, images, **kwargs):
        rec.teacher_in.append(images.detach().clone())
        return real_teacher_output(self, images, **kwargs)

    def prune(ckpt_dir, *args, **kwargs):
        rec.prunes.append(ckpt_dir)
        return real_prune(ckpt_dir, *args, **kwargs)

    def set_epoch(loader, sampler, epoch, seed, rank):
        rec.set_epochs.append((epoch, len(loader.dataset)))
        return entry_set_epoch(loader, sampler, epoch, seed, rank)

    def opt_step(self, *args, **kwargs):
        result = real_opt_step(self, *args, **kwargs)
        snap = lambda backbone: [None if p not in self.state else {k: v.clone() if torch.is_tensor(v) else v for k, v in self.state[p].items()}
                                 for g in self.param_groups if g["is_backbone"] == backbone for p in g["params"]]
        rec.opt.append(dict(backbone=snap(True), heads=snap(False)))
        return result

    ssl_data.SSLImageDataset.__getitem__, SSLMetaArch.update_ema, ssl_pretrain.apply_optim_scheduler = getitem, update_ema, apply
    SSLMetaArch.forward_backward, SSLMetaArch.get_teacher_output, ssl_pretrain.prune_checkpoints, ssl_pretrain.set_epoch = forward_backward, get_teacher_output, prune, set_epoch
    torch.optim.AdamW.step = opt_step
    try:
        yield rec
    finally:
        ssl_data.SSLImageDataset.__getitem__, SSLMetaArch.update_ema, ssl_pretrain.apply_optim_scheduler = real_getitem, real_update_ema, real_apply
        SSLMetaArch.forward_backward, SSLMetaArch.get_teacher_output, ssl_pretrain.prune_checkpoints, ssl_pretrain.set_epoch = real_forward_backward, real_teacher_output, real_prune, entry_set_epoch
        torch.optim.AdamW.step = real_opt_step


# parse: the six settings' defaults are the mixed-batch run (and what a checkpoint without the keys reads as); the mode resolves its own; every one is --resume-locked and none branch-free
assert tuple(getattr(P(), k) for k in X_ARK_KEYS) == X_DEFAULTS == tuple(ssl_pretrain.ARG_DEFAULTS[k] for k in X_ARK_KEYS) and P().momentum_teacher == 0.999
assert set(X_ARK_KEYS) <= set(ssl_pretrain.RESUME_MATCH) and not set(X_ARK_KEYS) & set(ssl_pretrain.BRANCH_FREE)
x_parsed = P("--cyclic_ark_teacher")
assert tuple(getattr(x_parsed, k) for k in X_ARK_KEYS) == (True, 0.8, True, "shared", False, False) and x_parsed.momentum_teacher is None  # no per-step EMA, no lock passes
assert vars(P("--cyclic_ark_teacher", "--teacher_clean_view")) == vars(x_parsed) == vars(P("--cyclic_ark_teacher", "--no_lock_half_pass"))  # implied / the default, given as well: the same run
assert P("--cyclic_ark_teacher", "--lock_half_pass").lock_half_pass is True and P("--cyclic_ark_teacher", "--lock_half_pass", "--no_lock_half_pass").lock_half_pass is False  # the last one wins
assert (P("--teacher_clean_view").teacher_clean_view, P("--teacher_clean_view").cyclic_ark_teacher, P("--teacher_clean_view").momentum_teacher) == (True, False, 0.999)  # alone: the ablation arm
assert (P("--cyclic_ark_teacher", "--ark_teacher_momentum", "0", "--heads", "per_dataset", "--keep_all_checkpoints").ark_teacher_momentum, P("--keep_all_checkpoints").keep_all_checkpoints) == (0.0, True)
for extra, flags in ((["--heads", "per_dataset"], ("--heads per_dataset", "--cyclic_ark_teacher")), (["--ark_teacher_momentum", "0.8"], ("--ark_teacher_momentum", "--cyclic_ark_teacher")),
                     (["--lock_half_pass"], ("--lock_half_pass", "--cyclic_ark_teacher")), (["--no_lock_half_pass"], ("--no_lock_half_pass", "--cyclic_ark_teacher")),
                     (["--cyclic_ark_teacher", "--momentum_teacher", "0.999"], ("--momentum_teacher", "--cyclic_ark_teacher")),
                     (["--cyclic_ark_teacher", "--data_set", "MIMIC"], ("--cyclic_ark_teacher", "--data_set", "MIMIC")),
                     (["--cyclic_ark_teacher", "--ark_teacher_momentum", "1"], ("--ark_teacher_momentum",)), (["--cyclic_ark_teacher", "--ark_teacher_momentum", "-0.1"], ("--ark_teacher_momentum",)),
                     (["--cyclic_ark_teacher", "--ark_teacher_momentum", "nan"], ("--ark_teacher_momentum",)), (["--heads", "both"], ("--heads",)),
                     (["--cyclic_ark_teacher", "--switch_lock_steps", "100"], ("unrecognized arguments: --switch_lock_steps 100",))):  # the step-count lock is gone
    with contextlib.redirect_stderr(io.StringIO()) as err:
        assert raises(SystemExit, P, *extra) == "2"
    complaint = err.getvalue().split("error: ", 1)[1]
    assert all(flag in complaint for flag in flags), (extra, err.getvalue())
assert "switch_lock_steps" not in vars(x_parsed) and "switch_lock_steps" not in ssl_pretrain.RESUME_MATCH and "switch_lock_steps" not in ssl_pretrain.ARG_DEFAULTS
print("OK (x) parse: the six settings default to the mixed-batch run (= ARG_DEFAULTS; --momentum_teacher 0.999), are --resume-locked and not branch-free; --cyclic_ark_teacher resolves clean views on, "
      "momentum 0.8, no lock passes, no per-step momentum, and the same args with --teacher_clean_view or --no_lock_half_pass given too; --lock_half_pass turns the lock passes on (the last of the pair "
      "wins); --teacher_clean_view alone keeps the per-step EMA; --heads per_dataset / --ark_teacher_momentum / --lock_half_pass / --no_lock_half_pass without the mode, --momentum_teacher or a repeated "
      "--data_set with it, a momentum outside [0, 1) or NaN and an unknown head mode each exit 2 naming the flag; --switch_lock_steps no longer exists")

# one run of the lock arm, 2 cycles, under the spies: the cycle, the passes, the teacher rule, the clean views, the lock and the heads are all read off it
X_A = os.path.join(TMP, "X_A")
with x_spying() as x_rec:
    x_summary, x_text = xgo(X_A, *ARK, "--stop_after_epochs", "2")
x_lines = x_text.splitlines()
x_steps, x_emas = x_rec.steps, x_rec.emas

# (a) the cycle: by image count, largest first, logged; per dataset a lock pass over a random half (the same draw on every rank, a fresh one per cycle), then the release pass over all of it; every
# batch one dataset (the tags, the CSV column and the images fetched for it agree); epoch_length = the sum of the passes
assert ("cycle (--cyclic_ark_teacher): each dataset in turn, largest first by image count, every batch from one dataset, a lock pass over a random half then a release pass over all of it: "
        "MIMIC 12 images: lock 3 + release 6 steps, CheXpert 8 images: lock 2 + release 4 steps, ChestXray14 6 images: lock 1 + release 3 steps, VinDrCXR 4 images: lock 1 + release 2 steps; "
        "22 steps a cycle (= one epoch)") in x_lines, x_text
assert x_summary["epoch_length"] == X_L == 22 and json.load(open(os.path.join(X_A, "args.json")))["epoch_length"] == 22 and "epoch_length 22 steps" in x_text
x_info = json.load(open(os.path.join(X_A, "args.json")))  # args.json records the six settings as resolved (and the per-step momentum as None); a mixed-batch run's, the defaults
assert tuple(x_info[k] for k in X_ARK_KEYS) == (True, 0.8, True, "per_dataset", True, False) and x_info["momentum_teacher"] is None
assert tuple(json.load(open(os.path.join(DIR_A, "args.json")))[k] for k in X_ARK_KEYS) == X_DEFAULTS
assert [(s["dataset"], s["phase"]) for s in x_steps] == X_PATTERN * 2, [(s["dataset"], s["phase"]) for s in x_steps]
for s in x_steps:
    assert len(s["paths"]) == 2 and {x_dataset_of(path) for path in s["paths"]} == {s["dataset"]}, s["paths"]  # the 2 images of the batch are both its dataset's
x_lock_seen = {}
for cycle in (0, 1):
    in_cycle = x_steps[X_L * cycle:X_L * (cycle + 1)]
    for place, name in enumerate(X_ORDER):
        seen = lambda phase: [path for s in in_cycle if (s["dataset"], s["phase"]) == (name, phase) for path in s["paths"]]
        assert sorted(seen("release")) == sorted(x_images(name)), (cycle, name)  # the release pass: every image once
        half = {x_images(name)[i] for i in ssl_pretrain.lock_subset(X_COUNTS[name], 100, cycle, place)}
        lock = seen("lock")
        assert len(lock) == len(set(lock)) == X_COUNTS[name] // 2 // 2 * 2 and set(lock) <= half and len(half) == X_COUNTS[name] // 2, (cycle, name, lock)  # half of it (drop_last: 2 of ChestXray14's 3)
        x_lock_seen[(cycle, name)] = set(lock)
assert all(x_lock_seen[(0, name)] != x_lock_seen[(1, name)] for name in ("MIMIC", "ChestXray14"))  # a fresh half each cycle (CheXpert's and VinDrCXR's draws happen to repeat)
assert ssl_pretrain.lock_subset(12, 100, 0, 0) == ssl_pretrain.lock_subset(12, 100, 0, 0) != ssl_pretrain.lock_subset(12, 100, 1, 0) and ssl_pretrain.lock_subset(12, 101, 0, 0) != ssl_pretrain.lock_subset(12, 100, 0, 0)
assert [s["paths"] for s in x_steps[:X_L]] != [s["paths"] for s in x_steps[X_L:]]
assert x_rec.set_epochs == [(8 * c + k, X_COUNTS[name] // (2 if phase == "lock" else 1)) for c in (0, 1) for k, (name, phase, _) in enumerate(X_PASSES)]  # each pass its own index
x_rows = read_csv(X_A)
x_col = {c: i for i, c in enumerate(x_rows[0])}
assert ",".join(x_rows[0]) == CSV_HEADER and all(len(r) == 16 for r in x_rows)
assert all(r[x_col["dataset"]] == "%s:%s" % (x_steps[int(r[0])]["dataset"], x_steps[int(r[0])]["phase"]) for r in x_rows[1:])  # MIMIC:lock, MIMIC:release, ...
x_boundary_rows = {X_L * c + e - 1 for c in (0, 1) for e in X_ENDS}  # 0-based step of every release pass's last step
assert x_boundary_rows <= {int(r[0]) for r in x_rows[1:]}  # each is a log row
assert all(r[x_col["momentum"]] == ("0.8" if int(r[0]) in x_boundary_rows else "1.0") for r in x_rows[1:])  # the momentum the step applied
assert all(r[x_col["dataset"]] == "-" and r[x_col["momentum"]] == "0.999" for r in rows_a[1:])  # the mixed-batch run (g): "-", the per-step momentum
# at the real list sizes and global batch 256 (64 x 4 ranks, the world faked as (q) does it; the lists are read, no image is: the run stops at its first fetch): VinDrCXR's lock pass is 29 steps and its
# release 58, MIMIC's 720 and 1441; epoch_length 3997 with the lock, 2665 without
from utils import get_config
for x_extra, x_cycle_steps in ((["--lock_half_pass"], 3997), ([], 2665)):
    x_real = os.path.join(TMP, "X_REAL%d" % len(x_extra))
    buf = io.StringIO()
    with as_rank(0, 4), contextlib.redirect_stdout(buf):
        raises(FileNotFoundError, ssl_pretrain.run, make_args(x_real, *ARK_OFF, *x_extra, "--batch_size_per_gpu", "64"), TINY_VIT, TINY_HEADS, get_config("datasets_config.yaml"),
               check_exists=False)
    x_cycle = [line for line in buf.getvalue().splitlines() if line.startswith("cycle (")]
    assert json.load(open(os.path.join(x_real, "args.json")))["epoch_length"] == x_cycle_steps and len(x_cycle) == 1, buf.getvalue()
    if x_extra:
        assert ("MIMIC 368960 images: lock 720 + release 1441 steps, CheXpert 223414 images: lock 436 + release 872 steps, ChestXray14 75312 images: lock 147 + release 294 steps, "
                "VinDrCXR 15000 images: lock 29 + release 58 steps; 3997 steps a cycle") in x_cycle[0], x_cycle
    else:
        assert "MIMIC 368960 images: release 1441 steps, CheXpert 223414 images: release 872 steps, ChestXray14 75312 images: release 294 steps, VinDrCXR 15000 images: release 58 steps; 2665 steps a cycle" in x_cycle[0]
print("OK (x) cycle: the logged order is MIMIC, CheXpert, ChestXray14, VinDrCXR by image count (12 / 8 / 6 / 4, not the --data_set order); per dataset a lock pass then a release pass (3 + 6, 2 + 4, 1 + 3, "
      "1 + 2 steps), epoch_length 22 = their sum (summary, args.json, which records the six settings as resolved); each of the 44 steps' batch is tagged with one dataset and phase and both of its images "
      "come from that dataset (an image-fetch hook); a release pass covers every image of its dataset once, a lock pass half of it (MIMIC 6 of 12, CheXpert 4 of 8, ChestXray14 2 of its half of 3, VinDrCXR "
      "2 of 4), all within lock_subset's draw, and MIMIC's and ChestXray14's halves differ between the cycles; set_epoch runs as each pass starts with its own index (0..15); the CSV's dataset column is "
      "<name>:<phase>, every release pass's last step is a row, the momentum column 0.8 there and 1.0 elsewhere (the mixed-batch run: '-' and 0.999); at the real list sizes and 64 x 4: VinDrCXR lock 29 "
      "+ release 58 steps, MIMIC 720 + 1441, 3997 a cycle (2665 without the lock)")

# (b) the teacher: bitwise constant through every pass, 0.8 x old + 0.2 x student after each release pass (the frozen first cycle's included; never after a lock pass), backbone and every head;
# no per-step EMA in the mode, one per step without it
assert [(e["step"], e["m"]) for e in x_emas] == [(X_L * c + e, 0.8) for c in (0, 1) for e in X_ENDS], [(e["step"], e["m"]) for e in x_emas]
assert all(x_steps[e["step"] - 1]["phase"] == "release" for e in x_emas)  # each blend follows a release pass's last step, none a lock pass's
for e in x_emas:
    for k, before in e["before"].items():
        want = before.mul(0.8).add_(e["student"][k], alpha=1 - 0.8) if k in e["params"] else before  # update_ema's foreach mul + add (buffers are not blended)
        assert torch.equal(e["after"][k], want), (e["step"], k)
    assert not same_state(e["after"], e["before"]), e["step"]
x_blend = {e["step"]: e for e in x_emas}
for i in range(1, 2 * X_L):  # the teacher each step sees: the one before it unless a blend came in between, then that blend's result
    assert same_state(x_steps[i]["teacher"], x_blend[i]["after"] if i in x_blend else x_steps[i - 1]["teacher"]), i
assert same_state(load_ckpt(X_A, "epoch_0002.pth")["teacher"], x_blend[2 * X_L]["after"]) and same_state(load_ckpt(X_A, "epoch_0001.pth")["teacher"], x_blend[X_L]["after"])
vin = "dino_head.VinDrCXR.mlp.0.weight"  # VinDrCXR's head trained in steps 19-21 only; the MIMIC blend of cycle 1 (after step 31) still moves its teacher copy: every head is blended
assert not torch.equal(x_blend[X_L + 9]["after"][vin], x_blend[X_L + 9]["before"][vin]) and torch.equal(x_steps[X_L + 8]["student"][vin], x_steps[X_L]["student"][vin])
with x_spying() as x_rec_mixed:
    go(os.path.join(TMP, "X_MIXED"), "--stop_after_epochs", "1")
assert [(e["step"], e["m"]) for e in x_rec_mixed.emas] == [(i, 0.999) for i in range(1, 7)]  # without the mode: after every step, --momentum_teacher
print("OK (x) teacher: update_ema runs only after steps 9, 15, 19, 22, 31, 37, 41, 44 (every release pass's last step, the frozen cycle's included; never a lock pass's) with m 0.8, and each time every teacher "
      "parameter is bitwise old x 0.8 + student x 0.2 (backbone and all 8 heads: VinDrCXR's head moves at the MIMIC blend too); between blends the teacher each step sees is bitwise the one before; "
      "the epoch checkpoints hold the cycle-end blend; a mixed-batch run calls it after each of its 6 steps with 0.999")

# (c) clean views: the transform's teacher crops are the student's crop + flip resized to the global size, the very base the Gram crops are cut from, with no distortion; nothing else changes
x_img = Image.open(ssl_data.image_paths("MIMIC", X_CFG["MIMIC"]["data_dir"], X_CFG["MIMIC"]["train_list"])[0]).convert("L")
aug_plain, aug_clean, x_bases = DataAugmentationCXR(), DataAugmentationCXR(teacher_no_distortions=True), []
x_geometry = aug_clean.geometric_augmentation_global
aug_clean.geometric_augmentation_global = lambda image: (lambda base: (x_bases.append(base), base)[1])(x_geometry(image))
out_plain, rng_plain = ng_seeded(aug_plain, x_img), torch.get_rng_state()
out_clean, rng_clean = ng_seeded(aug_clean, x_img), torch.get_rng_state()
assert torch.equal(rng_plain, rng_clean) and len(x_bases) == 2  # the clean crops draw nothing
for key in ("global_crops", "gram_teacher_crops", "local_crops"):
    assert all(torch.equal(a, b) for a, b in zip(out_plain[key], out_clean[key])), key  # the student's and the Gram teacher's crops: bitwise as without the flag
assert all(a is b for a, b in zip(out_plain["global_crops_teacher"], out_plain["global_crops"]))  # without it the teacher's "own" crops are the student's (Meta's default)
x_resize = v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC, antialias=True)
for i in range(2):
    clean, gram, student_crop = out_clean["global_crops_teacher"][i], out_clean["gram_teacher_crops"][i], out_clean["global_crops"][i]
    assert torch.equal(clean, aug_clean.normalize(aug_clean.resize_global(x_bases[i])))  # the crop + flip, resized to 224, normalized: nothing else
    assert torch.equal(gram, aug_clean.normalize(aug_clean.resize_gram_teacher(x_bases[i])))  # the Gram pipeline's undistorted output, from the same base
    assert clean.shape == student_crop.shape == (3, 224, 224) and gram.shape == (3, 448, 448)
    assert (clean - x_resize(gram)).abs().mean() < 0.02, i  # the Gram crop resized to 224 is the clean crop up to resampling
assert any(not torch.equal(c, s) for c, s in zip(out_clean["global_crops_teacher"], out_clean["global_crops"]))  # the student's are distorted
# ... and through run(): the EMA teacher's input is the batch's collated clean crops, never the student's distorted ones (the trunk above; --teacher_clean_view alone on mixed batches); a default run's
# teacher scores the student's crops. The clean-view run's student batches are bitwise the default run's (same seeds), its losses are not
assert all(torch.equal(s["teacher_in"].flatten(0, 1), s["data"]["collated_global_crops_teacher"]) and not torch.equal(s["teacher_in"].flatten(0, 1), s["data"]["collated_global_crops"])
           for s in x_steps)
with x_spying() as x_rec_cv:
    go(os.path.join(TMP, "X_CV"), "--teacher_clean_view", "--stop_after_epochs", "1")
assert all(torch.equal(s["teacher_in"].flatten(0, 1), s["data"]["collated_global_crops_teacher"]) and not torch.equal(s["teacher_in"].flatten(0, 1), s["data"]["collated_global_crops"])
           for s in x_rec_cv.steps) and len(x_rec_cv.emas) == 6
for s_cv, s_mixed in zip(x_rec_cv.steps, x_rec_mixed.steps):
    assert torch.equal(s_mixed["teacher_in"].flatten(0, 1), s_mixed["data"]["collated_global_crops"]) and "collated_global_crops_teacher" not in s_mixed["data"]
    for key in ("collated_global_crops", "collated_local_crops", "collated_masks", "collated_gram_teacher_crops"):
        assert torch.equal(s_cv["data"][key], s_mixed["data"][key]), key
assert read_csv(os.path.join(TMP, "X_CV"))[1][2] != read_csv(os.path.join(TMP, "X_MIXED"))[1][2]  # dino_global of step 0: the teacher saw other views
assert "teacher views: clean (--teacher_clean_view, implied by --cyclic_ark_teacher): the EMA teacher scores the global crops' crop + flip only, resized to 224" in x_text
print("OK (x) clean views: the clean teacher crops are bitwise normalize(resize to 224 of the recorded crop + flip base), the Gram crops bitwise normalize(resize to 448 of the same base), and the Gram crop "
      "resized to 224 matches the clean crop up to resampling (mean |diff| < 0.02); the flag draws no RNG and leaves the student's, the Gram teacher's and the local crops bitwise as they were; in run() the "
      "EMA teacher scores the collated clean crops at every step, not the distorted student crops (cyclic trunk and --teacher_clean_view alone), a default run's teacher the student's own; the clean-view "
      "run's student batches are bitwise the default run's and its losses differ")

# (d) the lock passes: backbone LR 0 exactly through the lock passes, nonzero through the release passes; the heads train throughout (outside the frozen cycle 0, where both passes run with the backbone
# at LR 0); one line per pass; and the arm without lock passes
x_lr = lambda s, backbone: {lr for is_backbone, lr, _ in s["lrs"] if is_backbone == backbone}
x_last_lr = lambda s: {lr for _, lr, is_last in s["lrs"] if is_last}  # every head's last layer (the optimizer's last-layer groups hold all 8)
for i, s in enumerate(x_steps):
    cycle = i // X_L
    assert (x_last_lr(s) == {0.0}) == (cycle == 0), i  # the last-layer freeze (FREEZE_LAST_LAYER_EPOCHS 1): cycle 0, every head
    if cycle == 0:
        assert x_lr(s, True) == {0.0}, i  # the frozen cycle
    else:
        assert (x_lr(s, True) == {0.0}) == (s["phase"] == "lock") and 0.0 not in x_lr(s, False), (i, s["phase"], s["lrs"])
        moved = not same_state(backbone_of(x_steps[i + 1]["student"] if i < 2 * X_L - 1 else load_ckpt(X_A, "epoch_0002.pth")["student"]), backbone_of(s["student"]))
        assert moved == (s["phase"] == "release"), i  # LR 0 leaves the backbone bitwise (AdamW, decoupled decay); a release step moves it
assert all(torch.equal(x_steps[X_L]["student"][k], v) for k, v in x_steps[0]["student"].items() if k.endswith("last_layer.weight"))  # 8 last layers untouched in cycle 0
pass_lines = [line for line in x_lines if line.startswith("dataset pass after step ")]
assert len(pass_lines) == 16 and pass_lines[:2] == [
    "dataset pass after step 0: MIMIC lock (cycle 0, dataset 1/4, a random half, 6 of 12 images, 3 steps): backbone frozen (LR 0, no optimizer step), heads train",
    "dataset pass after step 3: MIMIC release (cycle 0, dataset 1/4, all 12 images, 6 steps): backbone and heads train; the backbone at LR 0 in this frozen cycle"], pass_lines
assert pass_lines[-2:] == ["dataset pass after step 41: VinDrCXR lock (cycle 1, dataset 4/4, a random half, 2 of 4 images, 1 steps): backbone frozen (LR 0, no optimizer step), heads train",
                           "dataset pass after step 42: VinDrCXR release (cycle 1, dataset 4/4, all 4 images, 2 steps): backbone and heads train"], pass_lines
assert not any(line.startswith("NOTE: ") or "switch_lock" in line for line in x_lines)
# the lock is heads-only all the way through the optimizer (Codex r2 P2): across every lock pass, frozen cycle included, the backbone's AdamW state (exp_avg, exp_avg_sq, step) is bitwise what it was
# before the pass, and so are its weights (above), while the heads' state moves; over each release pass both move, except that in the frozen cycle 0 the release passes are the old LR-0 freeze: weights still,
# moments advancing (as on db2e33a, and in the lock-off arm below)
assert len(x_rec.opt) == 2 * X_L
x_pass_start = 0
for cycle in (0, 1):
    for name, phase, n in X_PASSES:
        first, last = X_L * cycle + x_pass_start, X_L * cycle + x_pass_start + n - 1
        before = x_rec.opt[first - 1] if first else dict(backbone=[None] * len(x_rec.opt[0]["backbone"]), heads=[None] * len(x_rec.opt[0]["heads"]))
        if phase == "lock":
            assert all(same_state(x_rec.opt[i]["backbone"], before["backbone"]) for i in range(first, last + 1)), (cycle, name)  # no moment, no step count moves
            assert not same_state(x_rec.opt[last]["heads"], before["heads"]), (cycle, name)  # the heads train
        else:
            assert not same_state(x_rec.opt[last]["backbone"], before["backbone"]) and all(m is not None for m in x_rec.opt[last]["backbone"]), (cycle, name)  # moments and steps move
            weights_after = backbone_of(x_steps[last + 1]["student"] if last + 1 < 2 * X_L else load_ckpt(X_A, "epoch_0002.pth")["student"])
            assert same_state(weights_after, backbone_of(x_steps[first]["student"])) == (cycle == 0), (cycle, name)  # LR 0 in the frozen cycle only
        x_pass_start = (x_pass_start + n) % X_L
assert x_rec.opt[2]["backbone"] == [None] * len(x_rec.opt[0]["backbone"])  # after MIMIC's first lock pass the backbone was never updated at all
x_lock_steps = sum(n for _, phase, n in X_PASSES if phase == "lock")
assert all(int(m["step"]) == 2 * X_L - 2 * x_lock_steps == 30 for m in x_rec.opt[-1]["backbone"])  # the backbone's AdamW steps: the release steps alone (cycle 0's frozen ones included)
assert "teacher blend after step 9 (end of MIMIC's release pass, cycle 0, dataset 1/4): teacher <- 0.8 x teacher + 0.2 x student, backbone and all heads" in x_lines
with x_spying() as x_rec_open:
    open_summary, open_text = xgo(os.path.join(TMP, "X_OPEN"), *ARK_OFF, "--stop_after_epochs", "2")
assert open_summary["epoch_length"] == 15 and [(s["dataset"], s["phase"]) for s in x_rec_open.steps] == [(name, "release") for name, phase, n in X_PASSES if phase == "release" for _ in range(n)] * 2
assert all(x_lr(s, True) == {0.0} for s in x_rec_open.steps[:15]) and all(0.0 not in x_lr(s, True) for s in x_rec_open.steps[15:])  # no lock: only the frozen cycle
assert all(not same_state(x_rec_open.opt[i]["backbone"], x_rec_open.opt[i - 1]["backbone"]) for i in range(1, 30))  # the old LR-0 freeze in cycle 0: the moments advance at every step
assert same_state(backbone_of(x_rec_open.steps[14]["student"]), backbone_of(x_rec_open.steps[0]["student"])) and all(int(m["step"]) == 30 for m in x_rec_open.opt[-1]["backbone"])
assert [e["step"] for e in x_rec_open.emas] == [15 * c + e for c in (0, 1) for e in (6, 10, 13, 15)] and len([l for l in open_text.splitlines() if l.startswith("dataset pass ")]) == 8
assert all(r[15].endswith(":release") for r in read_csv(os.path.join(TMP, "X_OPEN"))[1:]) and "its release pass only (--no_lock_half_pass)" in open_text
print("OK (x) lock: in cycle 1 the backbone groups are at LR 0 at every step of a lock pass and nonzero at every step of a release pass, the head groups nonzero at every step (and every head's last layer "
      "at LR 0 in cycle 0 only, untouched there); a lock step leaves the backbone bitwise, a release step moves it; in the frozen cycle 0 both passes run with the backbone at 0; 16 pass lines name the "
      "dataset, lock / release, the images and steps; no NOTE line; across every lock pass (cycle 0 included) the backbone's AdamW exp_avg / exp_avg_sq / step are bitwise unchanged while the heads' move, and over every release pass they move (the backbone's step count is the 30 release steps of the 44), with the weights still only in the frozen cycle 0; without --lock_half_pass a cycle is the 4 release passes (15 steps, blends after 6, 10, 13, 15), its backbone at LR 0 in cycle 0 only")

# (e) per-dataset heads: 4 + 4 (student, teacher), only the batch's pair gets gradients, every head in the optimizer with the shared heads' multipliers; the export is the stock key set
x_head_names = lambda sd, kind: {k.split(".")[1] for k in sd if k.startswith(kind + ".")}
for sd in (x_steps[0]["student"], x_steps[0]["teacher"], load_ckpt(X_A, "epoch_0002.pth")["student"], load_ckpt(X_A, "epoch_0002.pth")["teacher"]):
    assert x_head_names(sd, "dino_head") == x_head_names(sd, "ibot_head") == set(X_ORDER), x_head_names(sd, "dino_head")
for s in x_steps:
    for name, has in s["grads"].items():
        kind, owner = name.split(".")[:2]
        assert has == (kind == "backbone" or owner == s["dataset"]), (s["dataset"], name, has)
x_export = torch.load(os.path.join(X_A, "export", "teacher_backbone_epoch_0002.pth"), weights_only=True)
assert set(x_export) == set(tiny_stock) and all(torch.equal(v, load_ckpt(X_A, "epoch_0002.pth")["teacher"]["backbone." + k]) for k, v in x_export.items())
x_models = ng_seeded(build_models, TINY_VIT, None, TINY_HEADS, heads=X_ORDER)
shared_models = ng_seeded(build_models, TINY_VIT, None, TINY_HEADS)
assert isinstance(x_models[0].dino_head, torch.nn.ModuleDict) and list(x_models[0].dino_head) == X_ORDER and list(x_models[1].ibot_head) == X_ORDER
x_heads_built = [h for kind in ("dino_head", "ibot_head") for h in x_models[0][kind].values()]
assert all(not m.bias.any() for h in x_heads_built for m in h.mlp if isinstance(m, torch.nn.Linear))  # every head through init_weights (zero biases), as the shared ones
assert all(not torch.equal(a.last_layer.weight, b.last_layer.weight) for i, a in enumerate(x_heads_built) for b in x_heads_built[i + 1:])  # each its own random draw


def x_group_of(student):  # parameter name -> (lr_multiplier, wd_multiplier, is_last_layer, is_backbone) of its optimizer group; every parameter in exactly one group
    groups = ssl_pretrain.get_params_groups(student)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for _ in student.parameters())
    owner = {id(p): (g["lr_multiplier"], g["wd_multiplier"], g["is_last_layer"], g["is_backbone"]) for g in groups for p in g["params"]}
    return {name: owner[id(p)] for name, p in student.named_parameters()}, len(groups)


x_groups, x_n_groups = x_group_of(x_models[0])
shared_groups, shared_n_groups = x_group_of(shared_models[0])
assert x_n_groups == shared_n_groups and all(x_groups[n] == shared_groups[n.replace("." + name + ".", ".", 1) if n.split(".")[0] != "backbone" else n]
                                             for n in x_groups for name in [n.split(".")[1]])
if WEIGHTS is None:
    print("SKIP (x) ViT-B/16 per-dataset heads export: %s not found" % WEIGHT_NAME)
else:
    stock_x = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    teacher_x = build_models(DINOV3_VITB16_KWARGS, WEIGHTS, TINY_HEADS, gram=False, heads=X_ORDER)[1]
    exported_x = export_backbone(teacher_x.backbone)
    assert len(exported_x) == 188 == len(stock_x) and set(exported_x) == set(stock_x) and list(teacher_x.dino_head) == X_ORDER
    DinoVisionTransformer(**DINOV3_VITB16_KWARGS).load_state_dict(exported_x, strict=True)
    del stock_x, teacher_x, exported_x
print("OK (x) heads: the student and the teacher (built, and in the checkpoint) hold one DINO and one iBOT head per dataset; after each of the 44 backwards exactly the backbone and the batch's dataset's "
      "two heads have gradients; every head is its own random draw through init_weights; the optimizer puts every parameter in one group, %d groups as with shared heads, each head parameter with the shared head's multipliers (lr, wd, last layer); the "
      "export is the stock key set (tiny ViT; ViT-B/16: 188 keys, strict into DinoVisionTransformer)" % x_n_groups)

# (f) --keep_all_checkpoints: nothing pruned over 5 cycles; without it the old policy (the last 3 and every 10th)
with x_spying() as x_rec_keep:
    xgo(os.path.join(TMP, "X_KEEP"), *ARK_OFF, "--keep_all_checkpoints", "--epochs", "5")
with x_spying() as x_rec_prune:
    xgo(os.path.join(TMP, "X_PRUNE"), *ARK_OFF, "--epochs", "5")
assert [e for e, _ in ssl_pretrain.epoch_checkpoints(os.path.join(TMP, "X_KEEP", "ckpt"))] == [1, 2, 3, 4, 5] and x_rec_keep.prunes == []
assert [e for e, _ in ssl_pretrain.epoch_checkpoints(os.path.join(TMP, "X_PRUNE", "ckpt"))] == [3, 4, 5] and len(x_rec_prune.prunes) == 5
assert json.load(open(os.path.join(TMP, "X_KEEP", "args.json")))["keep_all_checkpoints"] is True
print("OK (x) keep-all: 5 cycles with --keep_all_checkpoints leave epoch_0001..0005 and never call prune_checkpoints; without it, prune_checkpoints after each epoch leaves 3, 4, 5")

# (g) a resume at a cycle boundary is bitwise the uninterrupted run, the cyclic samplers included (the same batches), with set_epoch as under --workers > 0 (the main process' RNGs carry
# over, so cycle 2's crops, drop path and RoPE draws start from the checkpoint's states); control: the same resume with the saved torch RNG state replaced differs
X_RA, X_RB, X_RC = (os.path.join(TMP, n) for n in ("X_RA", "X_RB", "X_RC"))
X_TRUNK = [*ARK, *RECIPE]
ssl_pretrain.set_epoch = set_epoch_main_rngs_untouched
try:
    xgo(X_RA, *X_TRUNK, "--stop_after_epochs", "1")
    shutil.copytree(X_RA, X_RC)
    ck_rc = load_ckpt(X_RC, "epoch_0001.pth")
    ck_rc["rng"]["torch"] = torch.Generator().manual_seed(12345).get_state()
    torch.save(ck_rc, os.path.join(X_RC, "ckpt", "epoch_0001.pth"))
    with x_spying() as x_rec_ra:
        _, text_ra = xgo(X_RA, *X_TRUNK, "--resume", "--stop_after_epochs", "2")
    with x_spying() as x_rec_rb:
        xgo(X_RB, *X_TRUNK, "--stop_after_epochs", "2")
    xgo(X_RC, *X_TRUNK, "--resume", "--stop_after_epochs", "2")
finally:
    ssl_pretrain.set_epoch = real_set_epoch
assert "resumed from" in text_ra and "dataset pass after step 22: MIMIC lock (cycle 1, dataset 1/4, a random half, 6 of 12 images, 3 steps)" in text_ra
assert [(s["dataset"], s["phase"], s["paths"]) for s in x_rec_ra.steps] == [(s["dataset"], s["phase"], s["paths"]) for s in x_rec_rb.steps[X_L:]]  # the same batches (lock halves included)
assert all(torch.equal(a["data"]["collated_global_crops"], b["data"]["collated_global_crops"]) for a, b in zip(x_rec_ra.steps, x_rec_rb.steps[X_L:]))
ck_ra, ck_rb, ck_rc2 = (load_ckpt(d, "epoch_0002.pth") for d in (X_RA, X_RB, X_RC))
assert not same_state(ck_rb["student"], load_ckpt(X_RB, "epoch_0001.pth")["student"])  # cycle 2 trained
for part in ("student", "teacher", "optimizer", "rng"):
    assert same_state(ck_ra[part], ck_rb[part]), part
assert same_state(*(torch.load(os.path.join(d, "export", "teacher_backbone_epoch_0002.pth"), weights_only=True) for d in (X_RA, X_RB)))
by_step_rb = {r[0]: r[2:13] + r[15:] for r in read_csv(X_RB)[1:]}
common_r = [r for r in read_csv(X_RA)[1:] if r[0] in by_step_rb]
assert len(common_r) == len(by_step_rb) and all(r[2:13] + r[15:] == by_step_rb[r[0]] for r in common_r)  # A's rows (its cycle 1 kept, cycle 2 resumed) are B's
assert not same_state(ck_rc2["student"], ck_rb["student"])  # the control
print("OK (x) resume: a lock-arm trunk (recipe on) stopped after cycle 1 and resumed equals 2 cycles straight with set_epoch as under --workers > 0: cycle 2's batches (dataset, phase, images, so the lock half too, collated crops) step for "
      "step, the epoch-2 student, teacher, optimizer, RNG and export bitwise, every CSV row's losses / lr / momentum / dataset; the same resume with a replaced torch RNG state differs")

# (h) branches of a cyclic trunk: --branch_from its cycle-1 checkpoint restores everything (8 heads, the teacher blended at that boundary, which the Gram teacher anchors to), refreshes at the next two
# boundaries copy the teacher as blended there; a --gram_weight 0 branch is bitwise the trunk resumed; a branch or resume that changes a mode setting is refused
X_T, X_BG, X_B0 = (os.path.join(TMP, n) for n in ("X_T", "X_BG", "X_B0"))
with x_spying() as x_rec_t:
    xgo(X_T, *X_TRUNK, "--keep_all_checkpoints", "--stop_after_epochs", "1")
X_SRC = os.path.join(X_T, "ckpt", "epoch_0001.pth")
x_src = torch.load(X_SRC, map_location="cpu", weights_only=False)
assert same_state(x_src["teacher"], x_rec_t.emas[-1]["after"]) and x_rec_t.emas[-1]["step"] == 22  # the checkpoint's teacher is the one blended after step 22
X_BRANCH = ["--branch_from", X_SRC, *RECIPE, *X_MODE, "--keep_all_checkpoints"]  # the trunk's line with --no_gram replaced
with x_spying() as x_rec_bg:
    summary_bg, text_bg = xgo(X_BG, *X_BRANCH, "--gram_refresh_steps", "44,66", "--stop_after_epochs", "3")  # K x epoch_length + i x epoch_length, i = 1, 2
assert (summary_bg["step"], summary_bg["epoch"], summary_bg["gram_refreshes"]) == (66, 3, [44, 66]) and "branched from %s: its step 22, epoch 1;" % X_SRC in text_bg, text_bg
first_bg = x_rec_bg.steps[0]
assert same_state(first_bg["student"], x_src["student"]) and same_state(first_bg["teacher"], x_src["teacher"])  # 8 heads included
assert same_state(first_bg["gram"], {k: v for k, v in x_src["teacher"].items() if k.startswith("backbone.")})  # anchored to the blended teacher
assert [e["step"] for e in x_rec_bg.emas] == [X_L * c + e for c in (0, 1) for e in X_ENDS]  # the branch's own steps: after global steps 31, 37, 41, 44, 53, 59, 63, 66
x_backbone = lambda sd: {k: v for k, v in sd.items() if k.startswith("backbone.")}  # the Gram teacher's keys
assert same_state(x_rec_bg.steps[X_L]["gram"], x_backbone(x_rec_bg.emas[3]["after"])) and not same_state(x_backbone(x_rec_bg.emas[3]["after"]), x_backbone(x_rec_bg.emas[3]["before"]))  # refresh after 44 = blend after 44
assert same_state(load_ckpt(X_BG, "epoch_0003.pth")["gram_teacher"], x_backbone(x_rec_bg.emas[-1]["after"]))  # and after step 66
assert "Gram refresh 1/2 after step 44:" in text_bg and [e for e, _ in ssl_pretrain.epoch_checkpoints(os.path.join(X_BG, "ckpt"))] == [2, 3]
xgo(X_B0, *X_BRANCH, "--gram_weight", "0", "--gram_refresh_steps", "100", "--stop_after_epochs", "2")
xgo(X_T, *X_TRUNK, "--keep_all_checkpoints", "--resume", "--stop_after_epochs", "2")
ck_t2, ck_b0 = load_ckpt(X_T, "epoch_0002.pth"), load_ckpt(X_B0, "epoch_0002.pth")
assert not same_state(ck_t2["student"], x_src["student"])
for part in ("student", "teacher", "optimizer", "rng"):
    assert same_state(ck_b0[part], ck_t2[part]), part
assert same_state(*(torch.load(os.path.join(d, "export", "teacher_backbone_epoch_0002.pth"), weights_only=True) for d in (X_T, X_B0)))
by_step_t2 = {r[0]: r for r in read_csv(X_T)[1:]}
common_b0 = [r for r in read_csv(X_B0)[1:] if r[0] in by_step_t2]
assert len(common_b0) >= 5 and all(r[:6] + r[7:12] + r[15:] == by_step_t2[r[0]][:6] + by_step_t2[r[0]][7:12] + by_step_t2[r[0]][15:] for r in common_b0)  # all but gram, gram_weight, timings
X_DROPPED = ["--branch_from", X_SRC, *RECIPE, "--keep_all_checkpoints"]  # the mode dropped (and with it --heads per_dataset, which needs it)
for name, extra, diffs in (("xbr_heads", [*X_BRANCH, "--heads", "shared"], ["heads: checkpoint 'per_dataset', now 'shared'"]),
                           ("xbr_lock", [*X_BRANCH, "--no_lock_half_pass"], ["lock_half_pass: checkpoint True, now False"]),
                           ("xbr_momentum", [*X_BRANCH, "--ark_teacher_momentum", "0.7"], ["ark_teacher_momentum: checkpoint 0.8, now 0.7"]),
                           ("xbr_dropped", X_DROPPED, ["cyclic_ark_teacher: checkpoint True, now False", "teacher_clean_view: checkpoint True, now False", "heads: checkpoint 'per_dataset', now 'shared'",
                                                     "lock_half_pass: checkpoint True, now False", "ark_teacher_momentum: checkpoint 0.8, now None"]),
                           ("xbr_keep", [a for a in X_BRANCH if a != "--keep_all_checkpoints"], ["keep_all_checkpoints: checkpoint True, now False"])):
    out = os.path.join(TMP, name)
    message = raises(ValueError, xgo, out, *extra, "--gram_refresh_steps", "100", "--stop_after_epochs", "2")
    assert "cannot --branch_from" in message and all(d in message for d in diffs) and not os.path.exists(out), (name, message)
before_xt = open(os.path.join(X_T, "args.json")).read()
message = raises(ValueError, xgo, X_T, *X_TRUNK, "--keep_all_checkpoints", "--heads", "shared", "--resume", "--stop_after_epochs", "3")
assert "cannot --resume" in message and "heads: checkpoint 'per_dataset', now 'shared'" in message and open(os.path.join(X_T, "args.json")).read() == before_xt, message
ck_old_mode = dict(ck_t, args={k: v for k, v in ck_t["args"].items() if k not in X_ARK_KEYS})  # the (t) trunk as written before the six keys existed
ssl_pretrain.check_resume(ck_old_mode, make_args(BR_T, "--no_gram"), 1, 6, ck_t["data_fingerprint"])  # reads as the mixed-batch defaults: passes
message = raises(ValueError, ssl_pretrain.check_resume, ck_old_mode, make_args(BR_T, *ARK), 1, 6, ck_t["data_fingerprint"])
assert all(d in message for d in ("cyclic_ark_teacher: checkpoint False, now True", "heads: checkpoint 'shared', now 'per_dataset'", "lock_half_pass: checkpoint False, now True")), message
print("OK (x) branch: --branch_from a lock-arm trunk's cycle-1 checkpoint (whose teacher is the blend after step 22) starts from its student, teacher (8 heads) and a Gram teacher equal to that teacher's "
      "backbone; Gram refreshes after steps 44 and 66 (the next two cycle boundaries) run and each copies the teacher as blended at that boundary; a --gram_weight 0 branch equals the trunk resumed "
      "(student, teacher, optimizer, RNG, export, CSV); a branch with --heads shared, --no_lock_half_pass, another --ark_teacher_momentum, without the mode or without --keep_all_checkpoints is refused "
      "naming the keys, nothing created; a resume with --heads shared is refused, args.json untouched; a checkpoint without the six keys reads as the mixed-batch run")

# (i) two real ranks (gloo on this CPU, loopback rendezvous; the driver written like (v)'s and (w)'s): every rank trains one dataset per batch, all ranks the same dataset at every step, the
# boundaries (blends) at the same steps; the two ranks' images are disjoint; the backbone and the 8 heads are each a DDP module with find_unused_parameters False, and DDP keeps the student
# and the teacher bitwise alike on both ranks (a head called past its DDP wrapper would not be all-reduced, and the ranks' different images would part them)
X_DRIVER, X_DRIVER_SOURCE = os.path.join(TMP, "x_driver.py"), '''# one rank of a real two-process (gloo, CPU) cyclic run, written by test_ssl_pretrain.py (x); argv: base dir, spec json
import argparse, json, os, sys
import torch
import torch.distributed as dist

base, spec = sys.argv[1], json.load(open(sys.argv[2]))
sys.path.insert(0, base)
os.chdir(base)
import ssl_pretrain
from ssl_dinov3 import data as ssl_data
from ssl_dinov3.meta import SSLMetaArch

rank, found = int(os.environ["RANK"]), dict(steps=[], blends=[])
fetched, real_getitem, real_fb, real_ema, real_init = [], ssl_data.SSLImageDataset.__getitem__, SSLMetaArch.forward_backward, SSLMetaArch.update_ema, SSLMetaArch.__init__
flag = lambda ddp: int(ddp._get_ddp_logging_data()["find_unused_parameters"])


def getitem(self, i):
    fetched.append(self.paths[i])
    return real_getitem(self, i)


def forward_backward(self, data, *, teacher_temp):
    found["steps"].append([data["dataset"], list(fetched), data["phase"]])
    fetched.clear()
    return real_fb(self, data, teacher_temp=teacher_temp)


def update_ema(self, m):
    found["blends"].append(len(found["steps"]))
    return real_ema(self, m)


def init(self, student, *args, **kwargs):  # the student's DDP modules as run() hands them over
    found["ddp"] = {"backbone": flag(student.backbone), **{"%s.%s" % (k, n): flag(h) for k in ("dino_head", "ibot_head") for n, h in student[k].items()}}
    found["types"] = sorted({type(h).__name__ for k in ("dino_head", "ibot_head") for h in student[k].values()} | {type(student.backbone).__name__})
    real_init(self, student, *args, **kwargs)
    found["meta"] = self


ssl_data.SSLImageDataset.__getitem__, SSLMetaArch.forward_backward, SSLMetaArch.update_ema, SSLMetaArch.__init__ = getitem, forward_backward, update_ema, init
ssl_pretrain.run(argparse.Namespace(**spec["args"]), spec["vit"], spec["heads"], spec["cfg"])
meta = found.pop("meta")
flat = lambda module: {k: v.detach().cpu() for k, v in module.state_dict().items()}
torch.save(dict(student={k: flat(ssl_pretrain.Accelerator.unwrap_model(m)) if not isinstance(m, torch.nn.ModuleDict) else
                         {n: flat(ssl_pretrain.Accelerator.unwrap_model(h)) for n, h in m.items()} for k, m in meta.student.items()},
                teacher=flat(meta.teacher)), os.path.join(spec["dump"], "models_rank%d.pt" % rank))
with open(os.path.join(spec["dump"], "found_rank%d.json" % rank), "w") as f:
    json.dump(found, f)
dist.destroy_process_group()
'''
with open(X_DRIVER, "w") as f:
    f.write(X_DRIVER_SOURCE)
x_dump, X_TWO = os.path.join(TMP, "x_dump"), os.path.join(TMP, "X_TWO")
os.makedirs(x_dump)
with open(os.path.join(TMP, "x_spec.json"), "w") as f:  # batch 1 per rank: 12 / 8 / 6 / 4 images (and halves) over 2 ranks give the same passes as batch 2 on one
    json.dump(dict(args=vars(make_args(X_TWO, *X_TRUNK, "--batch_size_per_gpu", "1", "--stop_after_epochs", "2")), vit=TINY_VIT, heads=TINY_HEADS, cfg=X_CFG, dump=x_dump), f)
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    x_env = dict(os.environ, WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(s.getsockname()[1]))
x_procs = [subprocess.Popen([sys.executable, X_DRIVER, BASE, os.path.join(TMP, "x_spec.json")], env=dict(x_env, RANK=str(r), LOCAL_RANK=str(r)), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True) for r in range(2)]
try:
    x_outputs = [p.communicate(timeout=180)[0] for p in x_procs]
finally:
    for p in x_procs:
        p.kill()  # a rank stuck in a collective must not outlive the test
        p.wait()
assert [p.returncode for p in x_procs] == [0, 0], x_outputs
x_found = [json.load(open(os.path.join(x_dump, "found_rank%d.json" % r))) for r in range(2)]
x_models_ranks = [torch.load(os.path.join(x_dump, "models_rank%d.pt" % r), weights_only=False) for r in range(2)]
for found in x_found:
    assert [(name, phase) for name, _, phase in found["steps"]] == X_PATTERN * 2 and found["blends"] == [X_L * c + e for c in (0, 1) for e in X_ENDS], found
    assert all(len(paths) == 1 and x_dataset_of(paths[0]) == name for name, paths, _ in found["steps"])
    assert found["ddp"] == dict.fromkeys(["backbone"] + ["%s.%s" % (k, n) for k in ("dino_head", "ibot_head") for n in X_ORDER], 0) and found["types"] == ["DistributedDataParallel"]
assert all(set(a[1]).isdisjoint(b[1]) for a, b in zip(*(f["steps"] for f in x_found)))  # each rank its own shard of the pass
for cycle in (0, 1):  # the lock halves: both ranks shard the same draw, lock_subset's (the two shards together are that half, less drop_last's leftover of an odd half)
    for place, name in enumerate(X_ORDER):
        shards = [[f["steps"][i][1][0] for i in range(X_L * cycle, X_L * (cycle + 1)) if f["steps"][i][0] == name and f["steps"][i][2] == "lock"] for f in x_found]
        half = {x_images(name)[i] for i in ssl_pretrain.lock_subset(X_COUNTS[name], 100, cycle, place)}
        assert set(shards[0]).isdisjoint(shards[1]) and set(shards[0]) | set(shards[1]) <= half and len(shards[0]) == len(shards[1]) == X_COUNTS[name] // 2 // 2, (cycle, name, shards)
        assert X_COUNTS[name] // 2 % 2 or set(shards[0]) | set(shards[1]) == half, (cycle, name)
assert same_state(x_models_ranks[0], x_models_ranks[1])  # student (backbone, 8 heads) and teacher, bitwise alike on both ranks
assert (load_ckpt(X_TWO, "epoch_0002.pth")["step"], load_ckpt(X_TWO, "epoch_0002.pth")["world_size"], len(load_ckpt(X_TWO, "epoch_0002.pth")["rng_per_rank"])) == (44, 2, 2)
print("OK (x) two real ranks (lock arm, recipe on, batch 1 per rank): both ranks train the cycle's 8 passes (MIMIC lock 3 + release 6, ..., VinDrCXR 1 + 2) twice, each batch's image from its dataset, the "
      "same dataset and phase on both ranks at every step and disjoint images; each lock pass's two shards are disjoint and together the half lock_subset draws (MIMIC 6, CheXpert 4, VinDrCXR 2; "
      "ChestXray14 2 of its 3); both blend after steps 9, 15, 19, 22, 31, 37, 41, 44; the backbone and the 8 heads are 9 DDP modules with find_unused_parameters False on both ranks, and "
      "the student (all heads) and the teacher end bitwise alike on both; the step-44 epoch-2 checkpoint holds both ranks' RNG states")

# the CSV of a run written before the dataset column (15 columns) resumes with its rows kept, "-" added
X_OLD = os.path.join(TMP, "X_OLD")
go(X_OLD, "--stop_after_epochs", "1")
old_rows = [r[:15] for r in read_csv(X_OLD)]
with open(os.path.join(X_OLD, "ssl_losses.csv"), "w", newline="") as f:
    csv.writer(f).writerows(old_rows)
go(X_OLD, "--resume", "--stop_after_epochs", "2")
new_rows = read_csv(X_OLD)
assert ",".join(new_rows[0]) == CSV_HEADER and new_rows[1:len(old_rows)] == [r + ["-"] for r in old_rows[1:]] and len(new_rows) > len(old_rows)
assert all(len(r) == 16 and r[15] == "-" for r in new_rows[1:])
print("OK (x) CSV: a resume of a run whose ssl_losses.csv has the 15 columns from before the dataset column keeps its %d rows (with '-' as their dataset) under the 16-column header" % (len(old_rows) - 1))

# the resume guard in the mode (Codex r1 P2): which dataset owns an image decides its heads and passes, so data_fingerprint hashes (dataset, path) per dataset in cycle order. A repartition that keeps
# the flattened list (--data_set order) and every pass length (the last 4 MIMIC images listed first in CheXpert's list instead: MIMIC 8, CheXpert 12) passed the flat hash; now it is refused both ways,
# on --resume and --branch_from, naming the image list. Outside the mode the flat hash is the one it always was
x_moved = os.path.join(TMP, "data_cyclic_moved")
os.makedirs(x_moved)
with open(X_CFG["MIMIC"]["train_list"]) as f:
    x_mimic_rows = f.read().splitlines()
with open(X_CFG["CheXpert"]["train_list"]) as f:
    x_chexpert_rows = f.read().splitlines()
x_moved_cfg = dict(X_CFG, MIMIC=dict(X_CFG["MIMIC"], train_list=os.path.join(x_moved, "MIMIC.list")), CheXpert=dict(X_CFG["CheXpert"], train_list=os.path.join(x_moved, "CheXpert.list")))
with open(x_moved_cfg["MIMIC"]["train_list"], "w") as f:  # header + the first 8 rows
    f.write("\n".join(x_mimic_rows[:9]) + "\n")
with open(x_moved_cfg["CheXpert"]["train_list"], "w") as f:  # header + the moved 4 as absolute paths (os.path.join keeps them) + its own 8
    f.write("\n".join([x_chexpert_rows[0]] + ["%s,F" % path for path in x_images("MIMIC")[8:]] + x_chexpert_rows[1:]) + "\n")
flat = lambda cfg: [p for name in ("VinDrCXR", "ChestXray14", "MIMIC", "CheXpert") for p in ssl_data.image_paths(name, cfg[name]["data_dir"], cfg[name]["train_list"])]
assert flat(x_moved_cfg) == flat(X_CFG) and len(ssl_data.image_paths("CheXpert", X_CFG["CheXpert"]["data_dir"], x_moved_cfg["CheXpert"]["train_list"])) == 12
x_flat_hash = hashlib.sha256("\n".join(flat(X_CFG)).encode()).hexdigest()  # the mixed-batch guard: the same for both partitions
X_FP, X_FPM = os.path.join(TMP, "X_FP"), os.path.join(TMP, "X_FPM")
fp_summary, _ = xgo(X_FP, *ARK_OFF, "--stop_after_epochs", "1")
fpm_summary, fpm_text = xgo(X_FPM, *ARK_OFF, "--stop_after_epochs", "1", cfg=x_moved_cfg)
assert fp_summary["epoch_length"] == fpm_summary["epoch_length"] == 15 and "CheXpert 12 images: release 6 steps, MIMIC 8 images: release 4 steps" in fpm_text  # same length, other partition
ck_fp, ck_fpm = load_ckpt(X_FP, "epoch_0001.pth"), load_ckpt(X_FPM, "epoch_0001.pth")
assert ck_fp["data_fingerprint"] == hashlib.sha256("\n".join("%s\t%s" % (name, p) for name in X_ORDER for p in x_images(name)).encode()).hexdigest() != ck_fpm["data_fingerprint"]
assert x_flat_hash not in (ck_fp["data_fingerprint"], ck_fpm["data_fingerprint"]) and ck2["data_fingerprint"] == hashlib.sha256("\n".join(flat(CFG)).encode()).hexdigest()  # (g)'s mixed run: the flat hash
for out, own_cfg, other_cfg, src in ((X_FP, X_CFG, x_moved_cfg, X_FPM), (X_FPM, x_moved_cfg, X_CFG, X_FP)):  # src: the other run, trained on other_cfg
    before_fp = open(os.path.join(out, "args.json")).read()
    message = raises(ValueError, xgo, out, *ARK_OFF, "--resume", "--stop_after_epochs", "2", cfg=other_cfg)
    assert "cannot --resume" in message and "training image list" in message and open(os.path.join(out, "args.json")).read() == before_fp, message
    branch_out = os.path.join(TMP, "xfp_branch_" + os.path.basename(out))
    message = raises(ValueError, xgo, branch_out, "--branch_from", os.path.join(src, "ckpt", "epoch_0001.pth"), *ARK_OFF[1:], "--gram_refresh_steps", "100", "--stop_after_epochs", "2",
                     cfg=own_cfg)  # a branch of the other run over this one's partition
    assert "cannot --branch_from" in message and "training image list" in message and not os.path.exists(branch_out), message
print("OK (x) fingerprint: two runs over the same flattened image list (the mixed-batch hash identical) but with 4 images moved from MIMIC's list to CheXpert's (the same 15-step cycle, another order and other "
      "heads per image) carry different data_fingerprints (the (dataset, path) hash in cycle order); each refuses a --resume and a --branch_from of the other, naming the training image list, args.json "
      "untouched and no branch dir created; a mixed-batch checkpoint's fingerprint is the flat hash as before")


# (k) launch scripts ------------------------------------------------------------------------------------------------------------
scripts = [os.path.join(REPO, "chain", f) for f in ("ssl_common.sh", "pilot_a.sh", "pilot_b.sh", "main.sh", "constlr.sh", "hires.sh")]
scripts.append(os.path.join(REPO, "run_d1_8s_timing_gaudi.sh"))
scripts.append(os.path.join(REPO, "run_d1_8hires_gram512_gaudi.sh"))  # the Gram-512 twin's sbatch wrapper; its contents are checked after the hires checks
scripts.append(os.path.join(REPO, "run_d1_8_constlr_nogram_gaudi.sh"))  # the no-Gram trunk's sbatch wrapper; its contents are checked after the constlr checks
scripts.append(os.path.join(REPO, "run_d1_8_branch_gram_gaudi.sh"))  # the Gram-anchored branches' sbatch wrapper; its contents are checked after the no-Gram wrapper's
scripts.append(os.path.join(REPO, "run_d1_8_constlr_nogram_plain_gaudi.sh"))  # the plain no-Gram trunk's sbatch wrapper; its contents are checked after the branch wrapper's
scripts.append(os.path.join(REPO, "run_d1_8_constlr_nogram_ablate_gaudi.sh"))  # the one-flag-off ablation arms' sbatch wrapper; its contents are checked after the Ark-teacher checks
for script in scripts:
    assert os.access(script, os.X_OK), script
    subprocess.run(["bash", "-n", script], check=True)
CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "HABANA_VISIBLE_", "HIRES_", "CONSTLR_")) and k != "ARK_ROOT"}
ALLOC4 = dict(CLEAN_ENV, SLURM_JOB_GPUS="0,1,2,3")  # the chain's 4-card allocation: ssl_launch counts its entries as Ark's ark_device_count does


def sandbox(weights=True, gaudi_executable=True, timing=False):
    """A fake repo root: copies of the chain scripts and Ark's common.sh, a non-empty weights file, and a gaudi.sh that records its argv."""
    root = tempfile.mkdtemp(dir=TMP)
    os.makedirs(os.path.join(root, "chain"))
    os.makedirs(os.path.join(root, "weights"))
    for name in ("main.sh", "pilot_a.sh", "pilot_b.sh", "constlr.sh", "hires.sh", "ssl_common.sh"):
        shutil.copy(os.path.join(REPO, "chain", name), os.path.join(root, "chain", name))
    os.makedirs(os.path.join(root, "experiment_scripts"))
    shutil.copy(os.path.join(REPO, "experiment_scripts", "common.sh"), os.path.join(root, "experiment_scripts"))  # ark_device_count
    if weights:
        open(os.path.join(root, "weights", WEIGHT_NAME), "w").write("x")
    recorders = {"gaudi.sh": 'printf "%s\\n" "$@" > "$PWD/argv.txt"', "gaudi_check_cards.sh": 'touch "$PWD/cards_checked"',
                 "gaudi_verify_busy.sh": 'printf "%s\\n" "$1" > "$PWD/busy_arg.txt"'}
    for name, body in recorders.items():
        if timing or name == "gaudi.sh":
            path = os.path.join(root, name)
            open(path, "w").write("#!/usr/bin/env bash\n" + body + "\n")
            os.chmod(path, 0o755 if gaudi_executable or name != "gaudi.sh" else 0o644)
    if timing:
        shutil.copy(os.path.join(REPO, "run_d1_8s_timing_gaudi.sh"), root)
    return root


def run_script(root, script, cwd=None, env=ALLOC4):
    return subprocess.run(["bash", script], cwd=cwd or root, env=env, capture_output=True, text=True)


def launched(root):
    path = os.path.join(root, "argv.txt")
    return open(path).read().splitlines() if os.path.exists(path) else None


def after(argv, flag):
    return argv[argv.index(flag) + 1]


COMMON = ["python", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "4", "ssl_pretrain.py", "--device", "hpu",
          "--data_set", "VinDrCXR", "--data_set", "ChestXray14", "--data_set", "MIMIC", "--data_set", "CheXpert",
          "--pretrained_weights", "/workspace/weights/" + WEIGHT_NAME, "--batch_size_per_gpu", "64", "--workers", "16",
          "--seed", "100", "--output_dir"]


def with_ranks(ranks):  # COMMON with another --nproc-per-node count
    return COMMON[:5] + [ranks] + COMMON[6:]


expanded = {}
for script, run_name, tail in (("pilot_a.sh", "ssl_dinov3_cxr_pilot_a_seed_100", ["--lr", "1e-3", "--epochs", "40", "--stop_after_epochs", "4"]),
                               ("pilot_b.sh", "ssl_dinov3_cxr_pilot_b_seed_100", ["--lr", "3e-4", "--no_lr_scaling", "--epochs", "40",
                                                                                "--stop_after_epochs", "4"])):
    root = sandbox()
    result = run_script(root, "chain/" + script)
    assert result.returncode == 0, result.stderr
    assert launched(root) == COMMON + ["/workspace/" + run_name, "--resume"] + tail, launched(root)
    expanded[script] = "./gaudi.sh " + " ".join(launched(root))

# one rank per allocated card, counted on the host as Ark's ark_device_count does: HABANA_VISIBLE_MODULES, else HABANA_VISIBLE_DEVICES, else
# SLURM_JOB_GPUS; with none of them it asks the container (gaudi.sh python -c ...) and, with no usable answer, refuses to launch
pilot_a_tail = ["/workspace/ssl_dinov3_cxr_pilot_a_seed_100", "--resume", "--lr", "1e-3", "--epochs", "40", "--stop_after_epochs", "4"]
for env, ranks in ((dict(CLEAN_ENV, SLURM_JOB_GPUS="0,1"), "2"), (dict(CLEAN_ENV, SLURM_JOB_GPUS="4,5,6,7"), "4"),
                   (dict(CLEAN_ENV, SLURM_JOB_GPUS="0,1,2,3", HABANA_VISIBLE_MODULES="5,6"), "2")):  # a host HABANA_VISIBLE_* wins, as in gaudi.sh
    root = sandbox()
    result = run_script(root, "chain/pilot_a.sh", env=env)
    assert result.returncode == 0 and launched(root) == with_ranks(ranks) + pilot_a_tail, (ranks, result.stderr, launched(root))
root = sandbox()  # nothing allocated on the host: the container is asked for its device count, the stub answers nothing -> nothing launched
result = run_script(root, "chain/pilot_a.sh", env=CLEAN_ENV)
assert result.returncode == 1 and "no usable devices" in result.stderr and launched(root)[:2] == ["python", "-c"], (result.stderr, launched(root))
root = sandbox()  # ... and a container that reports 3 devices gets 3 ranks
open(os.path.join(root, "gaudi.sh"), "w").write('#!/usr/bin/env bash\nif [[ $2 == -c ]]; then echo DEVICE_COUNT=3; else printf "%s\\n" "$@" > "$PWD/argv.txt"; fi\n')
result = run_script(root, "chain/pilot_a.sh", env=CLEAN_ENV)
assert result.returncode == 0 and launched(root) == with_ranks("3") + pilot_a_tail, (result.stderr, launched(root))

root = sandbox()  # main.sh: no main_lr.txt -> error naming it, gaudi.sh never reached
open(os.path.join(root, "chain", "main_epochs.txt"), "w").write("30\n")
result = run_script(root, "chain/main.sh")
assert result.returncode != 0 and "main_lr.txt" in result.stderr and launched(root) is None, result.stderr
os.remove(os.path.join(root, "chain", "main_epochs.txt"))  # ... and no main_epochs.txt
open(os.path.join(root, "chain", "main_lr.txt"), "w").write("2e-3\n")
result = run_script(root, "chain/main.sh")
assert result.returncode != 0 and "main_epochs.txt" in result.stderr and launched(root) is None, result.stderr
for lr, epochs, culprit in (("", "30", "main_lr.txt"), ("\n", "30", "main_lr.txt"), ("abc", "30", "main_lr.txt"),
                            ("0", "30", "main_lr.txt"), ("-1e-3", "30", "main_lr.txt"), ("1e-3 2e-3", "30", "main_lr.txt"),
                            ("2e-3", "", "main_epochs.txt"), ("2e-3", "0", "main_epochs.txt"), ("2e-3", "2.5", "main_epochs.txt"),
                            ("2e-3", "thirty", "main_epochs.txt")):
    open(os.path.join(root, "chain", "main_lr.txt"), "w").write(lr)
    open(os.path.join(root, "chain", "main_epochs.txt"), "w").write(epochs)
    result = run_script(root, "chain/main.sh")
    assert result.returncode == 1 and "ERROR" in result.stderr and culprit in result.stderr and launched(root) is None, (lr, epochs, result.stderr)
open(os.path.join(root, "chain", "main_lr.txt"), "w").write("0.00123\n")
open(os.path.join(root, "chain", "main_epochs.txt"), "w").write(" 30 \n")
result = run_script(root, "chain/main.sh")
assert result.returncode == 0, result.stderr
argv = launched(root)
assert argv == COMMON + ["/workspace/ssl_dinov3_cxr_main_seed_100", "--resume", "--lr", "0.00123", "--no_lr_scaling", "--epochs", "30"], argv
expanded["main.sh"] = "./gaudi.sh " + " ".join(argv)

root = sandbox()  # constlr.sh: main_lr.txt (the main run's LR) + constlr_stop_after.txt; either one missing -> error naming it, gaudi.sh never reached
open(os.path.join(root, "chain", "constlr_stop_after.txt"), "w").write("30\n")
result = run_script(root, "chain/constlr.sh")
assert result.returncode == 1 and "ERROR" in result.stderr and "main_lr.txt" in result.stderr and launched(root) is None, result.stderr
os.remove(os.path.join(root, "chain", "constlr_stop_after.txt"))  # ... and no constlr_stop_after.txt
open(os.path.join(root, "chain", "main_lr.txt"), "w").write("2e-3\n")
result = run_script(root, "chain/constlr.sh")
assert result.returncode == 1 and "ERROR" in result.stderr and "constlr_stop_after.txt" in result.stderr and launched(root) is None, result.stderr
for lr, stop, culprit in (("0", "30", "main_lr.txt"), ("2e-3", "", "constlr_stop_after.txt"), ("2e-3", "0", "constlr_stop_after.txt"),
                          ("2e-3", "2.5", "constlr_stop_after.txt"), ("2e-3", "thirty", "constlr_stop_after.txt"),
                          ("2e-3", "-3", "constlr_stop_after.txt"), ("2e-3", "1001", "constlr_stop_after.txt"),  # 1001: one past the 1000 ceiling
                          ("2e-3", "99999999999999999999", "constlr_stop_after.txt"),
                          ("2e-3", "18446744073709551617", "constlr_stop_after.txt")):  # 2**64 + 1: bash arithmetic would read it as 1
    open(os.path.join(root, "chain", "main_lr.txt"), "w").write(lr)
    open(os.path.join(root, "chain", "constlr_stop_after.txt"), "w").write(stop)
    result = run_script(root, "chain/constlr.sh")
    assert result.returncode == 1 and "ERROR" in result.stderr and culprit in result.stderr and launched(root) is None, (lr, stop, result.stderr)
assert "ceiling of 1000" in result.stderr, result.stderr  # the last one is refused for being above the ceiling, and the message names it


def constlr_argv(stop):  # what constlr.sh launches for this stop epoch
    return COMMON + ["/workspace/ssl_dinov3_cxr_constlr_seed_100", "--resume", "--lr", "0.002", "--min_lr", "0.002", "--no_lr_scaling",
                     "--epochs", "1000", "--stop_after_epochs", stop, "--gram_refresh_steps", "10000,20000,30000"]


open(os.path.join(root, "chain", "main_lr.txt"), "w").write("0.002\n")
open(os.path.join(root, "chain", "constlr_stop_after.txt"), "w").write(" 30 \n")
result = run_script(root, "chain/constlr.sh")
assert result.returncode == 0, result.stderr
argv = launched(root)
assert argv == constlr_argv("30"), argv
expanded["constlr.sh"] = "./gaudi.sh " + " ".join(argv)
assert not any(flag in argv for flag in RECIPE if flag.startswith("--"))  # the phase: none of the DINOv3 recipe flags the trunk and its branches pass
# the trainer reads that argv the way the phase means it: stop after epoch 30 of the 1000-epoch ceiling, Meta's three Gram refreshes (passed
# explicitly because --resume locks them; also the trainer's default, so the same as main.sh, which passes none), and an LR (and teacher
# temperature) that is flat once its warmup is over; epoch_length 10 stands in for the real one (2666 on 4 cards), only the shape matters
parsed = ssl_pretrain.parse_args(argv[argv.index("ssl_pretrain.py") + 1:])
default_steps = ssl_pretrain.parse_args(["--data_set", "VinDrCXR", "--pretrained_weights", "unused", "--output_dir", "unused"]).gram_refresh_steps
assert ssl_pretrain.check_args(parsed) == 30 and parsed.epochs == 1000
assert parsed.gram_refresh_steps == [10000, 20000, 30000] == default_steps
sch = build_schedules(lr=parsed.lr, min_lr=parsed.min_lr, epochs=parsed.epochs, warmup_epochs=parsed.warmup_epochs, epoch_length=10, global_batch_size=256,
                      scale_lr=not parsed.no_lr_scaling, teacher_temp_warmup_epochs=parsed.teacher_temp_warmup_epochs)
flat_lr, flat_temp = sch["lr"][parsed.warmup_epochs * 10:], sch["teacher_temp"][parsed.teacher_temp_warmup_epochs * 10:]
assert len(flat_lr) == (1000 - parsed.warmup_epochs) * 10 and len(flat_temp) == (1000 - parsed.teacher_temp_warmup_epochs) * 10  # not an empty slice
assert (flat_lr == 0.002).all() and (flat_temp == 0.07).all(), (flat_lr.min(), flat_lr.max(), flat_temp.min(), flat_temp.max())  # exactly, not just close
# what makes the extension knob safe: --resume locks the schedule (length, LR, warmup, Gram refreshes) but not where a run stops
assert "stop_after_epochs" not in ssl_pretrain.RESUME_MATCH
assert {"epochs", "lr", "min_lr", "gram_refresh_steps", "warmup_epochs"} <= set(ssl_pretrain.RESUME_MATCH)
# the boundary: a stop epoch equal to the ceiling launches, and is one the trainer's own check_args accepts
os.remove(os.path.join(root, "argv.txt"))
open(os.path.join(root, "chain", "constlr_stop_after.txt"), "w").write("1000\n")
result = run_script(root, "chain/constlr.sh")
boundary = launched(root)
assert result.returncode == 0 and boundary == constlr_argv("1000"), (result.stderr, boundary)
assert ssl_pretrain.check_args(ssl_pretrain.parse_args(boundary[boundary.index("ssl_pretrain.py") + 1:])) == 1000

# the no-Gram trunk ("DINO-X ConstLR 224, no Gram"): CONSTLR_NOGRAM=1 (what run_d1_8_constlr_nogram_gaudi.sh sets) makes constlr.sh launch its line with --no_gram in place of
# --gram_refresh_steps 10000,20000,30000, into its own run dir, with the stop epoch from its own cluster-only file (chain/constlr_stop_after.txt is then not read)
NOGRAM_ENV, NOGRAM_RUN = dict(ALLOC4, CONSTLR_NOGRAM="1"), "/workspace/ssl_dinov3_cxr_constlr_nogram_seed_100"
CONSTLR_FILES = {"main_lr.txt": "0.002\n", "constlr_stop_after.txt": "30\n", "constlr_nogram_stop_after.txt": " 100 \n"}  # 30 vs 100: the argv shows which file was read


def constlr_run(root, env=ALLOC4, **override):  # write the setting files (None: absent), run constlr.sh; returns (result, argv or None)
    for name, content in dict(CONSTLR_FILES, **override).items():
        path = os.path.join(root, "chain", name)
        if os.path.exists(path):
            os.remove(path)
        if content is not None:
            open(path, "w").write(content)
    if os.path.exists(os.path.join(root, "argv.txt")):
        os.remove(os.path.join(root, "argv.txt"))
    return run_script(root, "chain/constlr.sh", env=env), launched(root)


def nogram_argv_for(stop):  # constlr_argv(stop) with the run dir swapped and the Gram refresh list replaced by the DINOv3 recipe (RECIPE, (u)) and --no_gram
    argv = constlr_argv(stop)
    assert argv[-2:] == ["--gram_refresh_steps", "10000,20000,30000"]
    return [NOGRAM_RUN if arg == "/workspace/ssl_dinov3_cxr_constlr_seed_100" else arg for arg in argv[:-2]] + RECIPE + ["--no_gram"]


root = sandbox()
result, nogram_argv = constlr_run(root, env=NOGRAM_ENV)  # constlr_nogram_stop_after.txt " 100 ", constlr_stop_after.txt 30 (not read)
assert result.returncode == 0, result.stderr
assert nogram_argv == nogram_argv_for("100"), nogram_argv
assert result.stdout.splitlines() == ["run: ssl_dinov3_cxr_constlr_nogram_seed_100 (no Gram teacher: --no_gram replaces --gram_refresh_steps 10000,20000,30000; DINOv3 recipe: "
                                      "--drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm; "
                                      "stop after epoch 100 from chain/constlr_nogram_stop_after.txt)"] and result.stderr == "", (result.stdout, result.stderr)
expanded["constlr.sh with CONSTLR_NOGRAM=1"] = "./gaudi.sh " + " ".join(nogram_argv)
# the same stop epoch in the chain's phase (variable unset, or set but empty): the only differences are the run dir and the tail, and nothing new is printed
result, default_100 = constlr_run(root, **{"constlr_stop_after.txt": "100\n"})
assert result.returncode == 0 and default_100 == constlr_argv("100") and (result.stdout, result.stderr) == ("", ""), (result.stderr, default_100)
assert default_100[-2:] == ["--gram_refresh_steps", "10000,20000,30000"] and nogram_argv[-6:] == RECIPE + ["--no_gram"] and len(nogram_argv) == len(default_100) + 4
assert [(a, b) for a, b in zip(default_100[:-2], nogram_argv[:-6]) if a != b] == [("/workspace/ssl_dinov3_cxr_constlr_seed_100", NOGRAM_RUN)], (default_100, nogram_argv)
assert len(default_100[:-2]) == len(nogram_argv[:-6])  # the zip above covers the whole common part
result, empty_argv = constlr_run(root, env=dict(ALLOC4, CONSTLR_NOGRAM=""), **{"constlr_stop_after.txt": "100\n"})
assert result.returncode == 0 and empty_argv == default_100 and (result.stdout, result.stderr) == ("", ""), (result.stderr, empty_argv)
# a missing or invalid no-Gram stop file is refused naming itself (not constlr_stop_after.txt, which stays valid) before gaudi.sh is reached, and before any run: line
for stop in (None, "", "0", "2.5", "thirty", "-3", "1001", "18446744073709551617"):  # None: absent; 1001: one past the ceiling; 2**64 + 1: bash arithmetic would read it as 1
    result, refused_argv = constlr_run(root, env=NOGRAM_ENV, **{"constlr_nogram_stop_after.txt": stop})
    assert result.returncode == 1 and "ERROR" in result.stderr and "constlr_nogram_stop_after.txt" in result.stderr and refused_argv is None, (stop, result.stderr)
    assert "constlr_stop_after.txt" not in result.stderr and result.stdout == "", (stop, result.stderr, result.stdout)
assert "ceiling of 1000" in result.stderr, result.stderr  # the last one is refused for being above the ceiling, and the message names it
# the two stop files are independent: the no-Gram run does not need the phase's, and the phase does not take the no-Gram file instead of its own
result, solo_argv = constlr_run(root, env=NOGRAM_ENV, **{"constlr_stop_after.txt": None})
assert result.returncode == 0 and solo_argv == nogram_argv, (result.stderr, solo_argv)
result, refused_argv = constlr_run(root, **{"constlr_stop_after.txt": None})
assert result.returncode == 1 and "ERROR" in result.stderr and "constlr_stop_after.txt" in result.stderr and "nogram" not in result.stderr and refused_argv is None, result.stderr
# any other CONSTLR_NOGRAM is refused naming itself, before anything is read (the bare sandbox has no setting file to miss) and so before anything is launched (the other has them all)
bare = sandbox()
for value in ("0", "yes", "2", " 1"):
    expected_error = "ERROR: CONSTLR_NOGRAM holds '%s', which is not empty or 1\n" % value
    result, refused_argv = constlr_run(root, env=dict(ALLOC4, CONSTLR_NOGRAM=value))
    assert result.returncode == 1 and result.stderr == expected_error and result.stdout == "" and refused_argv is None, (value, result.stderr)
    result = run_script(bare, "chain/constlr.sh", env=dict(ALLOC4, CONSTLR_NOGRAM=value))
    assert result.returncode == 1 and result.stderr == expected_error and result.stdout == "" and launched(bare) is None, (value, result.stderr)
# the trainer reads the no-Gram argv as the run means it: stop after epoch 100 of the 1000-epoch ceiling, no Gram teacher at all (weight 0, no refreshes, the size left at the
# 2 x global default), and every other setting --resume locks the same as constlr.sh's argv gives it (world_size is the trainer's, not an arg)
nogram_parsed = ssl_pretrain.parse_args(nogram_argv[nogram_argv.index("ssl_pretrain.py") + 1:])
default_parsed = ssl_pretrain.parse_args(default_100[default_100.index("ssl_pretrain.py") + 1:])
assert ssl_pretrain.check_args(nogram_parsed) == 100 and nogram_parsed.epochs == 1000
assert (nogram_parsed.no_gram, nogram_parsed.gram_weight, nogram_parsed.gram_refresh_steps, nogram_parsed.gram_teacher_size) == (True, 0.0, [], 448)
assert (default_parsed.no_gram, default_parsed.gram_weight, default_parsed.gram_refresh_steps) == (False, 2.0, [10000, 20000, 30000])
assert "no_gram" in ssl_pretrain.RESUME_MATCH and "stop_after_epochs" not in ssl_pretrain.RESUME_MATCH  # --resume locks the Gram switch, not where a run stops
assert tuple(getattr(nogram_parsed, k) for k in RECIPE_KEYS) == (0.3, 2.0, True) and tuple(getattr(default_parsed, k) for k in RECIPE_KEYS) == (0.0, None, False)
locked_alike = set(ssl_pretrain.RESUME_MATCH) - {"no_gram", "gram_weight", "gram_refresh_steps", "world_size"} - set(RECIPE_KEYS)
assert all(getattr(nogram_parsed, k) == getattr(default_parsed, k) for k in locked_alike), \
    {k: (getattr(nogram_parsed, k), getattr(default_parsed, k)) for k in locked_alike if getattr(nogram_parsed, k) != getattr(default_parsed, k)}
assert nogram_parsed.pretrained_weights == default_parsed.pretrained_weights and nogram_parsed.stop_after_epochs == default_parsed.stop_after_epochs == 100
assert nogram_parsed.output_dir == NOGRAM_RUN != default_parsed.output_dir
# the wrapper: bash -n and the executable bit are checked with the other scripts above; here its sbatch header, its guards and its last line
nogram_lines = open(os.path.join(REPO, "run_d1_8_constlr_nogram_gaudi.sh")).read().splitlines()
assert [line[len("#SBATCH "):] for line in nogram_lines if line.startswith("#SBATCH ")] == [
    "--job-name=d1-8-nogram", "-A grp_jliang12", "-p gaudi", "-q public", "-t 7-00:00:00", "-N 1", "--gres=gpu:hl225:4", "-c 72", "--mem=128G"], nogram_lines
nogram_code = [line for line in nogram_lines if line and not line.startswith("#")]
assert nogram_code[-1] == "CONSTLR_ABLATE= CONSTLR_PLAIN= CONSTLR_ARK= CONSTLR_NOGRAM=1 exec ./chain/constlr.sh", nogram_code  # clears an inherited CONSTLR_ABLATE, CONSTLR_PLAIN and CONSTLR_ARK (checked by running it, with the ablation block below)
for needle in ("export PT_HPU_LAZY_MODE=1", "CONSTLR_NOGRAM=1", "exec ./chain/constlr.sh", "./gaudi_check_cards.sh || exit 1", "[[ -x ./gaudi_verify_busy.sh ]]", "./gaudi_verify_busy.sh 900 &"):
    assert any(needle in line for line in nogram_code), needle  # in a command, not only in a comment
assert all(line.startswith("#") for line in nogram_lines if "phases.txt" in line), nogram_lines  # not a chain phase: phases.txt is named in comments only

# the Gram-anchored branches (run_d1_8_branch_gram_gaudi.sh): CONSTLR_BRANCH_EPOCH=K / _STOP / _EVERY / _COUNT make constlr.sh launch the no-Gram trunk's line with --no_gram replaced by --branch_from
# <trunk>/ckpt/epoch_<K>.pth --gram_refresh_steps <absolute list>, into a run dir that names K and the cadence; the list starts at K x the epoch_length in the trunk's args.json (read there, never assumed)
import shlex
import stat

BRANCH_SCRIPT = os.path.join(REPO, "run_d1_8_branch_gram_gaudi.sh")
BRANCH_CD = "cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1"
BRANCH_TRUNK = "ssl_dinov3_cxr_constlr_nogram_seed_100"
BRANCH_E10 = "ssl_dinov3_cxr_constlr_nogram_e0010_gram_every10000x3_seed_100"  # the planned run: from epoch 10, Meta's 10k x 3
BRANCH_SRC10 = NOGRAM_RUN + "/ckpt/epoch_0010.pth"  # in-container path
branch_elsewhere = tempfile.mkdtemp(dir=TMP)  # sbatch starts the job in its spool dir, not the repo: the wrapper's own cd must do the work


def branch_root(epoch_length=2666, source=True):  # a sandbox with the wrapper (its cd pointed at the sandbox), main_lr.txt only, and a stub trunk
    root = sandbox(timing=True)  # also the gaudi_check_cards.sh / gaudi_verify_busy.sh recorders
    open(os.path.join(root, "chain", "main_lr.txt"), "w").write("0.002\n")  # the only setting file: a branch reads no stop file
    os.makedirs(os.path.join(root, BRANCH_TRUNK, "ckpt"))
    with open(os.path.join(root, BRANCH_TRUNK, "args.json"), "w") as f:  # the trainer's format (json.dump, indent=2): lists span lines, so epoch_length is not the only line
        json.dump(dict(no_gram=True, gram_refresh_steps=[], data_set=["VinDrCXR", "ChestXray14", "MIMIC", "CheXpert"], epoch_length=epoch_length,
                       total_steps=1000 * epoch_length), f, indent=2)
    if source:
        open(os.path.join(root, BRANCH_TRUNK, "ckpt", "epoch_0010.pth"), "w").close()
    text = open(BRANCH_SCRIPT).read()
    assert text.count(BRANCH_CD) == 1, BRANCH_CD
    open(os.path.join(root, "run_d1_8_branch_gram_gaudi.sh"), "w").write(text.replace(BRANCH_CD, "cd %s || exit 1" % shlex.quote(root)))
    return root


def branch_run(root, *args, env=ALLOC4):  # runs the wrapper copy with these arguments; returns (result, argv or None), a stale argv.txt removed first
    if os.path.exists(os.path.join(root, "argv.txt")):
        os.remove(os.path.join(root, "argv.txt"))
    result = subprocess.run(["bash", os.path.join(root, "run_d1_8_branch_gram_gaudi.sh")] + list(args), cwd=branch_elsewhere, env=env, capture_output=True, text=True)
    return result, launched(root)


def branch_argv_for(stop, run, source, steps):  # nogram_argv_for(stop) with the run dir swapped and the final --no_gram replaced by the branch pair
    argv = nogram_argv_for(stop)
    assert argv[-1:] == ["--no_gram"] and argv.count(NOGRAM_RUN) == 1
    return [("/workspace/" + run) if arg == NOGRAM_RUN else arg for arg in argv[:-1]] + ["--branch_from", source, "--gram_refresh_steps", steps]


def branch_refused(root, args, *needles, env=ALLOC4):  # the wrapper copy refuses these arguments: exit 1, an ERROR on stderr naming every needle, nothing printed, gaudi.sh never reached
    result, refused = branch_run(root, *args, env=env)
    assert result.returncode == 1 and "ERROR" in result.stderr and result.stdout == "" and refused is None, (args, result.returncode, result.stdout, result.stderr)
    assert all(needle in result.stderr for needle in needles), (args, needles, result.stderr)


# `10 40`: the trunk's line with the branch pair in place of --no_gram, into the run dir that names the cadence; this sandbox has no stop file at all (a branch reads none)
branch_dir = branch_root()
bres, bargv = branch_run(branch_dir, "10", "40")
assert bres.returncode == 0 and bres.stderr == "", (bres.stdout, bres.stderr)
assert bres.stdout.splitlines() == [
    "run: ssl_dinov3_cxr_constlr_nogram_e0010_gram_every10000x3_seed_100 (Gram-anchored branch of ssl_dinov3_cxr_constlr_nogram_seed_100 from its epoch 10: "
    "--branch_from /workspace/ssl_dinov3_cxr_constlr_nogram_seed_100/ckpt/epoch_0010.pth --gram_refresh_steps in place of --no_gram; stop after epoch 40)",
    "gram refresh steps: 36660,46660,56660 (branch step 26660 = epoch 10 x epoch_length 2666 from ssl_dinov3_cxr_constlr_nogram_seed_100/args.json, then every 10000 steps, 3 times)"], bres.stdout
assert bargv == branch_argv_for("40", BRANCH_E10, BRANCH_SRC10, "36660,46660,56660"), bargv
assert os.path.exists(os.path.join(branch_dir, "cards_checked"))
for _ in range(50):  # the busy check runs in the background
    if os.path.exists(os.path.join(branch_dir, "busy_arg.txt")):
        break
    subprocess.run(["sleep", "0.1"])
assert open(os.path.join(branch_dir, "busy_arg.txt")).read() == "900\n"
expanded["run_d1_8_branch_gram_gaudi.sh 10 40"] = "./gaudi.sh " + " ".join(bargv)
# ... which is the real trunk's line (constlr.sh with CONSTLR_NOGRAM=1, stop 40) except for the run dir and the tail
tres, trunk_argv = constlr_run(sandbox(), env=NOGRAM_ENV, **{"constlr_nogram_stop_after.txt": "40\n"})
assert tres.returncode == 0 and trunk_argv == nogram_argv_for("40"), (tres.stderr, trunk_argv)
assert trunk_argv[-1:] == ["--no_gram"] and bargv[-4:] == ["--branch_from", BRANCH_SRC10, "--gram_refresh_steps", "36660,46660,56660"] and len(bargv) == len(trunk_argv) + 3
assert [(a, b) for a, b in zip(trunk_argv[:-1], bargv[:-4]) if a != b] == [(NOGRAM_RUN, "/workspace/" + BRANCH_E10)], (trunk_argv, bargv)
# `10 40 2666 30`, one refresh per epoch for 30 epochs: the cadence is in the run dir's name, so it never shares a dir with the 10k x 3 branch from the same epoch
steps30 = ",".join(str(26660 + 2666 * i) for i in range(1, 31))
run30 = "ssl_dinov3_cxr_constlr_nogram_e0010_gram_every2666x30_seed_100"
bres, bargv30 = branch_run(branch_dir, "10", "40", "2666", "30")
assert bres.returncode == 0 and bres.stderr == "" and bargv30 == branch_argv_for("40", run30, BRANCH_SRC10, steps30) != bargv, (bres.stderr, bargv30)
assert len(bres.stdout.splitlines()) == 2 and bres.stdout.splitlines()[0].startswith("run: %s (Gram-anchored branch of " % run30), bres.stdout
assert bres.stdout.splitlines()[1] == ("gram refresh steps: %s (branch step 26660 = epoch 10 x epoch_length 2666 from ssl_dinov3_cxr_constlr_nogram_seed_100/args.json, "
                                       "then every 2666 steps, 30 times)" % steps30), bres.stdout
expanded["run_d1_8_branch_gram_gaudi.sh 10 40 2666 30"] = "./gaudi.sh " + " ".join(bargv30)
# the epoch length is read from the trunk's args.json, not assumed: 2665 moves every step; another K / STOP / EVERY / COUNT gets its own run dir, source and list
bres, bargv2665 = branch_run(branch_root(epoch_length=2665), "10", "40")
assert bres.returncode == 0 and bargv2665 == branch_argv_for("40", BRANCH_E10, BRANCH_SRC10, "36650,46650,56650"), (bres.stderr, bargv2665)
assert "gram refresh steps: 36650,46650,56650 (branch step 26650 = epoch 10 x epoch_length 2665 from " in bres.stdout, bres.stdout
open(os.path.join(branch_dir, BRANCH_TRUNK, "ckpt", "epoch_0007.pth"), "w").close()
bres, bargv7 = branch_run(branch_dir, "7", "12", "500", "2")
assert bres.returncode == 0 and bargv7 == branch_argv_for("12", "ssl_dinov3_cxr_constlr_nogram_e0007_gram_every500x2_seed_100", NOGRAM_RUN + "/ckpt/epoch_0007.pth", "19162,19662"), (bres.stderr, bargv7)
# the limits themselves launch: STOP at the 1000 ceiling, EVERY of 9 digits, COUNT of 4 (one more digit of either is refused below). Not in one launch: the last refresh must stay inside the 1000-epoch schedule (checked below),
# which 999999999 x 1000 cannot. EVERY of 9 digits gets the largest epoch_length (also 9 digits) for its one refresh, after the branch step 10 x 999999999; COUNT of 4 digits gets a small EVERY
steps_max = str(10 * 999999999 + 999999999)
bres, bargv_max = branch_run(branch_root(epoch_length=999999999), "10", "1000", "999999999", "1")
assert bres.returncode == 0 and bargv_max == branch_argv_for("1000", "ssl_dinov3_cxr_constlr_nogram_e0010_gram_every999999999x1_seed_100", BRANCH_SRC10, steps_max), (bres.stderr, bargv_max and bargv_max[-4:])
steps_count = ",".join(str(26660 + i) for i in range(1, 10000))
bres, bargv_count = branch_run(branch_dir, "10", "40", "1", "9999")
assert bres.returncode == 0 and bargv_count == branch_argv_for("40", "ssl_dinov3_cxr_constlr_nogram_e0010_gram_every1x9999_seed_100", BRANCH_SRC10, steps_count), (bres.stderr, bargv_count and bargv_count[-4:])
# the trainer reads the branch argv as meant: --branch_from the source, the absolute refresh list (all after the branch step), the Gram teacher ON at the trainer's weight 2.0 and teacher 448, stop after
# epoch 40 of the 1000-epoch ceiling, and every other setting --resume locks (the Gram ones, branch_from and the trainer's world_size aside) the same as the trunk's argv gives it
bparsed = ssl_pretrain.parse_args(bargv[bargv.index("ssl_pretrain.py") + 1:])
tparsed = ssl_pretrain.parse_args(trunk_argv[trunk_argv.index("ssl_pretrain.py") + 1:])
assert ssl_pretrain.check_args(bparsed) == 40 and bparsed.epochs == 1000 and ssl_pretrain.check_args(ssl_pretrain.parse_args(bargv_max[bargv_max.index("ssl_pretrain.py") + 1:])) == 1000
assert (bparsed.branch_from, bparsed.gram_refresh_steps, bparsed.no_gram, bparsed.gram_weight, bparsed.gram_teacher_size) == (BRANCH_SRC10, [36660, 46660, 56660], False, 2.0, 448)
assert tuple(getattr(bparsed, k) for k in RECIPE_KEYS) == (0.3, 2.0, True) and bargv[-9:-4] == RECIPE  # the trunk's recipe, which --branch_from requires to match
assert min(bparsed.gram_refresh_steps) > 10 * 2666 and bparsed.output_dir == "/workspace/" + BRANCH_E10 != tparsed.output_dir and bparsed.resume  # a resubmit resumes the branch
assert tparsed.branch_from is None and tparsed.no_gram is True
branch_locked_alike = set(ssl_pretrain.RESUME_MATCH) - set(ssl_pretrain.BRANCH_FREE) - {"world_size"}
assert all(getattr(bparsed, k) == getattr(tparsed, k) for k in branch_locked_alike), \
    {k: (getattr(bparsed, k), getattr(tparsed, k)) for k in branch_locked_alike if getattr(bparsed, k) != getattr(tparsed, k)}
assert bparsed.pretrained_weights == tparsed.pretrained_weights and bparsed.stop_after_epochs == tparsed.stop_after_epochs == 40
# refusals, each before anything is launched: exit 1, an ERROR naming the culprit, nothing on stdout, gaudi.sh never reached
usage_dir = branch_root()
for args in ((), ("10",), ("10", "40", "10000", "3", "5", "x")):
    branch_refused(usage_dir, args, "usage", "got %d arguments" % len(args))
assert not os.path.exists(os.path.join(usage_dir, "cards_checked")), "the card check ran before the usage guard"  # not even the card check ran, so the argument count is the wrapper's first guard
for value in ("abc", "0", "-1", "1.5", "010"):  # 010: printf %04d would read it as octal
    branch_refused(branch_dir, (value, "40"), "K (CONSTLR_BRANCH_EPOCH)", "holds '%s'" % value, *(["leading zero"] if value == "010" else []))
for value in ("x", "0", "2.5", "040"):
    branch_refused(branch_dir, ("10", value), "STOP (CONSTLR_BRANCH_STOP)", "holds '%s'" % value)
for value in ("10", "5"):  # equal to K and below it: the branch would train nothing
    branch_refused(branch_dir, ("10", value), "not after K")
for value in ("1001", "18446744073709551617"):  # one past the ceiling; 2**64 + 1, which bash arithmetic would read as 1
    branch_refused(branch_dir, ("10", value), "STOP (CONSTLR_BRANCH_STOP)", "ceiling of 1000")
for value in ("0", "abc", "1234567890"):  # the last has 10 digits
    branch_refused(branch_dir, ("10", "40", value), "EVERY (CONSTLR_BRANCH_EVERY)")
for value in ("0", "12345", "x"):
    branch_refused(branch_dir, ("10", "40", "10000", value), "COUNT (CONSTLR_BRANCH_COUNT)")
# the last refresh, K x epoch_length + COUNT x EVERY, is at most 1000 x epoch_length (the last step of the --epochs 1000 schedule); a later one could never run, even with the branch extended to the ceiling.
# `10 40 2666 990` ends exactly on it (26660 + 990 x 2666 = 2666000) and launches; one refresh more, or one 9-digit EVERY, passes it: refused naming both steps, before anything is printed
steps_end = ",".join(str(26660 + 2666 * i) for i in range(1, 991))
assert steps_end.endswith(",2666000")
bres, bargv_end = branch_run(branch_dir, "10", "40", "2666", "990")
assert bres.returncode == 0 and bres.stderr == "" and bargv_end == branch_argv_for("40", "ssl_dinov3_cxr_constlr_nogram_e0010_gram_every2666x990_seed_100", BRANCH_SRC10, steps_end), (bres.stderr, bargv_end and bargv_end[-4:])
branch_refused(branch_dir, ("10", "40", "2666", "991"), "past the end", "after step 2668666", "(step 2666000)")  # 26660 + 991 x 2666
branch_refused(branch_dir, ("10", "40", "999999999", "1"), "past the end", "after step 1000026659", "(step 2666000)")  # 26660 + 999999999
branch_refused(branch_root(epoch_length=2665), ("10", "40", "2666", "990"), "past the end", "(step 2665000)")  # the end is 1000 x the trunk's own epoch_length, not an assumed 2666
# no source: refused on a first start, naming it (the K the trunk never kept as well); once the branch has an epoch checkpoint of its own a resume needs no source
nosrc_dir = branch_root(source=False)
branch_refused(nosrc_dir, ("10", "40"), BRANCH_TRUNK + "/ckpt/epoch_0010.pth", "nothing to branch from")
branch_refused(branch_dir, ("5", "40"), BRANCH_TRUNK + "/ckpt/epoch_0005.pth", "nothing to branch from")
own_ckpt = os.path.join(nosrc_dir, BRANCH_E10, "ckpt")
os.makedirs(own_ckpt)
for name in ("epoch_0011.pth.tmp", "maxsteps_0000003.pth"):  # not epoch checkpoints (the trainer resumes from epoch_<digits>.pth only)
    open(os.path.join(own_ckpt, name), "w").close()
open(os.path.join(own_ckpt, "epoch_backup.pth"), "w").close()  # matches the glob epoch_*.pth, but is no epoch_<digits>.pth
os.mkdir(os.path.join(own_ckpt, "epoch_0011.pth"))  # nor is a directory of that name
branch_refused(nosrc_dir, ("10", "40"), BRANCH_TRUNK + "/ckpt/epoch_0010.pth")
os.rmdir(os.path.join(own_ckpt, "epoch_0011.pth"))
# a real one, which the glob lists before epoch_backup.pth: the loop's last iteration fails its test and constlr.sh must still go on under set -e
open(os.path.join(own_ckpt, "epoch_0011.pth"), "w").close()
bres, bargv_resume = branch_run(nosrc_dir, "10", "40")
assert bres.returncode == 0 and bres.stderr == "" and bargv_resume == bargv, (bres.stderr, bargv_resume)
# the branch step is never guessed: a trunk args.json that is missing, has no epoch_length line or a zero one is refused naming both
json_dir = branch_root()
trunk_args = os.path.join(json_dir, BRANCH_TRUNK, "args.json")
open(trunk_args, "w").write(json.dumps(dict(no_gram=True, epoch_length=0, total_steps=0), indent=2))
branch_refused(json_dir, ("10", "40"), "args.json", "epoch_length")
open(trunk_args, "w").write(json.dumps(dict(no_gram=True, total_steps=2666000), indent=2))
branch_refused(json_dir, ("10", "40"), "args.json", "epoch_length")
os.remove(trunk_args)
branch_refused(json_dir, ("10", "40"), "args.json", "epoch_length")
# a branch replaces --no_gram itself, so CONSTLR_NOGRAM beside it is refused (through the wrapper and in constlr.sh); a lone variable, or one of the four emptied, is refused naming the first one missing
env_dir = branch_root()
branch_refused(env_dir, ("10", "40"), "both set", env=NOGRAM_ENV)
branch_all = dict(CONSTLR_BRANCH_EPOCH="10", CONSTLR_BRANCH_STOP="40", CONSTLR_BRANCH_EVERY="10000", CONSTLR_BRANCH_COUNT="3")
for env, culprits in ((dict(NOGRAM_ENV, CONSTLR_BRANCH_EPOCH="10"), ["both set"]), (dict(ALLOC4, CONSTLR_BRANCH_EVERY="10000"), ["CONSTLR_BRANCH_EPOCH"]),
                      *((dict(ALLOC4, **dict(branch_all, **{name: ""})), [name]) for name in branch_all)):
    bres = run_script(env_dir, "chain/constlr.sh", env=env)
    assert bres.returncode == 1 and "ERROR" in bres.stderr and all(c in bres.stderr for c in culprits) and bres.stdout == "" and launched(env_dir) is None, (culprits, bres.stderr)
    assert culprits == ["both set"] or [name for name in branch_all if name in bres.stderr] == culprits, (culprits, bres.stderr)  # only the culprit is named
# the phase and the trunk are unchanged: with all four variables empty, constlr.sh launches and prints exactly as it did before they existed
no_branch = dict(CONSTLR_BRANCH_EPOCH="", CONSTLR_BRANCH_STOP="", CONSTLR_BRANCH_EVERY="", CONSTLR_BRANCH_COUNT="")
phase_dir = sandbox()
bres, phase_argv = constlr_run(phase_dir, env=dict(ALLOC4, **no_branch))
assert bres.returncode == 0 and phase_argv == constlr_argv("30") and (bres.stdout, bres.stderr) == ("", ""), (bres.stdout, bres.stderr, phase_argv)
bres, trunk_argv100 = constlr_run(phase_dir, env=dict(NOGRAM_ENV, **no_branch))
assert bres.returncode == 0 and trunk_argv100 == nogram_argv_for("100") and bres.stderr == "", (bres.stderr, trunk_argv100)
assert bres.stdout.splitlines() == ["run: ssl_dinov3_cxr_constlr_nogram_seed_100 (no Gram teacher: --no_gram replaces --gram_refresh_steps 10000,20000,30000; DINOv3 recipe: "
                                      "--drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm; "
                                    "stop after epoch 100 from chain/constlr_nogram_stop_after.txt)"], bres.stdout
# the wrapper: bash -n and the executable bit are checked with the other scripts above; here its mode, sbatch header, guards and last line
assert stat.S_IMODE(os.stat(BRANCH_SCRIPT).st_mode) == 0o755
branch_lines = open(BRANCH_SCRIPT).read().splitlines()
assert [line[len("#SBATCH "):] for line in branch_lines if line.startswith("#SBATCH ")] == [
    "--job-name=d1-8-branch", "-A grp_jliang12", "-p gaudi", "-q public", "-t 7-00:00:00", "-N 1", "--gres=gpu:hl225:4", "-c 72", "--mem=128G"], branch_lines
branch_code = [line for line in branch_lines if line and not line.startswith("#")]
assert branch_code[-1] == ("CONSTLR_BRANCH_EPOCH=$1 CONSTLR_BRANCH_STOP=$2 CONSTLR_BRANCH_EVERY=${3:-10000} CONSTLR_BRANCH_COUNT=${4:-3} CONSTLR_BRANCH_GRAM_EPOCH=${5:-} "
                            "exec ./chain/constlr.sh"), branch_code
branch_needles = ("export PT_HPU_LAZY_MODE=1", "[[ $# -ge 2 && $# -le 5 ]]", BRANCH_CD, "./gaudi_check_cards.sh || exit 1", "[[ -x ./gaudi_verify_busy.sh ]]", "./gaudi_verify_busy.sh 900 &")
for needle in branch_needles:
    assert any(needle in line for line in branch_code), needle  # in a command, not only in a comment
positions = [min(i for i, line in enumerate(branch_code) if needle in line) for needle in branch_needles]
assert all(a < b for a, b in zip(positions, positions[1:])), positions  # in this order: the usage guard comes before the cd and the card check
assert all(line.startswith("#") for line in branch_lines if "phases.txt" in line), branch_lines  # not a chain phase: phases.txt is named in comments only
print("OK (k) branch launcher: run_d1_8_branch_gram_gaudi.sh (mode 755, 4 cards, the usage guard first, ends in CONSTLR_BRANCH_EPOCH=$1 CONSTLR_BRANCH_STOP=$2 CONSTLR_BRANCH_EVERY=${3:-10000} "
      "CONSTLR_BRANCH_COUNT=${4:-3} CONSTLR_BRANCH_GRAM_EPOCH=${5:-} exec ./chain/constlr.sh, names phases.txt only in comments) makes constlr.sh launch the no-Gram trunk's line with only --no_gram replaced by --branch_from "
      "<trunk>/ckpt/epoch_<K>.pth --gram_refresh_steps <list> and no stop file read: `10 40` -> " + BRANCH_E10 + " refreshing after 36660,46660,56660 (10 x the epoch_length 2666 read from the "
      "trunk's args.json, 36650,... at 2665, then every 10000 steps, 3 times), `10 40 2666 30` -> ..._every2666x30_ (one refresh per epoch for 30 epochs), `7 12 500 2` -> ..._e0007_ 19162,19662; "
      "a refresh list must end inside the 1000-epoch schedule, whose last step is 1000 x the trunk's own epoch_length: `10 40 2666 990` (26660 + 990 x 2666 = 1000 x 2666) launches, 991 refreshes or a "
      "9-digit EVERY pass it and are refused; a branch's own epoch checkpoint is a regular file named epoch_<digits>.pth, the trainer's names (epoch_backup.pth, *.tmp, maxsteps_*.pth and a directory of "
      "that name leave a missing source refused); "
      "the argv parses to branch_from the source, Gram ON (2.0, 448), stop 40 of 1000, the trunk's DINOv3 recipe (drop path 0.3, rescale 2, untied local CLS norm), every other locked setting = the trunk's; refused before launching: a wrong argument count (before the "
      "card check), K / STOP / EVERY / COUNT that are not positive integers (no leading zero; EVERY at most 9 digits, COUNT 4, both accepted at the limit), STOP above the 1000 ceiling "
      "(accepted at 1000) or not after K, a trunk args.json without a positive epoch_length, CONSTLR_NOGRAM set as well, one of the four variables alone or emptied, and a missing source "
      "on a first start (a branch with an epoch checkpoint of its own resumes without it); the phase and the trunk launch and print exactly as before with all four variables empty")

# the plain no-Gram trunk ("DINO-X ConstLR 224, no Gram, plain"): CONSTLR_NOGRAM=1 CONSTLR_PLAIN=1 (what run_d1_8_constlr_nogram_plain_gaudi.sh sets) launches the phase's line with --no_gram in place of
# --gram_refresh_steps 10000,20000,30000 and NOTHING else changed, i.e. the no-Gram trunk's line without the DINOv3 recipe (RECIPE): the one-variable Gram-off control of ssl_dinov3_cxr_constlr_seed_100. Its own run dir
# and its own cluster-only stop file (neither of the other two is read); the phase, the no-Gram trunk and its branches (all above) launch and print exactly as before
PLAIN_ENV, PLAIN_RUN, PLAIN_STOP_FILE = dict(NOGRAM_ENV, CONSTLR_PLAIN="1"), "/workspace/ssl_dinov3_cxr_constlr_nogram_plain_seed_100", "constlr_nogram_plain_stop_after.txt"
RECIPE_FLAGS = [flag for flag in RECIPE if flag.startswith("--")]  # RECIPE also holds the values 0.3 and 2


def plain_run(root, env=PLAIN_ENV, **override):  # constlr_run with the plain stop file written too, " 60 " (None: absent): the argv shows which stop file was read (the phase's 30, the no-Gram trunk's 100, the plain one's 60)
    return constlr_run(root, env=env, **dict({PLAIN_STOP_FILE: " 60 \n"}, **override))


def plain_argv_for(stop):  # constlr_argv(stop) with the run dir swapped and the Gram refresh list replaced by --no_gram, nothing else
    argv = constlr_argv(stop)
    assert argv[-2:] == ["--gram_refresh_steps", "10000,20000,30000"]
    return [PLAIN_RUN if arg == "/workspace/ssl_dinov3_cxr_constlr_seed_100" else arg for arg in argv[:-2]] + ["--no_gram"]


plain_root = sandbox()
result, plain_argv = plain_run(plain_root)  # constlr_nogram_plain_stop_after.txt " 60 "; the phase's 30 and the no-Gram trunk's 100 are not read
assert result.returncode == 0 and result.stderr == "", result.stderr
assert plain_argv == plain_argv_for("60"), plain_argv
assert result.stdout.splitlines() == ["run: ssl_dinov3_cxr_constlr_nogram_plain_seed_100 (no Gram teacher: --no_gram replaces --gram_refresh_steps 10000,20000,30000; plain: no DINOv3 recipe flags, "
                                      "the one-variable Gram-off control of ssl_dinov3_cxr_constlr_seed_100; stop after epoch 60 from chain/constlr_nogram_plain_stop_after.txt)"], result.stdout
expanded["constlr.sh with CONSTLR_NOGRAM=1 CONSTLR_PLAIN=1"] = "./gaudi.sh " + " ".join(plain_argv)
assert plain_argv[-1:] == ["--no_gram"] and "--gram_refresh_steps" not in plain_argv and not any(flag in plain_argv for flag in RECIPE_FLAGS)  # --no_gram, and none of the three recipe flags
# the no-Gram trunk's line at the same stop epoch is this one with RECIPE in front of --no_gram (and its own run dir): the three recipe flags are all that separates the two no-Gram runs
result, plain_trunk_60 = constlr_run(plain_root, env=NOGRAM_ENV, **{"constlr_nogram_stop_after.txt": "60\n"})
assert result.returncode == 0 and plain_trunk_60 == nogram_argv_for("60"), (result.stderr, plain_trunk_60)
assert plain_trunk_60 == [NOGRAM_RUN if arg == PLAIN_RUN else arg for arg in plain_argv[:-1]] + RECIPE + ["--no_gram"], (plain_trunk_60, plain_argv)
# the trainer reads the plain argv as the run means it: stop after epoch 60 of the 1000-epoch ceiling, no Gram teacher (weight 0, no refreshes, the size left at the 448 default), the recipe at the trainer's
# defaults (no drop path, no RoPE rescale, tied local CLS norm), and every other parsed setting the phase's at the same stop epoch: only --no_gram (with the Gram weight and refresh list it fixes) and the run dir differ
result, plain_phase_60 = constlr_run(plain_root, **{"constlr_stop_after.txt": "60\n"})
assert result.returncode == 0 and plain_phase_60 == constlr_argv("60"), (result.stderr, plain_phase_60)
plain_parsed = ssl_pretrain.parse_args(plain_argv[plain_argv.index("ssl_pretrain.py") + 1:])
phase_parsed = ssl_pretrain.parse_args(plain_phase_60[plain_phase_60.index("ssl_pretrain.py") + 1:])
assert ssl_pretrain.check_args(plain_parsed) == 60 == ssl_pretrain.check_args(phase_parsed) and plain_parsed.epochs == 1000
assert (plain_parsed.no_gram, plain_parsed.gram_weight, plain_parsed.gram_refresh_steps, plain_parsed.gram_teacher_size) == (True, 0.0, [], 448)
assert tuple(getattr(plain_parsed, k) for k in RECIPE_KEYS) == (0.0, None, False) == tuple(getattr(phase_parsed, k) for k in RECIPE_KEYS)
plain_differs = {k for k in vars(plain_parsed) if getattr(plain_parsed, k) != getattr(phase_parsed, k)}
assert plain_differs == {"output_dir", "no_gram", "gram_weight", "gram_refresh_steps"}, plain_differs
assert plain_parsed.output_dir == PLAIN_RUN and plain_parsed.resume  # a resubmit resumes
# a missing or invalid plain stop file is refused naming itself (neither of the other two, which stay valid) before gaudi.sh is reached, and before any run: line
for stop in (None, "", "0", "2.5", "thirty", "-3", "1001", "18446744073709551617"):  # None: absent; 1001: one past the ceiling; 2**64 + 1: bash arithmetic would read it as 1
    result, refused_argv = plain_run(plain_root, **{PLAIN_STOP_FILE: stop})
    assert result.returncode == 1 and "ERROR" in result.stderr and PLAIN_STOP_FILE in result.stderr and refused_argv is None, (stop, result.stderr)
    assert "constlr_stop_after.txt" not in result.stderr and "constlr_nogram_stop_after.txt" not in result.stderr and result.stdout == "", (stop, result.stderr, result.stdout)
assert "ceiling of 1000" in result.stderr, result.stderr  # the last one is refused for being above the ceiling, and the message names it
# the three stop files are independent: the plain run needs neither of the other two ...
result, solo_argv = plain_run(plain_root, **{"constlr_stop_after.txt": None, "constlr_nogram_stop_after.txt": None})
assert result.returncode == 0 and solo_argv == plain_argv, (result.stderr, solo_argv)
# ... and the phase and the no-Gram trunk launch and print exactly as before, the plain stop file present (not read: the argv has their 30 and 100) or absent, CONSTLR_PLAIN unset or empty
for plain_file in (" 60 \n", None):
    for env in (NOGRAM_ENV, dict(NOGRAM_ENV, CONSTLR_PLAIN="")):
        result, kept_argv = plain_run(plain_root, env=env, **{PLAIN_STOP_FILE: plain_file})
        assert result.returncode == 0 and kept_argv == nogram_argv_for("100") and result.stderr == "", (plain_file, env, result.stderr, kept_argv)
        assert result.stdout.splitlines() == ["run: ssl_dinov3_cxr_constlr_nogram_seed_100 (no Gram teacher: --no_gram replaces --gram_refresh_steps 10000,20000,30000; DINOv3 recipe: "
                                              "--drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm; "
                                              "stop after epoch 100 from chain/constlr_nogram_stop_after.txt)"], (plain_file, env, result.stdout)
    for env in (ALLOC4, dict(ALLOC4, CONSTLR_PLAIN="")):
        result, kept_argv = plain_run(plain_root, env=env, **{PLAIN_STOP_FILE: plain_file})
        assert result.returncode == 0 and kept_argv == constlr_argv("30") and (result.stdout, result.stderr) == ("", ""), (plain_file, env, result.stderr, kept_argv)
bres, kept_bargv = branch_run(branch_dir, "10", "40")  # the branches (the wrapper and sandbox above): an empty CONSTLR_PLAIN changes neither the launch line nor the output
bres_empty, kept_bargv_empty = branch_run(branch_dir, "10", "40", env=dict(ALLOC4, CONSTLR_PLAIN=""))
assert bres.returncode == bres_empty.returncode == 0 and kept_bargv == kept_bargv_empty == bargv, (bres.stderr, bres_empty.stderr)
assert (bres.stdout, bres.stderr) == (bres_empty.stdout, bres_empty.stderr) and bres.stderr == "" and len(bres.stdout.splitlines()) == 2, (bres.stdout, bres_empty.stdout)
# CONSTLR_PLAIN is refused before anything is read (plain_bare has no setting file to miss) and so before anything is launched (plain_root has them all): a value other than empty or 1; 1 without
# CONSTLR_NOGRAM=1; 1 with any CONSTLR_BRANCH_* (with CONSTLR_NOGRAM=1 or without it)
plain_bare = sandbox()
PLAIN_NEEDS_NOGRAM = "ERROR: CONSTLR_PLAIN=1 is set without CONSTLR_NOGRAM=1: the plain trunk is the no-Gram trunk without the DINOv3 recipe, so set both\n"
PLAIN_NO_BRANCH = ("ERROR: CONSTLR_PLAIN and CONSTLR_BRANCH_* are both set: a branch continues the no-Gram trunk with the DINOv3 recipe, and branches of the plain trunk are a later feature, "
                   "so unset CONSTLR_PLAIN\n")
plain_refusals = [(dict(NOGRAM_ENV, CONSTLR_PLAIN=value), "ERROR: CONSTLR_PLAIN holds '%s', which is not empty or 1\n" % value) for value in ("0", "yes", "2", " 1")]
plain_refusals += [(dict(ALLOC4, CONSTLR_PLAIN="2"), "ERROR: CONSTLR_PLAIN holds '2', which is not empty or 1\n"),  # the value is checked first, with or without CONSTLR_NOGRAM
                   (dict(ALLOC4, CONSTLR_PLAIN="1"), PLAIN_NEEDS_NOGRAM), (dict(ALLOC4, CONSTLR_PLAIN="1", CONSTLR_NOGRAM=""), PLAIN_NEEDS_NOGRAM)]
plain_refusals += [(dict(base_env, CONSTLR_PLAIN="1", **branch_vars), PLAIN_NO_BRANCH) for base_env in (ALLOC4, NOGRAM_ENV)
                   for branch_vars in (dict(CONSTLR_BRANCH_EPOCH="10"), dict(CONSTLR_BRANCH_STOP="40"), branch_all)]
for env, expected_error in plain_refusals:
    result = run_script(plain_bare, "chain/constlr.sh", env=env)
    assert result.returncode == 1 and result.stderr == expected_error and result.stdout == "" and launched(plain_bare) is None, (env, result.stderr)
    result, refused_argv = plain_run(plain_root, env=env)
    assert result.returncode == 1 and result.stderr == expected_error and result.stdout == "" and refused_argv is None, (env, result.stderr)
branch_refused(branch_root(), ("10", "40"), "CONSTLR_PLAIN", "CONSTLR_BRANCH_*", "later feature", env=dict(ALLOC4, CONSTLR_PLAIN="1"))  # the branch wrapper sets all four: refused naming both
# the wrapper: bash -n and the executable bit are checked with the other scripts above; here its mode, sbatch header, guards and last line, that it is the no-Gram trunk's wrapper (nogram_code, above) but for
# the job name, the comments and CONSTLR_PLAIN=1, and that a copy with its cd pointed at a sandbox launches the plain line from outside the repo (as sbatch starts it)
PLAIN_SCRIPT = os.path.join(REPO, "run_d1_8_constlr_nogram_plain_gaudi.sh")
assert stat.S_IMODE(os.stat(PLAIN_SCRIPT).st_mode) == 0o755
plain_text = open(PLAIN_SCRIPT).read()
plain_lines = plain_text.splitlines()
assert [line[len("#SBATCH "):] for line in plain_lines if line.startswith("#SBATCH ")] == [
    "--job-name=d1-8-nogram-plain", "-A grp_jliang12", "-p gaudi", "-q public", "-t 7-00:00:00", "-N 1", "--gres=gpu:hl225:4", "-c 72", "--mem=128G"], plain_lines
plain_code = [line for line in plain_lines if line and not line.startswith("#")]
assert plain_code[-1] == "CONSTLR_NOGRAM=1 CONSTLR_PLAIN=1 exec ./chain/constlr.sh" and plain_code[:-1] == nogram_code[:-1], plain_code
for needle in ("export PT_HPU_LAZY_MODE=1", "CONSTLR_NOGRAM=1", "CONSTLR_PLAIN=1", "exec ./chain/constlr.sh", "./gaudi_check_cards.sh || exit 1", "[[ -x ./gaudi_verify_busy.sh ]]", "./gaudi_verify_busy.sh 900 &"):
    assert any(needle in line for line in plain_code), needle  # in a command, not only in a comment
assert all(line.startswith("#") for line in plain_lines if "phases.txt" in line), plain_lines  # not a chain phase: phases.txt is named in comments only
assert "chain/constlr_nogram_plain_stop_after.txt" in plain_text and "81.18 / 81.53 / 81.61 / 81.58" in plain_text  # the comments name the stop file and the Gram run's probes it is compared with
wrap_root = sandbox(timing=True)  # also the gaudi_check_cards.sh / gaudi_verify_busy.sh recorders
open(os.path.join(wrap_root, "chain", "main_lr.txt"), "w").write("0.002\n")
open(os.path.join(wrap_root, "chain", PLAIN_STOP_FILE), "w").write("60\n")  # the only stop file
assert plain_text.count(BRANCH_CD) == 1, BRANCH_CD
open(os.path.join(wrap_root, "plain_wrapper.sh"), "w").write(plain_text.replace(BRANCH_CD, "cd %s || exit 1" % shlex.quote(wrap_root)))
result = subprocess.run(["bash", os.path.join(wrap_root, "plain_wrapper.sh")], cwd=branch_elsewhere, env=ALLOC4, capture_output=True, text=True)
assert result.returncode == 0 and result.stderr == "" and launched(wrap_root) == plain_argv_for("60"), (result.stderr, launched(wrap_root))
assert result.stdout.startswith("run: ssl_dinov3_cxr_constlr_nogram_plain_seed_100 (no Gram teacher: ") and os.path.exists(os.path.join(wrap_root, "cards_checked")), result.stdout
for _ in range(50):  # the busy check runs in the background
    if os.path.exists(os.path.join(wrap_root, "busy_arg.txt")):
        break
    subprocess.run(["sleep", "0.1"])
assert open(os.path.join(wrap_root, "busy_arg.txt")).read() == "900\n"
print("OK (k) plain no-Gram trunk: constlr.sh with CONSTLR_NOGRAM=1 CONSTLR_PLAIN=1 (run_d1_8_constlr_nogram_plain_gaudi.sh: mode 755, 4 cards, the no-Gram trunk's wrapper but for the job name, the comments and "
      "CONSTLR_PLAIN=1, names phases.txt only in comments, launches the line below from outside the repo) launches the phase's line with --no_gram in place of the Gram refresh list, none of the three DINOv3 recipe flags "
      "(the no-Gram trunk's line is this one plus RECIPE) and nothing else changed, into ssl_dinov3_cxr_constlr_nogram_plain_seed_100 with the stop epoch from constlr_nogram_plain_stop_after.txt only (a missing or "
      "invalid file is refused naming itself, the other two stop files neither read nor needed); the argv parses to the phase's settings but for no_gram (and the Gram weight / refresh list it fixes) and the run dir, "
      "the recipe at the trainer's defaults; CONSTLR_PLAIN other than empty or 1, 1 without CONSTLR_NOGRAM=1 or 1 with any CONSTLR_BRANCH_* is refused before anything is read; the phase, the no-Gram trunk "
      "and its branches launch and print exactly as before, the plain stop file present or absent and CONSTLR_PLAIN empty")

# the Ark-teacher trunk ("DINO-X ConstLR 224, no Gram, Ark teacher"): CONSTLR_NOGRAM=1 CONSTLR_ARK=1 (what run_d1_8_constlr_arkteacher_gaudi.sh sets) launches the no-Gram trunk's line with the arm's ARK_FLAGS
# appended (--cyclic_ark_teacher --heads per_dataset <lock flag> --keep_all_checkpoints). CONSTLR_ARK_LOCK picks the arm (0 when unset or empty: --no_lock_half_pass; 1: --lock_half_pass), and the arm is in
# the run dir and the stop file (ssl_dinov3_cxr_constlr_arkteacher[_lock]_seed_100, chain/constlr_arkteacher[_lock]_stop_after.txt; none of the others read). Its Gram branches are CONSTLR_BRANCH_ARK=1 (with
# the same CONSTLR_ARK_LOCK) with the four CONSTLR_BRANCH_*, through the branch wrapper: that arm's trunk line with --no_gram replaced by the branch pair, ARK_FLAGS kept, the lock checked against the trunk's
# args.json. The phase, the no-Gram trunk, the plain trunk and the no-Gram trunk's branches launch and print exactly as before
ARK_ENV, ARK_STOP_FILE, ARK_LOCK_STOP_FILE = dict(NOGRAM_ENV, CONSTLR_ARK="1"), "constlr_arkteacher_stop_after.txt", "constlr_arkteacher_lock_stop_after.txt"
ARK_LOCK_ENV = dict(ARK_ENV, CONSTLR_ARK_LOCK="1")
ARK_TRUNKS = {"0": "ssl_dinov3_cxr_constlr_arkteacher_seed_100", "1": "ssl_dinov3_cxr_constlr_arkteacher_lock_seed_100"}
ARK_RUNS = {lock: "/workspace/" + trunk for lock, trunk in ARK_TRUNKS.items()}
ARK_FLAGS_OF = {lock: ["--cyclic_ark_teacher", "--heads", "per_dataset", flag, "--keep_all_checkpoints"] for lock, flag in (("0", "--no_lock_half_pass"), ("1", "--lock_half_pass"))}
ARK_LOCK_NOTES = {"0": "no lock passes: the full-pass release per dataset alone (CONSTLR_ARK_LOCK=0)", "1": "FX lock-release: half-pass lock + full-pass release per dataset (CONSTLR_ARK_LOCK=1)"}
NOGRAM_RUN_LINE = ("run: ssl_dinov3_cxr_constlr_nogram_seed_100 (no Gram teacher: --no_gram replaces --gram_refresh_steps 10000,20000,30000; DINOv3 recipe: --drop_path_rate 0.3 --rope_rescale_coords 2 "
                   "--untie_local_cls_norm; stop after epoch %s from chain/constlr_nogram_stop_after.txt)")


def ark_run_line(lock, stop):  # what constlr.sh prints for the arm
    return ("run: %s (no Gram teacher: --no_gram replaces --gram_refresh_steps 10000,20000,30000; DINOv3 recipe: --drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm; Ark teacher: %s "
            "after --no_gram (%s; clean teacher views implied, --ark_teacher_momentum 0.8 by default; an epoch is a cycle over the four datasets); stop after epoch %s from chain/%s)"
            % (ARK_TRUNKS[lock], " ".join(ARK_FLAGS_OF[lock]), ARK_LOCK_NOTES[lock], stop, ARK_LOCK_STOP_FILE if lock == "1" else ARK_STOP_FILE))


def ark_run(root, env=ARK_ENV, **override):  # constlr_run with all five stop files written (the phase's 30, the no-Gram trunk's 100, the plain 60, the Ark arms' " 70 " / " 80 "; None: absent)
    return constlr_run(root, env=env, **dict({PLAIN_STOP_FILE: "60\n", ARK_STOP_FILE: " 70 \n", ARK_LOCK_STOP_FILE: " 80 \n"}, **override))


def ark_argv_for(stop, lock="0"):  # nogram_argv_for(stop) with the arm's run dir and its ARK_FLAGS appended
    argv = nogram_argv_for(stop)
    assert argv[-1:] == ["--no_gram"] and argv.count(NOGRAM_RUN) == 1
    return [ARK_RUNS[lock] if arg == NOGRAM_RUN else arg for arg in argv] + ARK_FLAGS_OF[lock]


ark_root = sandbox()
ark_argvs = {}
for lock, env, stop in (("0", ARK_ENV, "70"), ("0", dict(ARK_ENV, CONSTLR_ARK_LOCK=""), "70"), ("0", dict(ARK_ENV, CONSTLR_ARK_LOCK="0"), "70"), ("1", ARK_LOCK_ENV, "80")):
    result, argv = ark_run(ark_root, env=env)
    assert result.returncode == 0 and result.stderr == "" and argv == ark_argv_for(stop, lock), (lock, env, result.stderr, argv)
    assert result.stdout.splitlines() == [ark_run_line(lock, stop)], result.stdout
    ark_argvs[lock] = argv
expanded["constlr.sh with CONSTLR_NOGRAM=1 CONSTLR_ARK=1"] = "./gaudi.sh " + " ".join(ark_argvs["0"])
expanded["constlr.sh with CONSTLR_NOGRAM=1 CONSTLR_ARK=1 CONSTLR_ARK_LOCK=1"] = "./gaudi.sh " + " ".join(ark_argvs["1"])
assert [(a, b) for a, b in zip(ark_argvs["0"], ark_argvs["1"]) if a != b] == [(ARK_RUNS["0"], ARK_RUNS["1"]), ("70", "80"), ("--no_lock_half_pass", "--lock_half_pass")]  # the run dir, its stop file, the flag
result, trunk_70 = constlr_run(ark_root, env=NOGRAM_ENV, **{"constlr_nogram_stop_after.txt": "70\n"})  # the no-Gram trunk at the same stop epoch: the arm's line minus ARK_FLAGS, its own run dir
assert result.returncode == 0 and trunk_70 == nogram_argv_for("70") and ark_argvs["0"] == [ARK_RUNS["0"] if arg == NOGRAM_RUN else arg for arg in trunk_70] + ARK_FLAGS_OF["0"], (trunk_70, ark_argvs["0"])
# the trainer reads both as meant: stop after cycle 70 / 80 of the 1000-epoch ceiling, the mode with its resolved settings (clean views, momentum 0.8, no per-step momentum), the arm's lock, and every other
# parsed setting the no-Gram trunk's: only the run dir, the stop epoch and the six mode settings (with --momentum_teacher, which the mode leaves None) differ
trunk_parsed_70 = ssl_pretrain.parse_args(trunk_70[trunk_70.index("ssl_pretrain.py") + 1:])
for lock, stop in (("0", 70), ("1", 80)):
    ark_parsed = ssl_pretrain.parse_args(ark_argvs[lock][ark_argvs[lock].index("ssl_pretrain.py") + 1:])
    assert ssl_pretrain.check_args(ark_parsed) == stop and ark_parsed.epochs == 1000 and ark_parsed.no_gram and ark_parsed.resume
    assert tuple(getattr(ark_parsed, k) for k in X_ARK_KEYS) == (True, 0.8, True, "per_dataset", lock == "1", True) and ark_parsed.momentum_teacher is None
    assert tuple(getattr(ark_parsed, k) for k in RECIPE_KEYS) == (0.3, 2.0, True)
    differs = {"output_dir", "momentum_teacher", *X_ARK_KEYS} - ({"lock_half_pass"} if lock == "0" else set()) | ({"stop_after_epochs"} if stop != 70 else set())  # lock 0 = the trunk's False
    assert {k for k in vars(ark_parsed) if getattr(ark_parsed, k) != getattr(trunk_parsed_70, k)} == differs, (lock, differs)
ark_parsed = ssl_pretrain.parse_args(ark_argvs["0"][ark_argvs["0"].index("ssl_pretrain.py") + 1:])  # the default arm, for the smoke check below
# a missing or invalid stop file of either arm is refused naming itself (none of the others, which stay valid) before gaudi.sh is reached, and before any run: line; each arm reads only its own
for lock, env, own in (("0", ARK_ENV, ARK_STOP_FILE), ("1", ARK_LOCK_ENV, ARK_LOCK_STOP_FILE)):
    for stop in (None, "", "0", "2.5", "thirty", "-3", "1001", "18446744073709551617"):
        result, refused_argv = ark_run(ark_root, env=env, **{own: stop})
        assert result.returncode == 1 and "ERROR" in result.stderr and own in result.stderr and refused_argv is None and result.stdout == "", (lock, stop, result.stderr)
        others = {"constlr_stop_after.txt", "constlr_nogram_stop_after.txt", PLAIN_STOP_FILE, ARK_STOP_FILE, ARK_LOCK_STOP_FILE} - {own}
        assert not any("chain/" + other in result.stderr for other in others), (lock, stop, result.stderr)
    assert "ceiling of 1000" in result.stderr, result.stderr
    result, solo_argv = ark_run(ark_root, env=env, **{name: None for name in {"constlr_stop_after.txt", "constlr_nogram_stop_after.txt", PLAIN_STOP_FILE, ARK_STOP_FILE, ARK_LOCK_STOP_FILE} - {own}})
    assert result.returncode == 0 and solo_argv == ark_argvs[lock], (lock, result.stderr, solo_argv)
# the others are byte-unchanged: the phase, the no-Gram trunk and the plain trunk with the Ark stop files present (not read) or absent and CONSTLR_ARK / CONSTLR_BRANCH_ARK / CONSTLR_ARK_LOCK empty,
# and the no-Gram trunk's branch
ark_empties = ({}, dict(CONSTLR_ARK=""), dict(CONSTLR_BRANCH_ARK=""), dict(CONSTLR_ARK_LOCK=""), dict(CONSTLR_ARK="", CONSTLR_BRANCH_ARK="", CONSTLR_ARK_LOCK=""))
for ark_file in (" 70 \n", None):
    for empty in ark_empties:
        files = {ARK_STOP_FILE: ark_file, ARK_LOCK_STOP_FILE: ark_file}
        result, kept_argv = ark_run(ark_root, env=dict(ALLOC4, **empty), **files)
        assert result.returncode == 0 and kept_argv == constlr_argv("30") and (result.stdout, result.stderr) == ("", ""), (ark_file, empty, result.stderr, kept_argv)
        result, kept_argv = ark_run(ark_root, env=dict(NOGRAM_ENV, **empty), **files)
        assert result.returncode == 0 and kept_argv == nogram_argv_for("100") and result.stderr == "" and result.stdout.splitlines() == [NOGRAM_RUN_LINE % "100"], (ark_file, empty, result.stdout)
        result, kept_argv = ark_run(ark_root, env=dict(PLAIN_ENV, **empty), **files)
        assert result.returncode == 0 and kept_argv == plain_argv_for("60") and result.stderr == "" and result.stdout.startswith("run: ssl_dinov3_cxr_constlr_nogram_plain_seed_100 ("), (ark_file, empty)
for empty in ark_empties:
    bres_ark, kept_bargv = branch_run(branch_dir, "10", "40", env=dict(ALLOC4, **empty))
    assert bres_ark.returncode == 0 and kept_bargv == bargv and (bres_ark.stdout, bres_ark.stderr) == (bres.stdout, bres.stderr), (empty, bres_ark.stderr)
# the refusals, each before anything is read (ark_bare has no setting file to miss) and so before anything is launched (ark_root has them all), with its exact text
ark_bare = sandbox()
ARK_NEEDS_NOGRAM = "ERROR: CONSTLR_ARK=1 is set without CONSTLR_NOGRAM=1: the Ark-teacher trunk is the no-Gram trunk with the cyclic Ark+ teacher, so set both\n"
ARK_NO_PLAIN = "ERROR: CONSTLR_ARK and CONSTLR_PLAIN are both set: the Ark-teacher trunk is the no-Gram trunk with the DINOv3 recipe, so unset CONSTLR_PLAIN\n"
ARK_NO_BRANCH = ("ERROR: CONSTLR_ARK and CONSTLR_BRANCH_* are both set: a branch of the Ark-teacher trunk is CONSTLR_BRANCH_ARK=1 with the CONSTLR_BRANCH_* variables, so unset CONSTLR_ARK\n")
BRANCH_ARK_NEEDS_BRANCH = ("ERROR: CONSTLR_BRANCH_ARK=1 is set without the CONSTLR_BRANCH_* variables: it picks the Ark-teacher trunk for a branch (run_d1_8_branch_gram_gaudi.sh sets the four)\n")
ARK_LOCK_ALONE = "ERROR: CONSTLR_ARK_LOCK is set without CONSTLR_ARK=1 or CONSTLR_BRANCH_ARK=1: it picks the Ark-teacher trunk's lock arm, so set one of them\n"
ark_refusals = [(dict(NOGRAM_ENV, CONSTLR_ARK=value), "ERROR: CONSTLR_ARK holds '%s', which is not empty or 1\n" % value) for value in ("0", "yes", "2", " 1")]
ark_refusals += [(dict(ALLOC4, CONSTLR_BRANCH_ARK=value), "ERROR: CONSTLR_BRANCH_ARK holds '%s', which is not empty or 1\n" % value) for value in ("0", "yes", " 1")]
ark_refusals += [(dict(ARK_ENV, CONSTLR_ARK_LOCK=value), "ERROR: CONSTLR_ARK_LOCK holds '%s', which is not empty, 0 or 1\n" % value) for value in ("2", "yes", "true", " 1", "01", "-1")]
ark_refusals += [(dict(ALLOC4, CONSTLR_ARK_LOCK=value), ARK_LOCK_ALONE) for value in ("0", "1")] + [(dict(NOGRAM_ENV, CONSTLR_ARK_LOCK="1"), ARK_LOCK_ALONE),
                                                                                                   (dict(PLAIN_ENV, CONSTLR_ARK_LOCK="0"), ARK_LOCK_ALONE)]
ark_refusals += [(dict(ALLOC4, CONSTLR_ARK="1"), ARK_NEEDS_NOGRAM), (dict(ALLOC4, CONSTLR_ARK="1", CONSTLR_NOGRAM=""), ARK_NEEDS_NOGRAM), (dict(PLAIN_ENV, CONSTLR_ARK="1"), ARK_NO_PLAIN),
                 (dict(ALLOC4, CONSTLR_BRANCH_ARK="1"), BRANCH_ARK_NEEDS_BRANCH), (dict(ARK_ENV, CONSTLR_BRANCH_ARK="1"), BRANCH_ARK_NEEDS_BRANCH)]
ark_refusals += [(dict(ALLOC4, CONSTLR_ARK="1", **branch_vars), ARK_NO_BRANCH) for branch_vars in (dict(CONSTLR_BRANCH_EPOCH="10"), branch_all, dict(branch_all, CONSTLR_BRANCH_ARK="1"))]
for env, expected_error in ark_refusals:
    result = run_script(ark_bare, "chain/constlr.sh", env=env)
    assert result.returncode == 1 and result.stderr == expected_error and result.stdout == "" and launched(ark_bare) is None, (env, result.stderr)
    result, refused_argv = ark_run(ark_root, env=env)
    assert result.returncode == 1 and result.stderr == expected_error and result.stdout == "" and refused_argv is None, (env, result.stderr)
branch_refused(branch_root(), ("10", "40"), "CONSTLR_ARK", "CONSTLR_BRANCH_*", env=dict(ALLOC4, CONSTLR_ARK="1"))  # the branch wrapper sets all four: refused naming both
branch_refused(branch_root(), ("10", "40"), "CONSTLR_ARK_LOCK", env=dict(ALLOC4, CONSTLR_ARK_LOCK="1"))  # a lock arm without CONSTLR_BRANCH_ARK=1 names nothing to pick
# a branch of either Ark-teacher arm: CONSTLR_BRANCH_ARK=1 (and the arm's CONSTLR_ARK_LOCK) through the wrapper reads that arm's trunk (args.json: the cycle's epoch_length and its lock_half_pass, which
# must be the arm's) and source, and launches its line with --no_gram replaced by the branch pair, ARK_FLAGS kept after it; without the variables the same arguments name the no-Gram trunk's source


def ark_branch_root(lock_json={"0": False, "1": True}, epoch_length={"0": 2665, "1": 3997}):  # branch_root() plus both stub Ark-teacher trunks, each with its epoch-1 checkpoint
    root = branch_root()
    for lock, trunk in ARK_TRUNKS.items():
        os.makedirs(os.path.join(root, trunk, "ckpt"))
        record = dict(no_gram=True, cyclic_ark_teacher=True, heads="per_dataset", data_set=["VinDrCXR", "ChestXray14", "MIMIC", "CheXpert"], epoch_length=epoch_length[lock],
                      total_steps=1000 * epoch_length[lock])
        if lock_json.get(lock) is not None:
            record["lock_half_pass"] = lock_json[lock]
        with open(os.path.join(root, trunk, "args.json"), "w") as f:
            json.dump(record, f, indent=2)
        open(os.path.join(root, trunk, "ckpt", "epoch_0001.pth"), "w").close()
    return root


ark_branch_dir = ark_branch_root()
for lock, env, length in (("0", dict(ALLOC4, CONSTLR_BRANCH_ARK="1"), 2665), ("1", dict(ALLOC4, CONSTLR_BRANCH_ARK="1", CONSTLR_ARK_LOCK="1"), 3997)):
    trunk = ARK_TRUNKS[lock]
    run_ab, src_ab = "%s_e0001_gram_every%dx30_seed_100" % (trunk[:-len("_seed_100")], length), ARK_RUNS[lock] + "/ckpt/epoch_0001.pth"
    steps_ab = ",".join(str(length + length * i) for i in range(1, 31))  # the next 30 cycle boundaries
    bres_ab, ark_bargv = branch_run(ark_branch_dir, "1", "40", str(length), "30", env=env)
    assert bres_ab.returncode == 0 and bres_ab.stderr == "", (lock, bres_ab.stderr)
    assert ark_bargv == [("/workspace/" + run_ab) if arg == ARK_RUNS[lock] else arg for arg in ark_argv_for("40", lock)[:-6]] + [
        "--branch_from", src_ab, "--gram_refresh_steps", steps_ab] + ARK_FLAGS_OF[lock], (lock, ark_bargv)
    assert ("_lock_e0001_" in run_ab) == (lock == "1") and bres_ab.stdout.splitlines() == [
        "run: %s (Gram-anchored branch of %s from its epoch 1: --branch_from %s --gram_refresh_steps in place of --no_gram; stop after epoch 40)" % (run_ab, trunk, src_ab),
        "gram refresh steps: %s (branch step %d = epoch 1 x epoch_length %d from %s/args.json, then every %d steps, 30 times)" % (steps_ab, length, length, trunk, length)], bres_ab.stdout
    expanded["CONSTLR_BRANCH_ARK=1 CONSTLR_ARK_LOCK=%s run_d1_8_branch_gram_gaudi.sh 1 40 %d 30" % (lock, length)] = "./gaudi.sh " + " ".join(ark_bargv)
    result, ark_trunk_40 = ark_run(sandbox(), env=dict(ARK_ENV, CONSTLR_ARK_LOCK=lock), **{ARK_LOCK_STOP_FILE if lock == "1" else ARK_STOP_FILE: "40\n"})  # the real trunk line of the arm at stop 40
    assert result.returncode == 0 and ark_trunk_40 == ark_argv_for("40", lock)
    ark_cut = ark_trunk_40.index("--no_gram")
    assert ark_bargv[:ark_cut] == [("/workspace/" + run_ab) if arg == ARK_RUNS[lock] else arg for arg in ark_trunk_40[:ark_cut]] and ark_bargv[ark_cut + 4:] == ark_trunk_40[ark_cut + 1:] == ARK_FLAGS_OF[lock]
    ark_bparsed = ssl_pretrain.parse_args(ark_bargv[ark_bargv.index("ssl_pretrain.py") + 1:])
    ark_tparsed = ssl_pretrain.parse_args(ark_trunk_40[ark_trunk_40.index("ssl_pretrain.py") + 1:])
    assert (ark_bparsed.branch_from, ark_bparsed.gram_refresh_steps, ark_bparsed.no_gram, ark_bparsed.lock_half_pass) == (src_ab, [length * i for i in range(2, 32)], False, lock == "1")
    assert all(getattr(ark_bparsed, k) == getattr(ark_tparsed, k) for k in branch_locked_alike), \
        {k: (getattr(ark_bparsed, k), getattr(ark_tparsed, k)) for k in branch_locked_alike if getattr(ark_bparsed, k) != getattr(ark_tparsed, k)}  # what --branch_from checks against the source
    branch_refused(ark_branch_dir, ("5", "40"), trunk + "/ckpt/epoch_0005.pth", "nothing to branch from", env=env)
branch_refused(ark_branch_dir, ("1", "40"), BRANCH_TRUNK + "/ckpt/epoch_0001.pth", "nothing to branch from")  # without CONSTLR_BRANCH_ARK=1: the no-Gram trunk's source
# the branch inherits its trunk's lock: a trunk whose args.json records the other arm's lock_half_pass, or none, is refused naming it, before anything is launched
for lock_json, env, words in (({"0": True}, dict(ALLOC4, CONSTLR_BRANCH_ARK="1"), ["lock_half_pass 'true'", "not the false of CONSTLR_ARK_LOCK=0"]),
                              ({"1": False}, dict(ALLOC4, CONSTLR_BRANCH_ARK="1", CONSTLR_ARK_LOCK="1"), ["lock_half_pass 'false'", "not the true of CONSTLR_ARK_LOCK=1"]),
                              ({"0": None}, dict(ALLOC4, CONSTLR_BRANCH_ARK="1"), ["lock_half_pass ''"])):
    branch_refused(ark_branch_root(lock_json=dict({"0": False, "1": True}, **lock_json)), ("1", "40", "2665", "30"), "args.json", *words, env=env)
# the wrappers: bash -n, the executable bit, the sbatch headers, the code (the trunk's: the no-Gram trunk's wrapper but for its last line), the comments' recipe, arms, reasons and stop files, and a copy
# with its cd pointed at a sandbox launching either arm's line from outside the repo; the smoke job launches the arm's flags for 2300 steps into the arm's throwaway run dir
ARK_SCRIPT, ARK_SMOKE = os.path.join(REPO, "run_d1_8_constlr_arkteacher_gaudi.sh"), os.path.join(REPO, "run_d1_8_arkteacher_smoke_gaudi.sh")
for script in (ARK_SCRIPT, ARK_SMOKE):
    assert os.access(script, os.X_OK) and stat.S_IMODE(os.stat(script).st_mode) == 0o755, script
    subprocess.run(["bash", "-n", script], check=True)
ark_text = open(ARK_SCRIPT).read()
ark_lines = ark_text.splitlines()
assert [line[len("#SBATCH "):] for line in ark_lines if line.startswith("#SBATCH ")] == [
    "--job-name=d1-8-arkteacher", "-A grp_jliang12", "-p gaudi", "-q public", "-t 7-00:00:00", "-N 1", "--gres=gpu:hl225:4", "-c 72", "--mem=128G"], ark_lines
ark_code = [line for line in ark_lines if line and not line.startswith("#")]
assert ark_code[-1] == "CONSTLR_NOGRAM=1 CONSTLR_ARK=1 exec ./chain/constlr.sh" and ark_code[:-1] == nogram_code[:-1], ark_code
assert all(line.startswith("#") for line in ark_lines if "phases.txt" in line), ark_lines
for needle in ("chain/constlr_arkteacher_stop_after.txt", "chain/constlr_arkteacher_lock_stop_after.txt", "ssl_dinov3_cxr_constlr_arkteacher_lock_seed_100", "CONSTLR_ARK_LOCK=1", "largest",
               "0.80 x teacher + 0.20 x student", "clean views", "per dataset", "a random half", "Gram off", "cycle 1", "4.3 GB"):
    assert needle in ark_text, needle
ark_wrap_root = sandbox(timing=True)
open(os.path.join(ark_wrap_root, "chain", "main_lr.txt"), "w").write("0.002\n")
open(os.path.join(ark_wrap_root, "chain", ARK_STOP_FILE), "w").write("70\n")  # the arms' stop files only
open(os.path.join(ark_wrap_root, "chain", ARK_LOCK_STOP_FILE), "w").write("80\n")
assert ark_text.count(BRANCH_CD) == 1, BRANCH_CD
open(os.path.join(ark_wrap_root, "ark_wrapper.sh"), "w").write(ark_text.replace(BRANCH_CD, "cd %s || exit 1" % shlex.quote(ark_wrap_root)))
for lock, env, stop in (("0", ALLOC4, "70"), ("1", dict(ALLOC4, CONSTLR_ARK_LOCK="1"), "80")):
    if os.path.exists(os.path.join(ark_wrap_root, "argv.txt")):
        os.remove(os.path.join(ark_wrap_root, "argv.txt"))
    result = subprocess.run(["bash", os.path.join(ark_wrap_root, "ark_wrapper.sh")], cwd=branch_elsewhere, env=env, capture_output=True, text=True)
    assert result.returncode == 0 and result.stderr == "" and launched(ark_wrap_root) == ark_argv_for(stop, lock) and result.stdout.splitlines() == [ark_run_line(lock, stop)], (lock, result.stderr)
assert os.path.exists(os.path.join(ark_wrap_root, "cards_checked"))
smoke_text = open(ARK_SMOKE).read()
assert [line[len("#SBATCH "):] for line in smoke_text.splitlines() if line.startswith("#SBATCH ")] == [
    "--job-name=d1-8-arkteacher-smoke", "-A grp_jliang12", "-p gaudi", "-q public", "-t 01:15:00", "-N 1", "--gres=gpu:hl225:4", "-c 72", "--mem=128G"]
smoke_root = sandbox(timing=True)
shutil.copy(ARK_SMOKE, smoke_root)
for lock, smoke_env, smoke_dir in (("0", dict(ALLOC4, ARK_ROOT=smoke_root), "ssl_dinov3_cxr_arkteacher_smoke_seed_100"),
                                   ("0", dict(ALLOC4, ARK_ROOT=smoke_root, CONSTLR_ARK_LOCK="0"), "ssl_dinov3_cxr_arkteacher_smoke_seed_100"),
                                   ("1", dict(ALLOC4, ARK_ROOT=smoke_root, CONSTLR_ARK_LOCK="1"), "ssl_dinov3_cxr_arkteacher_lock_smoke_seed_100")):
    if os.path.exists(os.path.join(smoke_root, "argv.txt")):
        os.remove(os.path.join(smoke_root, "argv.txt"))
    result = run_script(smoke_root, os.path.join(smoke_root, "run_d1_8_arkteacher_smoke_gaudi.sh"), cwd=branch_elsewhere, env=smoke_env)
    smoke_argv = launched(smoke_root)
    assert result.returncode == 0 and smoke_argv == COMMON + ["/workspace/" + smoke_dir, "--resume", *RECIPE, "--no_gram", *ARK_FLAGS_OF[lock], "--lr", "0.002", "--min_lr", "0.002",
                                                             "--no_lr_scaling", "--epochs", "1000", "--max_steps", "2300"], (lock, result.stderr, smoke_argv)
    smoke_parsed = ssl_pretrain.parse_args(smoke_argv[smoke_argv.index("ssl_pretrain.py") + 1:])
    trunk_arm = ssl_pretrain.parse_args(ark_argvs[lock][ark_argvs[lock].index("ssl_pretrain.py") + 1:])
    assert {k for k in vars(smoke_parsed) if getattr(smoke_parsed, k) != getattr(trunk_arm, k)} == {"output_dir", "max_steps", "stop_after_epochs"}  # the arm's trunk line, cut short
    expanded["run_d1_8_arkteacher_smoke_gaudi.sh (CONSTLR_ARK_LOCK=%s)" % lock] = "./gaudi.sh " + " ".join(smoke_argv)
# 2300 steps cross the first dataset boundary in both arms at 64 x 4: MIMIC's release alone (1441), or its lock + release (720 + 1441 = 2161), each followed by CheXpert's first pass
assert 368960 // 4 // 64 == 1441 and 368960 // 2 // 4 // 64 == 720 and 1441 < 720 + 1441 < 2300 < 1441 + 872 and 720 + 1441 + 223414 // 2 // 4 // 64 > 2300
for value in ("2", "yes", " 1"):
    os.remove(os.path.join(smoke_root, "argv.txt"))
    result = run_script(smoke_root, os.path.join(smoke_root, "run_d1_8_arkteacher_smoke_gaudi.sh"), cwd=branch_elsewhere, env=dict(ALLOC4, ARK_ROOT=smoke_root, CONSTLR_ARK_LOCK=value))
    assert result.returncode == 1 and result.stderr == "ERROR: CONSTLR_ARK_LOCK holds '%s', which is not empty, 0 or 1\n" % value and launched(smoke_root) is None, (value, result.stderr)
    open(os.path.join(smoke_root, "argv.txt"), "w").close()
for needle in ("cycle (--cyclic_ark_teacher)", "dataset pass after step 0: MIMIC lock", "teacher blend after step 1441 [2161]", "--max_steps 2300 reached at step 2300",
               "ssl_dinov3_cxr_arkteacher_lock_smoke_seed_100"):
    assert needle in smoke_text, needle  # the expected log lines it documents
print("OK (k) Ark-teacher trunk: constlr.sh with CONSTLR_NOGRAM=1 CONSTLR_ARK=1 (run_d1_8_constlr_arkteacher_gaudi.sh: mode 755, 4 cards, the no-Gram trunk's wrapper but for the job name, the comments and "
      "CONSTLR_ARK=1, names phases.txt only in comments, launches both arms from outside the repo) launches the no-Gram trunk's line with --cyclic_ark_teacher --heads per_dataset <lock flag> "
      "--keep_all_checkpoints appended; CONSTLR_ARK_LOCK picks the arm (unset, empty or 0: --no_lock_half_pass into ssl_dinov3_cxr_constlr_arkteacher_seed_100 with constlr_arkteacher_stop_after.txt; 1: "
      "--lock_half_pass into ssl_dinov3_cxr_constlr_arkteacher_lock_seed_100 with constlr_arkteacher_lock_stop_after.txt; the two lines differ in just those three), each arm reads only its own stop file "
      "(refused naming itself) and prints the flags and the arm in its run: line; the argv parses to the trunk's settings but the run dir and the mode; CONSTLR_ARK other than empty or 1, without "
      "CONSTLR_NOGRAM=1, with CONSTLR_PLAIN or any CONSTLR_BRANCH_*, CONSTLR_BRANCH_ARK other than empty or 1 or without the CONSTLR_BRANCH_*, CONSTLR_ARK_LOCK other than empty, 0 or 1 or set without "
      "CONSTLR_ARK / CONSTLR_BRANCH_ARK are refused with their exact text before anything is read; CONSTLR_BRANCH_ARK=1 through the branch wrapper (`1 40 <epoch_length> 30`) launches the arm's trunk line "
      "with --no_gram replaced by --branch_from its epoch_0001 --gram_refresh_steps at the next 30 cycle boundaries (2665 / 3997 from its args.json), ARK_FLAGS kept, the _lock token in the lock arm's "
      "branch dir, every --branch_from-checked setting = the trunk's, and a trunk args.json with the other lock_half_pass or none is refused; the phase, the no-Gram trunk, the plain trunk and the no-Gram "
      "branches launch and print exactly as before with the three variables empty and the Ark stop files present or absent; run_d1_8_arkteacher_smoke_gaudi.sh (75 min) launches the arm's flags for 2300 "
      "steps (past MIMIC's release, 1441, or lock + release, 2161) into ssl_dinov3_cxr_arkteacher[_lock]_smoke_seed_100 and refuses another CONSTLR_ARK_LOCK")

# the Gram teacher from an earlier trunk epoch (DINOv3's early-teacher rule): CONSTLR_BRANCH_GRAM_EPOCH=J, the branch wrapper's fifth argument GRAM_EPOCH. The line is the four-argument one plus
# --gram_teacher_from <trunk>/ckpt/epoch_<J %04d>.pth (last), the run dir gets _gt<J %04d> right after _e<K %04d>, the run: line names it; the no-Gram trunk and both Ark-teacher arms alike
gt_dir = branch_root()  # holds the trunk's epoch_0010.pth; the planned branch point epoch_0100.pth is added
open(os.path.join(gt_dir, BRANCH_TRUNK, "ckpt", "epoch_0100.pth"), "w").close()
GT_RUN4, GT_RUN5 = "ssl_dinov3_cxr_constlr_nogram_e0100_gram_every2666x3_seed_100", "ssl_dinov3_cxr_constlr_nogram_e0100_gt0010_gram_every2666x3_seed_100"
GT_SRC100 = NOGRAM_RUN + "/ckpt/epoch_0100.pth"
GT_REFRESH_LINE = ("gram refresh steps: 269266,271932,274598 (branch step 266600 = epoch 100 x epoch_length 2666 from ssl_dinov3_cxr_constlr_nogram_seed_100/args.json, then every 2666 steps, "
                   "3 times)")
gres4, gargv4 = branch_run(gt_dir, "100", "130", "2666", "3")
gres5, gargv5 = branch_run(gt_dir, "100", "130", "2666", "3", "10")
assert gres4.returncode == 0 == gres5.returncode and gres4.stderr == "" == gres5.stderr, (gres4.stderr, gres5.stderr)
assert gargv4 == branch_argv_for("130", GT_RUN4, GT_SRC100, "269266,271932,274598") and gargv4.count("/workspace/" + GT_RUN4) == 1, gargv4  # the four-argument line, as before
assert gargv5 == [("/workspace/" + GT_RUN5) if arg == "/workspace/" + GT_RUN4 else arg for arg in gargv4] + ["--gram_teacher_from", BRANCH_SRC10], gargv5
assert gres4.stdout.splitlines() == [
    "run: ssl_dinov3_cxr_constlr_nogram_e0100_gram_every2666x3_seed_100 (Gram-anchored branch of ssl_dinov3_cxr_constlr_nogram_seed_100 from its epoch 100: --branch_from "
    "/workspace/ssl_dinov3_cxr_constlr_nogram_seed_100/ckpt/epoch_0100.pth --gram_refresh_steps in place of --no_gram; stop after epoch 130)", GT_REFRESH_LINE], gres4.stdout
assert gres5.stdout.splitlines() == [
    "run: ssl_dinov3_cxr_constlr_nogram_e0100_gt0010_gram_every2666x3_seed_100 (Gram-anchored branch of ssl_dinov3_cxr_constlr_nogram_seed_100 from its epoch 100: --branch_from "
    "/workspace/ssl_dinov3_cxr_constlr_nogram_seed_100/ckpt/epoch_0100.pth --gram_refresh_steps in place of --no_gram; Gram teacher from trunk epoch 10 (ckpt/epoch_0010.pth), DINOv3's "
    "early-teacher rule; stop after epoch 130)", GT_REFRESH_LINE], gres5.stdout
expanded["run_d1_8_branch_gram_gaudi.sh 100 130 2666 3"] = "./gaudi.sh " + " ".join(gargv4)
expanded["run_d1_8_branch_gram_gaudi.sh 100 130 2666 3 10"] = "./gaudi.sh " + " ".join(gargv5)
# the trainer reads the two lines alike but for the run dir and the Gram teacher's file
gparsed4, gparsed5 = (ssl_pretrain.parse_args(argv[argv.index("ssl_pretrain.py") + 1:]) for argv in (gargv4, gargv5))
assert (gparsed5.gram_teacher_from, gparsed5.branch_from, gparsed4.gram_teacher_from) == (BRANCH_SRC10, GT_SRC100, None)
assert {k for k in vars(gparsed5) if getattr(gparsed5, k) != getattr(gparsed4, k)} == {"output_dir", "gram_teacher_from"}
# without a fifth argument nothing changes: an empty one, an inherited CONSTLR_BRANCH_GRAM_EPOCH (the wrapper's last line clears it) and constlr.sh with the variable empty launch and print the
# four-argument line; so does the planned `10 40`
for args, env in (((), dict(ALLOC4, CONSTLR_BRANCH_GRAM_EPOCH="")), (("",), ALLOC4), ((), dict(ALLOC4, CONSTLR_BRANCH_GRAM_EPOCH="10"))):
    gres, gargv = branch_run(gt_dir, "100", "130", "2666", "3", *args, env=env)
    assert gres.returncode == 0 and gargv == gargv4 and (gres.stdout, gres.stderr) == (gres4.stdout, ""), (args, gres.stderr, gargv)
gres, gargv = branch_run(branch_root(), "10", "40", env=dict(ALLOC4, CONSTLR_BRANCH_GRAM_EPOCH="5"))
assert gres.returncode == 0 and gargv == branch_argv_for("40", BRANCH_E10, BRANCH_SRC10, "36660,46660,56660") and "_gt" not in gres.stdout, (gres.stderr, gargv)
gt_env4 = dict(ALLOC4, CONSTLR_BRANCH_EPOCH="100", CONSTLR_BRANCH_STOP="130", CONSTLR_BRANCH_EVERY="2666", CONSTLR_BRANCH_COUNT="3")
for env, expected in ((dict(gt_env4, CONSTLR_BRANCH_GRAM_EPOCH=""), gargv4), (dict(gt_env4, CONSTLR_BRANCH_GRAM_EPOCH="10"), gargv5)):  # constlr.sh itself, without the wrapper
    if os.path.exists(os.path.join(gt_dir, "argv.txt")):
        os.remove(os.path.join(gt_dir, "argv.txt"))
    gres = run_script(gt_dir, "chain/constlr.sh", env=env)
    assert gres.returncode == 0 and launched(gt_dir) == expected, (gres.stderr, launched(gt_dir))
# both Ark-teacher arms (CONSTLR_BRANCH_ARK=1; every epoch checkpoint kept): `5 40 <epoch_length> 3 2` is `5 40 <epoch_length> 3` plus --gram_teacher_from <arm trunk>/ckpt/epoch_0002.pth, after
# ARK_FLAGS, with the same _gt0002 token
gt_ark_dir = ark_branch_root()
for lock, env, length in (("0", dict(ALLOC4, CONSTLR_BRANCH_ARK="1"), 2665), ("1", dict(ALLOC4, CONSTLR_BRANCH_ARK="1", CONSTLR_ARK_LOCK="1"), 3997)):
    trunk = ARK_TRUNKS[lock]
    for epoch in (2, 5):
        open(os.path.join(gt_ark_dir, trunk, "ckpt", "epoch_%04d.pth" % epoch), "w").close()
    ark_run4, ark_run5 = ("%s_e0005%s_gram_every%dx3_seed_100" % (trunk[:-len("_seed_100")], token, length) for token in ("", "_gt0002"))
    ark_src5, ark_src2 = ARK_RUNS[lock] + "/ckpt/epoch_0005.pth", ARK_RUNS[lock] + "/ckpt/epoch_0002.pth"
    ark_steps = ",".join(str(5 * length + length * i) for i in range(1, 4))
    ares4, aargv4 = branch_run(gt_ark_dir, "5", "40", str(length), "3", env=env)
    ares5, aargv5 = branch_run(gt_ark_dir, "5", "40", str(length), "3", "2", env=env)
    assert ares4.returncode == 0 == ares5.returncode and ares4.stderr == "" == ares5.stderr, (lock, ares4.stderr, ares5.stderr)
    assert aargv4 == [("/workspace/" + ark_run4) if arg == ARK_RUNS[lock] else arg for arg in ark_argv_for("40", lock)[:-6]] + [
        "--branch_from", ark_src5, "--gram_refresh_steps", ark_steps] + ARK_FLAGS_OF[lock], (lock, aargv4)  # the four-argument line, as before
    assert aargv5 == [("/workspace/" + ark_run5) if arg == "/workspace/" + ark_run4 else arg for arg in aargv4] + ["--gram_teacher_from", ark_src2], (lock, aargv5)
    assert ("_lock_e0005_gt0002_" in ark_run5) == (lock == "1") and ares5.stdout.splitlines() == [
        "run: %s (Gram-anchored branch of %s from its epoch 5: --branch_from %s --gram_refresh_steps in place of --no_gram; Gram teacher from trunk epoch 2 (ckpt/epoch_0002.pth), DINOv3's "
        "early-teacher rule; stop after epoch 40)" % (ark_run5, trunk, ark_src5), ares4.stdout.splitlines()[1]], ares5.stdout
    aparsed4, aparsed5 = (ssl_pretrain.parse_args(argv[argv.index("ssl_pretrain.py") + 1:]) for argv in (aargv4, aargv5))
    assert aparsed5.gram_teacher_from == ark_src2 and {k for k in vars(aparsed5) if getattr(aparsed5, k) != getattr(aparsed4, k)} == {"output_dir", "gram_teacher_from"}
    expanded["CONSTLR_BRANCH_ARK=1 CONSTLR_ARK_LOCK=%s run_d1_8_branch_gram_gaudi.sh 5 40 %d 3 2" % (lock, length)] = "./gaudi.sh " + " ".join(aargv5)
# refusals with their exact text, before anything is launched: the variable without the four (before anything is read: the bare sandbox has no setting file to miss; gt_dir has them all)
GT_ALONE = ("ERROR: CONSTLR_BRANCH_GRAM_EPOCH is set without the CONSTLR_BRANCH_* variables: it picks the trunk epoch a branch's Gram teacher is taken from (run_d1_8_branch_gram_gaudi.sh sets it from "
            "its fifth argument)\n")
gt_bare = sandbox()
for env in (dict(ALLOC4, CONSTLR_BRANCH_GRAM_EPOCH="10"), dict(ALLOC4, CONSTLR_BRANCH_GRAM_EPOCH="x"), dict(NOGRAM_ENV, CONSTLR_BRANCH_GRAM_EPOCH="10"),
            dict(ARK_ENV, CONSTLR_BRANCH_GRAM_EPOCH="10"), dict(ALLOC4, CONSTLR_BRANCH_ARK="", CONSTLR_BRANCH_GRAM_EPOCH="1")):
    gres = run_script(gt_bare, "chain/constlr.sh", env=env)
    assert gres.returncode == 1 and gres.stderr == GT_ALONE and gres.stdout == "" and launched(gt_bare) is None, (env, gres.stderr)
    gres, gargv = ark_run(ark_root, env=env)  # every stop file there: it would have launched
    assert gres.returncode == 1 and gres.stderr == GT_ALONE and gres.stdout == "" and gargv is None, (env, gres.stderr)


def gt_refused(root, args, expected_error, env=ALLOC4):  # the wrapper copy refuses: exit 1, exactly this stderr, nothing printed, gaudi.sh never reached
    gres, gargv = branch_run(root, *args, env=env)
    assert gres.returncode == 1 and gres.stderr == expected_error and gres.stdout == "" and gargv is None, (args, gres.returncode, gres.stdout, gres.stderr)


for value in ("abc", "0", "-1", "1.5", "010", " 10", "1e1"):  # 010: printf %04d would read it as octal
    gt_refused(gt_dir, ("100", "130", "2666", "3", value), "ERROR: J (CONSTLR_BRANCH_GRAM_EPOCH) holds '%s', which is not a positive integer without a leading zero (the trunk epoch the Gram teacher "
                                                           "is taken from)\n" % value)
for value in ("100", "101", "1000", "18446744073709551617"):  # J = K, after it, and 2**64 + 1 (bash arithmetic would read it as 1)
    gt_refused(gt_dir, ("100", "130", "2666", "3", value), "ERROR: J (CONSTLR_BRANCH_GRAM_EPOCH) holds '%s', which is not before K (CONSTLR_BRANCH_EPOCH) '100': the Gram teacher is taken from an "
                                                           "earlier trunk epoch than the branch point (DINOv3's early-teacher rule)\n" % value)
gt_refused(gt_dir, ("100", "130", "2666", "3", "10", "x"), "ERROR: usage: sbatch run_d1_8_branch_gram_gaudi.sh <K> <STOP> [EVERY=10000] [COUNT=3] [GRAM_EPOCH]; got 6 arguments\n")
# a missing ckpt/epoch_<J %04d>.pth, as a missing source: refused while the branch's own dir (the _gt one: the dir without the token is another run) has no epoch checkpoint; with one it launches
GT_MISSING = ("ERROR: ssl_dinov3_cxr_constlr_nogram_seed_100/ckpt/epoch_0020.pth not found and ssl_dinov3_cxr_constlr_nogram_e0100_gt0020_gram_every2666x3_seed_100 has no epoch checkpoint of its "
              "own yet: no Gram teacher to take from trunk epoch 20 (the trunk keeps every 10th epoch and its last 3)\n")
gt_refused(gt_dir, ("100", "130", "2666", "3", "20"), GT_MISSING)
os.makedirs(os.path.join(gt_dir, GT_RUN4, "ckpt"))
open(os.path.join(gt_dir, GT_RUN4, "ckpt", "epoch_0101.pth"), "w").close()
gt_refused(gt_dir, ("100", "130", "2666", "3", "20"), GT_MISSING)
os.makedirs(os.path.join(gt_dir, "ssl_dinov3_cxr_constlr_nogram_e0100_gt0020_gram_every2666x3_seed_100", "ckpt"))
open(os.path.join(gt_dir, "ssl_dinov3_cxr_constlr_nogram_e0100_gt0020_gram_every2666x3_seed_100", "ckpt", "epoch_0101.pth"), "w").close()
gres, gargv = branch_run(gt_dir, "100", "130", "2666", "3", "20")
assert gres.returncode == 0 and gargv[-2:] == ["--gram_teacher_from", NOGRAM_RUN + "/ckpt/epoch_0020.pth"] and "_gt0020_" in gargv[gargv.index("--output_dir") + 1], (gres.stderr, gargv)
gt_refused(gt_ark_dir, ("5", "40", "2665", "3", "3"), "ERROR: ssl_dinov3_cxr_constlr_arkteacher_seed_100/ckpt/epoch_0003.pth not found and ssl_dinov3_cxr_constlr_arkteacher_e0005_gt0003_gram_every2665x3"
           "_seed_100 has no epoch checkpoint of its own yet: no Gram teacher to take from trunk epoch 3 (the trunk keeps every 10th epoch and its last 3)\n", env=dict(ALLOC4, CONSTLR_BRANCH_ARK="1"))
# the wrapper's comments: the usage line, the paper's rule, the planned line, the resubmit hazard, and that without the fifth argument nothing changes
branch_text = open(BRANCH_SCRIPT).read()
for needle in ("# Usage: sbatch run_d1_8_branch_gram_gaudi.sh <K> <STOP> [EVERY] [COUNT] [GRAM_EPOCH]", "DINOv3's rule (Sec. 4.2; App. C p. 58)", "w_Gram 2", "every 10k steps, at most three times",
               "sbatch run_d1_8_branch_gram_gaudi.sh 100 130 2666 3 10  -> ssl_dinov3_cxr_constlr_nogram_e0100_gt0010_gram_every2666x3_seed_100", "269266,271932,274598",
               "Pass the same five arguments on every re-submit", "Without the fifth argument nothing changes"):
    assert any(needle in line for line in branch_text.splitlines() if line.startswith("#")), needle
print("OK (k) Gram teacher from an earlier trunk epoch: run_d1_8_branch_gram_gaudi.sh's fifth argument J (CONSTLR_BRANCH_GRAM_EPOCH) launches the four-argument line plus --gram_teacher_from "
      "<trunk>/ckpt/epoch_<J %04d>.pth, last, in the run dir with _gt<J %04d> after _e<K %04d> (`100 130 2666 3 10` -> " + GT_RUN5 + ", refreshes 269266,271932,274598), the run: line naming J "
      "and the file, the argv parsing to the four-argument one's but for output_dir and gram_teacher_from; the same for both Ark-teacher arms (`5 40 <epoch_length> 3 2`, after ARK_FLAGS, _gt0002 "
      "after _lock_e0005 in the lock arm); without a fifth argument (none, an empty one, an inherited variable, constlr.sh with it empty) the four-argument line and lines, byte for byte; refused with "
      "their exact text before launching: the variable without the four (before anything is read), J not a positive integer without a leading zero, J not below K, six arguments, and a missing "
      "ckpt/epoch_<J %04d>.pth while the _gt branch dir has no epoch checkpoint (the token-less dir's does not count; the _gt dir's own lets it launch); the wrapper's comments carry the usage, the "
      "rule, the planned line and the resubmit hazard")

# the one-flag-off recipe ablations ("DINO-X ConstLR 224, no Gram, one recipe flag off"): CONSTLR_NOGRAM=1 CONSTLR_ABLATE=drop_path | rope | local_norm (what run_d1_8_constlr_nogram_ablate_gaudi.sh sets, the arm
# coming from its sbatch line) launches the no-Gram trunk's line (RECIPE, --no_gram) with exactly that one RECIPE flag and its value removed, into the arm's own run dir with the arm's own cluster-only stop file (no
# other stop file is read). Refused before anything is read: another value, CONSTLR_NOGRAM unset, CONSTLR_PLAIN, CONSTLR_ARK, any CONSTLR_BRANCH_*; the wrapper also refuses to start without an arm. The phase,
# the no-Gram trunk, the plain trunk, both Ark-teacher arms and the no-Gram trunk's branches launch and print exactly as before
ABLATE_ARMS = {  # arm: (the RECIPE words it removes, the words that stay as the run: line prints them, the trainer's (drop_path_rate, rope_rescale_coords, untie_local_cls_norm), the one key that reads as its default)
    "drop_path": (["--drop_path_rate", "0.3"], "--rope_rescale_coords 2 --untie_local_cls_norm", (0.0, 2.0, True), "drop_path_rate"),
    "rope": (["--rope_rescale_coords", "2"], "--drop_path_rate 0.3 --untie_local_cls_norm", (0.3, None, True), "rope_rescale_coords"),
    "local_norm": (["--untie_local_cls_norm"], "--drop_path_rate 0.3 --rope_rescale_coords 2", (0.3, 2.0, False), "untie_local_cls_norm")}
ABLATE_STOPS = {"drop_path": "91", "rope": "92", "local_norm": "93"}  # each arm's stop epoch, in its file as " 91 \n": the argv shows which file was read (the phase's 30, the no-Gram trunk's 100, the plain 60, the Ark arms' 70 / 80)
ABLATE_FILES = {arm: "constlr_nogram_no_%s_stop_after.txt" % arm for arm in ABLATE_ARMS}
ABLATE_RUNS = {arm: "/workspace/ssl_dinov3_cxr_constlr_nogram_no_%s_seed_100" % arm for arm in ABLATE_ARMS}
ABLATE_ENVS = {arm: dict(NOGRAM_ENV, CONSTLR_ABLATE=arm) for arm in ABLATE_ARMS}
ABLATE_TRAINER_LINES = {"drop_path": "drop path 0, RoPE coordinate rescale 2, local-crop CLS norm untied", "rope": "drop path 0.3, RoPE coordinate rescale none, local-crop CLS norm untied",
                        "local_norm": "drop path 0.3, RoPE coordinate rescale 2, local-crop CLS norm tied"}  # the recipe (...) line the trainer prints for each arm (the f-string in run())
ALL_STOP_FILES = {"constlr_stop_after.txt", "constlr_nogram_stop_after.txt", PLAIN_STOP_FILE, ARK_STOP_FILE, ARK_LOCK_STOP_FILE, *ABLATE_FILES.values()}
ALL_RUNS = ["/workspace/ssl_dinov3_cxr_constlr_seed_100", NOGRAM_RUN, PLAIN_RUN, *ARK_RUNS.values(), *ABLATE_RUNS.values()]
assert len(ALL_STOP_FILES) == len(ALL_RUNS) == len(set(ALL_RUNS)) == 8  # the phase, the no-Gram trunk, the plain trunk, the two Ark arms and the three ablation arms: no run dir or stop file shared by any two
PLAIN_RUN_LINE = ("run: ssl_dinov3_cxr_constlr_nogram_plain_seed_100 (no Gram teacher: --no_gram replaces --gram_refresh_steps 10000,20000,30000; plain: no DINOv3 recipe flags, "
                  "the one-variable Gram-off control of ssl_dinov3_cxr_constlr_seed_100; stop after epoch 60 from chain/constlr_nogram_plain_stop_after.txt)")


def constlr_vars(env):  # for assertion messages: the CONSTLR_ settings, not the whole environment (it holds the caller's tokens)
    return {k: v for k, v in env.items() if k.startswith("CONSTLR_")}


def ablate_run(root, env, **override):  # ark_run with the three arms' stop files written too, " 91 \n" ... (None: absent); the argv shows which stop file was read
    return ark_run(root, env=env, **dict({ABLATE_FILES[arm]: " %s \n" % ABLATE_STOPS[arm] for arm in ABLATE_ARMS}, **override))


def without(argv, words):  # argv minus a run of consecutive words, which must be in it exactly once
    at = [i for i in range(len(argv) - len(words) + 1) if argv[i:i + len(words)] == words]
    assert len(at) == 1 and argv.count(words[0]) == 1, (words, at)
    return argv[:at[0]] + argv[at[0] + len(words):]


def ablate_argv_for(arm, stop):  # nogram_argv_for(stop) in the arm's run dir without the arm's flag (and its value), nothing else changed
    return [ABLATE_RUNS[arm] if arg == NOGRAM_RUN else arg for arg in without(nogram_argv_for(stop), ABLATE_ARMS[arm][0])]


ablate_root = sandbox()
ablate_argvs = {}
for arm, (off, kept, recipe_set, reset_key) in ABLATE_ARMS.items():
    stop = ABLATE_STOPS[arm]
    result, argv = ablate_run(ablate_root, ABLATE_ENVS[arm])  # the arm's stop file " 9x ": the other seven are not read
    assert result.returncode == 0 and result.stderr == "" and argv == ablate_argv_for(arm, stop), (arm, result.stderr, argv)
    assert result.stdout.splitlines() == ["run: ssl_dinov3_cxr_constlr_nogram_no_%s_seed_100 (no Gram teacher: --no_gram replaces --gram_refresh_steps 10000,20000,30000; recipe ablation: %s off (%s removed); "
                                          "DINOv3 recipe otherwise: %s; stop after epoch %s from chain/constlr_nogram_no_%s_stop_after.txt)" % (arm, arm, " ".join(off), kept, stop, arm)], result.stdout
    # the line spelled out, not derived: the arm's run dir and stop epoch, the two flags that stay and then --no_gram at the end, and neither the removed flag nor a Gram refresh list anywhere
    assert after(argv, "--output_dir") == ABLATE_RUNS[arm] and after(argv, "--stop_after_epochs") == stop and argv[-1 - len(kept.split()):] == kept.split() + ["--no_gram"], (arm, argv)
    assert off[0] not in argv and "--gram_refresh_steps" not in argv and all(argv.count(flag) == 1 for flag in kept.split() if flag.startswith("--")), (arm, argv)
    ablate_argvs[arm] = argv
    expanded["constlr.sh with CONSTLR_NOGRAM=1 CONSTLR_ABLATE=" + arm] = "./gaudi.sh " + " ".join(argv)
    # the no-Gram trunk's real line at this stop epoch (constlr.sh with CONSTLR_NOGRAM=1), minus the arm's flag and in the arm's run dir, is the arm's line byte for byte
    result, folded = ablate_run(ablate_root, NOGRAM_ENV, **{"constlr_nogram_stop_after.txt": stop + "\n"})
    assert result.returncode == 0 and folded == nogram_argv_for(stop), (arm, result.stderr, folded)
    assert argv == [ABLATE_RUNS[arm] if arg == NOGRAM_RUN else arg for arg in without(folded, off)] and len(folded) == len(argv) + len(off), (arm, argv, folded)
    # the trainer reads the arm's argv as meant: stop after epoch 9x of the 1000-epoch ceiling, no Gram teacher, the recipe with just that setting at its default, and every other parsed setting the
    # no-Gram trunk's at the same stop epoch (only the run dir and that one setting differ)
    parsed = ssl_pretrain.parse_args(argv[argv.index("ssl_pretrain.py") + 1:])
    folded_parsed = ssl_pretrain.parse_args(folded[folded.index("ssl_pretrain.py") + 1:])
    assert ssl_pretrain.check_args(parsed) == int(stop) and parsed.epochs == 1000 and parsed.no_gram and parsed.resume and parsed.output_dir == ABLATE_RUNS[arm]
    assert tuple(getattr(parsed, k) for k in RECIPE_KEYS) == recipe_set and tuple(getattr(folded_parsed, k) for k in RECIPE_KEYS) == (0.3, 2.0, True)
    assert {k for k in vars(parsed) if getattr(parsed, k) != getattr(folded_parsed, k)} == {"output_dir", reset_key}, arm
assert len({tuple(argv) for argv in ablate_argvs.values()}) == 3  # three different lines
# a missing or invalid stop file of an arm is refused naming itself (none of the other seven, which stay valid) before gaudi.sh is reached, and before any run: line; each arm reads only its own
for arm in ABLATE_ARMS:
    own = ABLATE_FILES[arm]
    for stop in (None, "", "0", "2.5", "thirty", "-3", "1001", "18446744073709551617"):  # None: absent; 1001: one past the ceiling; 2**64 + 1: bash arithmetic would read it as 1
        result, refused_argv = ablate_run(ablate_root, ABLATE_ENVS[arm], **{own: stop})
        assert result.returncode == 1 and "ERROR" in result.stderr and own in result.stderr and refused_argv is None and result.stdout == "", (arm, stop, result.stderr)
        assert not any("chain/" + other in result.stderr for other in ALL_STOP_FILES - {own}), (arm, stop, result.stderr)
    assert "ceiling of 1000" in result.stderr, result.stderr  # the last one is refused for being above the ceiling, and the message names it
    result, solo_argv = ablate_run(ablate_root, ABLATE_ENVS[arm], **{name: None for name in ALL_STOP_FILES - {own}})  # the arm needs none of the others
    assert result.returncode == 0 and solo_argv == ablate_argvs[arm], (arm, result.stderr, solo_argv)
# the others are byte-unchanged: the phase, the no-Gram trunk, the plain trunk and both Ark-teacher arms with the arms' stop files present (not read: the argv has the others' epochs) or absent and CONSTLR_ABLATE
# unset or empty, and the no-Gram trunk's branch
for ablate_file in (" 94 \n", None):
    files = {name: ablate_file for name in ABLATE_FILES.values()}
    for empty in ({}, dict(CONSTLR_ABLATE="")):
        result, kept_argv = ablate_run(ablate_root, dict(ALLOC4, **empty), **files)
        assert result.returncode == 0 and kept_argv == constlr_argv("30") and (result.stdout, result.stderr) == ("", ""), (ablate_file, empty, result.stderr, kept_argv)
        result, kept_argv = ablate_run(ablate_root, dict(NOGRAM_ENV, **empty), **files)
        assert result.returncode == 0 and kept_argv == nogram_argv_for("100") and result.stderr == "" and result.stdout.splitlines() == [NOGRAM_RUN_LINE % "100"], (ablate_file, empty, result.stdout)
        result, kept_argv = ablate_run(ablate_root, dict(PLAIN_ENV, **empty), **files)
        assert result.returncode == 0 and kept_argv == plain_argv_for("60") and result.stderr == "" and result.stdout.splitlines() == [PLAIN_RUN_LINE], (ablate_file, empty, result.stdout)
        for lock, env, stop in (("0", ARK_ENV, "70"), ("1", ARK_LOCK_ENV, "80")):
            result, kept_argv = ablate_run(ablate_root, dict(env, **empty), **files)
            assert result.returncode == 0 and kept_argv == ark_argv_for(stop, lock) and result.stderr == "" and result.stdout.splitlines() == [ark_run_line(lock, stop)], (ablate_file, empty, lock, result.stdout)
assert len(bres.stdout.splitlines()) == 2 and bres.stdout.startswith("run: %s (Gram-anchored branch of " % BRANCH_E10), bres.stdout  # bres: the `10 40` branch above
for empty in ({}, dict(CONSTLR_ABLATE="")):
    bres_ablate, kept_bargv = branch_run(branch_dir, "10", "40", env=dict(ALLOC4, **empty))
    assert bres_ablate.returncode == 0 and kept_bargv == bargv == branch_argv_for("40", BRANCH_E10, BRANCH_SRC10, "36660,46660,56660"), (empty, bres_ablate.stderr)
    assert (bres_ablate.stdout, bres_ablate.stderr) == (bres.stdout, bres.stderr) and bres_ablate.stderr == "", (empty, bres_ablate.stdout)
# the refusals, each before anything is read (ablate_bare has no setting file to miss) and so before anything is launched (ablate_root has them all), with its exact text
ablate_bare = sandbox()
ABLATE_BAD_VALUE = "ERROR: CONSTLR_ABLATE holds '%s', which is not empty, drop_path, rope or local_norm\n"
ABLATE_NO_PLAIN = ("ERROR: CONSTLR_ABLATE and CONSTLR_PLAIN are both set: an ablation arm is the no-Gram trunk with one DINOv3 recipe flag removed, and the plain trunk has no recipe flag left to remove, "
                   "so unset CONSTLR_PLAIN\n")
ABLATE_NO_ARK = "ERROR: CONSTLR_ABLATE and CONSTLR_ARK are both set: an ablation arm is the no-Gram trunk's line with one recipe flag removed and nothing added, so unset CONSTLR_ARK\n"
ABLATE_NO_BRANCH = "ERROR: CONSTLR_ABLATE and CONSTLR_BRANCH_* are both set: branches of an ablation arm are not supported, so unset CONSTLR_ABLATE\n"
ABLATE_NEEDS_NOGRAM = "ERROR: CONSTLR_ABLATE=%s is set without CONSTLR_NOGRAM=1: an ablation arm is the no-Gram trunk with one recipe flag removed, so set both\n"
bad_values = ("0", "1", "yes", "Rope", "drop-path", "local", "all", " rope", "rope ", "drop_path,rope")
ablate_refusals = [(dict(NOGRAM_ENV, CONSTLR_ABLATE=value), ABLATE_BAD_VALUE % value) for value in bad_values]
ablate_refusals += [(dict(ALLOC4, CONSTLR_ABLATE="yes"), ABLATE_BAD_VALUE % "yes")]  # the value is checked first, with or without CONSTLR_NOGRAM
ablate_refusals += [(dict(PLAIN_ENV, CONSTLR_ABLATE=arm), ABLATE_NO_PLAIN) for arm in ABLATE_ARMS]
ablate_refusals += [(dict(env, CONSTLR_ABLATE=arm), ABLATE_NO_ARK) for env in (ARK_ENV, ARK_LOCK_ENV) for arm in ABLATE_ARMS]
ablate_refusals += [(dict(ALLOC4, CONSTLR_ABLATE=arm, **branch_vars), ABLATE_NO_BRANCH) for arm in ABLATE_ARMS
                    for branch_vars in (dict(CONSTLR_BRANCH_EPOCH="10"), dict(CONSTLR_BRANCH_STOP="40"), branch_all, dict(branch_all, CONSTLR_BRANCH_ARK="1"))]
ablate_refusals += [(dict(ALLOC4, CONSTLR_ABLATE=arm), ABLATE_NEEDS_NOGRAM % arm) for arm in ABLATE_ARMS] + [(dict(ALLOC4, CONSTLR_ABLATE="rope", CONSTLR_NOGRAM=""), ABLATE_NEEDS_NOGRAM % "rope")]
for env, expected_error in ablate_refusals:
    result = run_script(ablate_bare, "chain/constlr.sh", env=env)
    assert result.returncode == 1 and result.stderr == expected_error and result.stdout == "" and launched(ablate_bare) is None, (constlr_vars(env), result.stderr)
    result, refused_argv = ablate_run(ablate_root, env)
    assert result.returncode == 1 and result.stderr == expected_error and result.stdout == "" and refused_argv is None, (constlr_vars(env), result.stderr)
branch_refused(branch_root(), ("10", "40"), "CONSTLR_ABLATE", "CONSTLR_BRANCH_*", env=dict(ALLOC4, CONSTLR_ABLATE="rope"))  # the branch wrapper sets all four: refused naming both
# the wrapper: bash -n and the executable bit are checked with the other scripts above; here its mode, sbatch header, guard, last line and comments, that it is the no-Gram trunk's wrapper but for the job
# name, the comments and one guard right after the export, and that a copy with its cd pointed at a sandbox launches each arm's line from outside the repo (as sbatch starts it) and refuses to start without an arm
ABLATE_SCRIPT = os.path.join(REPO, "run_d1_8_constlr_nogram_ablate_gaudi.sh")
assert stat.S_IMODE(os.stat(ABLATE_SCRIPT).st_mode) == 0o755
ablate_text = open(ABLATE_SCRIPT).read()
ablate_lines = ablate_text.splitlines()
assert [line[len("#SBATCH "):] for line in ablate_lines if line.startswith("#SBATCH ")] == [
    "--job-name=d1-8-nogram-no", "-A grp_jliang12", "-p gaudi", "-q public", "-t 7-00:00:00", "-N 1", "--gres=gpu:hl225:4", "-c 72", "--mem=128G"], ablate_lines  # the job name is completed on the sbatch line
ablate_code = [line for line in ablate_lines if line and not line.startswith("#")]
assert ablate_code[-1] == "CONSTLR_NOGRAM=1 exec ./chain/constlr.sh" and ablate_code[:-1] == nogram_code[:1] + ablate_code[1:2] + nogram_code[1:-1], ablate_code  # its last line keeps the CONSTLR_ABLATE the no-Gram trunk's clears
assert ablate_code[1].startswith("[[ -n ${CONSTLR_ABLATE:-} ]] || {") and ablate_code[1].endswith("exit 1; }") and all(arm in ablate_code[1] for arm in ABLATE_ARMS), ablate_code[1]  # names the three
assert all(line.startswith("#") for line in ablate_lines if "phases.txt" in line), ablate_lines  # not a chain phase: phases.txt is named in comments only
for arm, (off, kept, _, _) in ABLATE_ARMS.items():  # the comments: each arm's submit line (its stop file first, its own job name) and the note and trainer line its log must show
    assert ablate_lines.count("#   echo 100 > chain/%s && sbatch --job-name=d1-8-nogram-no-%s --export=ALL,CONSTLR_ABLATE=%s run_d1_8_constlr_nogram_ablate_gaudi.sh" % (ABLATE_FILES[arm], arm, arm)) == 1, arm
    assert "recipe ablation: %s off (%s removed); DINOv3 recipe otherwise: %s" % (arm, " ".join(off), kept) in ablate_text and "recipe (...): " + ABLATE_TRAINER_LINES[arm] in ablate_text, arm
for needle in ("the plain trunk is the all-off control and the no-Gram trunk the all-on one", "No smoke launcher", "run_d1_8_nogram_smoke_gaudi.sh", "HCCL", "pass the SAME CONSTLR_ABLATE",
               "ssl_dinov3_cxr_constlr_nogram_no_<value>_seed_100", "chain/constlr_nogram_no_<value>_stop_after.txt", "sbatch does not expand variables", "never export it in the shell"):
    assert needle in ablate_text, needle


def ablate_wrapper_root():  # a sandbox with all the arms' stop files and the no-Gram trunk's (so a launch that took no arm for "off" would show), and the wrapper with its cd pointed at it
    root = sandbox(timing=True)  # also the gaudi_check_cards.sh / gaudi_verify_busy.sh recorders
    open(os.path.join(root, "chain", "main_lr.txt"), "w").write("0.002\n")
    open(os.path.join(root, "chain", "constlr_nogram_stop_after.txt"), "w").write("100\n")
    for arm in ABLATE_ARMS:
        open(os.path.join(root, "chain", ABLATE_FILES[arm]), "w").write(ABLATE_STOPS[arm] + "\n")
    assert ablate_text.count(BRANCH_CD) == 1, BRANCH_CD
    open(os.path.join(root, "ablate_wrapper.sh"), "w").write(ablate_text.replace(BRANCH_CD, "cd %s || exit 1" % shlex.quote(root)))
    return root


def ablate_wrapper(root, env):  # the wrapper copy, started from elsewhere: (result, argv or None), a stale argv.txt removed first
    if os.path.exists(os.path.join(root, "argv.txt")):
        os.remove(os.path.join(root, "argv.txt"))
    return subprocess.run(["bash", os.path.join(root, "ablate_wrapper.sh")], cwd=branch_elsewhere, env=env, capture_output=True, text=True), launched(root)


for arm, (off, kept, _, _) in ABLATE_ARMS.items():
    wrap_root = ablate_wrapper_root()
    result, wrapped_argv = ablate_wrapper(wrap_root, dict(ALLOC4, CONSTLR_ABLATE=arm))
    assert result.returncode == 0 and result.stderr == "" and wrapped_argv == ablate_argv_for(arm, ABLATE_STOPS[arm]) == ablate_argvs[arm], (arm, result.stderr, wrapped_argv)
    assert result.stdout.startswith("run: ssl_dinov3_cxr_constlr_nogram_no_%s_seed_100 (no Gram teacher: " % arm) and ("recipe ablation: %s off (%s removed)" % (arm, " ".join(off))) in result.stdout, result.stdout
    assert os.path.exists(os.path.join(wrap_root, "cards_checked"))
    for _ in range(50):  # the busy check runs in the background
        if os.path.exists(os.path.join(wrap_root, "busy_arg.txt")):
            break
        subprocess.run(["sleep", "0.1"])
    assert open(os.path.join(wrap_root, "busy_arg.txt")).read() == "900\n"
for env in (ALLOC4, dict(ALLOC4, CONSTLR_ABLATE="")):  # no arm (unset or empty): refused before the cd and the card check, and the no-Gram trunk (its stop file is there) is not launched in its place
    wrap_root = ablate_wrapper_root()
    result, wrapped_argv = ablate_wrapper(wrap_root, env)
    assert result.returncode == 1 and result.stdout == "" and result.stderr.startswith("ERROR: CONSTLR_ABLATE is unset or empty") and result.stderr.count("\n") == 1, (constlr_vars(env), result.stderr)
    assert all(arm in result.stderr for arm in ABLATE_ARMS) and wrapped_argv is None and not os.path.exists(os.path.join(wrap_root, "cards_checked")), (constlr_vars(env), result.stderr)
wrap_root = ablate_wrapper_root()  # another value gets through the wrapper's guard and is refused by constlr.sh, naming the three; nothing is launched
result, wrapped_argv = ablate_wrapper(wrap_root, dict(ALLOC4, CONSTLR_ABLATE="rop"))
assert result.returncode == 1 and result.stdout == "" and result.stderr == ABLATE_BAD_VALUE % "rop" and wrapped_argv is None, result.stderr
print("OK (k) ablation arms: constlr.sh with CONSTLR_NOGRAM=1 CONSTLR_ABLATE=drop_path | rope | local_norm (run_d1_8_constlr_nogram_ablate_gaudi.sh: mode 755, 4 cards, the no-Gram trunk's wrapper but for the job name, "
      "the comments and one guard after the export that refuses to start without CONSTLR_ABLATE, names phases.txt only in comments, launches each arm's line from outside the repo) launches the no-Gram trunk's line "
      "with exactly that one flag and its value removed (byte for byte: the real no-Gram trunk line at the same stop epoch minus the flag, in the arm's run dir) into ssl_dinov3_cxr_constlr_nogram_no_<arm>_seed_100 "
      "with the stop epoch from constlr_nogram_no_<arm>_stop_after.txt only (a missing or invalid file is refused naming itself, the other seven stop files neither read nor needed; eight run dirs and eight stop "
      "files in all, none shared), printing the arm and the flag removed in its run: line; the argv parses to the no-Gram trunk's settings but for the run dir and that one recipe setting at its default; CONSTLR_ABLATE "
      "other than empty or the three values, without CONSTLR_NOGRAM=1, or with CONSTLR_PLAIN, CONSTLR_ARK or any CONSTLR_BRANCH_* is refused with its exact text before anything is read; the phase, the no-Gram trunk, "
      "the plain trunk, both Ark-teacher arms and the no-Gram trunk's branches launch and print exactly as before with CONSTLR_ABLATE unset or empty and the arms' stop files present or absent")

# an exported CONSTLR_ABLATE, CONSTLR_PLAIN or CONSTLR_ARK reaches every launcher (sbatch hands a job the submitting shell's environment): the no-Gram trunk's launcher clears all three on its last line and launches the trunk,
# never the arm, the plain trunk or the Ark-teacher trunk (every stop file is there, so a wrong launch would show in the argv); the plain and the Ark-teacher launchers refuse an exported CONSTLR_ABLATE (constlr.sh above:
# CONSTLR_PLAIN / CONSTLR_ARK set), as does the branch wrapper (checked above) and the chain's constlr phase (no CONSTLR_NOGRAM), so none of them turns into an arm; the two smoke launchers call ssl_launch with their own
# flags, not chain/constlr.sh
NOGRAM_SCRIPT = os.path.join(REPO, "run_d1_8_constlr_nogram_gaudi.sh")


def inherited_run(script, files, env):  # a copy of the launcher with its cd pointed at a sandbox holding main_lr.txt and these stop files, started from elsewhere: (result, argv or None)
    root = sandbox(timing=True)
    open(os.path.join(root, "chain", "main_lr.txt"), "w").write("0.002\n")
    for name, content in files.items():
        open(os.path.join(root, "chain", name), "w").write(content)
    text = open(script).read()
    assert text.count(BRANCH_CD) == 1, (script, BRANCH_CD)
    open(os.path.join(root, "launcher.sh"), "w").write(text.replace(BRANCH_CD, "cd %s || exit 1" % shlex.quote(root)))
    return subprocess.run(["bash", os.path.join(root, "launcher.sh")], cwd=branch_elsewhere, env=env, capture_output=True, text=True), launched(root)


every_stop_file = {"constlr_stop_after.txt": "30\n", "constlr_nogram_stop_after.txt": "100\n", PLAIN_STOP_FILE: "60\n", ARK_STOP_FILE: "70\n", ARK_LOCK_STOP_FILE: "80\n",
                   **{ABLATE_FILES[arm]: ABLATE_STOPS[arm] + "\n" for arm in ABLATE_ARMS}}  # every stop file there is, so that a launch of the wrong run would show in the argv
for env in (ALLOC4, *(dict(ALLOC4, CONSTLR_ABLATE=value) for value in (*ABLATE_ARMS, "", "bogus")),  # exported or not, an arm, empty or not even a value: the no-Gram trunk, always
            dict(ALLOC4, CONSTLR_PLAIN="1"), dict(ALLOC4, CONSTLR_ARK="1"), dict(ALLOC4, CONSTLR_PLAIN="1", CONSTLR_ARK="1", CONSTLR_ABLATE="rope")):  # likewise the plain and the Ark-teacher switches, alone and all three
    result, wrapped_argv = inherited_run(NOGRAM_SCRIPT, every_stop_file, env)
    assert result.returncode == 0 and result.stderr == "" and wrapped_argv == nogram_argv_for("100") and result.stdout.splitlines() == [NOGRAM_RUN_LINE % "100"], (constlr_vars(env), result.stderr, wrapped_argv)
for script, env, expected_error, expected_argv in ((PLAIN_SCRIPT, ALLOC4, ABLATE_NO_PLAIN, plain_argv_for("60")), (ARK_SCRIPT, ALLOC4, ABLATE_NO_ARK, ark_argv_for("70", "0")),
                                                   (ARK_SCRIPT, dict(ALLOC4, CONSTLR_ARK_LOCK="1"), ABLATE_NO_ARK, ark_argv_for("80", "1"))):
    result, wrapped_argv = inherited_run(script, every_stop_file, dict(env, CONSTLR_ABLATE="rope"))
    assert result.returncode == 1 and result.stdout == "" and result.stderr == expected_error and wrapped_argv is None, (script, result.stderr)
    result, wrapped_argv = inherited_run(script, every_stop_file, env)  # without it, the same launcher launches its own run
    assert result.returncode == 0 and result.stderr == "" and wrapped_argv == expected_argv, (script, result.stderr, wrapped_argv)
for script in (os.path.join(REPO, "run_d1_8_nogram_smoke_gaudi.sh"), ARK_SMOKE):
    assert not any("constlr.sh" in line or "CONSTLR_ABLATE" in line for line in open(script).read().splitlines() if line and not line.startswith("#")), script
# the memory request of the SSL launchers below: 128G with one comment line right after it (the same wording in each: what was measured, and why not 256G), the last directive and before any command. The others (the chain driver,
# the Gram-512 twin, the timing and sweep jobs, the older arms) keep their own
MEM_NOTE = ("# 128G: measured 2026-10-26 on the running no-Gram trunk (4 cards, 72 CPUs, job 66633863): proportional set size of all its processes 53.3 GB; sstat MaxRSS reports the request itself because page cache is "
            "charged to the job cgroup; 256G could not share a Gaudi node with another lab's 1-card jobs that allocate 511 of 515 GB; Sol's own floor is 2 GB per CPU, it accepts a lower request with a notice.")
MEM_LAUNCHERS = ("run_d1_8_constlr_arkteacher_gaudi.sh", "run_d1_8_arkteacher_smoke_gaudi.sh", "run_d1_8_constlr_nogram_gaudi.sh", "run_d1_8_constlr_nogram_plain_gaudi.sh", "run_d1_8_nogram_smoke_gaudi.sh",
                 "run_d1_8_branch_gram_gaudi.sh", "run_d1_8_constlr_nogram_ablate_gaudi.sh")
for name in MEM_LAUNCHERS:
    mem_lines = open(os.path.join(REPO, name)).read().splitlines()
    mem_at = mem_lines.index("#SBATCH --mem=128G")
    assert mem_lines.count("#SBATCH --mem=128G") == 1 and mem_lines[mem_at + 1] == MEM_NOTE and mem_lines.count(MEM_NOTE) == 1, name
    assert [line for line in mem_lines if line.startswith("#SBATCH ")][-1] == "#SBATCH --mem=128G" and mem_at < min(i for i, line in enumerate(mem_lines) if line and not line.startswith("#")), name
    assert not any("--mem=256G" in line for line in mem_lines if line.startswith("#SBATCH ")), name
print("OK (k) launchers: an exported CONSTLR_ABLATE (an arm, empty or not a value), CONSTLR_PLAIN=1 or CONSTLR_ARK=1 leaves run_d1_8_constlr_nogram_gaudi.sh launching the no-Gram trunk (its last line clears all three), while the plain and the Ark-teacher "
      "launchers (both arms) refuse it with the exact CONSTLR_ABLATE text and launch their own run without it; the two smoke launchers never reach chain/constlr.sh; the no-Gram trunk, plain, Ark-teacher, branch, "
      "ablation and both smoke launchers ask for --mem=128G with the same measured-memory comment line right after it (the last directive, before any command)")

# hires.sh ("DINO-X ConstLR 512 px"): a separate run from the stock weights, constlr.sh with only the crops and the batch changed. Four setting files,
# each missing or invalid one refused naming it before gaudi.sh is reached (the LR file is constlr's own main_lr.txt)
HIRES_FILES = {"main_lr.txt": "0.002\n", "hires_stop_after.txt": "30\n", "hires_gram_size.txt": "1024\n", "hires_batch_per_gpu.txt": "8\n"}


def hires_run(root, env=ALLOC4, **override):  # write the four files (None: absent), run hires.sh; returns (result, argv or None)
    for name, content in dict(HIRES_FILES, **override).items():
        path = os.path.join(root, "chain", name)
        if os.path.exists(path):
            os.remove(path)
        if content is not None:
            open(path, "w").write(content)
    if os.path.exists(os.path.join(root, "argv.txt")):
        os.remove(os.path.join(root, "argv.txt"))
    result = run_script(root, "chain/hires.sh", env=env)
    return result, launched(root)


root = sandbox()
for name in HIRES_FILES:
    result, argv = hires_run(root, **{name: None})
    assert result.returncode == 1 and "ERROR" in result.stderr and name in result.stderr and argv is None, (name, result.stderr)
for name, value in (("main_lr.txt", "0"), ("main_lr.txt", "abc"), ("hires_stop_after.txt", "0"), ("hires_stop_after.txt", "2.5"),
                    ("hires_stop_after.txt", "1001"), ("hires_stop_after.txt", "18446744073709551617"), ("hires_gram_size.txt", "1000"),
                    ("hires_gram_size.txt", "496"), ("hires_gram_size.txt", "1040"), ("hires_gram_size.txt", "abc"), ("hires_gram_size.txt", "99999"),
                    ("hires_batch_per_gpu.txt", "0"), ("hires_batch_per_gpu.txt", "abc"), ("hires_batch_per_gpu.txt", "12345")):
    result, argv = hires_run(root, **{name: value + "\n"})
    assert result.returncode == 1 and "ERROR" in result.stderr and name in result.stderr and argv is None, (name, value, result.stderr)
assert "ceiling of 1000" in hires_run(root, **{"hires_stop_after.txt": "1001\n"})[0].stderr


def hires_argv(gram, stop="30", batch="8", refresh="80000,160000,240000"):  # what hires.sh launches; base LR 0.001 = 0.002 / (4 x sqrt(256/1024))
    return COMMON + ["/workspace/ssl_dinov3_cxr_constlr512_seed_100", "--resume", "--pretrained_weights", "/workspace/weights/" + WEIGHT_NAME,
                     "--global_crops_size", "512", "--local_crops_size", "224", "--gram_teacher_size", gram, "--lr", "0.001", "--min_lr", "0.001",
                     "--epochs", "1000", "--stop_after_epochs", stop, "--gram_refresh_steps", refresh, "--batch_size_per_gpu", batch]


result, argv = hires_run(root)  # 8 per card on 4 cards: G = 32
assert result.returncode == 0 and argv == hires_argv("1024"), (result.stderr, argv)
assert "--no_lr_scaling" not in argv
assert ("LR: base 0.001 -> effective 0.000707107 at global batch 32 (8 x 4 cards; the trainer's sqrt_wrt_1024 rule; constlr: 0.002 at 256)"
        in result.stdout.splitlines()), result.stdout
assert "gram refresh steps: 80000,160000,240000 (constlr's 10k/20k/30k at batch 256 scaled by 256/32)" in result.stdout.splitlines(), result.stdout
assert "run: ssl_dinov3_cxr_constlr512_seed_100 (Gram 1024 from chain/hires_gram_size.txt)" in result.stdout.splitlines(), result.stdout
expanded["hires.sh"] = "./gaudi.sh " + " ".join(argv)
hires_lines = [line for line in result.stdout.splitlines() if line.startswith(("run:", "LR:", "gram refresh steps:"))]
# other batches and card counts: G = per-card batch x the cards ssl_launch counts (SLURM_JOB_GPUS here); 24 does not divide 256: rounded to the nearest step
for env, files, gram, stop, batch, refresh, g in (
        (ALLOC4, {"hires_gram_size.txt": " 512 \n", "hires_stop_after.txt": "1000\n", "hires_batch_per_gpu.txt": "16\n"}, "512", "1000", "16", "40000,80000,120000", 64),
        (ALLOC4, {"hires_batch_per_gpu.txt": "64\n"}, "1024", "30", "64", "10000,20000,30000", 256),  # constlr's own G: its own steps
        (ALLOC4, {"hires_batch_per_gpu.txt": "6\n"}, "1024", "30", "6", "106667,213333,320000", 24),
        (dict(CLEAN_ENV, SLURM_JOB_GPUS="0,1"), {}, "1024", "30", "8", "160000,320000,480000", 16)):
    result, argv_g = hires_run(root, env=env, **files)
    ranks = "2" if g == 16 else "4"
    assert result.returncode == 0 and argv_g == with_ranks(ranks) + hires_argv(gram, stop, batch, refresh)[len(COMMON):], (result.stderr, argv_g)
    steps = [int(x) for x in refresh.split(",")]
    assert all(abs(s_ * g - c * 256) <= g / 2 for s_, c in zip(steps, (10000, 20000, 30000))), (refresh, g)  # constlr's epoch fractions, to the step
    assert "at global batch %d" % g in result.stdout and "scaled by 256/%d)" % g in result.stdout, result.stdout
# the trainer reads it as meant: 512 / 224 / 1024, the file's batch (the last --batch_size_per_gpu wins over ssl_launch's 64), the stock weights, the base LR
# scaled (no --no_lr_scaling) on --lr and --min_lr alike, and every other schedule setting constlr.sh's argv gives the trainer (the same main_lr.txt here)
hires_parsed = ssl_pretrain.parse_args(argv[argv.index("ssl_pretrain.py") + 1:])
assert ssl_pretrain.check_args(hires_parsed) == 30
assert (hires_parsed.global_crops_size, hires_parsed.local_crops_size, hires_parsed.gram_teacher_size, hires_parsed.batch_size_per_gpu) == (512, 224, 1024, 8)
assert hires_parsed.init_from_checkpoint is None and hires_parsed.output_dir == "/workspace/ssl_dinov3_cxr_constlr512_seed_100"
assert hires_parsed.no_lr_scaling is False and hires_parsed.lr == hires_parsed.min_lr == 0.002 / lr_scaling_factor(256)
assert hires_parsed.gram_refresh_steps == [80000, 160000, 240000]
open(os.path.join(root, "chain", "constlr_stop_after.txt"), "w").write("30\n")
os.remove(os.path.join(root, "argv.txt"))
assert run_script(root, "chain/constlr.sh").returncode == 0
constlr_parsed = ssl_pretrain.parse_args(launched(root)[launched(root).index("ssl_pretrain.py") + 1:])
assert constlr_parsed.no_lr_scaling is True and constlr_parsed.lr == constlr_parsed.min_lr == 0.002


def resolved(parsed, g):  # the LR the trainer computes from these args at global batch g (run()'s build_schedules call; epoch_length 10 stands in)
    return build_schedules(lr=parsed.lr, min_lr=parsed.min_lr, epochs=parsed.epochs, warmup_epochs=parsed.warmup_epochs, epoch_length=10, global_batch_size=g,
                           scale_lr=not parsed.no_lr_scaling, teacher_temp_warmup_epochs=parsed.teacher_temp_warmup_epochs)


at_256, at_32, constlr_sch = resolved(hires_parsed, 256), resolved(hires_parsed, 32), resolved(constlr_parsed, 256)
assert at_256["peak_lr"] == at_256["end_lr"] == constlr_sch["peak_lr"] == constlr_sch["end_lr"] == 0.002  # exactly constlr's LR at its batch
assert math.isclose(at_32["peak_lr"], 0.002 * math.sqrt(32 / 256), rel_tol=1e-12) and at_32["end_lr"] == at_32["peak_lr"]  # --min_lr scaled too
assert (at_32["lr"][hires_parsed.warmup_epochs * 10:] == at_32["peak_lr"]).all() and (at_256["lr"] == constlr_sch["lr"]).all()  # flat after the warmup
lr_keys = {"lr", "min_lr", "no_lr_scaling", "gram_refresh_steps"}  # carried over to G, checked above
changed = {"global_crops_size", "local_crops_size", "gram_teacher_size", "batch_size_per_gpu", "world_size"} | lr_keys  # world_size: the trainer's, not an arg
assert hires_parsed.pretrained_weights == constlr_parsed.pretrained_weights and hires_parsed.stop_after_epochs == constlr_parsed.stop_after_epochs
assert all(getattr(hires_parsed, k) == getattr(constlr_parsed, k) for k in set(ssl_pretrain.RESUME_MATCH) - changed), \
    {k: (getattr(hires_parsed, k), getattr(constlr_parsed, k)) for k in set(ssl_pretrain.RESUME_MATCH) - changed if getattr(hires_parsed, k) != getattr(constlr_parsed, k)}
assert (constlr_parsed.epochs, constlr_parsed.warmup_epochs, constlr_parsed.freeze_backbone_epochs, constlr_parsed.gram_refresh_steps) == (1000, 4, 1, [10000, 20000, 30000])

# the Gram-512 twin: HIRES_GRAM=512 HIRES_RUN_SUFFIX=_gram512 (what run_d1_8hires_gram512_gaudi.sh sets) changes the Gram size and the run dir and nothing else; the file's 1024 is ignored
default_run, default_argv = hires_run(root)  # no override: the chain's phase, unchanged
assert default_run.returncode == 0 and after(default_argv, "--output_dir") == "/workspace/ssl_dinov3_cxr_constlr512_seed_100" and after(default_argv, "--gram_teacher_size") == "1024", default_argv
twin_run, twin_argv = hires_run(root, env=dict(ALLOC4, HIRES_GRAM="512", HIRES_RUN_SUFFIX="_gram512"))  # hires_gram_size.txt still holds 1024
assert twin_run.returncode == 0 and open(os.path.join(root, "chain", "hires_gram_size.txt")).read() == "1024\n", twin_run.stderr
assert after(twin_argv, "--gram_teacher_size") == "512" and after(twin_argv, "--output_dir") == "/workspace/ssl_dinov3_cxr_constlr512_gram512_seed_100", twin_argv
assert len(twin_argv) == len(default_argv) and [(a, b) for a, b in zip(default_argv, twin_argv) if a != b] == [
    ("/workspace/ssl_dinov3_cxr_constlr512_seed_100", "/workspace/ssl_dinov3_cxr_constlr512_gram512_seed_100"), ("1024", "512")], (default_argv, twin_argv)
assert "run: ssl_dinov3_cxr_constlr512_gram512_seed_100 (Gram 512 from HIRES_GRAM)" in twin_run.stdout.splitlines(), twin_run.stdout
expanded["hires.sh with HIRES_GRAM=512 HIRES_RUN_SUFFIX=_gram512"] = "./gaudi.sh " + " ".join(twin_argv)
# a bad override is refused before launching, naming itself (not the file), and never falls back to the file's valid 1024
for var, value in (("HIRES_GRAM", "1000"), ("HIRES_GRAM", "496"), ("HIRES_GRAM", "1040"), ("HIRES_GRAM", "abc"), ("HIRES_GRAM", "99999"),
                   ("HIRES_RUN_SUFFIX", "gram512"), ("HIRES_RUN_SUFFIX", "_Gram512"), ("HIRES_RUN_SUFFIX", "_g 512")):  # the suffixes: no underscore, uppercase, a space
    bad_run, bad_argv = hires_run(root, env=dict(ALLOC4, **{var: value}))
    assert bad_run.returncode == 1 and "ERROR" in bad_run.stderr and var in bad_run.stderr and "hires_gram_size.txt" not in bad_run.stderr and bad_argv is None, (var, value, bad_run.stderr)
# the wrapper: bash -n and the executable bit are checked with the other scripts above; here its sbatch header, its guards and its last line
twin_lines = open(os.path.join(REPO, "run_d1_8hires_gram512_gaudi.sh")).read().splitlines()
assert [line[len("#SBATCH "):] for line in twin_lines if line.startswith("#SBATCH ")] == [
    "--job-name=d1-8hires-g512", "-A grp_jliang12", "-p gaudi", "-q public", "-t 7-00:00:00", "-N 1", "--gres=gpu:hl225:4", "-c 72", "--mem=256G"], twin_lines
twin_code = [line for line in twin_lines if line and not line.startswith("#")]
assert twin_code[-1] == "HIRES_GRAM=512 HIRES_RUN_SUFFIX=_gram512 exec ./chain/hires.sh", twin_code
for needle in ("HIRES_GRAM=512", "HIRES_RUN_SUFFIX=_gram512", "exec ./chain/hires.sh", "./gaudi_check_cards.sh || exit 1", "./gaudi_verify_busy.sh 900 &"):
    assert any(needle in line for line in twin_code), needle  # in a command, not only in a comment (the busy verifier's guard line names the script too, hence the launch itself)
assert all(line.startswith("#") for line in twin_lines if "phases.txt" in line), twin_lines  # not a chain phase: phases.txt is named in comments only

root = sandbox(weights=False)  # the shared guards
result = run_script(root, "chain/pilot_a.sh")
assert result.returncode == 1 and "DINOv3 weights not found at ./weights/" + WEIGHT_NAME in result.stderr and launched(root) is None
root = sandbox(gaudi_executable=False)
result = run_script(root, "chain/pilot_a.sh")
assert result.returncode == 1 and "run from the repo root" in result.stderr and launched(root) is None

root, elsewhere = sandbox(timing=True), tempfile.mkdtemp(dir=TMP)  # the timing job, started from another directory
result = run_script(root, os.path.join(root, "run_d1_8s_timing_gaudi.sh"), cwd=elsewhere, env=dict(ALLOC4, ARK_ROOT=root))
assert result.returncode == 0, result.stderr
argv = launched(root)
assert argv == COMMON + ["/workspace/ssl_dinov3_cxr_timing_seed_100", "--resume", "--max_steps", "200", "--print_freq", "10"], argv
assert os.path.exists(os.path.join(root, "cards_checked"))
for _ in range(50):  # the busy check runs in the background
    if os.path.exists(os.path.join(root, "busy_arg.txt")):
        break
    subprocess.run(["sleep", "0.1"])
assert open(os.path.join(root, "busy_arg.txt")).read() == "900\n"
expanded["run_d1_8s_timing_gaudi.sh"] = "./gaudi.sh " + " ".join(argv)
result = run_script(root, os.path.join(root, "run_d1_8s_timing_gaudi.sh"), cwd=elsewhere)  # neither ARK_ROOT nor SLURM_SUBMIT_DIR
assert result.returncode != 0 and "sbatch from the repo root or set ARK_ROOT" in result.stderr
os.remove(os.path.join(root, "argv.txt"))
result = run_script(root, os.path.join(root, "run_d1_8s_timing_gaudi.sh"), cwd=elsewhere, env=dict(ALLOC4, SLURM_SUBMIT_DIR=root))
assert result.returncode == 0 and launched(root) == argv
result = run_script(root, os.path.join(root, "run_d1_8s_timing_gaudi.sh"), cwd=elsewhere, env=dict(ALLOC4, ARK_ROOT=root, SLURM_JOB_GPUS="0,1"))
assert result.returncode == 0 and launched(root) == with_ranks("2") + argv[len(COMMON):], (result.stderr, launched(root))  # sbatch --gres=gpu:hl225:2
expanded["run_d1_8s_timing_gaudi.sh (2 cards)"] = "./gaudi.sh " + " ".join(launched(root))
for script, command in expanded.items():
    print("     %s -> %s" % (script, command))
print("OK (k) scripts: bash -n passes and all are executable; pilot A/B, main (needs main_lr.txt + main_epochs.txt, rejects bad numbers "
      "before launching), constlr (needs main_lr.txt + constlr_stop_after.txt, rejects bad numbers and a stop epoch above the 1000 ceiling "
      "before launching, accepts 1000; its argv parses to stop 30 of --epochs 1000, Gram refreshes 10k/20k/30k (the trainer default, as main) and "
      "an LR flat at the peak after the warmup, and --resume does not lock "
      "--stop_after_epochs) and the timing job launch gaudi.sh with the expected argv; weights / gaudi.sh guards exit 1; --nproc-per-node = the "
      "allocated cards as Ark counts them (SLURM_JOB_GPUS 2 -> 2, 4 -> 4, a host HABANA_VISIBLE_MODULES wins; nothing set -> the container is "
      "asked and its count used, no count -> exit 1); hires (DINO-X ConstLR 512 px: constlr.sh's recipe from the stock weights, its main_lr.txt as the sqrt_wrt_1024 base that the trainer resolves "
      "to constlr's 0.002 at global batch 256, its Gram refreshes x 256/G, every other schedule setting = constlr.sh's, only the crops 512 / 224 / "
      "hires_gram_size.txt and the hires_batch_per_gpu.txt batch changed; printed: " + " | ".join(hires_lines) + ") refuses each "
      "missing or invalid setting file (the batch and Gram files included) and a stop epoch above the 1000 ceiling before launching; with HIRES_GRAM=512 "
      "HIRES_RUN_SUFFIX=_gram512 (the Gram-512 twin) only the Gram size and the run dir change, the file's Gram is ignored, and a bad override (Gram not x16 / outside "
      "512..1024 / not a number, a suffix without _ / uppercase / with a space) is refused naming itself; run_d1_8hires_gram512_gaudi.sh: 4 cards, ends in "
      "HIRES_GRAM=512 HIRES_RUN_SUFFIX=_gram512 exec ./chain/hires.sh, names phases.txt only in comments; constlr.sh with CONSTLR_NOGRAM=1 (the no-Gram trunk; "
      "run_d1_8_constlr_nogram_gaudi.sh: 4 cards, ends in CONSTLR_NOGRAM=1 exec ./chain/constlr.sh) launches its line with the DINOv3 recipe (--drop_path_rate 0.3 --rope_rescale_coords 2 "
      "--untie_local_cls_norm, which the phase's line lacks) and --no_gram in place of the Gram refreshes, into its own run dir "
      "and with the stop epoch from constlr_nogram_stop_after.txt only; a bad stop file or CONSTLR_NOGRAM is refused before launching; the argv parses to no_gram and the recipe with "
      "every other locked setting = constlr's")

# (l) data paths: our list parsing == Ark's dataset classes on the real lists -----------------------------------------------------------
try:
    import dataloader
except ImportError as error:
    print("SKIP (l) data paths: cannot import dataloader here (%s)" % error)
else:
    from utils import get_config
    real_config, total = get_config("datasets_config.yaml"), 0
    for name in DATASET_NAMES:
        cfg = real_config[name]
        ark_list = dataloader.dict_dataloarder[name](images_path=cfg["data_dir"], file_path=cfg["train_list"]).img_list
        ours = ssl_data.image_paths(name, cfg["data_dir"], cfg["train_list"])
        assert ours == ark_list and len(ours) > 0, name
        total += len(ours)
        print("     %s: %d images, first %s" % (name, len(ours), ours[0]))
    assert total == 682686, total
    print("OK (l) data paths: image_paths == the Ark dataset classes' img_list for all 4 train lists; %d images in total" % total)

print("ALL OK")
