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
CSV_HEADER = "step,epoch,dino_global,dino_local,koleo,ibot,gram,total,lr,wd,momentum,teacher_temp,gram_weight,step_time,data_time"

TMP = tempfile.mkdtemp()
atexit.register(shutil.rmtree, TMP, ignore_errors=True)


def make_synthetic_data(root):
    """12 grayscale images of a few non-square sizes (JPEG and PNG), 3 per dataset, list files in the 4 real formats."""
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
            for i in range(3):
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
    assert steps == sorted(set(steps)) and steps[0] == 0 and steps[-1] == 17 and len(rows[1]) == 15  # 0-based, no duplicates
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
        assert len(row) == 15 and row[ng_gram] == "0.0" and row[ng_gram_weight] == "0.0", row  # both columns present and numeric, never blank
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
assert set(ssl_pretrain.BRANCH_FREE) == {"no_gram", "gram_weight", "gram_refresh_steps", "gram_teacher_size", "branch_from"}
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
print("OK (t) parse: no --branch_from -> None (read as None in an old checkpoint, locked on --resume); --branch_from alone (with --gram_refresh_steps 5 or none) needs no "
      "--pretrained_weights and keeps gram_weight 2.0, Gram size 448; --no_gram, --init_from_checkpoint, or no --gram_refresh_steps with it exits 2 naming both flags")


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

# (k) launch scripts ------------------------------------------------------------------------------------------------------------
scripts = [os.path.join(REPO, "chain", f) for f in ("ssl_common.sh", "pilot_a.sh", "pilot_b.sh", "main.sh", "constlr.sh", "hires.sh")]
scripts.append(os.path.join(REPO, "run_d1_8s_timing_gaudi.sh"))
scripts.append(os.path.join(REPO, "run_d1_8hires_gram512_gaudi.sh"))  # the Gram-512 twin's sbatch wrapper; its contents are checked after the hires checks
scripts.append(os.path.join(REPO, "run_d1_8_constlr_nogram_gaudi.sh"))  # the no-Gram trunk's sbatch wrapper; its contents are checked after the constlr checks
scripts.append(os.path.join(REPO, "run_d1_8_branch_gram_gaudi.sh"))  # the Gram-anchored branches' sbatch wrapper; its contents are checked after the no-Gram wrapper's
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


def nogram_argv_for(stop):  # constlr_argv(stop) with the run dir swapped and the Gram refresh list replaced by --no_gram
    argv = constlr_argv(stop)
    assert argv[-2:] == ["--gram_refresh_steps", "10000,20000,30000"]
    return [NOGRAM_RUN if arg == "/workspace/ssl_dinov3_cxr_constlr_seed_100" else arg for arg in argv[:-2]] + ["--no_gram"]


root = sandbox()
result, nogram_argv = constlr_run(root, env=NOGRAM_ENV)  # constlr_nogram_stop_after.txt " 100 ", constlr_stop_after.txt 30 (not read)
assert result.returncode == 0, result.stderr
assert nogram_argv == nogram_argv_for("100"), nogram_argv
assert result.stdout.splitlines() == ["run: ssl_dinov3_cxr_constlr_nogram_seed_100 (no Gram teacher: --no_gram replaces --gram_refresh_steps 10000,20000,30000; "
                                      "stop after epoch 100 from chain/constlr_nogram_stop_after.txt)"] and result.stderr == "", (result.stdout, result.stderr)
expanded["constlr.sh with CONSTLR_NOGRAM=1"] = "./gaudi.sh " + " ".join(nogram_argv)
# the same stop epoch in the chain's phase (variable unset, or set but empty): the only differences are the run dir and the tail, and nothing new is printed
result, default_100 = constlr_run(root, **{"constlr_stop_after.txt": "100\n"})
assert result.returncode == 0 and default_100 == constlr_argv("100") and (result.stdout, result.stderr) == ("", ""), (result.stderr, default_100)
assert default_100[-2:] == ["--gram_refresh_steps", "10000,20000,30000"] and nogram_argv[-1:] == ["--no_gram"] and len(nogram_argv) == len(default_100) - 1
assert [(a, b) for a, b in zip(default_100[:-2], nogram_argv[:-1]) if a != b] == [("/workspace/ssl_dinov3_cxr_constlr_seed_100", NOGRAM_RUN)], (default_100, nogram_argv)
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
locked_alike = set(ssl_pretrain.RESUME_MATCH) - {"no_gram", "gram_weight", "gram_refresh_steps", "world_size"}
assert all(getattr(nogram_parsed, k) == getattr(default_parsed, k) for k in locked_alike), \
    {k: (getattr(nogram_parsed, k), getattr(default_parsed, k)) for k in locked_alike if getattr(nogram_parsed, k) != getattr(default_parsed, k)}
assert nogram_parsed.pretrained_weights == default_parsed.pretrained_weights and nogram_parsed.stop_after_epochs == default_parsed.stop_after_epochs == 100
assert nogram_parsed.output_dir == NOGRAM_RUN != default_parsed.output_dir
# the wrapper: bash -n and the executable bit are checked with the other scripts above; here its sbatch header, its guards and its last line
nogram_lines = open(os.path.join(REPO, "run_d1_8_constlr_nogram_gaudi.sh")).read().splitlines()
assert [line[len("#SBATCH "):] for line in nogram_lines if line.startswith("#SBATCH ")] == [
    "--job-name=d1-8-nogram", "-A grp_jliang12", "-p gaudi", "-q public", "-t 7-00:00:00", "-N 1", "--gres=gpu:hl225:4", "-c 72", "--mem=256G"], nogram_lines
nogram_code = [line for line in nogram_lines if line and not line.startswith("#")]
assert nogram_code[-1] == "CONSTLR_NOGRAM=1 exec ./chain/constlr.sh", nogram_code
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
# the limits themselves launch: STOP at the 1000 ceiling, EVERY of 9 digits, COUNT of 4 (one more digit of either is refused below)
steps_max = ",".join(str(26660 + 999999999 * i) for i in range(1, 1001))
bres, bargv_max = branch_run(branch_dir, "10", "1000", "999999999", "1000")
assert bres.returncode == 0 and bargv_max == branch_argv_for("1000", "ssl_dinov3_cxr_constlr_nogram_e0010_gram_every999999999x1000_seed_100", BRANCH_SRC10, steps_max), (bres.stderr, bargv_max and bargv_max[-4:])
# the trainer reads the branch argv as meant: --branch_from the source, the absolute refresh list (all after the branch step), the Gram teacher ON at the trainer's weight 2.0 and teacher 448, stop after
# epoch 40 of the 1000-epoch ceiling, and every other setting --resume locks (the Gram ones, branch_from and the trainer's world_size aside) the same as the trunk's argv gives it
bparsed = ssl_pretrain.parse_args(bargv[bargv.index("ssl_pretrain.py") + 1:])
tparsed = ssl_pretrain.parse_args(trunk_argv[trunk_argv.index("ssl_pretrain.py") + 1:])
assert ssl_pretrain.check_args(bparsed) == 40 and bparsed.epochs == 1000 and ssl_pretrain.check_args(ssl_pretrain.parse_args(bargv_max[bargv_max.index("ssl_pretrain.py") + 1:])) == 1000
assert (bparsed.branch_from, bparsed.gram_refresh_steps, bparsed.no_gram, bparsed.gram_weight, bparsed.gram_teacher_size) == (BRANCH_SRC10, [36660, 46660, 56660], False, 2.0, 448)
assert min(bparsed.gram_refresh_steps) > 10 * 2666 and bparsed.output_dir == "/workspace/" + BRANCH_E10 != tparsed.output_dir and bparsed.resume  # a resubmit resumes the branch
assert tparsed.branch_from is None and tparsed.no_gram is True
branch_locked_alike = set(ssl_pretrain.RESUME_MATCH) - set(ssl_pretrain.BRANCH_FREE) - {"world_size"}
assert all(getattr(bparsed, k) == getattr(tparsed, k) for k in branch_locked_alike), \
    {k: (getattr(bparsed, k), getattr(tparsed, k)) for k in branch_locked_alike if getattr(bparsed, k) != getattr(tparsed, k)}
assert bparsed.pretrained_weights == tparsed.pretrained_weights and bparsed.stop_after_epochs == tparsed.stop_after_epochs == 40
# refusals, each before anything is launched: exit 1, an ERROR naming the culprit, nothing on stdout, gaudi.sh never reached
usage_dir = branch_root()
for args in ((), ("10",), ("10", "40", "10000", "3", "x")):
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
# no source: refused on a first start, naming it (the K the trunk never kept as well); once the branch has an epoch checkpoint of its own a resume needs no source
nosrc_dir = branch_root(source=False)
branch_refused(nosrc_dir, ("10", "40"), BRANCH_TRUNK + "/ckpt/epoch_0010.pth", "nothing to branch from")
branch_refused(branch_dir, ("5", "40"), BRANCH_TRUNK + "/ckpt/epoch_0005.pth", "nothing to branch from")
own_ckpt = os.path.join(nosrc_dir, BRANCH_E10, "ckpt")
os.makedirs(own_ckpt)
for name in ("epoch_0011.pth.tmp", "maxsteps_0000003.pth"):  # not epoch checkpoints (the trainer resumes from epoch_<digits>.pth only)
    open(os.path.join(own_ckpt, name), "w").close()
branch_refused(nosrc_dir, ("10", "40"), BRANCH_TRUNK + "/ckpt/epoch_0010.pth")
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
assert bres.stdout.splitlines() == ["run: ssl_dinov3_cxr_constlr_nogram_seed_100 (no Gram teacher: --no_gram replaces --gram_refresh_steps 10000,20000,30000; "
                                    "stop after epoch 100 from chain/constlr_nogram_stop_after.txt)"], bres.stdout
# the wrapper: bash -n and the executable bit are checked with the other scripts above; here its mode, sbatch header, guards and last line
assert stat.S_IMODE(os.stat(BRANCH_SCRIPT).st_mode) == 0o755
branch_lines = open(BRANCH_SCRIPT).read().splitlines()
assert [line[len("#SBATCH "):] for line in branch_lines if line.startswith("#SBATCH ")] == [
    "--job-name=d1-8-branch", "-A grp_jliang12", "-p gaudi", "-q public", "-t 7-00:00:00", "-N 1", "--gres=gpu:hl225:4", "-c 72", "--mem=256G"], branch_lines
branch_code = [line for line in branch_lines if line and not line.startswith("#")]
assert branch_code[-1] == "CONSTLR_BRANCH_EPOCH=$1 CONSTLR_BRANCH_STOP=$2 CONSTLR_BRANCH_EVERY=${3:-10000} CONSTLR_BRANCH_COUNT=${4:-3} exec ./chain/constlr.sh", branch_code
branch_needles = ("export PT_HPU_LAZY_MODE=1", "[[ $# -ge 2 && $# -le 4 ]]", BRANCH_CD, "./gaudi_check_cards.sh || exit 1", "[[ -x ./gaudi_verify_busy.sh ]]", "./gaudi_verify_busy.sh 900 &")
for needle in branch_needles:
    assert any(needle in line for line in branch_code), needle  # in a command, not only in a comment
positions = [min(i for i, line in enumerate(branch_code) if needle in line) for needle in branch_needles]
assert all(a < b for a, b in zip(positions, positions[1:])), positions  # in this order: the usage guard comes before the cd and the card check
assert all(line.startswith("#") for line in branch_lines if "phases.txt" in line), branch_lines  # not a chain phase: phases.txt is named in comments only
print("OK (k) branch launcher: run_d1_8_branch_gram_gaudi.sh (mode 755, 4 cards, the usage guard first, ends in CONSTLR_BRANCH_EPOCH=$1 CONSTLR_BRANCH_STOP=$2 CONSTLR_BRANCH_EVERY=${3:-10000} "
      "CONSTLR_BRANCH_COUNT=${4:-3} exec ./chain/constlr.sh, names phases.txt only in comments) makes constlr.sh launch the no-Gram trunk's line with only --no_gram replaced by --branch_from "
      "<trunk>/ckpt/epoch_<K>.pth --gram_refresh_steps <list> and no stop file read: `10 40` -> " + BRANCH_E10 + " refreshing after 36660,46660,56660 (10 x the epoch_length 2666 read from the "
      "trunk's args.json, 36650,... at 2665, then every 10000 steps, 3 times), `10 40 2666 30` -> ..._every2666x30_ (one refresh per epoch for 30 epochs), `7 12 500 2` -> ..._e0007_ 19162,19662; "
      "the argv parses to branch_from the source, Gram ON (2.0, 448), stop 40 of 1000, every other locked setting = the trunk's; refused before launching: a wrong argument count (before the "
      "card check), K / STOP / EVERY / COUNT that are not positive integers (no leading zero; EVERY at most 9 digits, COUNT 4, both accepted at the limit), STOP above the 1000 ceiling "
      "(accepted at 1000) or not after K, a trunk args.json without a positive epoch_length, CONSTLR_NOGRAM set as well, one of the four variables alone or emptied, and a missing source "
      "on a first start (a branch with an epoch checkpoint of its own resumes without it); the phase and the trunk launch and print exactly as before with all four variables empty")

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
      "run_d1_8_constlr_nogram_gaudi.sh: 4 cards, ends in CONSTLR_NOGRAM=1 exec ./chain/constlr.sh) launches its line with --no_gram in place of the Gram refreshes, into its own run dir "
      "and with the stop epoch from constlr_nogram_stop_after.txt only; a bad stop file or CONSTLR_NOGRAM is refused before launching; the argv parses to no_gram with "
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
