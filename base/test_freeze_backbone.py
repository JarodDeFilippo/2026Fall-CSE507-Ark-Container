"""CPU checks for --freeze_backbone (arm D: frozen-backbone linear probe, only the projector and the heads train).
Run from base/: python test_freeze_backbone.py"""
import contextlib
import copy
import csv
import glob
import inspect
import io
import json
import math
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
import torch.distributed as dist

import engine
import models
import trainer
from accelerator import Accelerator
from models import ArkViT, DINOV3_VITB16_KWARGS, build_omni_model
from utils import cosine_scheduler, get_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAD_PREFIXES = ("projector.", "omni_heads.")
# the recipe's optimizer arguments (run_d1_8d_gaudi.sh): AdamW, wd 0.01, no layer decay, no patch-embed multiplier
RECIPE = SimpleNamespace(model_name="vit_base_dinov3", opt="adamw", lr=7e-5, opt_eps=1e-8, momentum=0.9,
                         weight_decay=0.01, layer_decay=None, patch_embed_lr_mult=1.0)
# same ViT with 2 blocks of width 64 on 32x32 inputs, for the steps and the file round trips
TINY = dict(DINOV3_VITB16_KWARGS, img_size=32, embed_dim=64, depth=2, num_heads=2, encoder_features=128)


class Wrapper(torch.nn.Module):  # stands in for DDP, which exposes the model as .module (parameter names get "module.")
    def __init__(self, module):
        super().__init__()
        self.module = module


class ListLog:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def rejects(exception, call, expected=None):
    """The exception message when call() raises `exception`; fails when it returns."""
    try:
        call()
    except exception as error:
        assert expected is None or expected in str(error), (expected, str(error))
        return str(error)
    raise AssertionError("expected {}".format(exception.__name__))


# (a) freezing, on the real ViT-B (no weights) ---------------------------------------------------------------------
model = ArkViT([6, 14, 14, 14], 1376, False, **DINOV3_VITB16_KWARGS)
named = list(model.named_parameters())
summary = engine._freeze_backbone(model)
for name, param in named:
    assert param.requires_grad == name.startswith(HEAD_PREFIXES), name  # backbone frozen, projector and heads trainable
frozen = [p for n, p in named if not n.startswith(HEAD_PREFIXES)]
trainable = [p for n, p in named if n.startswith(HEAD_PREFIXES)]
assert sum(p.numel() for p in trainable) == (1536 * 1376 + 1376) + (1376 * 6 + 6) + 3 * (1376 * 14 + 14) == 2181008
assert summary == "Backbone frozen: {} tensors ({:,} params) frozen, {} tensors ({:,} params) trainable".format(
    len(frozen), sum(p.numel() for p in frozen), len(trainable), sum(p.numel() for p in trainable))
print("OK (a) freeze: {}".format(summary))

# (b) optimizer: exactly the trainable parameters, nothing frozen, the recipe's schedule is accepted ---------------
optimizer = engine._create_frozen_backbone_optimizer(RECIPE, model)
assert isinstance(optimizer, torch.optim.AdamW)
held = [p for group in optimizer.param_groups for p in group["params"]]
assert {id(p) for p in held} == {id(p) for p in trainable} and len(held) == len(trainable)
assert sum(p.numel() for p in held) == sum(p.numel() for p in trainable) == 2181008
assert [g["weight_decay"] for g in optimizer.param_groups] == [0.0, 0.01]  # timm's add_weight_decay: 1-D / bias, then the rest
assert all(p.ndim == 1 for p in optimizer.param_groups[0]["params"]) and all(p.ndim == 2 for p in optimizer.param_groups[1]["params"])
wrapped = engine._create_frozen_backbone_optimizer(RECIPE, Wrapper(model))  # omni_engine passes the DDP wrapper
assert {id(p) for g in wrapped.param_groups for p in g["params"]} == {id(p) for p in trainable}
scheduler = engine.WarmupCosineScheduler(optimizer, total_cycles=5, warmup_cycles=2, start_lr=1e-6, peak_lr=7e-5, end_lr=1e-6)
warmup_lr, peak_lr = scheduler.step(0), scheduler.step(1)  # 5 cycles with 2 warmup cycles
assert math.isclose(warmup_lr, 1e-6, rel_tol=1e-9) and math.isclose(peak_lr, 7e-5, rel_tol=1e-9)
assert all(g["lr"] == peak_lr for g in optimizer.param_groups)  # timm's groups carry no lr_scale: both get the scheduled lr
assert math.isclose(scheduler.step(4), 1e-6, rel_tol=1e-9)  # the last cycle is at min_lr
rejects(RuntimeError, lambda: engine._create_frozen_backbone_optimizer(SimpleNamespace(**dict(vars(RECIPE), weight_decay=0.0)), model),
        "The optimizer holds 87,850,640 params but 2,181,008 are trainable")  # timm only skips frozen parameters when weight_decay > 0
print("OK (b) optimizer: AdamW over exactly the {:,} trainable params; with --weight-decay 0 timm hands over all {:,} and the check refuses".format(
    sum(p.numel() for p in held), sum(p.numel() for _, p in named)))

# (c) layer decay / patch-embed multiplier are refused together with the freeze ------------------------------------
refusal = "--freeze_backbone trains only projector and heads; do not pass --layer-decay / --patch-embed-lr-mult"
for changes in (dict(layer_decay=0.75), dict(patch_embed_lr_mult=0.2), dict(layer_decay=0.75, patch_embed_lr_mult=0.2)):
    assert rejects(ValueError, lambda: engine._create_frozen_backbone_optimizer(SimpleNamespace(**dict(vars(RECIPE), **changes)), model)) == refusal
opt_refusal = "--freeze_backbone supports --opt adamw, sgd or momentum, not '{}'"
for unsupported in ("adam", "nadam", "lamb", "lookahead_adamw", "fusedadam"):  # their saved groups would not validate on resume, so they never start
    assert rejects(ValueError, lambda: engine._create_frozen_backbone_optimizer(SimpleNamespace(**dict(vars(RECIPE), opt=unsupported)), model)) \
        == opt_refusal.format(unsupported)
for supported, expected_class, nesterov in (("adamw", torch.optim.AdamW, None), ("AdamW", torch.optim.AdamW, None),
                                            ("sgd", torch.optim.SGD, True), ("momentum", torch.optim.SGD, False)):
    built = engine._create_frozen_backbone_optimizer(SimpleNamespace(**dict(vars(RECIPE), opt=supported)), model)
    assert type(built) is expected_class and built.param_groups[0].get("nesterov") == nesterov, supported  # timm: sgd = Nesterov, momentum = plain
print("OK (c) --freeze_backbone with --layer-decay 0.75 / --patch-embed-lr-mult 0.2 raises ValueError; --opt adam (nadam, lamb, ...) raises ValueError, "
      "adamw / sgd / momentum build")

# (d) one step through the real trainer: backbone bit-identical, projector and head move, EMA keeps the teacher backbone --
torch.manual_seed(0)
student = ArkViT([6, 14], 32, False, **TINY)
teacher = copy.deepcopy(student)  # as in the run: the same stock backbone
teacher.requires_grad_(False)
teacher.eval()
engine._freeze_backbone(student)
tiny_optimizer = engine._create_frozen_backbone_optimizer(RECIPE, student)
engine.WarmupCosineScheduler(tiny_optimizer, total_cycles=5, warmup_cycles=2, start_lr=1e-6, peak_lr=7e-5, end_lr=1e-6).step(1)
before = {k: v.detach().clone() for k, v in student.state_dict().items()}
batch = (torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32), torch.rand(4, 6).round())
momentum_schedule = cosine_scheduler(0.9, 1, 5, 4)  # the run's: 5 cycles x 4 datasets, ema_mode epoch
trainer.train_one_epoch(student, 0, "ds", [batch], "cpu", torch.nn.BCEWithLogitsLoss(), tiny_optimizer, 0, "epoch", teacher,
                        momentum_schedule, 0, train_log=ListLog(), print_freq=1, clip_grad=3.0)  # trains head 0, then EMA
after = student.state_dict()
for key in before:
    if key.startswith(("projector.", "omni_heads.0.")):
        assert not torch.equal(before[key], after[key]), key  # trained
    else:  # backbone, buffers, and the heads of the datasets this step did not train (no gradient)
        assert torch.equal(before[key], after[key]), key
for it in range(1, len(momentum_schedule)):  # the run's remaining 19 EMA updates, student fixed
    trainer.ema_update_teacher(student, teacher, momentum_schedule, it)
teacher_state = teacher.state_dict()
backbone_keys = [k for k in after if not k.startswith(HEAD_PREFIXES)]
gap = max((teacher_state[k] - after[k]).abs().max().item() for k in backbone_keys if after[k].is_floating_point())
for key in backbone_keys:  # m*t + (1-m)*s is not bit-exact when t == s: equal to float rounding
    assert torch.allclose(teacher_state[key].float(), after[key].float(), rtol=1e-5, atol=1e-6), key
assert not any(p.requires_grad for p in teacher.parameters()) and not teacher.training
trainable_total = sum(p.numel() for p in student.parameters() if p.requires_grad)
print("OK (d) step: backbone + untouched heads bit-identical, projector + head 0 changed, teacher backbone within {:.1e} of the student's after 20 EMA updates".format(gap))

# (d2) DDP built after the freeze keeps the frozen backbone out of its reducer (a one-rank gloo group: no sockets) -----------
dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
saved_env = {key: os.environ.get(key) for key in ("RANK", "WORLD_SIZE")}
os.environ.update(RANK="0", WORLD_SIZE="2")  # makes Accelerator distributed, so wrap_model builds the DDP of a real run
try:
    accelerator = Accelerator("cpu")  # the non-CUDA branch of wrap_model, which the Gaudi run takes as well
    reducer_bytes = {}
    for order in ("freeze then wrap", "wrap then freeze"):
        torch.manual_seed(0)
        net = ArkViT([6, 14], 32, False, **TINY)
        if order == "freeze then wrap":
            engine._freeze_backbone(net)
        wrapped = accelerator.wrap_model(net)
        if order == "wrap then freeze":
            engine._freeze_backbone(net)
        logging_data = wrapped._get_ddp_logging_data()
        assert logging_data["find_unused_parameters"]  # wrap_model's setting; the unused heads (and any frozen parameter) are tolerated
        reducer_bytes[order] = logging_data["total_parameter_size_bytes"]
finally:
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    dist.destroy_process_group()
assert reducer_bytes["freeze then wrap"] == 4 * trainable_total == 19152
assert reducer_bytes["wrap then freeze"] == 4 * sum(p.numel() for p in student.parameters())  # all of it, frozen backbone included
omni_source = inspect.getsource(engine.omni_engine)  # so omni_engine freezes first
assert omni_source.index("_freeze_backbone(model)") < omni_source.index("accelerator.wrap_model(model)")
print("OK (d2) DDP: reducer holds {:,} B when frozen first vs {:,} B when frozen after wrapping; omni_engine freezes before wrap_model".format(
    reducer_bytes["freeze then wrap"], reducer_bytes["wrap then freeze"]))

# (e) resume guard: the flag is recorded in the manifest / checkpoint and a mismatch raises ---------------------------
run = SimpleNamespace(model_name="vit_base_dinov3", projector_features=1376, use_mlp=False, exp_name="x", seed=100)
metadata = {flag: engine._run_metadata(SimpleNamespace(**dict(vars(run), freeze_backbone=flag)), ["VinDrCXR"], [6], 5) for flag in (True, False)}
assert metadata[True]["freeze_backbone"] is True and metadata[False]["freeze_backbone"] is False
assert json.loads(json.dumps(metadata[True]))["freeze_backbone"] is True  # manifest.json is json.dump(metadata)
assert engine._run_metadata(run, ["VinDrCXR"], [6], 5)["freeze_backbone"] is False  # args from before the flag existed
for saved, requested in ((True, True), (False, False)):
    engine._validate_checkpoint_freeze_backbone(metadata[saved].copy(), "ckpt", SimpleNamespace(freeze_backbone=requested))
for saved, requested, advice in ((True, False, "--freeze_backbone true"), (False, True, "--freeze_backbone false")):
    rejects(ValueError, lambda: engine._validate_checkpoint_freeze_backbone(
        metadata[saved].copy(), "ckpt", SimpleNamespace(freeze_backbone=requested)), "freeze_backbone={}, but this run requests freeze_backbone={}; pass {} to match".format(saved, requested, advice))
legacy = {k: v for k, v in metadata[False].items() if k != "freeze_backbone"}  # a checkpoint from arms A-C
engine._validate_checkpoint_freeze_backbone(legacy, "ckpt", SimpleNamespace(freeze_backbone=False))
rejects(ValueError, lambda: engine._validate_checkpoint_freeze_backbone(legacy, "ckpt", SimpleNamespace(freeze_backbone=True)))
engine._validate_checkpoint_freeze_backbone(legacy, "ckpt", SimpleNamespace())  # args without the attribute behave as False
engine_source = inspect.getsource(engine.omni_engine)  # both resume paths (--resume_from and --resume) validate, like the strategy
assert engine_source.count("_validate_checkpoint_freeze_backbone(") == 2 == engine_source.count("_validate_checkpoint_training_strategy(")
assert "metadata = _run_metadata(" in engine_source and "checkpoint = metadata.copy()" in engine_source  # so the checkpoint dict carries it
print("OK (e) resume: freeze_backbone is in the manifest and checkpoint; frozen vs unfrozen mismatches raise ValueError both ways")

# (e1) resume guard for the input size: --input_size / --img_resize are recorded, a mismatch raises, a manifest from before them means 224 / 256 -----------
sized = engine._run_metadata(SimpleNamespace(**dict(vars(run), crop_size=448, resize=512)), ["VinDrCXR"], [6], 5)
assert (sized["input_size"], sized["img_resize"]) == (448, 512) and json.loads(json.dumps(sized))["input_size"] == 448
assert (engine._run_metadata(run, ["VinDrCXR"], [6], 5)["input_size"], engine._run_metadata(run, ["VinDrCXR"], [6], 5)["img_resize"]) == (224, 256)  # args without them
engine._validate_checkpoint_input_size(sized, "ckpt", SimpleNamespace(crop_size=448, resize=512))
for crop, resize, message in ((224, 512, "trained with input_size=448, but this run requests input_size=224; pass --input_size 448 to match"),
                              (448, 585, "trained with img_resize=512, but this run requests img_resize=585; pass --img_resize 512 to match")):
    rejects(ValueError, lambda: engine._validate_checkpoint_input_size(sized, "ckpt", SimpleNamespace(crop_size=crop, resize=resize)), "Checkpoint 'ckpt' was " + message)
legacy = {k: v for k, v in sized.items() if k not in ("input_size", "img_resize")}  # a manifest / checkpoint from before the keys: it trained at 224 / 256
engine._validate_checkpoint_input_size(legacy, "ckpt", SimpleNamespace(crop_size=224, resize=256))  # so a run started before them resumes unchanged
engine._validate_checkpoint_input_size(legacy, "ckpt", SimpleNamespace())  # args without the attributes: 224 / 256 too
rejects(ValueError, lambda: engine._validate_checkpoint_input_size(legacy, "ckpt", SimpleNamespace(crop_size=448, resize=512)), "trained with input_size=224, but this run requests input_size=448")
rejects(ValueError, lambda: engine._validate_checkpoint_input_size(legacy, "ckpt", SimpleNamespace(crop_size=224, resize=512)), "trained with img_resize=256, but this run requests img_resize=512")
engine_source = inspect.getsource(engine.omni_engine)  # both resume paths, and the resume from saved initial weights, check it like the strategy and the flag
assert engine_source.count("_validate_checkpoint_input_size(") == 2 and "_validate_checkpoint_input_size(manifest, weights_directory, args)" in inspect.getsource(engine._only_cycle0_saved)
print("OK (e1) resume: input_size / img_resize are in the manifest and checkpoint; a mismatch in either raises ValueError naming both values; a manifest without them is 224 / 256")

# (e2) a frozen run resumed with another optimizer is refused before load_state_dict can restore the old settings ----------
probe_net = ArkViT([6, 14], 32, False, **TINY)
engine._freeze_backbone(probe_net)


def frozen_optimizer(**changes):
    return engine._create_frozen_backbone_optimizer(SimpleNamespace(**dict(vars(RECIPE), **changes)), probe_net)


def saved_frozen_state(**changes):
    """checkpoint['optimizer'] as engine.py saves it, after one step."""
    built = frozen_optimizer(**changes)
    for p in probe_net.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    built.step()
    return engine._copy_to_cpu(built.state_dict())


adamw_state, sgd_state, momentum_state = (saved_frozen_state(opt=name) for name in ("adamw", "sgd", "momentum"))
for name, state in (("adamw", adamw_state), ("sgd", sgd_state), ("momentum", momentum_state)):  # every supported optimizer resumes
    engine._validate_grouped_optimizer_resume(frozen_optimizer(opt=name), state)  # same settings pass, though neither side's groups carry an lr_scale
    resumed = frozen_optimizer(opt=name)
    resumed.load_state_dict(state)
    assert [g["weight_decay"] for g in resumed.param_groups] == [0.0, 0.01]
requested = frozen_optimizer(weight_decay=0.05)
requested.load_state_dict(adamw_state)  # the finding: torch loads it and keeps the checkpoint's weight decay, not the requested one
assert requested.param_groups[1]["weight_decay"] == 0.01
rejects(ValueError, lambda: engine._validate_grouped_optimizer_resume(frozen_optimizer(opt="sgd"), adamw_state),
        "optimizer is AdamW, but this run built SGD")
rejects(ValueError, lambda: engine._validate_grouped_optimizer_resume(frozen_optimizer(), sgd_state),
        "optimizer is SGD, but this run built AdamW")
message = rejects(ValueError, lambda: engine._validate_grouped_optimizer_resume(frozen_optimizer(weight_decay=0.05), adamw_state), "group 1")
assert "0.01" in message and "0.05" in message and "must match the original run" in message
print("OK (e2) frozen resume: adamw / sgd / momentum validate and load with the same settings; --opt sgd and --weight-decay 0.05 against an adamw / 0.01 checkpoint raise ValueError")

# (f) phase-2 handoff: a full saved student state dict -> --pretrained_weights ----------------------------------------
stock_kwargs = models.DINOV3_VITB16_KWARGS
models.DINOV3_VITB16_KWARGS = TINY  # build_omni_model reads the module global, so it builds the small ViT
try:
    args = SimpleNamespace(model_name="vit_base_dinov3", projector_features=32, use_mlp=False, pretrained_weights=None)
    phase1 = build_omni_model(args, [6, 14])
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for name, param in phase1.named_parameters():
            if name.startswith(HEAD_PREFIXES):
                param.copy_(torch.randn(param.shape, generator=generator) + 5.0)  # distinctive, nothing like a fresh init
    with tempfile.TemporaryDirectory() as tmp:
        saved = os.path.join(tmp, "student_cycle_0005_all4_cyclic_dinov3_frozen_seed_100.pth")
        torch.save(engine._copy_to_cpu(phase1.state_dict()), saved)  # exactly what omni_engine writes
        args.pretrained_weights = saved
        student2, teacher2 = build_omni_model(args, [6, 14]), build_omni_model(args, [6, 14])  # engine.py builds both like this
        fresh = build_omni_model(SimpleNamespace(**dict(vars(args), pretrained_weights=None)), [6, 14])
        for key, value in phase1.state_dict().items():
            assert torch.equal(student2.state_dict()[key], value) and torch.equal(teacher2.state_dict()[key], value), key
            if key.startswith(HEAD_PREFIXES):
                assert not torch.equal(fresh.state_dict()[key], value), key  # loaded, not re-initialised
        assert sum(k.startswith(HEAD_PREFIXES) for k in student2.state_dict()) == 6
        # the guard still bites on a full-format file: a missing or an unexpected backbone key is an error
        full = torch.load(saved)
        for label, broken in (("missing", {k: v for k, v in full.items() if k != "blocks.0.attn.qkv.weight"}),
                              ("unexpected", dict(full, **{"extra.weight": torch.zeros(1)}))):
            bad = os.path.join(tmp, "bad_{}.pth".format(label))
            torch.save(broken, bad)
            rejects(RuntimeError, lambda: build_omni_model(SimpleNamespace(**dict(vars(args), pretrained_weights=bad)), [6, 14]),
                    "DINOv3 checkpoint does not match the backbone")
finally:
    models.DINOV3_VITB16_KWARGS = stock_kwargs
print("OK (f) phase-2 handoff: a saved student state dict passes the guard, loads projector + heads, student and teacher start identical; the guard still rejects broken files")

try:
    import main_ark
except ImportError as error:
    main_ark = None
    print("SKIP (g), (h): cannot import main_ark here ({})".format(error))
if main_ark is not None:
    # (g) argv from the real start scripts -> optparse; a bare --freeze_backbone is refused ---------------------------------
    assert shutil.which("bash"), "bash is needed to expand common.sh"
    wrapper_text = open(os.path.join(REPO, "run_d1_8d_gaudi.sh")).read()
    for line in ("#SBATCH --job-name=d1-8d-g", "#SBATCH -t 06:00:00", "#SBATCH --gres=gpu:hl225:4", "#SBATCH -c 72", "#SBATCH --mem=256G"):
        assert line in wrapper_text.splitlines(), line
    assert "./gaudi_check_cards.sh || exit 1" in wrapper_text and "./gaudi_verify_busy.sh 900 &" in wrapper_text
    assert "--layer-decay" not in wrapper_text and "--patch-embed-lr-mult" not in wrapper_text
    wrapper_env = dict(re.findall(r"^export (ARK_\w+)=(\S+)$", wrapper_text, re.M))

    def expand(experiment, *script_args):
        """experiment's start script, run with a stub container wrapper that prints its argv and 4 cards visible."""
        script = "start_{}.sh".format(experiment)
        with tempfile.TemporaryDirectory() as tmp:
            script_dir = os.path.join(tmp, "experiment_scripts", experiment)
            os.makedirs(script_dir)
            shutil.copy(os.path.join(REPO, "experiment_scripts", "common.sh"), os.path.join(tmp, "experiment_scripts"))
            with open(os.path.join(tmp, "gaudi.sh"), "w") as fake_wrapper:  # prints its argv instead of launching
                fake_wrapper.write('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
            os.chmod(os.path.join(tmp, "gaudi.sh"), 0o755)
            shutil.copy(os.path.join(REPO, "experiment_scripts", experiment, script), script_dir)
            env = dict(os.environ, HABANA_VISIBLE_MODULES="0,1,2,3", **wrapper_env)  # 4 cards
            return subprocess.run(["bash", os.path.join(script_dir, script)] + list(script_args), env=env, capture_output=True, text=True)

    def launched(finished):
        """(launcher, main_ark.py argv) printed by the stub wrapper."""
        assert finished.returncode == 0, finished.stderr
        out = finished.stdout.splitlines()
        return out[:out.index("main_ark.py")], out[out.index("main_ark.py") + 1:]

    def parse(argv):
        sys.argv = ["main_ark.py"] + list(argv)
        return main_ark.get_args_parser()

    def refused(argv):
        """The parser's stderr when it refuses argv: it has to exit with status 2, not raise."""
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                parse(argv)
        except SystemExit as stop:
            assert stop.code == 2, stop.code
            return stderr.getvalue()
        raise AssertionError("accepted {}".format(argv))

    launcher, argv = launched(expand("all4_cyclic_dinov3_frozen", "100", "gaudi"))
    assert launcher[-2:] == ["--nproc-per-node", "4"], launcher
    assert "--layer-decay" not in argv and "--patch-embed-lr-mult" not in argv
    assert argv[argv.index("--freeze_backbone") + 1] == "true"
    options = parse(argv)
    assert options.freeze_backbone is True
    assert options.opt == "adamw" and options.weight_decay == 0.01 and options.clip_grad == 3.0
    assert options.lr == 7e-5 and options.warmup_lr == 1e-6 and options.min_lr == 1e-6
    assert options.warmup_epochs == 2 and options.pretrain_epochs == 5 and options.eval_every == 1
    assert options.batch_size == 200 and options.workers == 16 and options.test_augment is False
    assert options.layer_decay is None and options.patch_embed_lr_mult == 1.0
    assert options.model_name == "vit_base_dinov3" and options.init == "dinov3" and options.projector_features == 1376
    assert options.pretrained_weights == "/workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    assert options.exp_name == "all4_cyclic_dinov3_frozen" and options.seed == 100 and options.device == "hpu" and options.resume is False
    assert options.dataset_list == ["VinDrCXR", "ChestXray14", "MIMIC", "CheXpert"]
    # the flag needs its value: a trailing or bare flag is a parser error, not an IndexError, and never swallows the next option
    for bare in (["--exp_name", "x", "--freeze_backbone"], ["--freeze_backbone", "--opt", "adamw", "--exp_name", "x"], ["--freeze_backbone", "adamw"]):
        assert "error: --freeze_backbone needs true or false" in refused(bare), bare
    assert parse(["--exp_name", "x"]).freeze_backbone is False  # arms A-C / old runs
    assert parse(["--freeze_backbone", "false"]).freeze_backbone is False and parse(["--freeze_backbone", "1"]).freeze_backbone is True
    assert parse(["--train_augment", "false", "--resume", "true"]).train_augment is False  # the shared boolean callback is unchanged
    print("OK (g) scripts: 4-card Slurm header, '--freeze_backbone true' + adamw / wd 0.01 / lr 7e-5 / 5 cycles / 2 warmup parse as specified; "
          "a bare or trailing flag is a parser error")

    # (g2) the probe start script: the frozen recipe unchanged (5 cycles, like arm D), with its own run name and the pilot export as weights ---
    export = "/workspace/ssl_dinov3_cxr_pilot_a_seed_100/export/teacher_backbone_epoch_0004.pth"
    launcher, probe_argv = launched(expand("all4_cyclic_dinov3_probe", "100", "gaudi", "all4_cyclic_dinov3_probe_a", export))
    assert launcher[-2:] == ["--nproc-per-node", "4"], launcher
    weights_args = [probe_argv[i + 1] for i, a in enumerate(probe_argv) if a == "--pretrained_weights"]
    assert weights_args == [wrapper_env["ARK_PRETRAINED_WEIGHTS"], export]  # common.sh's default first, the argument appended after it
    probe = parse(probe_argv)
    assert probe.pretrained_weights == export and probe.exp_name == "all4_cyclic_dinov3_probe_a" and probe.pretrain_epochs == 5
    assert probe.freeze_backbone is True and probe.warmup_epochs == 2 and probe.seed == 100
    assert {k for k in vars(options) if vars(options)[k] != vars(probe)[k]} == {"exp_name", "pretrained_weights"}
    probe_b = parse(launched(expand("all4_cyclic_dinov3_probe", "100", "gaudi", "all4_cyclic_dinov3_probe_b", export))[1])
    assert probe_b.exp_name == "all4_cyclic_dinov3_probe_b"  # the run dir differs per probe
    for count in (0, 1, 2, 3, 5):  # exactly <seed> <container> <experiment_name> <weights>
        usage = expand("all4_cyclic_dinov3_probe", *["100", "gaudi", "all4_cyclic_dinov3_probe_a", export, "extra"][:count])
        assert usage.returncode == 2 and "Usage:" in usage.stderr and usage.stdout == "", (count, usage.returncode, usage.stderr)
    print("OK (g2) probe script: identical to the frozen recipe (5 cycles, 2 warmup) except exp_name / --pretrained_weights (appended last, so it wins); "
          "usage error unless exactly 4 arguments")

    # (h) the whole engine on CPU (tiny ViT, random data): freeze, crash, refused resumes, resume, phase 2 ----------------
    class RandomData(torch.utils.data.Dataset):  # stands in for the CXR datasets: img_list / img_label like dataloader.py's
        def __init__(self, classes, count, seed):
            self.img_list = ["image_{}_{}".format(seed, i) for i in range(count)]
            self.img_label = (np.random.RandomState(seed).rand(count, classes) > 0.5).astype(np.float32)

        def __len__(self):
            return len(self.img_list)

        def __getitem__(self, index):
            return torch.randn(3, 32, 32), torch.randn(3, 32, 32), torch.from_numpy(self.img_label[index])

    class Crash(Exception):
        pass

    config = get_config("datasets_config.yaml")  # relative to base/
    names = ["VinDrCXR", "ChestXray14"]  # 6 and 14 classes, like the tiny student of (d)
    outputs = []

    def engine_run(run_dir, *flags, data=None):
        """omni_engine on CPU with the recipe's optimizer arguments, 5 cycles; what it printed is appended to outputs."""
        sys.argv = ["main_ark.py", "--mode", "train", "--exp_name", "smoke", "--seed", "1", "--device", "cpu", "--model", "vit_base_dinov3",
                    "--init", "dinov3", "--projector_features", "32", "--batch_size", "8", "--workers", "0", "--pretrain_epochs", "5",
                    "--warmup-epochs", "2", "--opt", "adamw", "--lr", "7e-5", "--warmup-lr", "1e-6", "--min-lr", "1e-6",
                    "--weight-decay", "0.01", "--clip-grad", "3.0", "--eval_every", "1", "--test_augment", "false",
                    "--pretrained_weights", stock_file, "--data_set", names[0], "--data_set", names[1]] + list(flags)
        splits = [[(data or RandomData)(len(config[name]["diseases"]), count, seed + i) for i, name in enumerate(names)]
                  for seed, count in ((10, 16), (20, 8), (30, 8))]  # train, validation, test
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                engine.omni_engine(main_ark.get_args_parser(), run_dir, run_dir, names, config, *splits)
        finally:
            outputs.append(printed.getvalue())

    def student_weights(run_dir, cycle):
        return torch.load(glob.glob(os.path.join(run_dir, "models", "weights", "epoch_%04d" % cycle, "student_*.pth"))[0])

    stock_train_one_epoch = engine.train_one_epoch

    def crash_in_cycle_3(*args, **kwargs):  # args[7] is the 0-based epoch
        if args[7] == 2:
            raise Crash
        return stock_train_one_epoch(*args, **kwargs)

    frozen_line = engine._freeze_backbone(student)  # the tiny ViT of (d), already frozen: same line the engine logs
    models.DINOV3_VITB16_KWARGS = TINY
    try:
        with tempfile.TemporaryDirectory() as tmp:
            frozen_dir, phase2_dir = os.path.join(tmp, "frozen"), os.path.join(tmp, "phase2")
            stock_file = os.path.join(tmp, "stock.pth")  # like the DINOv3 file: the backbone only, no projector or heads
            torch.save({k: v for k, v in ArkViT([6, 14], 32, False, **TINY).state_dict().items() if not k.startswith(HEAD_PREFIXES)}, stock_file)
            engine.train_one_epoch = crash_in_cycle_3
            rejects(Crash, lambda: engine_run(frozen_dir, "--freeze_backbone", "true"))  # cycles 1 and 2 finish, then the run dies
            engine.train_one_epoch = stock_train_one_epoch
            log = open(os.path.join(frozen_dir, "train.log")).read()
            for line in ("Freeze Backbone: True", frozen_line, "Optimizer holds {:,} params, equal to the trainable count".format(trainable_total),
                         "missing_keys=['projector.weight', 'projector.bias', 'omni_heads.0.weight'"):  # the stock-style file loads, heads stay new
                assert line in log, line
            checkpoint_path = glob.glob(os.path.join(frozen_dir, "models", "checkpoints", "cycle_0002_*.pth.tar"))[0]
            assert json.load(open(os.path.join(frozen_dir, "models", "weights", "epoch_0002", "manifest.json")))["freeze_backbone"] is True
            assert torch.load(checkpoint_path, weights_only=False)["freeze_backbone"] is True
            for flags in (("--resume", "true", "--freeze_backbone", "false"), ("--resume", "true"), ("--resume_from", checkpoint_path)):
                rejects(RuntimeError, lambda: engine_run(frozen_dir, *flags), "Resume setup failed on rank 0")  # both resume paths
                assert "trained with freeze_backbone=True, but this run requests freeze_backbone=False; pass --freeze_backbone true" in outputs[-1]
            assert json.load(open(os.path.join(frozen_dir, "models", "weights", "epoch_0002", "manifest.json")))["input_size"] == 224  # the default, recorded
            assert torch.load(checkpoint_path, weights_only=False)["img_resize"] == 256
            for resume_flags in (("--resume", "true"), ("--resume_from", checkpoint_path)):  # another input size: refused on both resume paths
                for other, reason in ((("--input_size", "448"), "trained with input_size=224, but this run requests input_size=448; pass --input_size 224 to match"),
                                      (("--img_resize", "512"), "trained with img_resize=256, but this run requests img_resize=512; pass --img_resize 256 to match")):
                    rejects(RuntimeError, lambda: engine_run(frozen_dir, *resume_flags, "--freeze_backbone", "true", *other), "Resume setup failed on rank 0")
                    assert reason in outputs[-1], (resume_flags, other)
            for other, reason in ((("--opt", "sgd"), "optimizer is AdamW, but this run built SGD"),
                                  (("--weight-decay", "0.05"), "Optimizer param group 1")):  # the one restore path serves both resume flags
                for resume_flags in (("--resume", "true"), ("--resume_from", checkpoint_path)):
                    rejects(ValueError, lambda: engine_run(frozen_dir, *resume_flags, "--freeze_backbone", "true", *other), reason)
            freezes_before = open(os.path.join(frozen_dir, "train.log")).read().count(frozen_line)  # the crashed launch + the refused ones
            engine_run(frozen_dir, "--resume", "true", "--freeze_backbone", "true")  # picks up at cycle 3
            log = open(os.path.join(frozen_dir, "train.log")).read()
            assert all("Cycle {:04d}: learning rate".format(cycle) in log for cycle in range(1, 6)) and log.count(frozen_line) == freezes_before + 1
            first, last = student_weights(frozen_dir, 1), student_weights(frozen_dir, 5)
            backbone = [k for k in last if not k.startswith(HEAD_PREFIXES)]
            assert all(torch.equal(first[k], last[k]) for k in backbone)  # untouched across the crash and the resume
            assert not torch.equal(first["projector.weight"], last["projector.weight"])
            student_file = glob.glob(os.path.join(frozen_dir, "models", "weights", "epoch_0005", "student_*.pth"))[0]
            engine_run(phase2_dir, "--pretrained_weights", student_file, "--pretrain_epochs", "3")  # phase 2: a new run, nothing frozen
            log = open(os.path.join(phase2_dir, "train.log")).read()
            assert "Freeze Backbone: False" in log and "Backbone frozen" not in log
            assert log.count("Loaded with msg: <All keys matched successfully>") == 2  # student and teacher
            assert any(not torch.equal(student_weights(phase2_dir, 3)[k], last[k]) for k in backbone)  # the backbone trains again
            rejects(ValueError, lambda: engine_run(os.path.join(tmp, "adam"), "--freeze_backbone", "true", "--opt", "adam"),
                    "--freeze_backbone supports --opt adamw, sgd or momentum, not 'adam'")  # a run that could not be resumed never starts
            for name in ("sgd", "momentum"):  # the other supported optimizers survive a crash + resume too
                other_dir = os.path.join(tmp, name)
                engine.train_one_epoch = crash_in_cycle_3
                rejects(Crash, lambda: engine_run(other_dir, "--freeze_backbone", "true", "--opt", name))
                engine.train_one_epoch = stock_train_one_epoch
                engine_run(other_dir, "--resume", "true", "--freeze_backbone", "true", "--opt", name)
                first_cycle, last_cycle = student_weights(other_dir, 1), student_weights(other_dir, 5)
                assert all(torch.equal(first_cycle[k], last_cycle[k]) for k in backbone), name
                assert not torch.equal(first_cycle["projector.weight"], last_cycle["projector.weight"]), name
    finally:
        engine.train_one_epoch = stock_train_one_epoch
        models.DINOV3_VITB16_KWARGS = stock_kwargs
    print("OK (h) engine: frozen run (from a stock-style backbone-only weights file) logs the freeze and the optimizer, a crash + resume keeps the backbone bit-identical, "
          "mismatched resumes (flag, --opt, --weight-decay, --input_size, --img_resize; --resume and --resume_from) are refused, a phase-2 run from the student file trains everything, "
          "--opt adam never starts, sgd and momentum crash + resume")

    # (i) --eval_cycle0: the initial weights are saved as epoch_0000 and scored as cycle 0 (evaluation_point init) before cycle 1 -------
    class FixedData(RandomData):  # the same images on every pass, so a score depends on the weights alone
        def __init__(self, classes, count, seed):
            super().__init__(classes, count, seed)
            self.seed = seed

        def __getitem__(self, index):
            image = torch.randn(3, 32, 32, generator=torch.Generator().manual_seed(1000 * self.seed + index))
            return image, image.clone(), torch.from_numpy(self.img_label[index])

    def seeded_run(run_dir, *flags):
        """engine_run for 3 cycles on FixedData, after seeding the three RNGs like main_ark.main."""
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        engine_run(run_dir, "--pretrain_epochs", "3", *flags, data=FixedData)

    def test_mode(weights_dir, out_dir):
        """main_ark --mode test on the saved weights (what eval_d1_6.sh runs), on engine_run's test split; returns the result rows."""
        sys.argv = ["main_ark.py", "--mode", "test", "--device", "cpu", "--model", "vit_base_dinov3", "--projector_features", "32",
                    "--batch_size", "8", "--workers", "0", "--test_augment", "false", "--pretrained_weights", weights_dir,
                    "--data_set", names[0], "--data_set", names[1]]
        test_split = [FixedData(len(config[name]["diseases"]), 8, 30 + i) for i, name in enumerate(names)]
        with contextlib.redirect_stdout(io.StringIO()):
            engine.omni_engine(main_ark.get_args_parser(), out_dir, out_dir, names, config, [], [], test_split)
        with open(engine._test_output_path(out_dir, weights_dir), newline="") as handle:
            return list(csv.reader(handle))

    def weights(run_dir, cycle):
        return os.path.join(run_dir, "models", "weights", "epoch_%04d" % cycle)

    def csv_text(run_dir, dataset, split):
        with open(os.path.join(run_dir, "evaluation", dataset, "{}_performance.csv".format(split)), newline="") as handle:
            return handle.read()

    def init_rows(run_dir):
        """{(dataset, split): the cycle-0 rows}, read from the evaluation CSVs."""
        return {(dataset, split): [row for row in csv.reader(io.StringIO(csv_text(run_dir, dataset, split))) if row[0] == "0"]
                for dataset in names for split in ("test", "val")}

    def file_bytes(directory):
        return {name: open(os.path.join(directory, name), "rb").read() for name in sorted(os.listdir(directory))}

    def tensors(path):
        return torch.load(path, weights_only=False)

    def same_value(a, b):
        return float(a) == float(b) or (math.isnan(float(a)) and math.isnan(float(b)))

    def log_of(run_dir):
        return open(os.path.join(run_dir, "train.log")).read()

    header = ["cycle", "epoch", "evaluation_point", "metric", "student", "teacher"]
    assert parse(["--exp_name", "x"]).eval_cycle0 is False and parse(["--eval_cycle0", "true"]).eval_cycle0 is True
    assert parse(["--eval_cycle0", "false", "--resume", "true"]).eval_cycle0 is False
    models.DINOV3_VITB16_KWARGS = TINY
    try:
        with tempfile.TemporaryDirectory() as tmp:
            stock_file = os.path.join(tmp, "stock.pth")  # the backbone only, like the DINOv3 file: student and teacher get their own random heads
            torch.save({k: v for k, v in ArkViT([6, 14], 32, False, **TINY).state_dict().items() if not k.startswith(HEAD_PREFIXES)}, stock_file)
            off, on = os.path.join(tmp, "off"), os.path.join(tmp, "on")
            seeded_run(off)
            seeded_run(on, "--eval_cycle0", "true")
            # flag off: no epoch_0000, no cycle-0 rows, no cycle-0 line
            assert not os.path.exists(weights(off, 0)) and "cycle 0 (init)" not in log_of(off)
            assert all(rows == [] for rows in init_rows(off).values())
            # flag on: the cycle-0 rows come first (epoch 0, init), one per metric, both columns filled; then the flag-off run's rows byte for byte,
            # since the RNGs are restored after the cycle-0 evaluation
            on_init = init_rows(on)
            for dataset in names:
                for split in ("test", "val"):
                    on_text, off_text = csv_text(on, dataset, split), csv_text(off, dataset, split)
                    on_rows = list(csv.reader(io.StringIO(on_text)))
                    init = on_init[dataset, split]
                    assert on_rows[0] == header and on_rows[1:1 + len(init)] == init, (dataset, split)
                    assert all(row[1:3] == ["0", "init"] and row[4] != "" and row[5] != "" for row in init), init
                    expected = (["validation_loss"] if split == "val" else []) + ["AUC_" + d for d in config[dataset]["diseases"]] + ["mAUC"]
                    assert [row[3] for row in init] == expected, (dataset, split)
                    assert on_text.endswith(off_text.split("\n", 1)[1]) and len(on_rows) == len(init) + off_text.count("\n"), (dataset, split)
            for cycle in (1, 2, 3):  # and the weights of every cycle are the flag-off run's
                for role in ("student", "teacher"):
                    on_state = tensors(glob.glob(os.path.join(weights(on, cycle), role + "_*.pth"))[0])
                    off_state = tensors(glob.glob(os.path.join(weights(off, cycle), role + "_*.pth"))[0])
                    assert on_state.keys() == off_state.keys() and all(torch.equal(on_state[k], off_state[k]) for k in on_state), (cycle, role)
            # epoch_0000: exactly epoch_0001's layout and manifest (cycle 0), holding the initial weights; the loaders accept it
            assert sorted(os.listdir(weights(on, 0))) == [name.replace("cycle_0001", "cycle_0000") for name in sorted(os.listdir(weights(on, 1)))] == [
                "manifest.json", "student_cycle_0000_smoke_seed_1.pth", "teacher_cycle_0000_smoke_seed_1.pth"]
            manifest = engine._load_weight_manifest(weights(on, 0))
            assert manifest == dict(engine._load_weight_manifest(weights(on, 1)), cycle=0)
            student0, teacher0, _ = engine._load_test_state_dicts(weights(on, 0))
            engine._validate_test_metadata(manifest, weights(on, 0), parse(["--model", "vit_base_dinov3", "--projector_features", "32"]),
                                           names, [6, 14], require_model_metadata=True)
            assert engine._test_output_path(tmp, weights(on, 0)) == os.path.join(tmp, "cycle_0000_smoke_seed_1.csv")
            stock = torch.load(stock_file)
            assert all(torch.equal(student0[k], stock[k]) and torch.equal(teacher0[k], stock[k]) for k in stock)  # the backbone file, untrained
            # student and teacher are built separately (two build_omni_model calls), so with a backbone-only file their projector and
            # heads start different: the two columns differ, and each is that model's own score
            assert any(not torch.equal(student0[k], teacher0[k]) for k in student0 if k.startswith(HEAD_PREFIXES))
            init_val = {row[3]: row for row in on_init[names[0], "val"]}
            assert init_val["validation_loss"][4] != init_val["validation_loss"][5]
            # an independent scoring of epoch_0000 by test mode (eval_d1_6.sh's path) gives the training run's cycle-0 test values
            scored = {(row[0], row[1], row[2] if row[2] == "mAUC" else "AUC_" + row[3]): row[4]
                      for row in test_mode(weights(on, 0), os.path.join(tmp, "scored"))[1:]}
            for dataset in names:
                for row in on_init[dataset, "test"]:
                    assert same_value(row[4], scored[dataset, "student", row[3]]) and same_value(row[5], scored[dataset, "teacher", row[3]]), row
            line = [line for line in log_of(on).splitlines() if "cycle 0 (init) evaluated" in line]
            assert len(line) == 1 and re.search(r" INFO cycle 0 \(init\) evaluated: test mAUC VinDrCXR student=\S+ teacher=\S+, ChestXray14 student=\S+ teacher=\S+; weights in .*epoch_0000$", line[0]), line
            # a crash in cycle 3, then --resume_from into a new run and --resume in place: epoch_0000 and the cycle-0 rows are kept, never redone
            crashed, branched = os.path.join(tmp, "crashed"), os.path.join(tmp, "branched")
            engine.train_one_epoch = crash_in_cycle_3
            rejects(Crash, lambda: seeded_run(crashed, "--eval_cycle0", "true"))
            engine.train_one_epoch = stock_train_one_epoch
            assert init_rows(crashed) == on_init and file_bytes(weights(crashed, 0)) == file_bytes(weights(on, 0))  # same seed, same init
            saved0 = file_bytes(weights(crashed, 0))
            checkpoint_path = glob.glob(os.path.join(crashed, "models", "checkpoints", "cycle_0002_*.pth.tar"))[0]
            seeded_run(branched, "--resume_from", checkpoint_path, "--eval_cycle0", "true")
            seeded_run(crashed, "--resume", "true", "--eval_cycle0", "true")
            for run_dir in (crashed, branched):
                assert file_bytes(weights(run_dir, 0)) == saved0 and init_rows(run_dir) == on_init, run_dir
                assert log_of(run_dir).count("cycle 0 (init) evaluated") == 1 and "Cycle 0003: learning rate" in log_of(run_dir), run_dir
                assert "=> loaded checkpoint" in log_of(run_dir) and "no valid cycle checkpoint yet" not in log_of(run_dir), run_dir  # cycle 2's, not epoch_0000
            # stopped after epoch_0000 but before the first cycle checkpoint (in cycle 1's second dataset, after cycle 1's first evaluation
            # and loss rows): a plain start is still refused; --resume drops the partial cycle 1, starts it from epoch_0000 without redoing
            # cycle 0, and ends where the uninterrupted run did, byte for byte. Without an epoch_0000 such a run still cannot be resumed.
            def crash_in_cycle_1_second_dataset(*args, **kwargs):  # args[1] is the dataset index, args[7] the 0-based epoch
                if args[7] == 0 and args[1] == 1:
                    raise Crash
                return stock_train_one_epoch(*args, **kwargs)

            stranded, plain = os.path.join(tmp, "stranded"), os.path.join(tmp, "plain")
            engine.train_one_epoch = crash_in_cycle_1_second_dataset
            rejects(Crash, lambda: seeded_run(stranded, "--eval_cycle0", "true"))
            rejects(Crash, lambda: seeded_run(plain))
            engine.train_one_epoch = stock_train_one_epoch
            assert not os.listdir(os.path.join(stranded, "models", "checkpoints")) and file_bytes(weights(stranded, 0)) == saved0
            assert any(row[0] == "1" for row in csv.reader(io.StringIO(csv_text(stranded, names[0], "test"))))  # cycle 1's partial rows
            rejects(FileExistsError, lambda: seeded_run(stranded, "--eval_cycle0", "true"), "Training run directory already exists")
            torch.manual_seed(5)
            other_stock = os.path.join(tmp, "other_stock.pth")
            torch.save({k: v for k, v in ArkViT([6, 14], 32, False, **TINY).state_dict().items() if not k.startswith(HEAD_PREFIXES)}, other_stock)
            seeded_run(stranded, "--resume", "true", "--eval_cycle0", "true", "--pretrained_weights", other_stock)
            assert log_of(stranded).count("cycle 0 (init) evaluated") == 1
            assert log_of(stranded).count("=> no valid cycle checkpoint yet: cycle 1 starts from the initial weights") == 1
            assert "ignored and removed checkpoint files" not in log_of(stranded)
            assert init_rows(stranded) == on_init and file_bytes(weights(stranded, 0)) == saved0
            for dataset in names:
                for split in ("test", "val"):
                    assert csv_text(stranded, dataset, split) == csv_text(on, dataset, split), (dataset, split)
            assert open(os.path.join(stranded, "loss.csv")).read() == open(os.path.join(on, "loss.csv")).read()
            for role in ("student", "teacher"):
                stranded_state = tensors(glob.glob(os.path.join(weights(stranded, 3), role + "_*.pth"))[0])
                on_state = tensors(glob.glob(os.path.join(weights(on, 3), role + "_*.pth"))[0])
                assert all(torch.equal(stranded_state[k], on_state[k]) for k in on_state), role
            rejects(RuntimeError, lambda: seeded_run(plain, "--resume", "true"), "Resume setup failed on rank 0")
            assert "No valid cycle checkpoint found" in outputs[-1]
            # the same stop, plus a cycle-1 checkpoint cut short by the kill: a changed recipe is refused as on a checkpoint resume, nothing
            # touched; with the run's recipe the partial file does not count as a checkpoint: it is reported and removed, and cycle 1
            # starts from epoch_0000 as above
            def tree_bytes(root):
                return {os.path.relpath(os.path.join(d, f), root): open(os.path.join(d, f), "rb").read() for d, _, fs in os.walk(root) for f in fs}

            partial = os.path.join(tmp, "partial")
            engine.train_one_epoch = crash_in_cycle_1_second_dataset
            rejects(Crash, lambda: seeded_run(partial, "--eval_cycle0", "true"))
            engine.train_one_epoch = stock_train_one_epoch
            whole = open(glob.glob(os.path.join(on, "models", "checkpoints", "cycle_0003_*.pth.tar"))[0], "rb").read()
            partial_file = os.path.join(partial, "models", "checkpoints", "cycle_0001_smoke_seed_1.pth.tar")
            with open(partial_file, "wb") as handle:
                handle.write(whole[:len(whole) // 2])
            before = tree_bytes(partial)
            for changed, reason in ((("--training_strategy", "joint"), "uses training strategy 'cyclic', but this run requests 'joint'"),
                                    (("--freeze_backbone", "true"), "was trained with freeze_backbone=False, but this run requests freeze_backbone=True"),
                                    (("--input_size", "448"), "was trained with input_size=224, but this run requests input_size=448")):
                rejects(RuntimeError, lambda: seeded_run(partial, "--resume", "true", "--eval_cycle0", "true", *changed), "Resume setup failed on rank 0")
                assert reason in outputs[-1] and "epoch_0000" in outputs[-1] and tree_bytes(partial) == before, changed
            seeded_run(partial, "--resume", "true", "--eval_cycle0", "true")
            reported = [line for line in log_of(partial).splitlines() if "=> no valid cycle checkpoint yet" in line]
            assert len(reported) == 1 and "ignored and removed checkpoint files that do not load: ['{}']".format(partial_file) in reported[0], reported
            assert log_of(partial).count("cycle 0 (init) evaluated") == 1 and init_rows(partial) == on_init
            for dataset in names:
                for split in ("test", "val"):
                    assert csv_text(partial, dataset, split) == csv_text(on, dataset, split), (dataset, split)
            assert [os.path.basename(p) for p in glob.glob(os.path.join(partial, "models", "checkpoints", "*"))] == ["cycle_0003_smoke_seed_1.pth.tar"]
            # stopped inside the cycle-0 evaluation (after the first dataset's rows): epoch_0000 is written only after it, so there is none,
            # and the run is not taken for one whose cycle 0 is complete
            stock_test_classification, calls = engine.test_classification, []

            def crash_in_third_test_pass(*args, **kwargs):
                calls.append(1)
                if len(calls) == 3:
                    raise Crash
                return stock_test_classification(*args, **kwargs)

            # killed between the student and the teacher save of epoch_0000: the weights are published by renaming a complete sibling
            # directory, so there is no epoch_0000 (only the hidden .partial-epoch_0000, without a manifest) and --resume is refused
            stock_save, saves = torch.save, []

            def crash_in_second_save(*args, **kwargs):
                saves.append(1)
                if len(saves) == 2:
                    raise Crash
                return stock_save(*args, **kwargs)

            mid_save = os.path.join(tmp, "mid_save")
            torch.save = crash_in_second_save
            try:
                rejects(Crash, lambda: seeded_run(mid_save, "--eval_cycle0", "true"))
            finally:
                torch.save = stock_save
            assert os.listdir(os.path.join(mid_save, "models", "weights")) == [".partial-epoch_0000"]
            assert os.listdir(os.path.join(mid_save, "models", "weights", ".partial-epoch_0000")) == ["student_cycle_0000_smoke_seed_1.pth"]
            rejects(RuntimeError, lambda: seeded_run(mid_save, "--resume", "true", "--eval_cycle0", "true"), "Resume setup failed on rank 0")
            assert "No valid cycle checkpoint found" in outputs[-1]
            assert all(not name.startswith(".partial") for run_dir in (on, off, stranded, partial) for name in os.listdir(os.path.join(run_dir, "models", "weights")))
            mid_eval = os.path.join(tmp, "mid_eval")
            engine.test_classification = crash_in_third_test_pass
            try:
                rejects(Crash, lambda: seeded_run(mid_eval, "--eval_cycle0", "true"))
            finally:
                engine.test_classification = stock_test_classification
            first_rows = [row for row in csv.reader(io.StringIO(csv_text(mid_eval, names[0], "test"))) if row[0] == "0"]
            assert first_rows == on_init[names[0], "test"] and not os.path.exists(weights(mid_eval, 0))
            rejects(RuntimeError, lambda: seeded_run(mid_eval, "--resume", "true", "--eval_cycle0", "true"), "Resume setup failed on rank 0")
            # the resume truncation keeps cycle 0 (0 <= max_cycle)
            trimmed = os.path.join(tmp, "trim.csv")
            for max_cycle, kept in ((2, ["0", "1", "2"]), (0, ["0"])):
                with open(trimmed, "w", newline="") as handle:
                    csv.writer(handle).writerows([header] + [[c, 0, "init" if c == 0 else "after_x", "mAUC", 0.5, 0.5] for c in range(4)])
                engine._rewrite_csv_through_cycle(trimmed, max_cycle)
                with open(trimmed, newline="") as handle:
                    assert [row[0] for row in csv.reader(handle)] == ["cycle"] + kept, max_cycle
            # initial weights that hold the projector and heads too (a saved student file): student == teacher at init, and both columns agree
            full = os.path.join(tmp, "full")
            student_file = glob.glob(os.path.join(weights(on, 3), "student_*.pth"))[0]
            seeded_run(full, "--eval_cycle0", "true", "--pretrained_weights", student_file)
            student0, teacher0, _ = engine._load_test_state_dicts(weights(full, 0))
            assert all(torch.equal(student0[k], teacher0[k]) for k in student0)
            for (dataset, split), init in init_rows(full).items():
                assert init and all(row[4] == row[5] for row in init), (dataset, split, init)
    finally:
        engine.train_one_epoch = stock_train_one_epoch
        models.DINOV3_VITB16_KWARGS = stock_kwargs
    print("OK (i) --eval_cycle0: epoch_0000 (epoch_0001's layout, manifest cycle 0, accepted by the test-mode loader) holds the initial weights; "
          "the cycle-0 rows (epoch 0, init) come first in test/val CSVs, then the flag-off run's rows byte for byte and bit-identical cycle 1-3 weights; "
          "test mode re-scores epoch_0000 to the same values; a backbone-only init gives different student / teacher columns, a full init equal ones; "
          "--resume and --resume_from keep cycle 0 and never redo it; a run stopped after epoch_0000 and before its first checkpoint "
          "resumes from epoch_0000 to the uninterrupted run's outputs (a partial cycle-1 checkpoint is reported and removed, a changed recipe refused); "
          "flag off: no epoch_0000, no cycle-0 rows")

    # (j) an in-training evaluation leaves the training RNG streams where they were (engine._preserved_rng): a run evaluated every cycle trains exactly like
    # one evaluated every 2nd cycle. engine_run's RandomData draws every image from the global torch RNG, the evaluation sets' too, so any draw an
    # evaluation leaves behind would show in what training draws next
    entry_states = []

    def recording_train_one_epoch(*args, **kwargs):  # the RNG states each dataset pass starts from, read (not drawn): (cycle, dataset, random, numpy, torch)
        entry_states.append((args[7] + 1, args[1], random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state().numpy().tobytes()))
        return stock_train_one_epoch(*args, **kwargs)

    def cadence_run(run_dir, every):
        """engine_run for 4 cycles evaluated every `every` cycles (and the last), seeded like main_ark.main: (entry states, cycle-4 student, evaluated cycles)."""
        del entry_states[:]
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        engine.train_one_epoch = recording_train_one_epoch
        try:
            engine_run(run_dir, "--pretrain_epochs", "4", "--eval_every", every)
        finally:
            engine.train_one_epoch = stock_train_one_epoch
        evaluated = sorted({int(row[0]) for row in csv.reader(io.StringIO(csv_text(run_dir, names[0], "test"))) if row[0].isdigit()})
        return list(entry_states), student_weights(run_dir, 4), evaluated

    stock_preserved_rng = engine._preserved_rng
    models.DINOV3_VITB16_KWARGS = TINY
    try:
        with tempfile.TemporaryDirectory() as tmp:
            stock_file = os.path.join(tmp, "stock.pth")
            torch.save({k: v for k, v in ArkViT([6, 14], 32, False, **TINY).state_dict().items() if not k.startswith(HEAD_PREFIXES)}, stock_file)
            every1, every2 = cadence_run(os.path.join(tmp, "every1"), "1"), cadence_run(os.path.join(tmp, "every2"), "2")
            assert (every1[2], every2[2]) == ([1, 2, 3, 4], [2, 4]), (every1[2], every2[2])  # the cadences really differ
            assert len(every1[0]) == len(every2[0]) == 4 * len(names) and [state[:2] for state in every1[0]] == [state[:2] for state in every2[0]]
            assert every1[0] == every2[0], [state[:2] for a, state in zip(every1[0], every2[0]) if a != state]  # every pass starts from the same RNG states
            assert list(every1[1]) == list(every2[1]) and all(torch.equal(every1[1][k], every2[1][k]) for k in every1[1])  # and trains to bit-identical weights
            # the negative control: without the restore, the extra evaluations of the every-cycle run move what training draws next
            engine._preserved_rng = lambda device: contextlib.nullcontext()
            unpreserved = cadence_run(os.path.join(tmp, "unpreserved"), "1")
            engine._preserved_rng = stock_preserved_rng
            # (the first pass is the same; the second already starts after cycle 1's after_VinDrCXR evaluation, which the every-2nd-cycle run does not do)
            assert unpreserved[0][0] == every2[0][0] and unpreserved[0][1] != every2[0][1], "the check cannot see a leaked evaluation draw"
            assert any(not torch.equal(unpreserved[1][k], every2[1][k]) for k in every2[1])
    finally:
        engine._preserved_rng = stock_preserved_rng
        engine.train_one_epoch = stock_train_one_epoch
        models.DINOV3_VITB16_KWARGS = stock_kwargs
    omni_source = inspect.getsource(engine.omni_engine)  # cycle 0, the cyclic after_<dataset> and the joint after_joint evaluations, test + validation inside one block each
    assert omni_source.count("with _preserved_rng(device):") == 3 and omni_source.count("_evaluate_test_sets(") == omni_source.count("_evaluate_validation_sets(") == 3
    assert "random.setstate" not in omni_source and "torch.set_rng_state" not in omni_source  # one helper does it
    print("OK (j) RNG and evaluation cadence: with eval_every 1 (cycles 1-4 evaluated) and eval_every 2 (cycles 2, 4) every training pass starts from the same "
          "random / numpy / torch RNG states and cycle 4's weights are bit-identical; without the restore the every-cycle run diverges from its first evaluation on; "
          "omni_engine wraps its three evaluation blocks (cycle 0, after_<dataset>, after_joint) in _preserved_rng; cycle 0 as in (i)")

print("ALL OK")
