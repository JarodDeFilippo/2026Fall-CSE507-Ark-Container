"""CPU checks for the arm-B DINOv3 fine-tuning recipe (AdamW, layer-wise LR decay, grad-norm clipping).
Run from base/: python test_dinov3_finetune_recipe.py"""
import inspect
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import timm.optim
import torch

import engine
import joint_training
import trainer
from models import ArkViT, DINOV3_VITB16_KWARGS, vit_param_groups
from utils import clip_or_measure_grad_norm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENS = ("cls_token", "storage_tokens", "mask_token")


def build_vit():
    return ArkViT([6, 14, 14, 14], 1376, False, **DINOV3_VITB16_KWARGS)


# (a) param groups -------------------------------------------------------------------------------------------------
model = build_vit()
named = list(model.named_parameters())
groups = vit_param_groups(model, 0.04, 0.9, 0.2)
group_of = {}
for group in groups:
    assert set(group) == {"params", "weight_decay", "lr_scale"}
    for p in group["params"]:
        assert id(p) not in group_of  # every param in exactly one group
        group_of[id(p)] = group
assert set(group_of) == {id(p) for _, p in named}
assert len(groups) == 29 and len({(g["lr_scale"], g["weight_decay"]) for g in groups}) == 29
assert groups[0]["lr_scale"] == 1.0  # param_groups[0]["lr"] is what train.log and loss.csv record


def scale_of(prefix):
    scales = {group_of[id(p)]["lr_scale"] for n, p in named if n.startswith(prefix)}
    assert len(scales) == 1, (prefix, scales)
    return scales.pop()


for i in range(12):  # blocks.0 = 0.9**12 ... blocks.11 = 0.9**1
    assert math.isclose(scale_of("blocks.%d." % i), 0.9 ** (12 - i), rel_tol=1e-12)
for prefix in ("norm.", "projector.", "omni_heads."):
    assert scale_of(prefix) == 1.0
assert math.isclose(scale_of("patch_embed."), 0.9 ** 13 * 0.2, rel_tol=1e-12)
for token in TOKENS:
    assert math.isclose(scale_of(token), 0.9 ** 13, rel_tol=1e-12)
for n, p in named:
    assert group_of[id(p)]["weight_decay"] == (0.0 if p.ndim <= 1 or n in TOKENS else 0.04), n


class Wrapper(torch.nn.Module):  # stands in for DDP, which exposes the model as .module
    def __init__(self, module):
        super().__init__()
        self.module = module


def signature(param_groups):
    return [(g["lr_scale"], g["weight_decay"], [id(p) for p in g["params"]]) for g in param_groups]


assert signature(vit_param_groups(Wrapper(model), 0.04, 0.9, 0.2)) == signature(groups)
assert {g["lr_scale"] for g in vit_param_groups(model, 0.04, None, 0.2)} == {1.0, 0.2}  # None = no layer decay
model.mask_token.requires_grad_(False)
assert id(model.mask_token) not in {id(p) for g in vit_param_groups(model, 0.04, 0.9, 0.2) for p in g["params"]}
model.mask_token.requires_grad_(True)
print("OK (a) param groups: 29 groups, scales and no-decay sets as specified")

# (b) scheduler ----------------------------------------------------------------------------------------------------
p1, p2, p3 = (torch.nn.Parameter(torch.zeros(1)) for _ in range(3))
optimizer = torch.optim.SGD([{"params": [p1], "lr_scale": 1.0}, {"params": [p2], "lr_scale": 0.5}, {"params": [p3]}], lr=1.0)
scheduler = engine.WarmupCosineScheduler(optimizer, total_cycles=10, warmup_cycles=3, start_lr=1e-6, peak_lr=1e-4, end_lr=1e-6)
for cycle in range(10):
    lr = scheduler.step(cycle)
    assert optimizer.param_groups[0]["lr"] == lr
    assert optimizer.param_groups[1]["lr"] == lr / 2
    assert optimizer.param_groups[2]["lr"] == lr == scheduler.learning_rate(cycle)  # no lr_scale key: the old value
print("OK (b) scheduler: lr_scale 1.0 / 0.5 give lr and lr/2; a group without lr_scale is unchanged")

# (c) grad-norm helper ---------------------------------------------------------------------------------------------
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
net(torch.randn(5, 8)).pow(2).sum().backward()
before = [p.grad.clone() for p in net.parameters()]
gn = clip_or_measure_grad_norm(net.parameters(), None)
assert torch.is_tensor(gn) and gn.item() > 0.1
assert all(torch.equal(p.grad, b) for p, b in zip(net.parameters(), before))  # clip None: grads bitwise unchanged
reference = math.sqrt(sum(float((b.double() ** 2).sum()) for b in before))
assert abs(gn.item() - reference) < 1e-5 * reference
assert abs(clip_or_measure_grad_norm(net.parameters(), 0.1).item() - gn.item()) < 1e-6 * gn.item()  # returns the pre-clip norm
post = math.sqrt(sum(float((p.grad.double() ** 2).sum()) for p in net.parameters()))
assert post <= 0.1 + 1e-6, post
for p, b in zip(net.parameters(), before):  # back to the unclipped grads
    p.grad.copy_(b)
clip_or_measure_grad_norm(net.parameters(), None)
assert all(torch.equal(p.grad, b) for p, b in zip(net.parameters(), before))  # None is still read-only after a clip
clip_or_measure_grad_norm(net.parameters(), 0.0)
zeroed = math.sqrt(sum(float((p.grad.double() ** 2).sum()) for p in net.parameters()))
assert zeroed <= 1e-12, zeroed  # 0.0 is a threshold, not "off": it zeroes the grads
no_grad_param = torch.nn.Parameter(torch.zeros(1))
assert clip_or_measure_grad_norm([no_grad_param], None).item() == 0.0 == clip_or_measure_grad_norm([no_grad_param], 1.0).item()
print("OK (c) grad norm: clip None leaves grads bitwise intact, clip 0.1 gives post-clip norm {:.8f}, clip 0.0 zeroes the grads".format(post))

# (d) argv from the real scripts -> optparse ------------------------------------------------------------------------
try:
    import main_ark
except ImportError as error:
    print("SKIP (d): cannot import main_ark here ({})".format(error))
else:
    assert shutil.which("bash"), "bash is needed to expand common.sh"
    wrapper_env = {}
    for line in open(os.path.join(REPO, "run_d1_8a_class.sh")):
        match = re.match(r"export (ARK_\w+)=(\S+)", line)
        if match:
            wrapper_env[match.group(1)] = match.group(2)
    with tempfile.TemporaryDirectory() as tmp:
        script_dir = os.path.join(tmp, "experiment_scripts", "all4_cyclic_dinov3_adamw")
        os.makedirs(script_dir)
        shutil.copy(os.path.join(REPO, "experiment_scripts", "common.sh"), os.path.join(tmp, "experiment_scripts"))
        with open(os.path.join(tmp, "nvidia.sh"), "w") as fake_wrapper:  # prints its argv instead of launching
            fake_wrapper.write('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
        os.chmod(os.path.join(tmp, "nvidia.sh"), 0o755)
        for kind in ("start", "resume"):
            script = "{}_all4_cyclic_dinov3_adamw.sh".format(kind)
            shutil.copy(os.path.join(REPO, "experiment_scripts", "all4_cyclic_dinov3_adamw", script), script_dir)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", **wrapper_env)
            out = subprocess.run(["bash", os.path.join(script_dir, script), "100"], env=env, check=True,
                                 capture_output=True, text=True).stdout.splitlines()
            argv = out[out.index("main_ark.py") + 1:]
            assert argv.count("--opt") == 2 and argv[argv.index("--opt") + 1] == "sgd"  # common.sh's value comes first
            sys.argv = ["main_ark.py"] + argv
            options = main_ark.get_args_parser()
            assert options.opt == "adamw" and options.weight_decay == 0.04 and options.layer_decay == 0.9
            assert options.patch_embed_lr_mult == 0.2 and options.clip_grad == 3.0
            assert abs(options.lr - 3.5e-4) < 1e-12 and options.warmup_lr == 1e-6 and options.min_lr == 1e-6 and options.warmup_epochs == 20
            assert options.model_name == "vit_base_dinov3" and options.exp_name == "all4_cyclic_dinov3_adamw"
            assert options.resume is (kind == "resume")
    sys.argv = ["main_ark.py", "--exp_name", "x"]
    options = main_ark.get_args_parser()
    assert options.layer_decay is None and options.patch_embed_lr_mult == 1.0 and options.clip_grad is None  # arm A / old runs
    print("OK (d) optparse: last --opt wins; adamw / wd 0.04 / layer-decay 0.9 / clip 3.0 / lr 3.5e-4 for start and resume")

# (e) optimizer builds, steps, and survives a checkpoint round trip -------------------------------------------------
args = SimpleNamespace(model_name="vit_base_dinov3", opt="adamw", lr=1e-4, opt_eps=1e-8, momentum=0.9,
                       weight_decay=0.04, layer_decay=0.9, patch_embed_lr_mult=0.2)
optimizer = engine._create_vit_finetune_optimizer(args, model)
assert isinstance(optimizer, torch.optim.AdamW)
default_betas = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))]).defaults["betas"]
assert all(g["betas"] == default_betas for g in optimizer.param_groups)  # args without opt_betas: torch's default
for betas, expected in ((None, default_betas), ([0.8, 0.95], (0.8, 0.95))):  # --opt-betas arrives as a list, None when unset
    built = engine._create_vit_finetune_optimizer(SimpleNamespace(**dict(vars(args), opt_betas=betas)), model)
    assert all(g["betas"] == expected for g in built.param_groups), (betas, built.param_groups[0]["betas"])
print(engine._param_group_summary(optimizer))
scheduler = engine.WarmupCosineScheduler(optimizer, total_cycles=10, warmup_cycles=3, start_lr=1e-6, peak_lr=1e-4, end_lr=1e-6)
lr = scheduler.step(4)
assert all(g["lr"] == lr * g["lr_scale"] for g in optimizer.param_groups)
model.train()
optimizer.zero_grad()
sum(o.pow(2).sum() for o in model(torch.randn(1, 3, 224, 224))).backward()
weight_before = model.blocks[0].attn.proj.weight.detach().clone()
assert torch.isfinite(clip_or_measure_grad_norm(model.parameters(), 3.0))
optimizer.step()
assert not torch.equal(weight_before, model.blocks[0].attn.proj.weight)
fresh = engine._create_vit_finetune_optimizer(args, build_vit())  # a fresh run resuming from the checkpoint
fresh.load_state_dict(engine._copy_to_cpu(optimizer.state_dict()))
assert [(g["lr_scale"], g["weight_decay"], g["lr"]) for g in fresh.param_groups] == \
       [(g["lr_scale"], g["weight_decay"], g["lr"]) for g in optimizer.param_groups]
for name in ("sgd", "momentum"):  # same Nesterov convention as timm 0.5.4
    ours = engine._create_vit_finetune_optimizer(SimpleNamespace(**dict(vars(args), opt=name)), model)
    reference = timm.optim.create_optimizer_v2(torch.nn.Linear(2, 2), opt=name, lr=0.1, momentum=0.9)
    assert isinstance(ours, torch.optim.SGD) and ours.param_groups[0]["nesterov"] == reference.param_groups[0]["nesterov"]
for bad in (dict(model_name="swin_base"), dict(opt="lamb")):
    try:
        engine._create_vit_finetune_optimizer(SimpleNamespace(**dict(vars(args), **bad)), model)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for {}".format(bad))
print("OK (e) optimizer: AdamW builds and steps, opt_betas reaches it (None = torch default), lr_scale/weight_decay/lr survive state_dict -> load_state_dict, sgd/momentum match timm, bad configs raise")

# (f) the real step paths: bitwise-old with clip None, clipped with clip set, GN in the log line --------------------
class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(6, 5)
        self.omni_heads = torch.nn.ModuleList([torch.nn.Linear(5, 3), torch.nn.Linear(5, 2)])

    def forward(self, x, head_n=None, return_all=False, return_features=False):
        feat = self.fc(x)
        if return_features:
            return feat
        if return_all:
            return feat, [head(feat) for head in self.omni_heads]
        return feat, self.omni_heads[head_n](feat)


class ListLog:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def run_step_path(joint, clip_grad):
    """One batch through train_one_epoch / train_one_epoch_joint with plain SGD (lr 1, so the update is the grad)."""
    torch.manual_seed(0)
    student, teacher = TinyNet(), TinyNet()
    teacher.load_state_dict(student.state_dict())
    teacher.requires_grad_(False)
    start = [p.detach().clone() for p in student.parameters()]
    batch = (torch.randn(4, 6), torch.randn(4, 6), torch.rand(4, 3).round())
    opt = torch.optim.SGD(student.parameters(), lr=1.0)
    log = ListLog()
    schedule = np.array([0.92, 0.95])
    if joint:
        joint_training.train_one_epoch_joint(student, [batch + (torch.tensor([0, 1, 0, 1]),)], [3, 2],
                                             ["multi-label classification"] * 2, "cpu", opt, 0, "epoch", teacher,
                                             schedule, 0, train_log=log, clip_grad=clip_grad)
        logged = re.search(r"GN=([0-9.]+) \(([0-9.]+)\)", log.messages[-1])
    else:
        trainer.train_one_epoch(student, 0, "ds", [batch], "cpu", torch.nn.BCEWithLogitsLoss(), opt, 0, "epoch", teacher,
                                schedule, 0, train_log=log, print_freq=1, clip_grad=clip_grad)
        logged = re.search(r"GN=([0-9.]+)\(([0-9.]+)\) Elapsed=", log.messages[-1])
    assert logged, log.messages[-1]
    moved = math.sqrt(sum(float(((p - s) ** 2).sum()) for p, s in zip(student.parameters(), start)))
    return student, moved, float(logged.group(1)), float(logged.group(2))


def old_cyclic_step():
    """The step as it was before the GN/clip change: zero_grad, backward, step."""
    torch.manual_seed(0)
    student, teacher = TinyNet(), TinyNet()
    teacher.load_state_dict(student.state_dict())
    samples1, samples2, targets = torch.randn(4, 6), torch.randn(4, 6), torch.rand(4, 3).round()
    coff = (0.92 - 0.9) * 5
    with torch.no_grad():
        feat_t, _ = teacher(samples2, 0)
    feat_s, pred_s = student(samples1, 0)
    loss = (1 - coff) * torch.nn.BCEWithLogitsLoss()(pred_s, targets) + coff * torch.nn.MSELoss()(feat_s, feat_t)
    opt = torch.optim.SGD(student.parameters(), lr=1.0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return student


old = old_cyclic_step()
for joint in (False, True):
    student, moved, gn_value, gn_average = run_step_path(joint, None)
    assert abs(moved - gn_value) < 1e-3 and gn_value == gn_average and gn_value > 0.01  # update == grad, GN is its norm
    if not joint:
        assert all(torch.equal(a, b) for a, b in zip(student.parameters(), old.parameters()))  # bitwise-old
    _, moved_clipped, gn_clipped, _ = run_step_path(joint, 0.01)
    assert abs(gn_clipped - gn_value) < 1e-3  # GN reports the pre-clip norm
    assert moved_clipped <= 0.01 + 1e-6, moved_clipped  # the step saw the clipped grads
print("OK (f) step paths: cyclic with clip None is bitwise the old step; both paths clip before the step and log GN")

# (g) a resume with other grouped-optimizer settings raises before load_state_dict can overwrite them ----------------
resume_args = dict(model_name="vit_base_dinov3", opt="adamw", lr=1e-4, opt_eps=1e-8, momentum=0.9,
                   weight_decay=0.04, layer_decay=0.9, patch_embed_lr_mult=0.2)


def build_grouped(**changes):
    return engine._create_vit_finetune_optimizer(SimpleNamespace(**dict(resume_args, **changes)), model)


def saved_state_of(**changes):
    """checkpoint['optimizer'] as engine.py saves it, after one step (zero grads still give every param its state)."""
    built = build_grouped(**changes)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    built.step()
    return engine._copy_to_cpu(built.state_dict())


def rejection(optimizer, saved_state):
    """The helper's ValueError message, or None when it accepts the saved state."""
    try:
        engine._validate_grouped_optimizer_resume(optimizer, saved_state)
    except ValueError as error:
        return str(error)
    return None


adamw_state, sgd_state = saved_state_of(), saved_state_of(opt="sgd")
for changes, saved in ((dict(), adamw_state), (dict(opt="sgd"), sgd_state)):  # same settings as the checkpoint: accepted
    message = rejection(build_grouped(**changes), saved)
    assert message is None, message
build_grouped().load_state_dict(adamw_state)
requested = build_grouped(layer_decay=0.8)
requested.load_state_dict(adamw_state)  # the finding: torch loads it and keeps the checkpoint's lr_scale, not the requested one
assert requested.param_groups[2]["lr_scale"] == adamw_state["param_groups"][2]["lr_scale"] == 0.9
assert build_grouped(layer_decay=0.8).param_groups[2]["lr_scale"] == 0.8

message = rejection(build_grouped(layer_decay=0.8), adamw_state)
assert message and "group 2" in message and "0.9" in message and "0.8" in message, message  # names the group and both values
no_lr_scale_state = dict(adamw_state, param_groups=[{k: v for k, v in g.items() if k != "lr_scale"}
                                                    for g in adamw_state["param_groups"]])
for changes, saved, expected in (
        (dict(patch_embed_lr_mult=0.5), adamw_state, "lr_scale"),
        (dict(weight_decay=0.05), adamw_state, "group 0"),  # the first decayed group
        (dict(layer_decay=None), adamw_state, "has 29 optimizer param groups, but this run built 4"),  # torch refuses this too, less clearly
        (dict(opt="sgd"), adamw_state, "optimizer is AdamW, but this run built SGD"),  # torch loads it, the first step dies (KeyError)
        (dict(opt="adamw"), sgd_state, "optimizer is SGD, but this run built AdamW"),  # torch's own load dies (KeyError)
        (dict(), no_lr_scale_state, "nan")):  # a group that carries no lr_scale never matches
    message = rejection(build_grouped(**changes), saved)
    assert message and expected in message and "must match the original run" in message and "--layer-decay" in message, (changes, message)
resume_site = inspect.getsource(engine.omni_engine)  # the one place that restores optimizer state (--resume and --resume_from)
assert resume_site.count("optimizer.load_state_dict(") == 1 == resume_site.count("_validate_grouped_optimizer_resume(")
assert resume_site.index("_validate_grouped_optimizer_resume(") < resume_site.index("optimizer.load_state_dict(")  # validate, then load
print("OK (g) resume guard: changed --layer-decay / --patch-embed-lr-mult / --weight-decay / --opt are rejected before load_state_dict, identical settings pass")

print("ALL OK")
