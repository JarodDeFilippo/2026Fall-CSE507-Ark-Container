"""The STUDENT backbone of a Stage A epoch checkpoint as a flat backbone file, laid out like export/teacher_backbone_epoch_NNNN.pth (ssl_pretrain.save_state exports only the EMA teacher's),
so the probes can take it as their weights. Run from base/ (nvidia.sh does; the repo-root export_student.sh submits it as a short CPU job):
    python -m ssl_dinov3.export_student --ckpt <run>/ckpt/epoch_NNNN.pth --check <run>/export/teacher_backbone_epoch_NNNN.pth --out <run>/export/student_backbone_epoch_NNNN.pth
--check proves the mapping before it is used: the same mapping applied to the checkpoint's teacher must give that file, tensor for tensor. Any mismatch, or an existing --out, exits 1 and writes nothing."""
import argparse
import os
import sys
from types import SimpleNamespace

import torch

from ssl_dinov3.model import export_backbone


def backbone_of(state):
    """The flat backbone state dict inside the state dict of a student / teacher nn.ModuleDict (backbone., dino_head., ibot_head. keys, as save_state writes them; the student is saved unwrapped,
    so no DDP module. prefix, and the trainer loads these dicts without stripping any): export_backbone(module.backbone) run on the state dict, so local_cls_norm is left out exactly as there."""
    return export_backbone(SimpleNamespace(state_dict=lambda: {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}))


def problem(got, want, exact):
    """Why the flat state dict `got` is not `want`'s layout (same keys, shapes and dtypes; with `exact` the same values too), or None."""
    if got.keys() != want.keys():
        return f"keys differ, only in the checkpoint's: {sorted(got.keys() - want.keys())[:3]}, only in the check file's: {sorted(want.keys() - got.keys())[:3]}"
    for k, w in want.items():
        if got[k].shape != w.shape or got[k].dtype != w.dtype:
            return f"{k}: {tuple(got[k].shape)} {got[k].dtype}, the check file's {tuple(w.shape)} {w.dtype}"
        if exact and not torch.equal(got[k], w):
            return f"{k}: same shape, different values"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--ckpt", "--check", "--out"):
        parser.add_argument(flag, required=True)
    args = parser.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists: not overwriting it")
    # weights_only=False: the cluster image's pickle protocol 4 (and the checkpoint's RNG states), as ssl_pretrain loads its checkpoints and models.py its exports
    ckpt, check = (torch.load(f, map_location="cpu", weights_only=False) for f in (args.ckpt, args.check))
    teacher, student = backbone_of(ckpt["teacher"]), backbone_of(ckpt["student"])
    del ckpt  # only the two backbone copies above are used from here; the rest of the file (optimizer state, heads) is most of its GBs
    for what, got, exact in (("teacher: the mapping does not reproduce", teacher, True), ("student: not laid out like", student, False)):
        why = problem(got, check, exact)
        if why:
            sys.exit(f"{what} {args.check} ({why}); nothing written")
    torch.save(student, args.out + ".tmp")  # as ssl_pretrain.save_atomic: the file appears whole or not at all
    os.replace(args.out + ".tmp", args.out)
    flat = lambda sd: torch.cat([sd[k].double().flatten() for k in check if check[k].is_floating_point()])
    s, t = flat(student), flat(teacher)
    print(f"{args.out}: {len(student)} keys, mean abs diff {(s - t).abs().mean().item():.6g}, cosine {torch.cosine_similarity(s, t, dim=0).item():.8f}")


if __name__ == "__main__":
    main()
