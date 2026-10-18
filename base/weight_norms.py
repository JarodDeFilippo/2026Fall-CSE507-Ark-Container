"""Per-block Frobenius norm of the DINOv3 backbone weights for saved Ark+ cycles, student and teacher (CPU only).

Run from base/:  python weight_norms.py <run_dir> <cycle> [<cycle> ...] [--out norms.csv] [--verbose]   or  --selfcheck
Reads <run_dir>/models/weights/epoch_<NNNN>/{student,teacher}_cycle_*.pth, which engine.py saves as plain CPU state dicts.
Counted: 2-D tensors under blocks.<i>. (attn.qkv, attn.proj, mlp.fc1, mlp.fc2) and patch_embed.proj.weight (a 4-D conv kernel,
flattened); "total" is the joint norm of those. Biases, norms, LayerScale gammas, tokens, rope, projector and omni_heads are
skipped; --verbose lists both groups for the first checkpoint. Exits 3 if torch.load fails (e.g. HPU-tagged tensors):
rerun inside the Gaudi container.
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

import torch

# Real keys are bare "blocks.<i>." (ArkViT subclasses DinoVisionTransformer); the optional prefix tolerates module./backbone.
BLOCK = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
PATCH = re.compile(r"(?:^|\.)patch_embed\.proj\.weight$")


def find_weight(weights_dir, role):
    # Same lookup as engine._find_saved_weight: <role>.pth if present, else exactly one <role>_cycle_*.pth.
    legacy = os.path.join(weights_dir, role + ".pth")
    if os.path.isfile(legacy):
        return legacy
    paths = glob.glob(os.path.join(weights_dir, role + "_cycle_*.pth"))
    if len(paths) != 1:
        sys.exit("Expected one {} weight file in {}, found {}".format(role, weights_dir, len(paths)))
    return paths[0]


def load_state_dict(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # same call as engine._load_test_state_dicts
    except Exception as exc:
        print("torch.load failed on {}: {}: {} (HPU-saved? rerun inside the Gaudi container)".format(
            path, type(exc).__name__, " ".join(str(exc).split())), file=sys.stderr)
        sys.exit(3)


def norms(state_dict, verbose=False):
    """{block index | "patch_embed" | "total": (Frobenius norm, n_params)} over the counted tensors."""
    sq, count, skipped = {}, {}, set()
    for key, tensor in state_dict.items():
        m = BLOCK.search(key)  # block index for 2-D block tensors, else the patch-embed kernel, else not counted
        lab = int(m.group(1)) if m and tensor.ndim == 2 else "patch_embed" if PATCH.search(key) else None
        if lab is None:
            skipped.add(re.sub(r"blocks\.\d+\.", "blocks.*.", key))
            continue
        if verbose:
            print("  counted {} {}".format(key, tuple(tensor.shape)))
        sq[lab] = sq.get(lab, 0.0) + tensor.double().pow(2).sum().item()
        count[lab] = count.get(lab, 0) + tensor.numel()
    if verbose:
        print("\n".join("  skipped " + key for key in sorted(skipped)))
    blocks = sorted(b for b in sq if b != "patch_embed")
    if not blocks:
        sys.exit("No 2-D blocks.<i>. weights in this state dict; first keys: {}".format(list(state_dict)[:5]))
    result = {b: (math.sqrt(sq[b]), count[b]) for b in blocks + (["patch_embed"] if "patch_embed" in sq else [])}
    result["total"] = (math.sqrt(sum(sq.values())), sum(count.values()))
    return result


def table(res, cycles):
    first = cycles[0]
    print("student, Frobenius norm of the counted weights; c<N>/c{0} = ratio to cycle {0}".format(first))
    print("{:<12}".format("block") + "".join("{:>11}".format("c{}".format(c)) for c in cycles)
          + "".join("{:>11}".format("c{}/c{}".format(c, first)) for c in cycles[1:]))
    for b, (base, _) in res[first, "student"].items():
        vals = [res[c, "student"][b][0] for c in cycles]
        print("{:<12}".format(b) + "".join("{:11.4f}".format(v) for v in vals)
              + "".join("{:11.4f}".format(v / base if base else float("nan")) for v in vals[1:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("cycles", nargs="+", type=int)
    ap.add_argument("--out", default="norms.csv")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    res, rows = {}, []
    for cycle in a.cycles:
        weights_dir = os.path.join(a.run_dir, "models", "weights", "epoch_{:04d}".format(cycle))
        for role in ("student", "teacher"):
            res[cycle, role] = norms(load_state_dict(find_weight(weights_dir, role)), a.verbose and not res)
            rows += [(cycle, role, b, nv, n) for b, (nv, n) in res[cycle, role].items()]
    with open(a.out, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["cycle", "model", "block", "norm", "n_params"])
        out.writerows(rows)
    table(res, a.cycles)
    print("wrote {}".format(a.out))


def selfcheck():
    full = torch.full
    sd = {"blocks.0.attn.qkv.weight": full((2, 2), 3.0),  # 4 * 3^2 = 36 -> norm 6, 4 params
          "blocks.0.attn.qkv.bias": full((2,), 100.0), "blocks.0.norm1.weight": full((2,), 100.0),  # 1-D: skipped
          "blocks.1.mlp.fc1.weight": full((1, 4), 2.0), "blocks.1.mlp.fc2.weight": full((4, 1), 2.0),  # 16 + 16, 8 params
          "patch_embed.proj.weight": full((2, 1, 2, 2), 1.0),  # 4-D kernel: 8 ones -> sqrt(8)
          "projector.weight": full((2, 2), 9.0)}  # not backbone: skipped
    got = norms(sd)
    want = {0: (6.0, 4), 1: (math.sqrt(32), 8), "patch_embed": (math.sqrt(8), 8), "total": (math.sqrt(36 + 32 + 8), 20)}
    assert list(got) == list(want), list(got)
    for b in want:
        assert got[b][1] == want[b][1] and math.isclose(got[b][0], want[b][0]), (b, got[b], want[b])
    assert norms({"module.blocks.3.attn.proj.weight": full((2, 2), 1.0)})[3] == (2.0, 4)  # wrapper prefix
    print("selfcheck ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selfcheck"]:
        selfcheck()
    else:
        main()
