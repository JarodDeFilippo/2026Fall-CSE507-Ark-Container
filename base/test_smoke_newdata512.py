"""CPU checks for the multi-card memory smoke of the 512 px Ark-teacher run (run_d1_8_newdata512_smoke_gaudi.sh) and the trainer's --log_peak_memory: the sbatch header, the smoke's launch line against the REAL
launcher's (chain/newdata512_cropsrc.sh, which is not edited; both run in a fake repo with a recording gaudi.sh, and the flags that differ are exactly the documented ones), the real line unchanged, 2 / 4 / 8 cards,
the guards (STEPS must stay inside MIMIC's pass, counted as the trainer's loader counts it), the summary (allocator peaks, hl-smi peaks, speed, verdict) behind stub trainer output, with and without an allocator counter,
and a tiny end-to-end trainer run on the smoke's flag set: --log_peak_memory prints per rank (and nothing without it) from torch.hpu's counter, else Habana's memory_stats MaxInUse (bytes), else `unavailable` with the run going on,
the backbone and the heads' last layers get gradients with the freezes on, and AdamW holds moments only for the heads that have stepped.
Run from base/: python test_smoke_newdata512.py (needs torch, timm, torchvision, PIL; no Slurm, no Gaudi). The test_ functions are also collected by pytest."""
import contextlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import types

sys.dont_write_bytecode = True
import numpy as np
import torch
from PIL import Image

import probe_resolution_memory as res
import ssl_pretrain
from models import DINOV3_VITB16_KWARGS
from ssl_dinov3.data import build_data_loader, image_paths
from ssl_dinov3.model import build_models
from utils import get_config

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
SMOKE = os.path.join(REPO, "run_d1_8_newdata512_smoke_gaudi.sh")
REAL_RUN = "ssl_dinov3_cxr_newdata512_cropsrc_arkteacher_seed_100"
INIT = "/workspace/ssl_dinov3_cxr_constlr_arkteacher_seed_100/ckpt/epoch_0200.pth"
WEIGHT_NAME = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
SEVEN = ["MIMIC", "ReXGradient", "CheXpert", "TAIXRay", "PadChest", "ChestXray14", "VinDrCXR"]
MIMIC_LIST = "base/dataset/MIMIC/mimic-cxr-2.0.0-train.csv"  # datasets_config.yaml's MIMIC train_list, from base/ where the trainer runs: the smoke counts its rows
MIMIC_ROWS = 368960  # the real list's today (and a header row); the fake repos' list has this many, so the default 200 steps are inside MIMIC's pass at every batch and card count tried
GiB = 2 ** 30
TINY_VIT = dict(DINOV3_VITB16_KWARGS, embed_dim=64, depth=2, num_heads=2, ffn_ratio=2)
TINY_HEADS = dict(out_dim=128, hidden_dim=64, bottleneck_dim=32, nlayers=3)
ENV = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "HABANA_VISIBLE_", "SSL_DATASETS", "EXTRA_BINDS", "FAKE_", "HLSMI_", "CROPSRC_"))}
CARDS = {2: "0,1", 4: "0,1,2,3", 8: "0,1,2,3,4,5,6,7"}
DIFFERENT = {"--batch_size_per_gpu", "--output_dir", "--stop_after_epochs", "--datasets_config", "--max_steps", "--freeze_backbone_epochs", "--log_peak_memory"}  # what the smoke's header lists
GAUDI_STUB = r'''#!/usr/bin/env bash
printf "%s\n" "$@" > "$PWD/argv.txt"; printf "%s" "${EXTRA_BINDS:-}" > "$PWD/extra_binds.txt"
if [[ -n ${FAKE_TRAINER:-} ]]; then  # what the trainer leaves behind in its run dir and prints, then its exit
  out=; prev=; for a in "$@"; do [[ $prev == --output_dir ]] && out=${a#/workspace/}; prev=$a; done
  mkdir -p "$PWD/$out/ckpt" "$PWD/$out/export"; touch "$PWD/$out/ckpt/maxsteps_0000200.pth" "$PWD/$out/export/teacher_backbone_maxsteps_0000200.pth"
  printf 'step,epoch,dino_global,dino_local,koleo,ibot,gram,total,lr,wd,momentum,teacher_temp,gram_weight,step_time,data_time,dataset\n' > "$PWD/$out/ssl_losses.csv"
  printf '0,0,1,1,1,1,0,1,0,0.04,1,0.04,0,30.2,5.2,MIMIC:release\n199,0,1,1,1,1,0,1,0,0.04,1,0.04,0,2.5,0.03,MIMIC:release\n' >> "$PWD/$out/ssl_losses.csv"
  for step in 1 50 200; do for r in 0 1 2 3; do echo "peak_memory rank=$r step=$step GiB=${FAKE_PEAK:-$((80 + step / 50 + r)).50}"; done; done  # FAKE_PEAK=unavailable: the trainer found no counter
  sleep 2
  [[ -z ${FAKE_ERR:-} ]] || echo "$FAKE_ERR" >&2
  exit "${FAKE_RC:-0}"
fi
'''
HLSMI_STUB = r'''#!/usr/bin/env bash
n=$(cat "$HLSMI_COUNT" 2>/dev/null || echo 0); echo $((n + 1)) > "$HLSMI_COUNT"
echo "index, module_id, memory.used [MiB]"
for i in 0 1 2 3 4 5 6 7; do echo "$i, $((i + 2)), $((768 + n * 1000 + i * 10)) MiB"; done
'''


def put(path, text, mode=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    if mode:
        os.chmod(path, mode)


def mimic_list(root, rows):
    """A MIMIC list in the format image_paths reads: a header row, then `rows` images."""
    put(os.path.join(root, MIMIC_LIST), "path,subject\n" + "x,1\n" * rows)


def fake_repo(tmp, init=True, trainer=True, settings=True, mimic=MIMIC_ROWS):
    """A fake repo root: the real launcher, ssl_common.sh, Ark's common.sh and gaudi_hlsmi.sh, a recording gaudi.sh, a card check that records it ran, the stock weights, the trunk's epoch-200 checkpoint (init=False:
    absent), the trainer's text (trainer=False: without --log_peak_memory), the real launcher's two settings files (settings=False: absent, which the smoke does not need) and a MIMIC list of `mimic` rows
    (mimic=False: absent)."""
    root = os.path.join(tmp, "repo")
    for name in ("chain/newdata512_cropsrc.sh", "chain/ssl_common.sh", "experiment_scripts/common.sh", "gaudi_hlsmi.sh"):
        os.makedirs(os.path.dirname(os.path.join(root, name)), exist_ok=True)
        shutil.copy(os.path.join(REPO, name), os.path.join(root, name))
    put(os.path.join(root, "gaudi.sh"), GAUDI_STUB, 0o755)
    put(os.path.join(root, "gaudi_check_cards.sh"), '#!/usr/bin/env bash\ntouch "$PWD/cards_checked"\n', 0o755)
    put(os.path.join(root, "weights", WEIGHT_NAME), "x")
    if init:
        put(os.path.join(root, INIT.replace("/workspace/", "")), "x")
    text = open(os.path.join(BASE, "ssl_pretrain.py")).read()
    put(os.path.join(root, "base", "ssl_pretrain.py"), text if trainer else text.replace('"--log_peak_memory"', '"--gone"'))
    if mimic:
        mimic_list(root, mimic)
    if settings:
        put(os.path.join(root, "chain", "newdata512_cropsrc_stop_after.txt"), "5\n")
        put(os.path.join(root, "chain", "newdata512_cropsrc_batch_per_gpu.txt"), "16\n")
    put(os.path.join(tmp, "bin", "hl-smi"), HLSMI_STUB, 0o755)
    os.makedirs(os.path.join(tmp, "datasets-new"), exist_ok=True)
    return root


def smoke_script(tmp, root, datasets="datasets-new"):
    """The smoke wrapper with its cd pointed at the fake repo, its datasets check at the fake directory and its 5 s sample interval at 0.2 s (the stub trainer runs 2 s)."""
    text = open(SMOKE).read()
    cd, check, sleep = "cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1", "[[ -d /scratch/$USER/datasets-new ]]", "sleep 5; done"
    assert text.count(cd) == 1 and text.count(check) == 1 and text.count(sleep) == 1
    path = os.path.join(root, "smoke.sh")
    put(path, text.replace(cd, "cd '%s' || exit 1" % root).replace(check, "[[ -d '%s' ]]" % os.path.join(tmp, datasets)).replace(sleep, "sleep 0.2; done"))
    return path


def env_for(tmp, cards=4, **extra):
    return dict(ENV, SLURM_JOB_GPUS=CARDS[cards], SLURM_JOB_ID="4242", USER="tester", PATH=os.path.join(tmp, "bin") + os.pathsep + ENV["PATH"], HLSMI_COUNT=os.path.join(tmp, "hlsmi_count"), **extra)


def run(script, root, env, *args, cwd=None):
    path = os.path.join(root, "argv.txt")
    if os.path.exists(path):
        os.remove(path)  # what an earlier launch recorded must not pass for this one's
    result = subprocess.run(["bash", script] + list(args), cwd=cwd or root, env=env, capture_output=True, text=True)
    return result, (open(path).read().splitlines() if os.path.exists(path) else None)


def flags(argv):
    """The trainer's part of a launch line as {flag: [the value list of each time it was given]}."""
    out, current = {}, None
    for word in argv[argv.index("ssl_pretrain.py") + 1:]:
        if word.startswith("--"):
            current = out.setdefault(word, [])
            current.append([])
        else:
            current[-1].append(word)
    return out


def changed(a, b):
    fa, fb = flags(a), flags(b)
    return {flag for flag in fa.keys() | fb.keys() if fa.get(flag) != fb.get(flag)}


parse = lambda argv: ssl_pretrain.parse_args(argv[argv.index("ssl_pretrain.py") + 1:])


def test_smoke_header():
    subprocess.run(["bash", "-n", SMOKE], check=True)
    assert os.access(SMOKE, os.X_OK) and stat.S_IMODE(os.stat(SMOKE).st_mode) == 0o755
    text = open(SMOKE).read()
    lines = text.splitlines()
    assert [line[len("#SBATCH "):] for line in lines if line.startswith("#SBATCH ")] == [
        "--job-name=d1-8-newdata512-smoke", "-A grp_jliang12", "-p gaudi", "-q public", "-t 00:30:00", "-N 1", "--gres=gpu:hl225:4", "-c 72", "--mem=256G"], lines
    real = open(os.path.join(REPO, "run_d1_8_newdata512_cropsrc_arkteacher_gaudi.sh")).read()
    assert "sbatch --gres=gpu:hl225:4 -c 72 --mem=256G run_d1_8_newdata512_cropsrc_arkteacher_gaudi.sh" in real  # the 4-card request line the run documents
    shared = lambda text: [l for l in text.splitlines() if l.startswith("#SBATCH ") and l.split()[1] in ("-A", "-p", "-q", "-N")]
    assert shared(real) == shared(text) and len(shared(text)) == 4  # account, partition, QOS, nodes: the run's
    code = [line for line in lines if line and not line.startswith("#")]
    for needle in ("export PT_HPU_LAZY_MODE=1", "cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1", "source ./chain/ssl_common.sh", "./gaudi_check_cards.sh || exit 1",
                   "export EXTRA_BINDS=/scratch/$USER/datasets-new:/scratch/$USER/datasets-new:ro", "--freeze_backbone_epochs 0 --log_peak_memory", "--max_steps"):
        assert any(needle in line for line in code), needle
    assert not any("newdata512_cropsrc_stop_after" in line or "newdata512_cropsrc_batch_per_gpu" in line or "datasets_config" in line for line in code)  # no settings file, no TAIX-Ray native list
    for needle in ("sbatch run_d1_8_newdata512_smoke_gaudi.sh", "--gres=gpu:hl225:2 -c 36 --mem=128G", "--gres=gpu:hl225:8 -c 144 --mem=480G", "srun -A grp_jliang12 -p gaudi -q public -N 1 --gres=gpu:hl225:4 -c 72 --mem=256G -t 00:30:00",
                   "ADD 2.06 GiB", "grep -E '^(SMOKE|peak_memory)'", "LR-0 freezes", "datasets_config.yaml", "STEPS stays below MIMIC's pass", "GiB=unavailable"):
        assert needle in text, needle
    assert "newdata512_smoke" not in [line.split("#")[0].strip() for line in open(os.path.join(REPO, "chain", "phases.txt"))]
    assert "MIMIC_LIST=" + MIMIC_LIST + " " in text and MIMIC_LIST == "base/" + get_config(os.path.join(BASE, "datasets_config.yaml"))["MIMIC"]["train_list"]  # the list the trainer reads


def test_launch_line_against_the_real_launcher():
    with tempfile.TemporaryDirectory() as tmp:
        root = fake_repo(tmp)
        env = env_for(tmp)
        result, real = run("chain/newdata512_cropsrc.sh", root, env)
        assert result.returncode == 0, result.stderr
        expected = (["python", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "4", "ssl_pretrain.py", "--device", "hpu"] + [x for name in SEVEN for x in ("--data_set", name)]
                    + ["--pretrained_weights", "/workspace/weights/" + WEIGHT_NAME, "--batch_size_per_gpu", "64", "--workers", "16", "--seed", "100", "--output_dir", "/workspace/" + REAL_RUN, "--resume",
                       "--global_crops_size", "512", "--local_crops_size", "224", "--gram_teacher_size", "512", "--decode_min_side", "1024", "--batch_size_per_gpu", "16", "--lr", "0.0001", "--min_lr", "0.0001",
                       "--epochs", "1000", "--stop_after_epochs", "5", "--warmup_epochs", "2", "--teacher_temp_warmup_epochs", "2",
                       "--drop_path_rate", "0.3", "--rope_rescale_coords", "2", "--untie_local_cls_norm", "--no_gram", "--cyclic_ark_teacher", "--heads", "per_dataset", "--no_lock_half_pass",
                       "--keep_all_checkpoints", "--init_from_checkpoint", INIT, "--init_new_heads_from", "MIMIC", "--datasets_config", "datasets_config_taix_original.yaml"])
        assert real == expected, real  # the real argument line, unchanged: no smoke flag in it
        real_args = parse(real)
        assert real_args.log_peak_memory is False and real_args.max_steps is None and real_args.freeze_backbone_epochs == 1 and ssl_pretrain.check_args(real_args) == 5
        script = smoke_script(tmp, root)
        result, smoke = run(script, root, dict(env, FAKE_TRAINER="1"), "32", cwd=tempfile.mkdtemp(dir=tmp))  # from elsewhere, as sbatch does
        assert result.returncode == 0 and result.stderr == "", (result.stdout, result.stderr)
        run_dir = "smoke_newdata512_b32_x4_4242"
        assert changed(real, smoke) == DIFFERENT, changed(real, smoke)
        assert flags(smoke)["--output_dir"] == [["/workspace/" + run_dir]] and flags(smoke)["--batch_size_per_gpu"] == [["64"], ["32"]]
        assert flags(smoke)["--max_steps"] == [["200"]] and flags(smoke)["--freeze_backbone_epochs"] == [["0"]] and flags(smoke)["--log_peak_memory"] == [[]]
        assert "--stop_after_epochs" not in flags(smoke) and "--datasets_config" not in flags(smoke) and "--pre_resize" not in flags(smoke)
        assert smoke[:smoke.index("--output_dir")] == real[:real.index("--output_dir")]  # cards, trainer, datasets, weights, workers, seed: as the run's
        smoke_args = parse(smoke)
        differs = {k for k in vars(real_args) if vars(real_args)[k] != vars(smoke_args)[k]}
        assert differs == {"batch_size_per_gpu", "output_dir", "stop_after_epochs", "datasets_config", "max_steps", "freeze_backbone_epochs", "log_peak_memory"}, differs  # every other trainer setting, defaults included
        assert (smoke_args.batch_size_per_gpu, smoke_args.max_steps, smoke_args.freeze_backbone_epochs, smoke_args.log_peak_memory, smoke_args.datasets_config) == (32, 200, 0, True, "datasets_config.yaml")
        assert ssl_pretrain.check_args(smoke_args) == 1000
        assert (smoke_args.global_crops_size, smoke_args.local_crops_size, smoke_args.gram_teacher_size, smoke_args.decode_min_side, smoke_args.no_gram, smoke_args.heads, smoke_args.data_set) == (
            512, 224, 512, 1024, True, "per_dataset", SEVEN)
        assert smoke_args.init_from_checkpoint == INIT and smoke_args.init_new_heads_from == "MIMIC" and not smoke_args.lock_half_pass and not smoke_args.bf16  # fp32, no lock pass, the start of the run
        # a throwaway dir that cannot be the real run's, nor resume into it: other name, neither a prefix of the other, and the real dir is never made
        assert run_dir != REAL_RUN and not run_dir.startswith(REAL_RUN) and not REAL_RUN.startswith(run_dir) and not os.path.exists(os.path.join(root, REAL_RUN))
        assert open(os.path.join(root, "extra_binds.txt")).read() == "/scratch/tester/datasets-new:/scratch/tester/datasets-new:ro" and os.path.exists(os.path.join(root, "cards_checked"))
        assert "SMOKE plan: 4 cards x per-card batch 32 (global batch 128), 200 steps, run dir " + run_dir in result.stdout


def test_cards_and_batches():
    with tempfile.TemporaryDirectory() as tmp:
        root = fake_repo(tmp, settings=False)  # neither settings file of the real run is needed
        script = smoke_script(tmp, root)
        for cards in (2, 4, 8):
            result, argv = run(script, root, env_for(tmp, cards, FAKE_TRAINER="1"), "24", "300")
            assert result.returncode == 0 and argv[argv.index("--nproc-per-node") + 1] == str(cards), (cards, result.stderr)
            assert flags(argv)["--output_dir"] == [["/workspace/smoke_newdata512_b24_x%d_4242" % cards]] and flags(argv)["--max_steps"] == [["300"]]
            assert "SMOKE plan: %d cards x per-card batch 24 (global batch %d), 300 steps" % (cards, 24 * cards) in result.stdout
            shutil.rmtree(os.path.join(root, "smoke_newdata512_b24_x%d_4242" % cards))  # the next card count starts clean (the job id is fixed here)
        # refused before anything is launched
        os.remove(os.path.join(root, "cards_checked"))
        for args in ((), ("0",), ("08",), ("abc",), ("12345",), ("32", "0"), ("32", "abc")):
            result, argv = run(script, root, env_for(tmp), *args)
            assert result.returncode == 2 and argv is None and "usage:" in result.stderr and not os.path.exists(os.path.join(root, "cards_checked")), (args, result.stderr)
        for label, kwargs, word in (("no start checkpoint", dict(init=False), "nothing to start from"), ("a trainer without the flag", dict(trainer=False), "--log_peak_memory"),
                                    ("no MIMIC list", dict(mimic=False), "mimic-cxr-2.0.0-train.csv")):
            bare = fake_repo(tempfile.mkdtemp(dir=tmp), **kwargs)
            result, argv = run(smoke_script(os.path.dirname(bare), bare), bare, env_for(os.path.dirname(bare)), "32")
            assert result.returncode == 1 and argv is None and "ERROR" in result.stderr and word in result.stderr, (label, result.stderr)
        result, argv = run(smoke_script(tmp, root, datasets="nowhere"), root, env_for(tmp), "32")
        assert result.returncode == 1 and argv is None and "datasets-new not found" in result.stderr, result.stderr
        script = smoke_script(tmp, root)  # the plain wrapper again
        os.makedirs(os.path.join(root, "smoke_newdata512_b32_x4_4242"))  # the run dir is new for every job
        result, argv = run(script, root, env_for(tmp), "32")
        assert result.returncode == 1 and argv is None and "exists" in result.stderr


def pass_steps(rows, cards, batch):
    """MIMIC's pass in steps as the trainer gets it: run() builds each pass's loader with build_data_loader (DistributedSampler and DataLoader, both drop_last) for `cards` ranks; a length needs no item read."""
    return len(build_data_loader(range(rows), batch, 0, 0, cards, 100, False, None)[0])


def test_steps_stay_inside_mimic_pass():
    """The summary says only MIMIC's heads have optimizer state, true while every step is inside MIMIC's pass (the cycle's first; its last step blends the teacher, which the smoke does not run): a STEPS at or past the pass
    is refused (exit 2) before the run dir is made or anything launches, and one below it runs. The pass is the trainer's own count at the allocation's cards and the batch, on the rows of the list it reads."""
    real = os.path.join(BASE, "dataset", "MIMIC", "mimic-cxr-2.0.0-train.csv")
    rows = len(image_paths("MIMIC", "", real))  # the trainer's reader: the header skipped, a row an image
    assert (rows, pass_steps(rows, 8, 32), pass_steps(rows, 4, 32)) == (MIMIC_ROWS, 1441, 2882)  # the fake lists' size, and the numbers the smoke's header gives
    with tempfile.TemporaryDirectory() as tmp:
        root = fake_repo(tmp, settings=False)
        script = smoke_script(tmp, root)

        def attempt(cards, batch, steps, rows_read):  # one wrapper run; True if it launched (the stub trainer exits at once), its run dir removed again
            run_dir = os.path.join(root, "smoke_newdata512_b%d_x%d_4242" % (batch, cards))
            result, argv = run(script, root, env_for(tmp, cards), str(batch), *([str(steps)] if steps else []))
            if result.returncode == 0:
                assert argv is not None and flags(argv)["--max_steps"] == [[str(steps or 200)]], (cards, batch, steps, result.stderr)
                shutil.rmtree(run_dir)
                return True
            assert result.returncode == 2 and argv is None and not os.path.exists(run_dir), (cards, batch, steps, result.stderr)
            pass_len = pass_steps(rows_read, cards, batch)
            assert "usage:" not in result.stderr and "do not stay below MIMIC's pass of %d steps (%d images / %d cards / per-card batch %d" % (pass_len, rows_read, cards, batch) in result.stderr, result.stderr
            assert result.stderr.rstrip().endswith("use at most %d steps" % (pass_len - 1)), result.stderr
            return False

        for cards, batch in ((8, 32), (4, 24), (2, 8), (8, 9999)):  # the cards and the batch both set the pass; 8 x 9999 leaves 4 steps, which the default 200 is past
            last = pass_steps(MIMIC_ROWS, cards, batch)
            assert not attempt(cards, batch, last, MIMIC_ROWS) and attempt(cards, batch, last - 1, MIMIC_ROWS), (cards, batch, last)  # the edge: the pass's own length is out, one fewer is in
            assert attempt(cards, batch, 0, MIMIC_ROWS) == (200 < last)  # the default 200: in where it is below the pass, refused where not
        assert not attempt(8, 32, 999999, MIMIC_ROWS) and attempt(8, 32, 1000, MIMIC_ROWS)  # far past, and a mid value
        for rows_here in (4 * 24 * 52 - 1, 4 * 24 * 52):  # the list is read at run time (not a constant) and its header row is no image: an image short of 52 whole batches of 4 x 24 the pass is 51 steps, at 52 whole batches 52
            mimic_list(root, rows_here)
            last = pass_steps(rows_here, 4, 24)
            assert last in (51, 52) and not attempt(4, 24, last, rows_here) and attempt(4, 24, last - 1, rows_here)
        os.remove(os.path.join(root, MIMIC_LIST))
        os.symlink(real, os.path.join(root, MIMIC_LIST))  # the real list now: the wrapper's count of its rows is the trainer's (the awk over it is slow on the Mac, so only here)
        assert not attempt(8, 32, 1441, rows) and attempt(8, 32, 1440, rows)


def test_summary():
    with tempfile.TemporaryDirectory() as tmp:
        root = fake_repo(tmp)
        script = smoke_script(tmp, root)
        run_dir = os.path.join(root, "smoke_newdata512_b32_x4_4242")
        result, argv = run(script, root, env_for(tmp, FAKE_TRAINER="1"), "32")
        lines = result.stdout.splitlines()
        assert result.returncode == 0, result.stderr
        for rank in range(4):  # the allocator peak per rank: the largest of its printed values (80 + step / 50 + rank, at 1 / 50 / 200)
            assert "SMOKE allocator peak, rank %d: %.2f GiB" % (rank, 84.5 + rank) in lines, lines
        peaks = {int(l.split("card index ")[1].split(":")[0]): int(l.split("peak ")[1].split(" MiB")[0]) for l in lines if l.startswith("SMOKE hl-smi card index")}
        assert sorted(peaks) == [0, 1, 2, 3], peaks  # the allocated cards only: not 4 to 7
        assert all(peaks[i] >= 768 + 1000 + i * 10 and (peaks[i] - 768 - i * 10) % 1000 == 0 for i in peaks), peaks  # the largest of several samples (the stub's reading grows by 1000 MiB a call)
        assert "SMOKE speed: row at step 199: 2.500 s/step = 0.400 steps/s, 12.8 images/s per card" in "\n".join(lines)  # the last row: 32 / 2.5
        assert "SMOKE verdict: FITS: 4 cards x per-card batch 32 ran 200 steps (exit 0)" in lines and any(l.startswith("SMOKE not held here") for l in lines)
        assert sorted(os.listdir(run_dir)) == ["hlsmi.txt", "ssl_losses.csv", "train.log"], os.listdir(run_dir)  # ckpt/ and export/ are gone, the evidence stays
        assert "peak_memory rank=3 step=200 GiB=87.50" in open(os.path.join(run_dir, "train.log")).read()
        shutil.rmtree(run_dir)
        # the trainer found no allocator counter (its lines say GiB=unavailable): no peak is made up from them, the summary says so per rank, and the hl-smi peaks, the speed and the verdict are all still there
        result, _ = run(script, root, env_for(tmp, FAKE_TRAINER="1", FAKE_PEAK="unavailable"), "32")
        lines = result.stdout.splitlines()
        assert result.returncode == 0, result.stderr
        assert "peak_memory rank=3 step=200 GiB=unavailable" in open(os.path.join(run_dir, "train.log")).read()
        assert [l for l in lines if l.startswith("SMOKE allocator peak")] == [
            "SMOKE allocator peak, rank %d: unavailable (no allocator counter on this stack: the hl-smi peaks below are the measure)" % rank for rank in range(4)], lines
        assert sorted(int(l.split("card index ")[1].split(":")[0]) for l in lines if l.startswith("SMOKE hl-smi card index")) == [0, 1, 2, 3] and any("SMOKE speed: row at step 199" in l for l in lines)
        assert "SMOKE verdict: FITS: 4 cards x per-card batch 32 ran 200 steps (exit 0)" in lines and not any("0.00 GiB" in l for l in lines)
        shutil.rmtree(run_dir)
        result, _ = run(script, root, env_for(tmp, FAKE_TRAINER="1", FAKE_RC="1", FAKE_ERR="RuntimeError: Habana: no storage tensor attached to lazy tensor"), "32")
        assert result.returncode == 1 and "SMOKE verdict: OUT OF MEMORY: 4 cards x per-card batch 32 (exit 1)" in result.stdout and "FITS" not in result.stdout
        shutil.rmtree(run_dir)
        result, _ = run(script, root, env_for(tmp, FAKE_TRAINER="1", FAKE_RC="3", FAKE_ERR="ValueError: boom"), "32")
        assert result.returncode == 3 and "SMOKE verdict: FAILED (exit 3) and not obviously memory" in result.stdout


def make_data(root):
    """3 tiny grey images per dataset, the list files in the formats image_paths reads."""
    rng, config = np.random.RandomState(0), {}
    for name, ext, header in (("ChestXray14", "png", None), ("VinDrCXR", "jpeg", None), ("CheXpert", "jpg", "Path,Sex"), ("MIMIC", "jpg", "path,subject"),
                              ("PadChest", "png", "path,x"), ("TAIXRay", "png", "path,x"), ("ReXGradient", "png", "path,x")):
        data_dir, listing = os.path.join(root, name), os.path.join(root, name + ".list")
        os.makedirs(data_dir)
        with open(listing, "w") as f:
            if header:
                f.write(header + "\n")
            for i in range(3):
                rel = ("v%d.%s" if name == "VinDrCXR" else "x%d.%s") % (i, ext)
                Image.fromarray(rng.randint(0, 255, (80, 96), dtype=np.uint8)).save(os.path.join(data_dir, rel))
                f.write({"ChestXray14": rel + " 0\n", "VinDrCXR": "v%d 0\n" % i}.get(name, rel + ",F\n"))
        config[name] = dict(data_dir=data_dir, train_list=listing)
    return config


def tiny_run(tmp, config, tag, *extra):
    """ssl_pretrain.run on CPU with the smoke's flags (the seven sets, per-dataset heads, the recipe, --no_gram, the cyclic Ark teacher, 512 / 224 / 512 scaled down to 32 / 16 / 32, no start checkpoint), 4 steps.
    Returns (stdout, the AdamW calls [(backbone lr, last-layer lr, backbone params with a gradient / all, others with a gradient, last-layer params with a gradient)], the optimizer, args.json)."""
    out, calls, seen = os.path.join(tmp, tag), [], []
    argv = [x for name in SEVEN for x in ("--data_set", name)] + [
        "--pretrained_weights", "unused", "--output_dir", out, "--device", "cpu", "--workers", "0", "--batch_size_per_gpu", "1", "--global_crops_size", "32", "--local_crops_size", "16",
        "--gram_teacher_size", "32", "--lr", "0.0001", "--min_lr", "0.0001", "--epochs", "1000", "--max_steps", "4", "--warmup_epochs", "2", "--teacher_temp_warmup_epochs", "2",
        "--drop_path_rate", "0.3", "--rope_rescale_coords", "2", "--untie_local_cls_norm", "--no_gram", "--cyclic_ark_teacher", "--heads", "per_dataset", "--no_lock_half_pass",
        "--keep_all_checkpoints", "--print_freq", "2"] + list(extra)
    real_step = torch.optim.AdamW.step

    def step(self, *a, **kw):
        groups, seen_opt = self.param_groups, self
        back = [p for g in groups if g["is_backbone"] for p in g["params"]]
        last = [p for g in groups if g["is_last_layer"] for p in g["params"]]
        rest = [p for g in groups if not g["is_backbone"] for p in g["params"]]
        calls.append((max(g["lr"] for g in groups if g["is_backbone"]), max(g["lr"] for g in groups if g["is_last_layer"]), sum(p.grad is not None for p in back), len(back),
                      sum(p.grad is not None for p in rest), sum(p.grad is not None for p in last)))
        seen[:] = [seen_opt]
        return real_step(self, *a, **kw)

    torch.optim.AdamW.step = step
    buffer = io.StringIO()
    args = ssl_pretrain.parse_args(argv)
    args.pretrained_weights = None  # random init (there are no stock weights to load into the tiny ViT)
    try:
        with contextlib.redirect_stdout(buffer):
            ssl_pretrain.run(args, TINY_VIT, TINY_HEADS, config)
    finally:
        torch.optim.AdamW.step = real_step
    return buffer.getvalue(), calls, seen[0], json.load(open(os.path.join(out, "args.json")))


@contextlib.contextmanager
def gaudi_counters(torch_hpu=None, htcore_hpu=None):
    """The tiny CPU run reading the memory counters of a Gaudi stack: probe_resolution_memory's hpu attempt chain stands in for the cpu's (torch.cpu has no counter), over stand-ins for torch.hpu and for
    habana_frameworks.torch.core.hpu, each given as a dict of its attributes ({}: the module is there without the counter; None: not there at all)."""
    saved = getattr(torch, "hpu", None), sys.modules.pop(res.HTCORE, None)
    if hasattr(torch, "hpu"):
        del torch.hpu
    if torch_hpu is not None:
        torch.hpu = types.SimpleNamespace(**torch_hpu)
    if htcore_hpu is not None:
        sys.modules[res.HTCORE] = types.ModuleType(res.HTCORE)
        sys.modules[res.HTCORE].hpu = types.SimpleNamespace(**htcore_hpu)
    res.PEAK["cpu"] = res.PEAK["hpu"]
    try:
        yield
    finally:
        del res.PEAK["cpu"]
        sys.modules.pop(res.HTCORE, None)
        if hasattr(torch, "hpu"):
            del torch.hpu
        if saved[0] is not None:
            torch.hpu = saved[0]
        if saved[1] is not None:
            sys.modules[res.HTCORE] = saved[1]


def peak_lines_of(text):
    return [l for l in text.splitlines() if l.startswith("peak_memory")]


def test_probe_module_not_loaded_by_the_trainer():
    """The counter chain comes from probe_resolution_memory, imported at the first logged step of a run with --log_peak_memory: importing the trainer (every run's start) does not load it."""
    code = "import sys, ssl_pretrain; sys.exit(int('probe_resolution_memory' in sys.modules))"
    subprocess.run([sys.executable, "-B", "-c", code], cwd=BASE, check=True, capture_output=True)


def test_trainer_tiny_run():
    per_head = len(list(build_models(TINY_VIT, None, TINY_HEADS)[0].dino_head.parameters()))  # tensors in one head (the shared-heads student has a single DINO head)
    per_last = len(list(build_models(TINY_VIT, None, TINY_HEADS)[0].dino_head.last_layer.parameters()))
    with tempfile.TemporaryDirectory() as tmp:
        config = make_data(os.path.join(tmp, "data"))
        with gaudi_counters({"max_memory_allocated": lambda: 3 * GiB}, {"memory_stats": lambda: {"MaxInUse": 9 * GiB}}):  # torch.hpu answers: its figure, not Habana's
            text, calls, opt, info = tiny_run(tmp, config, "smoke", "--freeze_backbone_epochs", "0", "--log_peak_memory")
        assert peak_lines_of(text) == ["peak_memory rank=0 step=%d GiB=3.00" % s for s in (1, 2, 3, 4)], peak_lines_of(text)  # the logged steps (the first, every print_freq, a pass's last, the last), this rank's card
        assert info["log_peak_memory"] is True and info["freeze_backbone_epochs"] == 0 and info["heads"] == "per_dataset" and info["no_gram"] and info["data_set"] == SEVEN
        assert len(calls) == 4
        n_back = calls[0][3]
        for lr_back, lr_last, back_with_grad, back_total, rest_with_grad, last_with_grad in calls:
            assert back_with_grad == back_total == n_back  # every backbone tensor has a gradient at every optimizer step
            assert rest_with_grad == 2 * per_head and last_with_grad == 2 * per_last  # the batch's DINO and iBOT heads, last layers included (their LR is 0 in cycle 0: the freeze is the LR only)
            assert lr_last == 0.0
        assert calls[0][0] == 0.0 and all(c[0] > 0 for c in calls[1:])  # freeze off: the backbone's LR is the warmup's (0 at step 0), then above 0 from step 1
        # AdamW holds moments for the backbone and the heads that have stepped: MIMIC (3 steps), then ReXGradient (step 4); the other 5 datasets' 10 heads have none
        total = len([p for g in opt.param_groups for p in g["params"]])
        assert total == n_back + 14 * per_head and len(opt.state) == n_back + 4 * per_head, (total, len(opt.state), n_back, per_head)
        quiet, _, _, quiet_info = tiny_run(tmp, config, "quiet", "--freeze_backbone_epochs", "0")  # without the flag: nothing printed, the record says off
        assert "peak_memory" not in quiet and quiet_info["log_peak_memory"] is False
        # torch.hpu without max_memory_allocated (the Gaudi stack that broke the first version): Habana's memory_stats()["MaxInUse"], in bytes, printed in GiB (5.25 GiB; Limit and InUse are other numbers)
        with gaudi_counters({}, {"memory_stats": lambda: {"Limit": 96 * GiB, "InUse": 1 * GiB, "MaxInUse": 5 * GiB + GiB // 4}}):
            frozen, frozen_calls, _, _ = tiny_run(tmp, config, "frozen", "--log_peak_memory")  # the trainer's default freeze (cycle 0): LR 0 only, the gradients and the moments are the same
        assert [c[0] for c in frozen_calls] == [0.0] * 4 and [c[2:] for c in frozen_calls] == [c[2:] for c in calls]
        assert peak_lines_of(frozen) == ["peak_memory rank=0 step=%d GiB=5.25" % s for s in (1, 2, 3, 4)], peak_lines_of(frozen)
        # no counter at all (torch.hpu without the API, no Habana module): the line says unavailable at every logged step, nothing raises, and the run trains all four steps
        with gaudi_counters({}):
            none, none_calls, _, _ = tiny_run(tmp, config, "none", "--freeze_backbone_epochs", "0", "--log_peak_memory")
        assert peak_lines_of(none) == ["peak_memory rank=0 step=%d GiB=unavailable" % s for s in (1, 2, 3, 4)], peak_lines_of(none)
        assert len(none_calls) == 4 and "--max_steps 4 reached at step 4" in none


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("OK", name)
    print("ALL OK")
