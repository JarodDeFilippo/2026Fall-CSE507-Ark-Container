# About this fork

This fork of the [CSE 507 course repository](https://github.com/jliangclass-intro/2026Fall-CSE507-Ark-Container) adds a DINOv3 ViT-B/16 backbone (`ArkViT` in `base/models.py`, built on vendored Meta code in `base/dinov3/`); the Demo 1 experiment scripts under `experiment_scripts/all4_*`, namely the Swin-B concurrent run (`all4_concurrent_random_sampling`) and the DINOv3 cyclic arms A1/A2 with SGD (`all4_concurrent_random_sampling_dinov3`, which keeps its historical name but trains cyclically, and `all4_cyclic_dinov3_v2`) and B with AdamW (`all4_cyclic_dinov3_adamw`); and a Gaudi2/HPU path (`containers/gaudi/`, `gaudi.sh`, `gaudi_hlsmi.sh`, `gaudi_check_cards.sh`, `gaudi_verify_busy.sh`, `gaudi_smoke.sh`, `build_gaudi_image.sh`, `run_d1_8a_gaudi.sh` and `resume_d1_8a_gaudi.sh`, described under "Gaudi on Sol" below). The `run_*` and `resume_*` scripts at the repository root are the Slurm launchers for the Practice 1 (`p1_*`) and Demo 1 (`d1_*`) runs on Sol; their `#SBATCH -A` lines name the course and lab accounts, so change them to your own.

## Licenses

The course repository has no license file. The Ark-derived code is under the ASU non-commercial license in `LICENSE-ASU-Ark.txt`, and `base/dinov3/` is Meta's DINOv3 code under the DINOv3 License in `base/dinov3/LICENSE.md`. The DINOv3 weights are gated by Meta and are not in this repository: download them yourself into `weights/` (gitignored; the scripts expect `weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`).

## Run on Gaudi

```bash
# Gaudi2 needs the lab research account: -A grp_jliang12 -p gaudi -q public --gres=gpu:hl225:N (the scripts below already set it)
sbatch build_gaudi_image.sh   # build containers/gaudi/gaudi.sif
sbatch gaudi_smoke.sh         # 1-card smoke test, about 30 minutes
sbatch run_d1_8a_gaudi.sh     # DINOv3 arm B on 4 cards; sbatch resume_d1_8a_gaudi.sh to continue
```

Guides: [Getting our training onto Gaudi (short, PDF)](docs/gaudi-vs-a100.pdf) and [the step-by-step port with every number's provenance (long, PDF)](docs/sol-gaudi-guide.pdf); Markdown sources sit next to them in docs/.

Companion repo: https://github.com/JarodDeFilippo/UniMiSS-code (UniMiSS+ on Gaudi)

---

# CSE 507 Ark+ Container

This repository contains modified Ark+ code for CSE 507. The container for this code is not included in this repository: see the README of the [upstream course repository](https://github.com/jliangclass-intro/2026Fall-CSE507-Ark-Container) for the container download, and save it as `containers/nvidia/nvidia.sif`.

After downloading the above container, you can run experiments using the following instructions

## Experiment Scripts

The experiment_scripts directory provides scripts for running experiments. Scripts are provided for easily running the following five experiments:

- vindr_cxr_only: Train a student/teacher Ark model on VinDr-CXR only.
- chestxray14_only: Train a student/teacher Ark model on ChestX-ray14 only.
- vindr_cxr_chestxray14_cyclic: Train a student/teacher Ark model on VinDr-CXR and ChestX-ray14 cyclically.
- vindr_cxr_chestxray14_concurrent_random_sampling: Train a student/teacher Ark model on VinDr-CXR and ChestX-ray14 concurrently with random sampling.
- vindr_cxr_chestxray14_concurrent_equal_sampling: Train a student/teacher Ark model on VinDr-CXR and ChestX-ray14 concurrently with equal sampling.

(For practice 1/demo 1 you should run the experiments in the above order)

### Running Experiments

A start and resume script are provided for each of the experiments, and they both require a seed as an argument. They should be run by running the following command from the root of the repository:

```bash
# Start experiment (replace vindr_cxr_only with the experiment name, and enter any seed)
./experiment_scripts/vindr_cxr_only/start_vindr_cxr_only.sh 100

# Resume experiment (use the same seed as when you started the experiment)
./experiment_scripts/vindr_cxr_only/resume_vindr_cxr_only.sh 100
```

The resume scripts will resume training from the latest completed epoch/cycle.

## Experiment Output

Each training run creates a directory in the repository root named
`<experiment>_seed_<seed>`, such as `vindr_cxr_only_seed_100`. The main
outputs are organized as follows:

```text
<experiment>_seed_<seed>/
├── train.log
├── loss.csv
├── evaluation/
│   └── <dataset>/
│       ├── val_performance.csv
│       └── test_performance.csv
├── models/
│   ├── weights/
│   │   └── epoch_<cycle>/
│   │       ├── manifest.json
│   │       ├── student_cycle_<cycle>_<experiment>_seed_<seed>.pth
│   │       └── teacher_cycle_<cycle>_<experiment>_seed_<seed>.pth
│   └── checkpoints/
│       └── cycle_<cycle>_<experiment>_seed_<seed>.pth.tar
└── snapshots/
    └── cycle_<cycle>/
        └── [<dataset>/]
            ├── student.jpeg
            └── teacher.jpeg
```

- `train.log` contains the configuration, training progress, validation and
  test metrics, and checkpoint messages.
- `loss.csv` contains the per-batch classification, consistency, and total
  losses, their percentages, learning rate, and teacher momentum.
- `evaluation/<dataset>/` contains CSV results for both the student and
  teacher. Rows include the cycle, epoch, evaluation point, metric name, and
  student/teacher values.
- `models/weights/` stores student and teacher weights after every cycle.
- `models/checkpoints/` stores resumable checkpoints every 10 cycles and the
  latest completed cycle.

When a dataset's validation and test samples are identical, the separate
validation pass is skipped and the test measurement is also recorded in the
validation performance file. Otherwise, validation and test evaluation are
recorded separately after each task or joint update.

## Analyzing Results of Experiments

Plotting scripts are also provided for easily analyzing the results of experiments you run using this repository. Matplotlib is required for creating the plots. On sol, you can use matplotlib by running the following commands:

```bash
ml mamba
conda create -n plot python matplotlib
source activate plot
```

See the plot/ directory for more information on how to run the plotting scripts.

## References

Original Ark+ Repo

```text
https://github.com/jlianglab/Ark
```

## Gaudi on Sol

Intel Gaudi2 (HL-225) cards on sol are reachable through the lab account only: `-A grp_jliang12 -p gaudi -q public` (the class account has no Gaudi QOS). Each node has 8 cards and about 18 cores per card, and the partition limit is 7 days. Request cards with `-N 1 --gres=gpu:hl225:N`.

```bash
# Build containers/gaudi/gaudi.sif (Habana's PyTorch base image; the pip constraints in the .def keep Habana's torch intact)
sbatch build_gaudi_image.sh

# 1-card smoke test (~30 min); it trains a throwaway all4_cyclic_dinov3_adamw_seed_1/, so delete that directory afterwards
sbatch gaudi_smoke.sh

# Launch on 4 cards, and continue with the resume script if the job ends early
sbatch run_d1_8a_gaudi.sh
sbatch resume_d1_8a_gaudi.sh
```

The experiment scripts take `gaudi` as a second argument (`./experiment_scripts/<experiment>/start_<experiment>.sh <seed> gaudi`) and then train on the `hpu` device through `gaudi.sh`, the counterpart of `nvidia.sh`. `gaudi.sh` runs the container with `--cleanenv` and forwards only `HABANA_VISIBLE_MODULES`, `HABANA_VISIBLE_DEVICES`, `OMP_NUM_THREADS` and `PT_HPU_LAZY_MODE` from the host. Slurm on Sol sets no `HABANA_VISIBLE_*` and the device cgroup does not hide the other cards (a 1-card job saw all 8), so when neither variable is set `gaudi.sh` maps `SLURM_JOB_GPUS` to `HABANA_VISIBLE_MODULES` through `hl-smi` (assuming Slurm's gres index equals the `hl-smi` index) and refuses to run if it cannot. The smoke test and the launchers first run `gaudi_check_cards.sh`, which fails closed: it exits 1 unless every allocated card reads idle in `hl-smi` before the container touches any card, and the restriction reached the container with exactly as many entries as Slurm allocated cards (`torch.hpu.device_count()` ignores the restriction and counts every card on the node), and when a `HABANA_VISIBLE_*` variable is set on the host it also checks card identity against Slurm's allocation through `hl-smi`. Once training runs, `gaudi_verify_busy.sh` reports through `hl-smi` whether every allocated card is busy (after 10 minutes in the smoke test, 15 in the launchers). That check is report-only: it never cancels or kills anything, it logs a `WARNING` (to stdout and stderr) that says to inspect the job and `scancel` it by hand if confirmed.

`PT_HPU_LAZY_MODE=1` (lazy mode) is required. The lab's Gaudi_Demo measured eager mode about 9 to 10 AUC points worse on Ark+ training (VinDr-CXR student 0.9412 lazy vs 0.8482 eager), so `gaudi.sh` defaults to lazy mode.
