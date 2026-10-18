#!/usr/bin/env bash
# Sourced by chain/pilot_a.sh, pilot_b.sh, main.sh, constlr.sh, hires.sh and run_d1_8s_timing_gaudi.sh: shared guards plus ssl_launch, which starts
# base/ssl_pretrain.py on every card of the Slurm allocation (2 or 4), and read_setting, which main.sh, constlr.sh and hires.sh read their settings
# files with. cwd must be the repo root. No SBATCH header: these phases run inside an allocation that someone else (the chain script)
# already holds.
set -euo pipefail
export PT_HPU_LAZY_MODE=1
WEIGHTS=/workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth  # in-container path (/workspace = this repo root)

[[ -x ./gaudi.sh ]] || { echo "ERROR: ./gaudi.sh not found or not executable here; run from the repo root" >&2; exit 1; }
source ./experiment_scripts/common.sh  # ark_device_count: the rank count Ark's own launch path uses
# fail fast (seconds, not an 8 h slot) if the DINOv3 checkpoint is not in place yet
HOST_WEIGHTS="${WEIGHTS/#\/workspace\//./}"
if [[ ! -s "$HOST_WEIGHTS" ]]; then
  echo "ERROR: DINOv3 weights not found at $HOST_WEIGHTS (host path for $WEIGHTS); requeue after the download lands" >&2
  exit 1
fi

# ssl_launch <run_name> <extra ssl_pretrain.py args...>; the run dir is /workspace/<run_name>, i.e. <repo root>/<run_name>.
# It execs gaudi.sh, so run each phase as its own process (bash chain/pilot_a.sh), never source one into a chain script.
# Several phases can then run back to back in one job: the card count and gaudi.sh's HABANA_VISIBLE_MODULES are both
# derived from SLURM_JOB_GPUS in each phase's own process every time, so nothing leaks from one phase to the next.
# gaudi.sh runs the container with --pwd /workspace/base, hence the relative ssl_pretrain.py. --resume is always passed: a
# completed phase exits at once ("nothing to do"), a re-submitted chain continues where the last one stopped, and a first
# run just starts from scratch. Flags in "$@" come last, so they win over the defaults above them.
ssl_launch() {
  local run_name="$1" cards
  shift
  # one rank per allocated card, counted on the host as Ark's launch path does (ark_device_count): HABANA_VISIBLE_MODULES,
  # else HABANA_VISIBLE_DEVICES, else SLURM_JOB_GPUS, else ask the container. It is an argument, so --cleanenv drops nothing.
  cards=$(ark_device_count gaudi)
  exec ./gaudi.sh python -m torch.distributed.run --standalone --nproc-per-node "$cards" ssl_pretrain.py --device hpu \
    --data_set VinDrCXR --data_set ChestXray14 --data_set MIMIC --data_set CheXpert \
    --pretrained_weights "$WEIGHTS" --batch_size_per_gpu 64 --workers 16 --seed 100 \
    --output_dir "/workspace/$run_name" --resume "$@"
}

# read_setting <file> <extended regex> <description>: prints the file's content without whitespace, or exits 1 with an ERROR
read_setting() {
  local file="$1" pattern="$2" what="$3" value
  if [[ ! -s "$file" ]]; then
    echo "ERROR: $file is missing or empty (run from the repo root); it must hold $what" >&2
    exit 1
  fi
  value=$(tr -d '[:space:]' < "$file")
  if [[ ! "$value" =~ $pattern ]]; then
    echo "ERROR: $file holds '$value', which is not $what" >&2
    exit 1
  fi
  printf '%s' "$value"
}
