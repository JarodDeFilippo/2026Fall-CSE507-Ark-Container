#!/usr/bin/env bash
# Parity check of a converted DINOv3 .pth against the Hugging Face checkpoint it came from (base/check_hf_dinov3_parity.py), as a CPU job on Sol:
#   sbatch ... --wrap "./check_hf_dinov3_parity.sh <model> <HF dir> <converted .pth> [--input-size N] [--seed N] [--tol X] [--dtype float32|bfloat16]"
# A plain script with no #SBATCH lines (it can be an sbatch script's body too), so the job is sized on the command line. <model> is vit_base_dinov3, vit_large_dinov3, vit_huge_plus_dinov3 or
# vit_7b_dinov3; everything after it goes to the Python script as is. The paths are the in-container ones (/workspace = this repo root). The check prints its memory estimate first (about 54 GB
# for the 7B in float32; --dtype bfloat16 needs an explicit --tol). The Python is CPU-only. nvidia.sh passes --nv (weight_norms.sh asks for a GPU because of it), but three CPU jobs on htc ran
# ./nvidia.sh fine without one (66354710-12, 2026-10-22), so a CPU node is acceptable.
# ARK_ROOT overrides the repo root (default /scratch/$USER/2026Fall-CSE507-Ark-Container).
# transformers is not in the container image. It is pip-installed into a directory of its own per pin, /workspace/.pylib-transformers-<TRANSFORMERS> (./.pylib-transformers-<TRANSFORMERS> on the host), and
# nothing else goes into it: the directory holds transformers if and only if exactly that pin was installed, a new pin gets a new directory, an old one can be deleted by hand, and nothing is merged.
# The install is staged in a temporary directory (.pylib.XXXXXX) and published only when pip has finished, so a job that dies half way leaves no cache that looks complete. Two first-time jobs race
# on the publishing: mkdir of the cache directory is the lock. The job whose mkdir succeeds moves the stage's entries in, transformers last, so "transformers is there" means everything is; a job whose
# mkdir fails discards its own stage and uses the other's copy once transformers is in it (it waits up to 30 s).
#   ponytail: a job killed between its mkdir and its last mv leaves a cache without transformers; the next run says so (after the 30 s) and the directory must be removed by hand.
#   TRANSFORMERS is the pin: DINOv3ViTModel arrived in transformers 4.56.0, and 4.56.2 (2025-09-19) is the last 4.56 release (4.57.0 was yanked). It still takes torch_dtype, with a one-time
#   deprecation warning that dtype replaces it: a later pin should change that one call in check_hf_dinov3_parity.py. Tried 2026-10-22 on the Mac with torch 2.11.0, numpy 2.2.6 and Python 3.12
#   (the container's versions) through this same --target install: the real ViT-B checkpoint gave PARITY OK, largest max |diff| 1.1e-05.
#   numpy is deleted from the install: the container pins its own (containers/nvidia/requirements.txt: 2.2.6, next to opencv-python-headless 4.12.0.88, which declares numpy<2.3), and PYTHONPATH
#   would put pip's copy in front of it (pip --target took numpy 2.5.4 on 2026-10-22). The other dependencies (huggingface-hub, tokenizers, safetensors, ...) are new to the container or pure Python.
# PYTHONPATH is set INSIDE the container command: nvidia.sh runs apptainer with --cleanenv, which drops the host's environment, so an `export PYTHONPATH=...` here would never reach python.
set -euo pipefail
TRANSFORMERS=4.56.2
CACHE=.pylib-transformers-$TRANSFORMERS
if [[ $# -lt 3 ]]; then
    echo "Usage: sbatch ... --wrap \"$0 <model> <in-container HF dir> <in-container converted .pth> [--input-size N] [--seed N] [--tol X] [--dtype float32|bfloat16]\"" >&2
    exit 2
fi
cd "${ARK_ROOT:-/scratch/$USER/2026Fall-CSE507-Ark-Container}"
if [[ ! -d $CACHE/transformers ]]; then
    stage=$(mktemp -d .pylib.XXXXXX)
    trap 'rm -rf "$stage"' EXIT
    ./nvidia.sh python -m pip install --target "/workspace/$stage" -q "transformers==$TRANSFORMERS"
    rm -rf "$stage"/numpy "$stage"/numpy.libs "$stage"/numpy-*.dist-info
    if mkdir "$CACHE" 2>/dev/null; then
        shopt -s dotglob
        for entry in "$stage"/*; do [[ ${entry##*/} == transformers ]] || mv "$entry" "$CACHE"/; done
        mv "$stage"/transformers "$CACHE"/
    else
        for _ in $(seq 30); do [[ -d $CACHE/transformers ]] && break; sleep 1; done
    fi
    rm -rf "$stage"
    trap - EXIT
    [[ -d $CACHE/transformers ]] || { echo "ERROR: $CACHE has no transformers (an install killed half way leaves it so): remove $CACHE and rerun" >&2; exit 1; }
fi
exec ./nvidia.sh env PYTHONPATH=/workspace/$CACHE python check_hf_dinov3_parity.py --model "$1" "${@:2}"
