#!/usr/bin/env bash
#SBATCH --job-name=b2-emb-c
#SBATCH -A class_cse49478170fall2026
#SBATCH -p public,htc
#SBATCH -q class
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH -c 12
#SBATCH --mem=64G
# The A100 sibling of run_emb_gaudi.sh: the same usage, extras allow-list, validation, results preflight, ARK_EMB_INPUT_SIZE / ARK_EMB_NO_FIT and fit; see run_emb_gaudi.sh for the contract.
# What differs: the header (one A100 on the class account, the 1-GPU header of resume_d1_6_class_1gpu.sh with -c 12); nvidia.sh instead of gaudi.sh; --device cuda; --workers 12 = -c; and no Gaudi card check
# (Slurm's GPU gres confines the job, and nvidia.sh forwards CUDA_VISIBLE_DEVICES). The four fits get OMP_NUM_THREADS=1 through `env` inside the container, because nvidia.sh's --cleanenv forwards no OMP_NUM_THREADS.
# The class QOS counts the REQUESTED -t of a user's running GPU jobs against 960 GPU-minutes, so pass a realistic -t (the 04:00:00 above is only the default); htc's limit is 4 h, so a longer -t runs on public only.
# e.g. ARK_EMB_INPUT_SIZE=768 sbatch -t 14:00:00 run_emb_class.sh arkplus6 swin_large_768 /workspace/weights/Ark6_swinLarge768_ep50.pth.tar imagenet --datasets MIMIC
#      (an Ark+ Swin-L, one job per dataset as in run_emb_gaudi.sh: ARK_EMB_NO_FIT=1 in front of ARK_EMB_INPUT_SIZE on every such job but the last, which runs without it, with sbatch --dependency=afterok:<the other jobs' ids>)
# --- begin: shared with run_emb_gaudi.sh / run_emb_class.sh (test_ark_finetune.py checks the two copies are identical) ---
TAG_RE='^[A-Za-z0-9][A-Za-z0-9_.-]*$'   # the tag is a directory name: no path separators, no spaces
MODEL_RE='^[A-Za-z0-9_]+$'
NORMALIZATION_RE='^(imagenet|chestx-ray|none|raddino)$'
EXTRA_RE='^--(max_images|datasets|workers)(=.+)?$'   # the only options that may follow <normalization>
if [[ $# -lt 4 || ! $1 =~ $TAG_RE || ! $2 =~ $MODEL_RE || $3 != /workspace/* || ! $4 =~ $NORMALIZATION_RE ]]; then
  echo "Usage: sbatch run_emb_gaudi.sh|run_emb_class.sh <tag: letters, digits, . _ - only> <model> <in-container weights path under /workspace> <imagenet|chestx-ray|none|raddino> [--max_images N] [--datasets NAME...] [--workers N]" >&2
  exit 2
fi
refused=()
for arg in "${@:5}"; do
  [[ $arg == -* && ! $arg =~ $EXTRA_RE ]] && refused+=("$arg")
done
if [[ ${#refused[@]} -gt 0 ]]; then
  echo "ERROR: ${refused[*]}: not accepted after <normalization>. The wrapper owns --tag --model --weights --normalization --device; --embeddings_root, --results_root, --input_size, --img_resize and --batch_size are fixed (they move the cache or change the embeddings the fit reads). Only --max_images N, --datasets NAME... and --workers N may follow" >&2
  exit 2
fi
EMB_SIZE_ARGS=()
if [[ -n ${ARK_EMB_INPUT_SIZE:-} ]]; then
  if [[ ! $ARK_EMB_INPUT_SIZE =~ ^[1-9][0-9]{0,3}$ ]]; then
    echo "ERROR: ARK_EMB_INPUT_SIZE='$ARK_EMB_INPUT_SIZE' is not a positive integer of at most 4 digits (the crop size in pixels, e.g. 512)" >&2
    exit 2
  fi
  EMB_SIZE_ARGS=(--input_size "$ARK_EMB_INPUT_SIZE" --img_resize $(( (ARK_EMB_INPUT_SIZE * 16 + 7) / 14 )))  # round(size x 8/7)
fi
if [[ -n ${ARK_EMB_NO_FIT:-} && $ARK_EMB_NO_FIT != 1 ]]; then
  echo "ERROR: ARK_EMB_NO_FIT='$ARK_EMB_NO_FIT' must be unset, empty or 1" >&2
  exit 2
fi
TAG=$1
MODEL=$2
WEIGHTS=$3
NORMALIZATION=$4
datasets=(ChestXray14 CheXpert MIMIC VinDrCXR)
cd "${ARK_ROOT:-/scratch/$USER/2026Fall-CSE507-Ark-Container}" || exit 1
# fail fast (seconds, not the slot) if the weights are not in place; $WEIGHTS is the in-container path (/workspace = this repo root)
HOST_WEIGHTS="${WEIGHTS/#\/workspace\//./}"
if [[ ! -s "$HOST_WEIGHTS" ]]; then
  echo "ERROR: weights not found at $HOST_WEIGHTS (host path for $WEIGHTS); requeue after they land" >&2
  exit 1
fi
# fail fast the other way round too: results that exist for any dataset to be fit would make run_linearprobe refuse AFTER the extraction had spent the card. These are the two paths per dataset
# that it checks (run_linearprobe.results_of, kept in step by the wrapper test): the dataset's JSON, and the directory of the shipped script's own files. A dataset's .log alone is no result.
taken=()
for dataset in "${datasets[@]}"; do
  for path in "results/$TAG/$dataset.json" "results/$TAG/probe/$dataset"; do
    [[ -e $path ]] && taken+=("$path")
  done
done
if [[ ${#taken[@]} -gt 0 ]]; then
  echo "ERROR: results already exist for tag $TAG: ${taken[*]}; refusing to start the extraction, because the fit would refuse to overwrite them afterwards. Remove them or use another tag" >&2
  exit 1
fi
# --- end: shared ---
# 1. the extraction on the GPU; it refuses an existing cache before the model loads, and a failure ends the job here: no fit on a half-written cache
./nvidia.sh python -m ark_finetune.extract_embeddings --device cuda --tag "$TAG" --model "$MODEL" --weights "$WEIGHTS" --normalization "$NORMALIZATION" --workers 12 "${EMB_SIZE_ARGS[@]}" "${@:5}" || exit $?
if [[ ${ARK_EMB_NO_FIT:-} == 1 ]]; then  # extraction only: no fit and no results/<tag> here, the job that fits (the last dataset's, without the switch) finds all four caches
  echo "ARK_EMB_NO_FIT=1: extraction only, no fit for tag $TAG; the job that extracts the last dataset of this tag without ARK_EMB_NO_FIT fits all four"
  exit 0
fi
# 2. the fit: four CPU processes, one thread each (the loop is bound by Python overhead on small tensors; set through env inside the container, nvidia.sh's --cleanenv drops a host OMP_NUM_THREADS), each with its own log; run_linearprobe refuses a (tag, dataset) that has results
mkdir -p "results/$TAG" || exit 1
pids=()
for dataset in "${datasets[@]}"; do
  ./nvidia.sh env OMP_NUM_THREADS=1 python -m ark_linearprobe.run_linearprobe --tag "$TAG" --datasets "$dataset" > "results/$TAG/$dataset.log" 2>&1 &
  pids+=($!)
done
status=0
for i in "${!datasets[@]}"; do
  if wait "${pids[$i]}"; then
    echo "fit ${datasets[$i]}: done"
  else
    echo "ERROR: fit ${datasets[$i]} failed (see results/$TAG/${datasets[$i]}.log)" >&2
    status=1
  fi
done
grep -h '^PROBE ' "results/$TAG"/*.log   # the one-line summaries; the JSON is results/<tag>/<dataset>.json
exit $status
