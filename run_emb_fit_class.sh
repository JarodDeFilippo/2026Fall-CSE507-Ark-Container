#!/usr/bin/env bash
#SBATCH --job-name=b2-fit-c
#SBATCH -A class_cse49478170fall2026
#SBATCH -p public,htc
#SBATCH -q class
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=64G
# The fit half of run_emb_class.sh as a job of its own: CPU only, so NO --gres and no constraint, and no A100 sits idle behind it. B2 (CSE 494 IP, Step VIII): the Ark+ lab's embedding linear probe (Ark_Plus/Linearprobing/main_linearprobing.py, as shipped)
# on the embeddings that run_emb_class.sh (ARK_EMB_NO_FIT=1) or run_emb_gaudi.sh already extracted under embeddings/<tag>/. It is step 2 of run_emb_class.sh unchanged: base/ark_linearprobe/run_linearprobe.py, 10 runs of 100 epochs per dataset,
# one process per dataset and one thread each, the datasets in parallel -> results/<tag>/<dataset>.json, with <dataset>.log (the runner's stdout) and probe/<dataset>/ (the script's own files: its per-batch log.txt, run_<i>/ckpt.pth) beside it.
# Why a job of its own: the one-job probe holds an A100 for the extraction (about 1 h for a ViT-B at 224 px) and then through the fits (about 1.5 h more on the CPU, the card idle), and the class QOS counts the requested -t of running GPU jobs against its
# GPU-minute cap (run_emb_class.sh); this job asks for no GPU. The lab's own probe is fit-only by construction (its script reads saved embeddings). submit_probe_class.sh submits the extraction and this fit together, the fit behind it with afterok.
# -t 03:00:00 is a default, not a measurement: the longest of the four fits, MIMIC, took 1.6 to 1.8 h on Sol (a one-job probe, 2026-10-26), and the four run in parallel, so the job takes as long as MIMIC. Raise it for a slower node (sbatch -t, or PROBE_FIT_TIME for
# the wrapper); htc's limit is 4 h, so a longer -t runs on public only.
# usage: sbatch run_emb_fit_class.sh <tag> [--datasets NAME...]
#   <tag>        the backbone tag the embeddings were extracted under: letters, digits, . _ - only (as in run_emb_class.sh).
#   --datasets   the only option: a non-empty subset of ChestXray14 CheXpert MIMIC VinDrCXR, each at most once; without it all four. Anything else exits 2 before anything runs.
#   e.g. sbatch run_emb_fit_class.sh nogram_e0060
#        sbatch -t 04:00:00 run_emb_fit_class.sh nogram_e0060 --datasets MIMIC
# Checks, in this order, each a stop that names its cause: (1) the arguments, and ARK_EMB_FIT_RESUME below (exit 2); (2) results that already exist for a dataset to be fit: results/<tag>/<dataset>.json or results/<tag>/probe/<dataset>, the two paths run_linearprobe refuses
# and run_emb_class.sh checks before its extraction (exit 1; ARK_EMB_FIT_RESUME=1 replaces this stop by the resume rule); (3) the cache of every dataset to be fit, embeddings/<tag>/<dataset>/meta.json, which the extractor writes last, so it is there only once that
# dataset's extraction has finished (exit 1, naming each missing one; run_linearprobe then checks all the cache's files itself). A failed fit exits 1 after the others have finished, as in run_emb_class.sh; the job log ends with the PROBE lines of results/<tag>/*.log.
# ARK_EMB_FIT_RESUME (environment): unset or empty = a fresh fit, check (2) above. Exactly 1 = finish a fit that stopped half way (a time limit, a node failure, one failed fit): per dataset, results/<tag>/<dataset>.json there = finished, skipped with a line, never touched;
# not there = whatever else is left of that dataset belongs to a dead fit, so its results/<tag>/probe/<dataset> is removed (with a line) and the dataset is fitted again from the start, because run_linearprobe cannot resume one (log.txt appends, ckpt.pth is overwritten, and it
# refuses to start on the directory). That directory is the fit's own output under results/<tag>/ (the shipped script's log.txt and run_<i>/ckpt.pth), never a dataset root, and it is the only thing this script ever removes, and only in this mode; the cache check comes first,
# so nothing is removed for a fit that cannot start. Re-submit only after the earlier fit job has ended: a half-done directory is taken to be a dead job's, and removing a live job's would break it. Any other value exits 2.
# --check as the FIRST argument (submit_probe_class.sh's check before it submits anything; under sbatch it would only run an empty job): runs checks (1) and (2) and exits 0, silently, without check (3) (the extraction has not run yet when the wrapper asks) and without
# writing or removing anything.
# ARK_ROOT overrides the repo root (the Mac dry run in base/ark_finetune/test_ark_finetune.py points it at a fake repo).
# The lines TAG_RE=, datasets=(, cd "${ARK_ROOT..., and the loop that fills taken below are the same text as in the block run_emb_gaudi.sh and run_emb_class.sh share (test_ark_finetune.py checks it); the rest of that block is the extraction's.
check_only=0
if [[ ${1:-} == --check ]]; then check_only=1; shift; fi
TAG_RE='^[A-Za-z0-9][A-Za-z0-9_.-]*$'   # the tag is a directory name: no path separators, no spaces
datasets=(ChestXray14 CheXpert MIMIC VinDrCXR)
member() { local word; for word in "${@:2}"; do [[ $word == "$1" ]] && return 0; done; return 1; }   # member WORD LIST...: WORD is one of LIST
if [[ $# -lt 1 || ! $1 =~ $TAG_RE || ( $# -gt 1 && ( $2 != --datasets || $# -lt 3 ) ) ]]; then
  echo "Usage: sbatch run_emb_fit_class.sh <tag: letters, digits, . _ - only> [--datasets NAME...]   (NAME: ChestXray14 CheXpert MIMIC VinDrCXR, each at most once)" >&2
  exit 2
fi
TAG=$1
if [[ $# -gt 1 ]]; then
  chosen=()
  for name in "${@:3}"; do
    if ! member "$name" "${datasets[@]}"; then
      echo "ERROR: '$name' is not a dataset of the probe (ChestXray14 CheXpert MIMIC VinDrCXR)" >&2
      exit 2
    fi
    if member "$name" "${chosen[@]}"; then
      echo "ERROR: --datasets lists $name twice" >&2
      exit 2
    fi
    chosen+=("$name")
  done
  datasets=("${chosen[@]}")
fi
if [[ -n ${ARK_EMB_FIT_RESUME:-} && $ARK_EMB_FIT_RESUME != 1 ]]; then
  echo "ERROR: ARK_EMB_FIT_RESUME='$ARK_EMB_FIT_RESUME' must be unset, empty or 1" >&2
  exit 2
fi
cd "${ARK_ROOT:-/scratch/$USER/2026Fall-CSE507-Ark-Container}" || exit 1
# results that exist for a dataset to be fit: the two paths per dataset that run_linearprobe checks (run_linearprobe.results_of, kept in step by the wrapper test): the dataset's JSON, and the directory of the shipped script's own files. A dataset's .log alone is no result.
taken=()
for dataset in "${datasets[@]}"; do
  for path in "results/$TAG/$dataset.json" "results/$TAG/probe/$dataset"; do
    [[ -e $path ]] && taken+=("$path")
  done
done
if [[ ${#taken[@]} -gt 0 && ${ARK_EMB_FIT_RESUME:-} != 1 ]]; then
  echo "ERROR: results already exist for tag $TAG: ${taken[*]}; refusing to fit over them. To finish a fit that stopped half way set ARK_EMB_FIT_RESUME=1 (finished datasets are skipped, a half-done one is removed and fitted again); otherwise remove them or use another tag" >&2
  exit 1
fi
[[ $check_only == 1 ]] && exit 0
missing=()
for dataset in "${datasets[@]}"; do
  [[ -f embeddings/$TAG/$dataset/meta.json ]] || missing+=("embeddings/$TAG/$dataset/meta.json")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "ERROR: no embeddings for tag $TAG: ${missing[*]} missing; extract them first (ARK_EMB_NO_FIT=1 sbatch run_emb_class.sh $TAG <model> <weights> <normalization> [--datasets NAME...])" >&2
  exit 1
fi
# the datasets to fit now: all of them in a fresh fit (taken is empty there); in the resume rule the finished ones are skipped and a half-done one is cleared first (the comment above says why that directory may be removed)
todo=()
for dataset in "${datasets[@]}"; do
  if [[ -e results/$TAG/$dataset.json ]]; then
    echo "skip $dataset: results/$TAG/$dataset.json exists, the fit is finished"
    continue
  fi
  if [[ -e results/$TAG/probe/$dataset ]]; then
    echo "refit $dataset: removing results/$TAG/probe/$dataset, left half done by an earlier fit"
    rm -rf "results/$TAG/probe/$dataset"
  fi
  todo+=("$dataset")
done
# the fit: four CPU processes, one thread each (the loop is bound by Python overhead on small tensors; set through env inside the container, nvidia.sh's --cleanenv drops a host OMP_NUM_THREADS), each with its own log; run_linearprobe refuses a (tag, dataset) that has results
mkdir -p "results/$TAG" || exit 1
pids=()
for dataset in "${todo[@]}"; do
  ./nvidia.sh env OMP_NUM_THREADS=1 python -m ark_linearprobe.run_linearprobe --tag "$TAG" --datasets "$dataset" > "results/$TAG/$dataset.log" 2>&1 &
  pids+=($!)
done
status=0
for i in "${!todo[@]}"; do
  if wait "${pids[$i]}"; then
    echo "fit ${todo[$i]}: done"
  else
    echo "ERROR: fit ${todo[$i]} failed (see results/$TAG/${todo[$i]}.log)" >&2
    status=1
  fi
done
grep -h '^PROBE ' "results/$TAG"/*.log   # the one-line summaries; the JSON is results/<tag>/<dataset>.json
exit $status
