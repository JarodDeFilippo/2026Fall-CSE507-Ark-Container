#!/usr/bin/env bash
# The frozen-backbone segmentation probe (base/dense/seg_probe.py) as ONE Slurm job: one A100 on the class account, one (backbone, dataset) run. This script has no #SBATCH header of its own: it checks its arguments, generates the
# job script and gives it to sbatch on stdin. NOT an sbatch script: run it on the login node, from the repo copy (it refuses inside a Slurm job). It prints sbatch's own line and exits; nothing is left running here.
# usage: ./launch_dense.sh [--check] <candidptx|jsrt_lung> <tag> <model> <in-container weights path under /workspace> <imagenet|chestx-ray|none|raddino> [--iters N] [--seed S] [--runs N] [--size N]
#   <tag>     names the backbone and is a directory name (letters, digits, . _ - only): the job writes results/dense/<tag>/<dataset>/seed<S>/result.json and head_best.pt for S = --seed (0 without it), --seed + 1, ... , --seed + --runs - 1 (one run without --runs)
#   <model>   vit_base_dinov3, vit_base_dinov2, vit_base_raddino or vit_large_raydino (base/models.py names; seg_probe's --model)
#   --iters N, --seed S, --runs N, --size N   the only options, each at most once; without them seg_probe's defaults apply (5000 iterations, seed 0, one run, 224 px); --runs N runs the N seeds S, S + 1, ... one after another in the one job; --size N is the input side in pixels
#   The results path does not carry the size (it stays results/dense/<tag>/<dataset>/seed<S>), so name the size in the <tag> (e.g. newdata512_e0005_512px); a larger size is slower, so raise DENSE_TIME with it.
#   ponytail: --iters, --seed, --runs and --size are checked as integers only; seg_probe applies its own minimums when the job starts, and checks then that --size is a multiple of the backbone's patch size (16 for vit_base_dinov3 and vit_large_raydino, so 512; 14 for vit_base_dinov2 and vit_base_raddino, so 518): a size off that grid is refused only after the queue wait.
#   e.g. ./launch_dense.sh candidptx stock_dinov3 vit_base_dinov3 /workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth imagenet
#   e.g. ./launch_dense.sh candidptx newdata512_e0005_512px vit_base_dinov3 /workspace/ssl_dinov3_cxr_newdata512_arkteacher_seed_100/export/teacher_backbone_epoch_0005.pth imagenet --size 512 --seed 0 --runs 10
# The job: cd to ARK_ROOT (default /scratch/$USER/2026Fall-CSE507-Ark-Container) and ./nvidia.sh python -m dense.seg_probe on the arguments, with --workers 12 (= -c 12) and --out /workspace/results/dense (results/dense in ARK_ROOT).
# Its header is run_emb_class.sh's: -A class_cse49478170fall2026 -p public,htc -q class -N 1 --gres=gpu:1 --constraint=a100_80 -c 12 --mem=64G, then -t and -J dense-<tag>-<dataset>.
# environment: DENSE_TIME (-t, default 02:00:00: a default, not a measurement, so set it from the first run's wall time times --runs: the class QOS counts the requested -t of a user's running GPU jobs against its GPU-minute cap);
#   DENSE_ACCOUNT (-A, default class_cse49478170fall2026; a group member in another class account sets it, and it must carry the class QOS); ARK_ROOT (the repo copy the job runs in).
# Checks, each a stop that names its cause (exit 2: the arguments, DENSE_TIME, DENSE_ACCOUNT; exit 1: the rest): the arguments' shape and values (the regexes of run_emb_class.sh, plus the datasets, the models and the four options); not inside a Slurm job
#   (a real submission only); then, if ARK_ROOT exists, that the weights file is under it (a missing file would otherwise fail only after the queue wait) and that none of results/dense/<tag>/<dataset>/seed<S>, for the job's seeds, exists in any form (seg_probe refuses to run when one does,
#   after the same wait). A real submission also needs ARK_ROOT to exist, and sbatch.
# --check as the FIRST argument: every check runs, then the sbatch line and the job script are printed, as a heredoc that repeats the submission, and nothing is submitted (exit 0). No Slurm command is called, so it runs on any machine; where ARK_ROOT does
#   not exist (a laptop) it says on stderr that the weights file and the results were not checked.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1   # the repo root: sbatch writes slurm-<id>.out in the directory it is run from
check_only=0
if [[ ${1:-} == --check ]]; then check_only=1; shift; fi
USAGE="Usage: ./launch_dense.sh [--check] <candidptx|jsrt_lung> <tag: letters, digits, . _ - only> <vit_base_dinov3|vit_base_dinov2|vit_base_raddino|vit_large_raydino> <in-container weights path under /workspace> <imagenet|chestx-ray|none|raddino> [--iters N] [--seed S] [--runs N] [--size N]"
DATASET_RE='^(candidptx|jsrt_lung)$'
TAG_RE='^[A-Za-z0-9][A-Za-z0-9_.-]*$'   # the tag is a directory name: no path separators, no spaces
MODEL_RE='^(vit_base_dinov3|vit_base_dinov2|vit_base_raddino|vit_large_raydino)$'
NORMALIZATION_RE='^(imagenet|chestx-ray|none|raddino)$'
ITERS_RE='^[1-9][0-9]{0,6}$'
RUNS_RE='^[1-9][0-9]{0,2}$'
SEED_RE='^(0|[1-9][0-9]{0,8})$'   # no leading zeros: seg_probe names the folder seed<int>, so 007 and 7 would be one run
SIZE_RE='^[1-9][0-9]{0,3}$'
TIME_RE='^([0-9]+-)?[0-9]+(:[0-9]{2}){0,2}$'   # a Slurm -t: minutes, M:SS, H:MM:SS, D-H, D-H:MM or D-H:MM:SS
ACCOUNT_RE='^[A-Za-z0-9_]+$'
die() { echo "ERROR: $*" >&2; exit 2; }
[[ $# -ge 5 ]] || { echo "$USAGE" >&2; exit 2; }
DATASET=$1
TAG=$2
MODEL=$3
WEIGHTS=$4
NORMALIZATION=$5
[[ $DATASET =~ $DATASET_RE ]] || die "dataset '$DATASET' is not candidptx or jsrt_lung"
[[ $TAG =~ $TAG_RE ]] || die "tag '$TAG' is not letters, digits, . _ - only (and starts with a letter or digit)"
[[ $MODEL =~ $MODEL_RE ]] || die "model '$MODEL' is not vit_base_dinov3, vit_base_dinov2, vit_base_raddino or vit_large_raydino"
[[ $WEIGHTS == /workspace/* ]] || die "weights '$WEIGHTS' is not an in-container path under /workspace"
[[ $NORMALIZATION =~ $NORMALIZATION_RE ]] || die "normalization '$NORMALIZATION' is not imagenet, chestx-ray, none or raddino"
extra=("${@:6}")
iters=
seed=
runs=
size=
i=0
while [[ $i -lt ${#extra[@]} ]]; do
  value=${extra[$((i + 1))]:-}
  case ${extra[$i]} in
    --iters) [[ -z $iters && $value =~ $ITERS_RE ]] || die "--iters takes one positive integer, once (got '$value')"; iters=$value ;;
    --seed) [[ -z $seed && $value =~ $SEED_RE ]] || die "--seed takes one non-negative integer without leading zeros, once (got '$value')"; seed=$value ;;
    --runs) [[ -z $runs && $value =~ $RUNS_RE ]] || die "--runs takes one positive integer, once (got '$value')"; runs=$value ;;
    --size) [[ -z $size && $value =~ $SIZE_RE ]] || die "--size takes one positive integer, once (got '$value')"; size=$value ;;
    *) die "${extra[$i]}: not accepted after <normalization>; only --iters N, --seed S, --runs N and --size N may follow" ;;
  esac
  i=$((i + 2))
done
time_limit=${DENSE_TIME:-02:00:00}
[[ $time_limit =~ $TIME_RE ]] || die "DENSE_TIME='$time_limit' is not a Slurm time limit (minutes, H:MM:SS or D-HH:MM:SS)"
account=${DENSE_ACCOUNT:-class_cse49478170fall2026}
[[ $account =~ $ACCOUNT_RE ]] || die "DENSE_ACCOUNT='$account' is not a Slurm account name"
if [[ $check_only == 0 && -n ${SLURM_JOB_ID:-} ]]; then
  echo "ERROR: launch_dense.sh submits the job itself: run it on the login node, not inside Slurm job $SLURM_JOB_ID" >&2
  exit 1
fi
ARK_ROOT=${ARK_ROOT:-/scratch/$USER/2026Fall-CSE507-Ark-Container}
if [[ -d $ARK_ROOT ]]; then
  HOST_WEIGHTS=$ARK_ROOT/${WEIGHTS#/workspace/}   # $WEIGHTS is the in-container path (/workspace = the repo copy)
  if [[ ! -f $HOST_WEIGHTS || ! -s $HOST_WEIGHTS ]]; then
    echo "ERROR: weights not found at $HOST_WEIGHTS (host path for $WEIGHTS); nothing was submitted" >&2
    exit 1
  fi
  for ((s = ${seed:-0}; s < ${seed:-0} + ${runs:-1}; s++)); do   # seg_probe's folder for each seed of the job, refused when it exists in any form; its default is seed 0, one run
    RESULT_DIR=results/dense/$TAG/$DATASET/seed$s
    if [[ -e $ARK_ROOT/$RESULT_DIR || -L $ARK_ROOT/$RESULT_DIR ]]; then
      echo "ERROR: $RESULT_DIR exists in $ARK_ROOT and seg_probe will not run over it: delete $RESULT_DIR there to rerun, or use another tag or seed; nothing was submitted" >&2
      exit 1
    fi
  done
elif [[ $check_only == 1 ]]; then
  echo "note: $ARK_ROOT is not a directory here, so the weights file and existing results were not checked" >&2
else
  echo "ERROR: $ARK_ROOT is not a directory: set ARK_ROOT to your copy of the repo on Sol; nothing was submitted" >&2
  exit 1
fi
cmd=(./nvidia.sh python -m dense.seg_probe --model "$MODEL" --weights "$WEIGHTS" --normalization "$NORMALIZATION" --dataset "$DATASET" --tag "$TAG" --workers 12 --out /workspace/results/dense)
[[ -n $iters ]] && cmd+=(--iters "$iters")
[[ -n $seed ]] && cmd+=(--seed "$seed")
[[ -n $runs ]] && cmd+=(--runs "$runs")
[[ -n $size ]] && cmd+=(--size "$size")
body=$(printf '#!/usr/bin/env bash\ncd %q && ' "$ARK_ROOT"; printf '%q ' "${cmd[@]}")
body=${body% }
opts=(-A "$account" -p "public,htc" -q class -N 1 --gres=gpu:1 --constraint=a100_80 -c 12 --mem=64G -t "$time_limit" -J "dense-$TAG-$DATASET")
if [[ $check_only == 1 ]]; then
  echo "sbatch ${opts[*]} <<'JOB'"
  printf '%s\n' "$body"
  echo JOB
  exit 0
fi
command -v sbatch > /dev/null || { echo "ERROR: sbatch not found: run this on the Sol login node (--check needs no Slurm); nothing was submitted" >&2; exit 1; }
sbatch "${opts[@]}" <<< "$body" || { echo "ERROR: sbatch refused the job; nothing was submitted" >&2; exit 1; }
