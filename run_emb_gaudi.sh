#!/usr/bin/env bash
#SBATCH --job-name=b2-emb
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:1
#SBATCH -c 36
#SBATCH --mem=128G
# B2 (CSE 494 IP, Step VIII): the Ark+ lab's embedding linear probe (Ark_Plus/Linearprobing/main_linearprobing.py, as shipped) for ONE backbone, on 1 Gaudi card and the node's CPUs.
#   1. extraction on the card (base/ark_finetune/extract_embeddings.py): the frozen backbone's 1536-d (ViT-B) or 2048-d (ViT-L) vector (CLS ++ mean patch), or a Swin's pooled output before its projector (1024-d Swin-B, 1536-d Swin-L), for every image of the full official train list and the official
#      test list of ChestXray14, CheXpert, MIMIC and VinDrCXR, ONE center-crop embedding per image, no validation split -> /workspace/embeddings/<tag>/<dataset>/
#   2. the fit on the CPU (base/ark_linearprobe/run_linearprobe.py): the shipped script, 10 runs of 100 epochs (float64 linear layer, Adam 1e-4 with the shipped cosine schedule whose lr RISES, batch 512),
#      in each run the epoch with the best TEST mAUC is the one reported (the script has no validation split); one process per dataset, the four in parallel
#      -> /workspace/results/<tag>/<dataset>.json, with <dataset>.log (the runner's stdout) and probe/<dataset>/ (the script's own files: its per-batch log.txt, run_<i>/ckpt.pth) beside it.
# The 8 h limit is a bound, not a measurement of the extraction (717 k images, no Gaudi timing yet). The fit was timed on an M3 Pro, one thread per dataset, on synthetic embeddings of the real
# list sizes, other jobs running beside it: the full 10 runs x 100 epochs took 10 min (ChestXray14), 21 min (CheXpert), 31 min (MIMIC) and 1.5 min (VinDrCXR), so the four in parallel take as long
# as MIMIC; Sol's cores may be slower. The shipped script logs a line per batch: results/<tag>/probe/<dataset>/log.txt came to 17 / 48 / 78 / 4 MB.
# usage: sbatch run_emb_gaudi.sh <tag> <model> <in-container weights path> <normalization> [--max_images N] [--datasets NAME...] [--workers N]
#   <tag>            names the backbone: letters, digits, . _ - only. The cache and the results are keyed by it. Results that already exist for the tag stop the job in seconds, before the card
#                    check and the extraction (the fit would refuse to overwrite them after the extraction had spent the allocation); an existing cache is refused by the extractor.
#   <model>          base/models.py's model name, passed through: vit_base_dinov3, vit_base_dinov2, vit_base_raddino, vit_large_raydino, swin_base, swin_large, swin_large_384, swin_large_768 (an Ark+ .pth.tar: its TEACHER
#                    is embedded, see models.build_omni_model_from_checkpoint).
#   <weights>        under /workspace (= the repo root on the host): the stock .pth, a Stage A export/teacher_backbone_epoch_NNNN.pth or an Ark+ .pth.tar.
#   <normalization>  imagenet | chestx-ray | none | raddino.
#   e.g. sbatch run_emb_gaudi.sh stock_dinov3 vit_base_dinov3 /workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth imagenet
#        sbatch run_emb_gaudi.sh rad_dino vit_base_raddino /workspace/weights/rad_dino_vitb14_pretrain.pth raddino
#        sbatch run_emb_gaudi.sh raydino vit_large_raydino /workspace/weights/xray_dino_vitl16_pretrained-ad31c2b0.pth imagenet
#        (RayDINO: Meta's xray_dino_vitl16 file, trained with ImageNet mean/std, so no new normalization; its block keys are renamed on load, see models.flatten_chunked_blocks)
#        ARK_EMB_INPUT_SIZE=768 sbatch -t 1-00:00:00 run_emb_gaudi.sh arkplus6 swin_large_768 /workspace/weights/Ark6_swinLarge768_ep50.pth.tar imagenet
#        (an Ark+ Swin-L, the lab's public file name: Ark+ trains with ImageNet mean/std, the lab's main_ark.py default, so no new normalization; a Swin is built for ONE input size, so ARK_EMB_INPUT_SIZE must equal it, 768 here,
#        and the extractor refuses otherwise)
#   smoke: sbatch run_emb_gaudi.sh stock_smoke vit_base_dinov3 /workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth imagenet --max_images 512
# After <normalization> only --max_images N, --datasets NAME... and --workers N may follow (--opt=value too); they reach extract_embeddings.py as they are, after the arguments below, so
# --workers 16 wins over the wrapper's 32. Any other argument exits 2 before anything else runs, the card check included: --tag --model --weights --normalization --device are the wrapper's own
# (a second --tag would send the cache where the fit does not look), --embeddings_root and --results_root move the cache or the results away from where the fit looks, and --input_size,
# --img_resize and --batch_size are the extractor's fixed embedding settings (224 / 256 / 64), which every result of a tag must share. An allow-list, not a deny-list: argparse takes an
# abbreviation (--emb) for the option it abbreviates, and --help would exit 0 and start the fit on nothing.
# The fit always covers all four datasets, from the metadata the extractor wrote (--max_images included): --datasets is for finishing an extraction a dead job left half done (delete that dataset's
# directory under embeddings/<tag>/ first, as the refusal says), and the job should then be the first to fit this tag. To redo one dataset's fit by hand, remove its JSON and
# results/<tag>/probe/<dataset> and run:  ./gaudi.sh python -m ark_linearprobe.run_linearprobe --tag <tag> --datasets <dataset>
# The maintenance reservation (Oct 12 08:00 to Oct 15 15:00) means Slurm will not start a job whose limit crosses it: a job that cannot end before Oct 12 08:00 waits until Oct 15 15:00.
# ARK_EMB_INPUT_SIZE (environment, default 224): the center crop the backbone embeds at, with the resize before it = round(size x 8/7) (224 -> 256, 448 -> 512,
# 512 -> 585), passed as the extractor's --input_size / --img_resize and recorded in meta.json (transform.crop / transform.resize; the fit refuses a tag whose
# datasets differ there). Unset, the extractor keeps its own 224 / 256, so every existing probe number stays comparable; a 512 probe is a second table, under
# its own tag, and slower (raise -t). e.g. ARK_EMB_INPUT_SIZE=512 sbatch -t 1-00:00:00 run_emb_gaudi.sh <tag> vit_base_dinov3 <weights> imagenet
# ARK_EMB_NO_FIT (environment): unset or empty = the job above (extraction, then the fit of all four datasets). Exactly 1 = extraction only: once the extraction has succeeded the job prints an "extraction only" line
# and exits 0, with no fit and no results/<tag> (the results preflight still runs); any other value exits 2 with the other environment checks, before the weights check, the preflight and the card check.
# It splits a long extraction (a Swin-L at 768 px) into one job per dataset: ARK_EMB_NO_FIT=1 sbatch ... <tag> <model> <weights> <normalization> --datasets <one dataset> for each dataset but the last, then the last
# dataset's job without ARK_EMB_NO_FIT and with sbatch --dependency=afterok:<the other jobs' ids> ...: it extracts its dataset and fits all four. Every job of a tag takes the same <model>, <weights>, <normalization> and
# ARK_EMB_INPUT_SIZE: the fit refuses the caches of one tag that were not extracted the same way.
# ARK_ROOT overrides the repo root (the Mac dry run in base/ark_finetune/test_ark_finetune.py points it at a fake repo).
# The A100 sibling is run_emb_class.sh (one A100 on the class account, through nvidia.sh): the same usage, checks and fit, and the block between the two "shared" marker lines below is identical in both scripts (test_ark_finetune.py checks it).
export PT_HPU_LAZY_MODE=1
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
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated card before the extraction starts
./gaudi_check_cards.sh || exit 1
# 1. the extraction; it refuses an existing cache before the model loads, and a failure ends the job here: no fit on a half-written cache
./gaudi.sh python -m ark_finetune.extract_embeddings --device hpu --tag "$TAG" --model "$MODEL" --weights "$WEIGHTS" --normalization "$NORMALIZATION" --workers 32 "${EMB_SIZE_ARGS[@]}" "${@:5}" || exit $?
if [[ ${ARK_EMB_NO_FIT:-} == 1 ]]; then  # extraction only: no fit and no results/<tag> here, the job that fits (the last dataset's, without the switch) finds all four caches
  echo "ARK_EMB_NO_FIT=1: extraction only, no fit for tag $TAG; the job that extracts the last dataset of this tag without ARK_EMB_NO_FIT fits all four"
  exit 0
fi
# 2. the fit: four CPU processes, one thread each (the loop is bound by Python overhead on small tensors), each with its own log; run_linearprobe refuses a (tag, dataset) that has results
mkdir -p "results/$TAG" || exit 1
pids=()
for dataset in "${datasets[@]}"; do
  OMP_NUM_THREADS=1 ./gaudi.sh python -m ark_linearprobe.run_linearprobe --tag "$TAG" --datasets "$dataset" > "results/$TAG/$dataset.log" 2>&1 &
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
