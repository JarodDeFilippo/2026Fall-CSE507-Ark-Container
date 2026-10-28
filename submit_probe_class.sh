#!/usr/bin/env bash
# The frozen linear probe on the class account as TWO Slurm jobs, so the A100 is held for the extraction only: the extraction (run_emb_class.sh with ARK_EMB_NO_FIT=1, the one part that needs the GPU), then, behind it with
# --dependency=afterok, the fit (run_emb_fit_class.sh: the four CPU fits, no GPU). The one-job form (sbatch run_emb_class.sh ...) holds the card through the fits too: about 1 h of extraction, then about 1.5 h of CPU fits with
# the A100 idle, and the class QOS counts the requested -t of running GPU jobs against its GPU-minute cap (run_emb_class.sh). The lab's own probe is fit-only by construction, so this split is the native shape.
# NOT an sbatch script: run it on the login node (it refuses inside a Slurm job). It submits, prints one line and exits; nothing is left running here.
# usage: ./submit_probe_class.sh <tag> <model> <in-container weights path under /workspace> <imagenet|chestx-ray|none|raddino> [--datasets NAME...]
#   the first four are run_emb_gaudi.sh's <tag> <model> <weights> <normalization> (see there for the contract); --datasets is the only option and goes to both jobs (NAME: ChestXray14 CheXpert MIMIC VinDrCXR). The smoke form (--max_images)
#   and the per-dataset split of a long Swin-L extraction are not here: run run_emb_class.sh directly (run_emb_gaudi.sh's header describes them).
#   e.g. ./submit_probe_class.sh stock_dinov3 vit_base_dinov3 /workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth imagenet
#        ./submit_probe_class.sh nogram_e0060 vit_base_dinov3 /workspace/ssl_dinov3_cxr_constlr_nogram_seed_100/export/teacher_backbone_epoch_0060.pth imagenet
#   the second runs these two commands and prints the last line:
#        ARK_EMB_NO_FIT=1 sbatch --parsable -J probe-nogram_e0060-x -p public,htc -q class --constraint=a100 -t 01:15:00 run_emb_class.sh nogram_e0060 vit_base_dinov3 <that weights path> imagenet
#        sbatch --parsable -J probe-nogram_e0060-fit -t 03:00:00 --dependency=afterok:<the extraction's id> run_emb_fit_class.sh nogram_e0060
#        probe nogram_e0060: extraction <id1>, fit <id2> (afterok)
# Follow the two ids with squeue -u $USER (or sacct -j <id1>,<id2>). The fit starts only if the extraction ends with exit 0; if it fails, the fit stays pending on a dependency that can never be met (DependencyNeverSatisfied) until Slurm
# cancels it or you scancel it. The fit job's .out (slurm-<id2>.out) ends with one PROBE line per dataset; the same lines are in results/<tag>/<dataset>.log and the full numbers in results/<tag>/<dataset>.json.
# Nothing is submitted unless the launchers' own checks pass; they are run here, not repeated: run_emb_class.sh --check (usage, extras, ARK_EMB_*, the weights file, results that already exist; with ARK_EMB_NO_FIT=1 as the extraction will have it) and
# run_emb_fit_class.sh --check (tag, --datasets, results). The extraction's cache is not checked: the extractor refuses an existing one, in seconds, and the fit then never starts. If the fit cannot be submitted after the extraction was, the extraction is
# cancelled (scancel), so no extraction is left without its fit. The wrapper never submits the fit without the dependency. It does not look for a pair already queued for the tag: run twice before the first extraction has written anything, it
# submits twice (the second extraction is then refused by the extractor, which leaves the first pair's cache alone, and the second fit stays pending on a dependency that is never met).
# environment: ARK_EMB_INPUT_SIZE (as run_emb_gaudi.sh; reaches the checks and the extraction; a larger crop is slower, so raise PROBE_X_TIME with it); PROBE_X_TIME (the extraction's -t, default 01:15:00: the one ViT-B extraction at 224 px timed on an A100
#   on Sol took 56 min (2026-10-26), plus a third; the class QOS counts the requested time of GPU jobs); PROBE_X_CONSTRAINT (the extraction's --constraint, default a100: any A100, see run_emb_class.sh for why not MIG slices or H100s);
#   PROBE_FIT_TIME (the fit's -t, default 03:00:00, see run_emb_fit_class.sh); PROBE_ACCOUNT (the Slurm account of both jobs, -A right after each -J; unset or empty = no -A, so the launchers' own #SBATCH -A applies, class_cse49478170fall2026; it must carry the class QOS,
#   so the lab account grp_jliang12 is not a substitute; an account sbatch refuses fails the first submission and nothing is queued): a group member in another class account runs
#   PROBE_ACCOUNT=class_bme50769722fall2026 ./submit_probe_class.sh <tag> <model> <weights> <normalization>
#   ARK_EMB_FIT_RESUME is unset here: a pair is always a fresh fit (to finish a fit that stopped half way, submit run_emb_fit_class.sh alone with it set).
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1   # the repo root: the launchers sit here and sbatch is given their names
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  echo "ERROR: submit_probe_class.sh submits the jobs itself: run it on the login node, not inside Slurm job $SLURM_JOB_ID" >&2
  exit 1
fi
if [[ $# -lt 4 || ( $# -gt 4 && $5 != --datasets ) ]]; then   # the shape only; what each argument may be is the launchers' check below
  echo "Usage: ./submit_probe_class.sh <tag> <model> <in-container weights path under /workspace> <imagenet|chestx-ray|none|raddino> [--datasets NAME...]" >&2
  exit 2
fi
unset ARK_EMB_FIT_RESUME
tag=$1
ARK_EMB_NO_FIT=1 ./run_emb_class.sh --check "$@"
status=$?
[[ $status -eq 0 ]] && { ./run_emb_fit_class.sh --check "$tag" "${@:5}"; status=$?; }
if [[ $status -ne 0 ]]; then
  echo "submit_probe_class.sh: a launcher's check refused the arguments (exit $status); nothing was submitted" >&2
  exit $status
fi
account=(); [[ -n ${PROBE_ACCOUNT:-} ]] && account=(-A "$PROBE_ACCOUNT")   # no -A when unset or empty: the launchers' own #SBATCH -A applies
# --parsable prints "<id>" or "<id>;<cluster>"
x_id=$(ARK_EMB_NO_FIT=1 sbatch --parsable -J "probe-$tag-x" "${account[@]}" -p public,htc -q class --constraint="${PROBE_X_CONSTRAINT:-a100}" -t "${PROBE_X_TIME:-01:15:00}" run_emb_class.sh "$@") || { echo "ERROR: sbatch refused the extraction job; nothing was submitted" >&2; exit 1; }
x_id=${x_id%%;*}
if [[ ! $x_id =~ ^[0-9]+$ ]]; then
  echo "ERROR: sbatch printed '$x_id' for the extraction, not a job id; the fit was not submitted" >&2
  exit 1
fi
fit_id=$(sbatch --parsable -J "probe-$tag-fit" "${account[@]}" -t "${PROBE_FIT_TIME:-03:00:00}" --dependency="afterok:$x_id" run_emb_fit_class.sh "$tag" "${@:5}")
fit_id=${fit_id%%;*}
if [[ ! $fit_id =~ ^[0-9]+$ ]]; then
  scancel "$x_id" || echo "WARNING: scancel $x_id failed: cancel the extraction by hand" >&2   # an extraction with no fit behind it would only spend the card
  echo "ERROR: the fit job was not submitted (sbatch printed '$fit_id'); extraction job $x_id cancelled. Fix the cause and run this again" >&2
  exit 1
fi
echo "probe $tag: extraction $x_id, fit $fit_id (afterok)"
