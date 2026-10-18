#!/usr/bin/env bash
# Standalone phase script "sweep_run": ONE arm of the Step VIII fine-tuning sweep (DINOv3 ViT-B/16 fine-tuned with Ark's cyclic training on the four
# data sets, the row's seed; two Gaudi cards, or 4 / 8 through run_d1_8sw4_gaudi.sh / run_d1_8sw8_gaudi.sh). usage: chain/sweep_run.sh <sweep id>
# NOT a chain phase: it is deliberately absent from chain/phases.txt, so the chain driver never runs it. It lives in chain/ only so that
# base/test_chain_scripts.py can test it like the chain phase scripts.
# The arm is the row <sweep id> of sweep/wave1.tsv (tab-separated, columns: id opt lr wd layer_decay patch_embed_mult warmup cycles model init weights
# normalization seed eval_every eval_cycle0 input_size; weights is the in-container path, /workspace = this repo root, or STAGEA_FINAL = the Stage A main run's final
# export, resolved below). The first twelve columns are read by position (the header has to start with exactly them); seed, eval_every, eval_cycle0 and
# input_size are read by their header names, anywhere after those. seed is the trainer's --seed and names the run dir (all4_cyclic_dinov3_sweep_<id>_seed_<seed>), eval_every
# is --eval_every (every Nth cycle plus the last), eval_cycle0 (true | false) is --eval_cycle0 (save and evaluate the initial weights as cycle 0) and
# input_size is --input_size (the crop, in pixels) with --img_resize = round(input_size * 8 / 7), the resize before the crop, which keeps the 256 / 224 ratio of
# the arms before the column existed (224 -> 256, 448 -> 512, 512 -> 585).
# This script owns the rest of the ARK_* recipe, as armD_long.sh does.
# run_d1_8sw_gaudi.sh runs it inside an allocation that has already done cd /scratch/$USER/2026Fall-CSE507-Ark-Container, gaudi_check_cards.sh and
# gaudi_verify_busy.sh and exported PT_HPU_LAZY_MODE=1, so none of that is repeated here.
# Starts the run, or resumes it from its latest checkpoint when an interrupted attempt left its run dir behind (the engine refuses to
# start into an existing run dir); when no cycle had finished yet, from the initial weights an eval_cycle0 run saves as models/weights/epoch_0000.
# Never deletes anything.
# Exit 0: finished now, or already done. 1: the arm's weights file is missing (for STAGEA_FINAL also: the main phase's chain/done/main marker or
# chain/main_epochs.txt is missing), or the run dir holds neither a resumable checkpoint nor complete saved initial weights.
# 2: usage error, unknown sweep id, or a malformed table / row, schedule included (nothing is launched). Otherwise the start / resume script's exit status.
set -uo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <sweep id, a row of sweep/wave1.tsv>" >&2
  exit 2
fi
ID="$1"
TSV=sweep/wave1.tsv  # relative to the repo root (the cwd), like the weights and the run dir below
COLUMNS_LINE="id opt lr wd layer_decay patch_embed_mult warmup cycles model init weights normalization"  # the header starts with these, read by position
ID_RE='^[A-Za-z0-9][A-Za-z0-9_-]*$'  # becomes part of the experiment name and the run dir
NUM_RE='^([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$'  # a non-negative number such as 2e-3 or 0.0003 (chain/main.sh's pattern)
INT_RE='^[0-9]{1,4}$'  # at most 4 digits: the done marker below is epoch_<4 digits>
SEED_RE='^[0-9]{1,9}$'  # the run dir is ..._seed_<seed>; read as decimal below, like the cycle counts
LR_FLOOR=1e-6  # the start script's --warmup-lr and --min-lr

bad() { echo "ERROR: $*" >&2; exit 2; }
refuse() { echo "ERROR: $*" >&2; exit 1; }
# need <column> <value> <extended regex> <what it must be>
need() { [[ $2 =~ $3 ]] || bad "row '$ID' of $TSV: $1 '$2' is not $4"; }
# by_name <column>: sets COLUMN to the row's value under the header column of that name (the header has to have exactly one)
by_name() {
  local k n=0
  for k in "${!HEADER[@]}"; do
    if [[ ${HEADER[k]} == "$1" ]]; then COLUMN=${FIELDS[k]}; n=$((n + 1)); fi
  done
  (( n == 1 )) || bad "row '$ID' of $TSV: the header has $n columns named '$1' (exactly one is needed; the header is: ${FIRST//$'\t'/ })"
}

[[ $ID =~ $ID_RE ]] || bad "'$ID' is not a valid sweep id (letters, digits, _ and -)"
[[ -r $TSV ]] || bad "$TSV is missing or unreadable (run from the repo root)"
IFS= read -r FIRST < "$TSV"
HEAD12="${COLUMNS_LINE// /$'\t'}"
[[ ($FIRST == "$HEAD12" || $FIRST == "$HEAD12"$'\t'*) && $FIRST != *$'\r'* ]] || bad "$TSV does not start with the tab-separated header '$COLUMNS_LINE' (its first 12 columns are read by position; CR line ends are not accepted)"
[[ $FIRST != *$'\t\t'* && $FIRST != *$'\t' ]] || bad "$TSV: the header row has an empty column name"
IFS=$'\t' read -r -a HEADER <<< "$FIRST"
ROW=$(awk -F'\t' -v id="$ID" 'NR > 1 && ($1 "") == (id "")' "$TSV")
[[ -n $ROW ]] || bad "sweep id '$ID' is not a row of $TSV (rows: $(awk -F'\t' 'NR > 1 { printf "%s%s", sep, $1; sep = " " }' "$TSV"))"
[[ $ROW != *$'\n'* ]] || bad "sweep id '$ID' is listed more than once in $TSV"
TABS=${ROW//[^$'\t']/}
if (( ${#TABS} + 1 != ${#HEADER[@]} )) || [[ $ROW == *$'\t\t'* || $ROW == *$'\t' ]]; then
  bad "row '$ID' of $TSV needs exactly ${#HEADER[@]} non-empty tab-separated fields (${FIRST//$'\t'/ })"
fi
IFS=$'\t' read -r -a FIELDS <<< "$ROW"
IFS=$'\t' read -r _ OPT LR WD LD PE WARMUP CYCLES MODEL INIT WEIGHTS NORM _ <<< "$ROW"  # the trailing _ takes the named columns, so NORM is the 12th field alone
by_name seed; SEED=$COLUMN
by_name eval_every; EVAL_EVERY=$COLUMN
by_name eval_cycle0; EVAL_CYCLE0=$COLUMN
by_name input_size; INPUT_SIZE=$COLUMN
need opt "$OPT" '^(adamw|momentum)$' "adamw or momentum"
need lr "$LR" "$NUM_RE" "a number"
awk -v v="$LR" 'BEGIN { exit !(v + 0 > 0) }' || bad "row '$ID' of $TSV: lr '$LR' is not positive"
need wd "$WD" "$NUM_RE" "a number"
need layer_decay "$LD" "$NUM_RE" "a number"
need patch_embed_mult "$PE" "$NUM_RE" "a number"
need warmup "$WARMUP" "$INT_RE" "a whole number of at most 4 digits"
need cycles "$CYCLES" "$INT_RE" "a whole number of at most 4 digits"
need model "$MODEL" '^[A-Za-z0-9_]+$' "a model name"
need init "$INIT" '^[A-Za-z0-9_]+$' "an init name"
need weights "$WEIGHTS" '^(STAGEA_FINAL|/workspace/[^[:space:]]+)$' "STAGEA_FINAL or an in-container path under /workspace/"
# The column stays so later rows can be revisited, but only imagenet is accepted: with --train_augment true (the default) the dataset loaders in
# dataloader.py hard-code the ImageNet mean/std for training, while --normalization only reaches the val/test transforms
need normalization "$NORM" '^imagenet$' "supported: only imagenet is accepted, because with --train_augment true the dataset loaders hard-code ImageNet mean/std for training, so any other preset would normalize train and val/test differently"
need seed "$SEED" "$SEED_RE" "a whole number of at most 9 digits"
need eval_every "$EVAL_EVERY" "$INT_RE" "a whole number of at most 4 digits"
need eval_cycle0 "$EVAL_CYCLE0" '^(true|false)$' "true or false"
need input_size "$INPUT_SIZE" '^[0-9]{3,4}$' "a whole number of 3 or 4 digits (pixels)"
# main_ark.py's optparse reads a leading 0 as octal ("030" is 24 cycles, "08" is an error), and bash does too in $(( )): force decimal once, here
# (the seed too: "007" is seed 7 to the trainer, whose run dir is ..._seed_7)
WARMUP=$((10#$WARMUP))
CYCLES=$((10#$CYCLES))
SEED=$((10#$SEED))
EVAL_EVERY=$((10#$EVAL_EVERY))
INPUT_SIZE=$((10#$INPUT_SIZE))
(( EVAL_EVERY >= 1 )) || bad "row '$ID' of $TSV: eval_every $EVAL_EVERY is below 1 (the trainer evaluates every Nth cycle plus the final one, so N is at least 1)"
# The conditions of engine.py's WarmupCosineScheduler, which only refuses a bad schedule after the container and the data sets are up: at least 3 cycles,
# a warmup of at least 2 cycles and fewer than the cycles, and a peak LR that is not below the fixed --warmup-lr / --min-lr of the start script
(( CYCLES >= 3 )) || bad "row '$ID' of $TSV: cycles $CYCLES is below 3 (WarmupCosineScheduler needs at least 3 cycles)"
(( WARMUP >= 2 )) || bad "row '$ID' of $TSV: warmup $WARMUP is below 2 (WarmupCosineScheduler needs a warmup of at least 2 cycles)"
(( WARMUP < CYCLES )) || bad "row '$ID' of $TSV: warmup $WARMUP is not below cycles $CYCLES (WarmupCosineScheduler needs a warmup shorter than the run)"
awk -v v="$LR" -v lo="$LR_FLOOR" 'BEGIN { exit !(v + 0 >= lo + 0) }' || bad "row '$ID' of $TSV: lr '$LR' is below $LR_FLOOR, the start script's fixed --warmup-lr and --min-lr (WarmupCosineScheduler needs the peak LR not below them)"
# the ViT-B/16 cuts the crop into 16-pixel patches, so the crop is a whole number of patches; 112 (7 x 7 patches) is the floor
(( INPUT_SIZE % 16 == 0 )) || bad "row '$ID' of $TSV: input_size $INPUT_SIZE is not a multiple of 16 (the ViT-B/16 patch size)"
(( INPUT_SIZE >= 112 )) || bad "row '$ID' of $TSV: input_size $INPUT_SIZE is below 112"
# the resize before the crop: input_size * 8 / 7 rounded to the nearest integer ((8x + 3) / 7 in integer arithmetic: 8x / 7 never ends in exactly .5)
IMG_RESIZE=$(( (INPUT_SIZE * 8 + 3) / 7 ))

export ARK_BATCH_SIZE=200
export ARK_LR="$LR"
export ARK_TEST_AUGMENT=false
# evaluate every eval_every-th cycle plus the final one (engine.py always evaluates the last cycle); eval_cycle0 true also saves the initial weights as
# epoch_0000 and evaluates them as cycle 0 (common.sh passes it as --eval_cycle0; the rows that ran before the column existed say false)
export ARK_EVAL_EVERY="$EVAL_EVERY"
export ARK_EVAL_CYCLE0="$EVAL_CYCLE0"
# common.sh passes them as --input_size / --img_resize (the train, validation and test transforms); the engine records both in every manifest and checkpoint,
# refuses a resume at another size, and offline test mode (eval_d1_6.sh) reads them back from the saved cycle's manifest
export ARK_INPUT_SIZE="$INPUT_SIZE"
export ARK_IMG_RESIZE="$IMG_RESIZE"
export ARK_WARMUP_EPOCHS="$WARMUP"
# 16 data workers per rank, as in D-long and run_d1_8c_gaudi.sh (8 left arm B data-bound on 4 cards). The Gaudi wrappers request 18 cores per card, so the 2 ranks
# of run_d1_8sw_gaudi.sh (-c 36) run 2 x 16 = 32 workers on 36 cores (4 x 16 = 64 of 72 for run_d1_8sw4_gaudi.sh, 8 x 16 = 128 of 144 for run_d1_8sw8_gaudi.sh);
# 32 workers per rank would be 64 on 36 cores and oversubscribe them
export ARK_WORKERS=16
export ARK_MODEL="$MODEL"
export ARK_INIT="$INIT"
# main_ark.py names the run dir <exp_name>_seed_<seed> in the repo root (the seed is the row's). engine.py ends every cycle by writing manifest.json, student_*.pth, teacher_*.pth and,
# last of all, the cycle checkpoint models/checkpoints/cycle_<n>_*.pth.tar (the evaluations of a cycle, the final one included, come before the weights), so
# the arm is done when the final cycle's student weights AND its checkpoint exist. Residual window: torch.save is not atomic, so a kill during that very last
# write leaves a truncated checkpoint that this still counts as done (student, teacher and evaluations are complete by then); noticing that is manual, and so is the resubmit.
RUN="all4_cyclic_dinov3_sweep_${ID}_seed_${SEED}"
LAST=$(printf '%04d' "$CYCLES")
if compgen -G "$RUN/models/weights/epoch_$LAST/student_cycle_${LAST}_*.pth" > /dev/null && compgen -G "$RUN/models/checkpoints/cycle_${LAST}_*.pth.tar" > /dev/null; then
  echo "sweep_run $ID already done"
  exit 0
fi
# STAGEA_FINAL: the Stage A main run's last export (chain/main.sh: ssl_dinov3_cxr_main_seed_100). The trainer counts epochs from 1, so the last export is
# teacher_backbone_epoch_<chain/main_epochs.txt, what main.sh passed as --epochs>.pth. It exists only once the chain driver has written the main phase's
# done marker (on exit 0 of chain/main.sh), so both that marker and the epochs file have to be there.
if [[ $WEIGHTS == STAGEA_FINAL ]]; then
  [[ -e chain/done/main ]] || refuse "sweep $ID: its weights are STAGEA_FINAL (the Stage A main run's final export), but the main phase is not done: chain/done/main is missing (the chain driver writes it when chain/main.sh exits 0); requeue after it exists"
  [[ -s chain/main_epochs.txt ]] || refuse "sweep $ID: its weights are STAGEA_FINAL, but chain/main_epochs.txt (the main run's epoch count, written by chain/gate.py) is missing or empty"
  MAIN_EPOCHS=$(tr -d '[:space:]' < chain/main_epochs.txt)
  [[ $MAIN_EPOCHS =~ $INT_RE ]] || refuse "sweep $ID: chain/main_epochs.txt holds '$MAIN_EPOCHS', which is not a whole number of at most 4 digits"
  MAIN_EPOCHS=$((10#$MAIN_EPOCHS))
  (( MAIN_EPOCHS >= 1 )) || refuse "sweep $ID: chain/main_epochs.txt holds $MAIN_EPOCHS, which is not a positive epoch count"
  WEIGHTS=$(printf '/workspace/ssl_dinov3_cxr_main_seed_100/export/teacher_backbone_epoch_%04d.pth' "$MAIN_EPOCHS")
fi
export ARK_PRETRAINED_WEIGHTS="$WEIGHTS"
# fail fast (seconds, not a whole slot) if the arm's weights are not in place yet (a resume builds the model from them too).
# ARK_PRETRAINED_WEIGHTS is the in-container path (/workspace = this repo root)
HOST_WEIGHTS="${ARK_PRETRAINED_WEIGHTS/#\/workspace\//./}"
if [[ ! -s "$HOST_WEIGHTS" ]]; then
  echo "ERROR: sweep $ID: weights file $HOST_WEIGHTS is missing or empty (host path for $ARK_PRETRAINED_WEIGHTS); it has to land first (the DINOv3 download, or the Stage A main run's final export), then requeue" >&2
  exit 1
fi
ARGS=("$ID" "$SEED" gaudi "$OPT" "$WD" "$LD" "$PE" "$WARMUP" "$CYCLES" "$NORM")
if [[ -d "$RUN" ]]; then
  # an interrupted attempt: continue it, from its latest cycle checkpoint or, when it stopped before the first one, from the initial weights the engine saved as
  # models/weights/epoch_0000 (eval_cycle0 true; the engine writes the manifest, then the student, then the teacher, once cycle 0 is evaluated, and resumes at
  # cycle 0 from them without evaluating again). A dir with neither can be neither resumed nor started over; leave it for a human.
  INITIAL="$RUN/models/weights/epoch_0000"
  if compgen -G "$RUN/models/checkpoints/*.pth.tar" > /dev/null; then
    echo "sweep_run $ID: resuming ./$RUN from its latest checkpoint"
  elif [[ -s "$INITIAL/manifest.json" ]] && compgen -G "$INITIAL/student_cycle_0000_*.pth" > /dev/null && compgen -G "$INITIAL/teacher_cycle_0000_*.pth" > /dev/null; then
    echo "sweep_run $ID: resuming ./$RUN from its saved initial weights ($INITIAL; no cycle has finished yet)"
  else
    echo "ERROR: run dir ./$RUN exists but holds no resumable checkpoint ($RUN/models/checkpoints/*.pth.tar) and no complete saved initial weights ($INITIAL with manifest.json, student_cycle_0000_*.pth and teacher_cycle_0000_*.pth); neither resuming nor starting over it, nothing deleted: inspect or remove it, then requeue" >&2
    exit 1
  fi
  ./experiment_scripts/all4_cyclic_dinov3_sweep/resume_all4_cyclic_dinov3_sweep.sh "${ARGS[@]}"
else
  ./experiment_scripts/all4_cyclic_dinov3_sweep/start_all4_cyclic_dinov3_sweep.sh "${ARGS[@]}"
fi
