#!/usr/bin/env bash
# Constant-LR run (Meta's DINOv3 recipe): continued SSL on the chest X-rays whose LR stays flat after the 4-epoch warmup, with weight decay and
# teacher momentum constant (the trainer always has them so) and the Gram teacher refreshed every 10,000 steps at most three times (steps 10k /
# 20k / 30k, about epochs 3.75 / 7.5 / 11.25 on 4 cards), then fixed: DINOv3 Appendix C ("every 10k steps for a maximum of three updates") and
# the released config's gram.max_updates: 3, the same as the main run. This run is meant to be extended for weeks, indefinitely and without a
# code change, and --gram_refresh_steps is locked on --resume, so the list is passed explicitly rather than left to the trainer's default.
# Peak LR = chain/main_lr.txt (absolute, as in main.sh). --min_lr = --lr makes the trainer's warmup + cosine schedule end where it peaks, i.e.
# warmup + constant; --no_lr_scaling keeps both absolute.
# --epochs 1000 (EPOCHS below) is a ceiling, not a plan: the trainer locks --epochs, --lr, --min_lr and --gram_refresh_steps on --resume but not
# --stop_after_epochs, so the schedule is sized once, beyond what the Sol access allows (it ends Dec 4: at most ~850 epochs at 1.4 h each),
# and chain/constlr_stop_after.txt (a positive integer, at most the ceiling) is the epoch this invocation stops after.
# Like main_lr.txt, constlr_stop_after.txt exists only on the cluster (never in the repo, so an rsync cannot overwrite a raised value): create
# it before the chain reaches this phase (echo 30 > chain/constlr_stop_after.txt). To extend the run: edit that number, rm chain/done/constlr,
# sbatch run_d1_8_chain_gaudi.sh; it resumes from its last epoch checkpoint. Both files are read relative to the repo root (cwd), and a
# missing, empty or invalid one stops the phase before anything is launched; so does a stop epoch above the ceiling, which the trainer would
# only refuse after the allocation had started.
# Env overrides, all unset in the chain's phase (so the phase's run dir, launch line and messages are unchanged): CONSTLR_NOGRAM=1 makes this the no-Gram trunk,
# "DINO-X ConstLR 224, no Gram": the same launch line with --no_gram (no Gram teacher built, loaded or run) in place of --gram_refresh_steps 10000,20000,30000, into its own
# run dir ssl_dinov3_cxr_constlr_nogram_seed_100, with the stop epoch read from chain/constlr_nogram_stop_after.txt instead (the same rules, cluster-only too; chain/constlr_stop_after.txt
# is not read then), and one `run:` line printed before the launch. Any other non-empty value is refused before anything is read. run_d1_8_constlr_nogram_gaudi.sh sets it: its own
# 4-card job outside the chain, never a phase.
# CONSTLR_BRANCH_EPOCH=K, CONSTLR_BRANCH_STOP=STOP, CONSTLR_BRANCH_EVERY=EVERY and CONSTLR_BRANCH_COUNT=COUNT (all four; run_d1_8_branch_gram_gaudi.sh sets them, never the
# chain) make this a Gram-anchored branch of the no-Gram trunk: the trunk's launch line with --no_gram replaced by --branch_from
# /workspace/ssl_dinov3_cxr_constlr_nogram_seed_100/ckpt/epoch_<K %04d>.pth --gram_refresh_steps <list>, into the run dir
# ssl_dinov3_cxr_constlr_nogram_e<K %04d>_gram_every<EVERY>x<COUNT>_seed_100 (the cadence is in the name, so two cadences from the same K never share a dir), stopping
# after epoch STOP (no stop file is read). The list is the branch step + i x EVERY for i = 1..COUNT, the branch step being K x the trunk's real epoch_length, read from
# the trunk's args.json (the trainer writes it there; never assumed), so every resubmit passes the same list, which --resume locks. Refused before anything is launched:
# CONSTLR_NOGRAM set as well (the branch's line replaces --no_gram itself); any of the four empty or invalid once one is set (K and STOP positive integers without a
# leading zero, which printf %04d would read as octal; EVERY at most 9 digits, COUNT at most 4); STOP above the ceiling or not after K; a trunk args.json without a
# positive epoch_length; a last refresh (K x epoch_length + COUNT x EVERY) past the end of the 1000-epoch schedule (1000 x epoch_length), which could never run even with the
# branch extended to the ceiling (a last refresh exactly on that step is accepted); and a missing source checkpoint while the branch dir has no epoch checkpoint of its own
# yet (a regular file named epoch_<digits>.pth, the names the trainer's epoch_checkpoints() takes; once it has one, the trainer resumes from it and never reads the source again, so the
# trunk may have pruned it). `run:` and `gram refresh steps:` lines are printed before the launch.
set -euo pipefail
NOGRAM=${CONSTLR_NOGRAM:-}  # first, before anything is read or launched
[[ -z $NOGRAM || $NOGRAM == 1 ]] || { echo "ERROR: CONSTLR_NOGRAM holds '$NOGRAM', which is not empty or 1" >&2; exit 1; }
BRANCH=${CONSTLR_BRANCH_EPOCH:-}${CONSTLR_BRANCH_STOP:-}${CONSTLR_BRANCH_EVERY:-}${CONSTLR_BRANCH_COUNT:-}  # non-empty: a branch (all four are then checked below)
[[ -z $BRANCH || -z $NOGRAM ]] || { echo "ERROR: CONSTLR_NOGRAM and CONSTLR_BRANCH_* are both set: a branch launches the no-Gram trunk's line with --branch_from in place of --no_gram, so set only the CONSTLR_BRANCH_* variables" >&2; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/ssl_common.sh"

EPOCHS=1000  # the --epochs ceiling (never reached): the stop epoch's upper limit and the launch line both use it, so they cannot drift
TRUNK=ssl_dinov3_cxr_constlr_nogram_seed_100  # the no-Gram trunk's run dir (CONSTLR_NOGRAM=1), which every branch starts from
RUN=ssl_dinov3_cxr_constlr_seed_100
STOP_FILE=chain/constlr_stop_after.txt
GRAM_ARGS=(--gram_refresh_steps 10000,20000,30000)
if [[ -n $NOGRAM ]]; then
  RUN=$TRUNK
  STOP_FILE=chain/constlr_nogram_stop_after.txt
  GRAM_ARGS=(--no_gram)
fi
LR=$(read_setting chain/main_lr.txt '^([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$' 'a positive number such as 2e-3 or 0.0003')
awk -v v="$LR" 'BEGIN { exit !(v + 0 > 0) }' || { echo "ERROR: chain/main_lr.txt holds '$LR', which is not positive" >&2; exit 1; }
if [[ -z $BRANCH ]]; then
  STOP_AFTER=$(read_setting "$STOP_FILE" '^[1-9][0-9]*$' 'a positive integer')
else
  # branch_value <name> <variable> <regex> <what>: prints the variable's value, or exits 1 with an ERROR naming both
  branch_value() { [[ ${!2:-} =~ $3 ]] || { echo "ERROR: $1 ($2) holds '${!2:-}', which is not $4" >&2; exit 1; }; printf '%s' "${!2}"; }
  K=$(branch_value K CONSTLR_BRANCH_EPOCH '^[1-9][0-9]*$' 'a positive integer without a leading zero (the trunk epoch to branch from)')
  STOP_AFTER=$(branch_value STOP CONSTLR_BRANCH_STOP '^[1-9][0-9]*$' 'a positive integer without a leading zero (the epoch the branch stops after)')
  EVERY=$(branch_value EVERY CONSTLR_BRANCH_EVERY '^[1-9][0-9]{0,8}$' 'a positive integer of at most 9 digits (the steps between Gram refreshes)')
  COUNT=$(branch_value COUNT CONSTLR_BRANCH_COUNT '^[1-9][0-9]{0,3}$' 'a positive integer of at most 4 digits (the number of Gram refreshes)')
  STOP_FILE="STOP (CONSTLR_BRANCH_STOP)"  # what the ceiling check below names
fi
# awk, not (( )): bash arithmetic wraps silently past 2^63 (18446744073709551617 would read as 1)
awk -v v="$STOP_AFTER" -v cap="$EPOCHS" 'BEGIN { exit !(v + 0 <= cap + 0) }' || { echo "ERROR: $STOP_FILE holds '$STOP_AFTER', which is above the --epochs ceiling of $EPOCHS" >&2; exit 1; }
if [[ -n $BRANCH ]]; then
  awk -v k="$K" -v s="$STOP_AFTER" 'BEGIN { exit !(k + 0 < s + 0) }' || { echo "ERROR: STOP (CONSTLR_BRANCH_STOP) holds '$STOP_AFTER', which is not after K (CONSTLR_BRANCH_EPOCH) '$K': the branch would train nothing" >&2; exit 1; }
  # from here K < STOP <= 1000, so (( )) and printf %04d are safe. The trainer writes args.json with json.dump(indent=2): one "epoch_length": N line
  TRUNK_ARGS=$TRUNK/args.json
  EPOCH_LENGTH=$(sed -n 's/^ *"epoch_length": *\([0-9][0-9]*\),\{0,1\} *$/\1/p' "$TRUNK_ARGS" 2>/dev/null || true)
  [[ $EPOCH_LENGTH =~ ^[1-9][0-9]{0,8}$ ]] || { echo "ERROR: $TRUNK_ARGS is missing or holds no positive epoch_length (the trainer writes it when the trunk starts); refusing to guess the branch step" >&2; exit 1; }
  EK=$(printf %04d "$K")
  SOURCE=$TRUNK/ckpt/epoch_$EK.pth
  RUN=${TRUNK%_seed_100}_e${EK}_gram_every${EVERY}x${COUNT}_seed_100
  # the branch's own epoch checkpoints: once there is one, the trainer resumes from it and never reads the source again. The name test is the trainer's epoch_checkpoints():
  # epoch_<digits>.pth (epoch_backup.pth, epoch_0011.pth.tmp and maxsteps_*.pth do not count), and it must be a regular file, not a directory of that name
  OWN=
  for f in "$RUN"/ckpt/epoch_*.pth; do if [[ -f $f && ${f##*/} =~ ^epoch_[0-9]+\.pth$ ]]; then OWN=$f; fi; done  # if, not &&: a false test leaves status 0 under set -e
  [[ -n $OWN || -f $SOURCE ]] || { echo "ERROR: $SOURCE not found and $RUN has no epoch checkpoint of its own yet: nothing to branch from (the trunk keeps every 10th epoch and its last 3)" >&2; exit 1; }
  BRANCH_STEP=$(( K * EPOCH_LENGTH ))
  STEPS=()
  for (( i = 1; i <= COUNT; i++ )); do STEPS+=("$(( BRANCH_STEP + i * EVERY ))"); done
  REFRESH=$(IFS=,; echo "${STEPS[*]}")
  # a refresh after the last step of the --epochs schedule (EPOCHS x EPOCH_LENGTH) could never run, even with the branch extended to the ceiling; equal to it is fine. The digit caps keep every product below 2^63
  LAST=${STEPS[COUNT - 1]}
  END=$(( EPOCHS * EPOCH_LENGTH ))
  (( LAST <= END )) || { echo "ERROR: the last Gram refresh, after step $LAST, is past the end of the $EPOCHS-epoch schedule (step $END): it could never run; lower EVERY (CONSTLR_BRANCH_EVERY) or COUNT (CONSTLR_BRANCH_COUNT)" >&2; exit 1; }
  GRAM_ARGS=(--branch_from "/workspace/$SOURCE" --gram_refresh_steps "$REFRESH")
  echo "run: $RUN (Gram-anchored branch of $TRUNK from its epoch $K: --branch_from /workspace/$SOURCE --gram_refresh_steps in place of --no_gram; stop after epoch $STOP_AFTER)"
  echo "gram refresh steps: $REFRESH (branch step $BRANCH_STEP = epoch $K x epoch_length $EPOCH_LENGTH from $TRUNK_ARGS, then every $EVERY steps, $COUNT times)"
fi
[[ -z $NOGRAM ]] || echo "run: $RUN (no Gram teacher: --no_gram replaces --gram_refresh_steps 10000,20000,30000; stop after epoch $STOP_AFTER from $STOP_FILE)"

ssl_launch "$RUN" --lr "$LR" --min_lr "$LR" --no_lr_scaling --epochs "$EPOCHS" --stop_after_epochs "$STOP_AFTER" "${GRAM_ARGS[@]}"
