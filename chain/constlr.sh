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
# "DINO-X ConstLR 224, no Gram": the same launch line with --no_gram (no Gram teacher built, loaded or run) in place of --gram_refresh_steps 10000,20000,30000 and, ahead of it,
# three settings of DINOv3's training recipe the phase does not pass (RECIPE below: --drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm; Jarod, 2026-10-26, so
# the trunk is no longer an exact ConstLR-224 control: the Gram teacher is not the only difference), into its own
# run dir ssl_dinov3_cxr_constlr_nogram_seed_100, with the stop epoch read from chain/constlr_nogram_stop_after.txt instead (the same rules, cluster-only too; chain/constlr_stop_after.txt
# is not read then), and one `run:` line printed before the launch. Any other non-empty value is refused before anything is read. run_d1_8_constlr_nogram_gaudi.sh sets it: its own
# 4-card job outside the chain, never a phase.
# CONSTLR_PLAIN=1, only together with CONSTLR_NOGRAM=1, makes the no-Gram trunk the plain one, "DINO-X ConstLR 224, no Gram, plain": the same launch line WITHOUT RECIPE, so --no_gram in place of
# --gram_refresh_steps 10000,20000,30000 is the only difference from the phase, the one-variable Gram-off control of ssl_dinov3_cxr_constlr_seed_100 (Jarod, 2026-10-26; the hypothesis: Gram anchoring
# from step 0 capped the probe), into its own run dir ssl_dinov3_cxr_constlr_nogram_plain_seed_100, with the stop epoch read from chain/constlr_nogram_plain_stop_after.txt (the same rules, cluster-only
# too; neither of the other two stop files is read then) and its own `run:` line. TRUNK below stays the no-Gram trunk's run dir (the one with RECIPE), which every branch starts from. Refused before anything is read: any other
# non-empty value, CONSTLR_NOGRAM unset or empty, and any CONSTLR_BRANCH_* set (branches of the plain trunk are a later feature). run_d1_8_constlr_nogram_plain_gaudi.sh sets both CONSTLR_NOGRAM and
# CONSTLR_PLAIN: its own 4-card job outside the chain, never a phase.
# CONSTLR_BRANCH_EPOCH=K, CONSTLR_BRANCH_STOP=STOP, CONSTLR_BRANCH_EVERY=EVERY and CONSTLR_BRANCH_COUNT=COUNT (all four; run_d1_8_branch_gram_gaudi.sh sets them, never the
# chain) make this a Gram-anchored branch of the no-Gram trunk: the trunk's launch line (RECIPE included, which --branch_from requires to match the source) with --no_gram replaced by --branch_from
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
# CONSTLR_ARK=1, only together with CONSTLR_NOGRAM=1, makes the no-Gram trunk the Ark-teacher trunk, "DINO-X ConstLR 224, no Gram, Ark teacher" (Jarod, 2026-10-26): the no-Gram trunk's launch line
# (RECIPE included) with ARK_FLAGS below appended, --cyclic_ark_teacher --heads per_dataset <lock flag> --keep_all_checkpoints (the trainer's cyclic Ark+ / FX teacher: one dataset per batch, the
# datasets in turn largest first, a cycle per epoch, the teacher frozen through every pass and blended 0.8 / 0.2 with the student after each dataset's release pass, on clean views, per-dataset heads,
# every epoch checkpoint kept; the clean views and the 0.8 are the mode's own defaults, so the line does not pass them). CONSTLR_ARK_LOCK (0 or 1; empty or unset = 0) is the lock arm, a per-run
# ablation (Jarod, 2026-10-26): 1 passes --lock_half_pass (Foundation X's lock-release: per dataset per cycle a lock pass over a random half with the backbone frozen (LR 0, no optimizer step), then the full release pass; about
# 3997 steps a cycle at global batch 256), 0 passes --no_lock_half_pass (the release passes alone, 2665 steps a cycle). The arm is in the run dir and the stop file, so the two never share either and a
# resume never flips it: ssl_dinov3_cxr_constlr_arkteacher_seed_100 with chain/constlr_arkteacher_stop_after.txt for 0, ssl_dinov3_cxr_constlr_arkteacher_lock_seed_100 with
# chain/constlr_arkteacher_lock_stop_after.txt for 1 (the same rules as the other stop files, cluster-only too; no other stop file is read then), and its own `run:` line says which. Refused before
# anything is read: any other non-empty CONSTLR_ARK, CONSTLR_NOGRAM unset or empty, CONSTLR_PLAIN set (the Ark-teacher trunk keeps the recipe) and any CONSTLR_BRANCH_* set (its branches are
# CONSTLR_BRANCH_ARK=1); a CONSTLR_ARK_LOCK other than empty, 0 or 1, and one set without CONSTLR_ARK=1 or CONSTLR_BRANCH_ARK=1. run_d1_8_constlr_arkteacher_gaudi.sh sets CONSTLR_NOGRAM and
# CONSTLR_ARK (sbatch --export=ALL,CONSTLR_ARK_LOCK=1 for the lock arm): its own 4-card job outside the chain, never a phase.
# CONSTLR_BRANCH_ARK=1, only together with the four CONSTLR_BRANCH_* (sbatch --export=ALL,CONSTLR_BRANCH_ARK=1[,CONSTLR_ARK_LOCK=1] run_d1_8_branch_gram_gaudi.sh <K> <STOP> [EVERY] [COUNT]), makes
# the branch one of an Ark-teacher trunk instead: TRUNK is the CONSTLR_ARK_LOCK arm's run dir (its args.json, its ckpt/epoch_<K %04d>.pth; the branch's run dir ${TRUNK%_seed_100}_e<K %04d>_gram_every
# <EVERY>x<COUNT>_seed_100 carries its _lock token), whose args.json must record that lock_half_pass (the branch inherits it: refused if it is missing or the other one), and the line is that trunk's
# with --no_gram replaced by the branch pair, ARK_FLAGS kept after it, so a branch launches its trunk's line plus --branch_from as before (--branch_from requires every mode flag to match the source).
# Its epoch_length is the cycle's, read from that args.json (2665, or 3997 with the lock, at global batch 256), so `1 40 <epoch_length> 30` refreshes the Gram teacher at the next 30 cycle
# boundaries. Refused before anything is read: any other non-empty value, and CONSTLR_BRANCH_ARK=1 without the CONSTLR_BRANCH_* variables.
# CONSTLR_ABLATE (drop_path, rope or local_norm; empty or unset = off), only together with CONSTLR_NOGRAM=1, makes the no-Gram trunk a one-flag-off recipe ablation, "DINO-X ConstLR 224, no Gram, one recipe flag off" (Jarod,
# 2026-10-26; for group members, one arm each): the no-Gram trunk's launch line (RECIPE and --no_gram) with exactly one RECIPE flag and its value removed, drop_path --drop_path_rate 0.3, rope --rope_rescale_coords 2, local_norm
# --untie_local_cls_norm, the other two kept in RECIPE's order, so each flag's effect on the frozen probe and the cosine maps reads on its own: the plain trunk (CONSTLR_PLAIN) is the all-off control and the no-Gram trunk the all-on one.
# Each arm has its own run dir ssl_dinov3_cxr_constlr_nogram_no_<value>_seed_100 and its own stop file chain/constlr_nogram_no_<value>_stop_after.txt (the same rules, cluster-only too; no other stop file is read then), so no two
# arms, and no arm and a trunk, share either, and its own `run:` line says which flag is off. TRUNK below stays the no-Gram trunk's run dir, and branches of an ablation arm are not supported. Refused before anything is read: any other
# non-empty value (the message names the three), CONSTLR_NOGRAM unset or empty, CONSTLR_PLAIN or CONSTLR_ARK set, and any CONSTLR_BRANCH_* set. run_d1_8_constlr_nogram_ablate_gaudi.sh sets CONSTLR_NOGRAM and refuses to start without
# CONSTLR_ABLATE (sbatch --export=ALL,CONSTLR_ABLATE=<value>): its own 4-card job outside the chain, never a phase.
set -euo pipefail
NOGRAM=${CONSTLR_NOGRAM:-}  # first, before anything is read or launched
[[ -z $NOGRAM || $NOGRAM == 1 ]] || { echo "ERROR: CONSTLR_NOGRAM holds '$NOGRAM', which is not empty or 1" >&2; exit 1; }
PLAIN=${CONSTLR_PLAIN:-}
[[ -z $PLAIN || $PLAIN == 1 ]] || { echo "ERROR: CONSTLR_PLAIN holds '$PLAIN', which is not empty or 1" >&2; exit 1; }
BRANCH=${CONSTLR_BRANCH_EPOCH:-}${CONSTLR_BRANCH_STOP:-}${CONSTLR_BRANCH_EVERY:-}${CONSTLR_BRANCH_COUNT:-}  # non-empty: a branch (all four are then checked below)
[[ -z $PLAIN || -z $BRANCH ]] || { echo "ERROR: CONSTLR_PLAIN and CONSTLR_BRANCH_* are both set: a branch continues the no-Gram trunk with the DINOv3 recipe, and branches of the plain trunk are a later feature, so unset CONSTLR_PLAIN" >&2; exit 1; }
[[ -z $PLAIN || -n $NOGRAM ]] || { echo "ERROR: CONSTLR_PLAIN=1 is set without CONSTLR_NOGRAM=1: the plain trunk is the no-Gram trunk without the DINOv3 recipe, so set both" >&2; exit 1; }
[[ -z $BRANCH || -z $NOGRAM ]] || { echo "ERROR: CONSTLR_NOGRAM and CONSTLR_BRANCH_* are both set: a branch launches the no-Gram trunk's line with --branch_from in place of --no_gram, so set only the CONSTLR_BRANCH_* variables" >&2; exit 1; }
ARK=${CONSTLR_ARK:-}
[[ -z $ARK || $ARK == 1 ]] || { echo "ERROR: CONSTLR_ARK holds '$ARK', which is not empty or 1" >&2; exit 1; }
BRANCH_ARK=${CONSTLR_BRANCH_ARK:-}
[[ -z $BRANCH_ARK || $BRANCH_ARK == 1 ]] || { echo "ERROR: CONSTLR_BRANCH_ARK holds '$BRANCH_ARK', which is not empty or 1" >&2; exit 1; }
[[ -z $ARK || -z $PLAIN ]] || { echo "ERROR: CONSTLR_ARK and CONSTLR_PLAIN are both set: the Ark-teacher trunk is the no-Gram trunk with the DINOv3 recipe, so unset CONSTLR_PLAIN" >&2; exit 1; }
[[ -z $ARK || -z $BRANCH ]] || { echo "ERROR: CONSTLR_ARK and CONSTLR_BRANCH_* are both set: a branch of the Ark-teacher trunk is CONSTLR_BRANCH_ARK=1 with the CONSTLR_BRANCH_* variables, so unset CONSTLR_ARK" >&2; exit 1; }
[[ -z $ARK || -n $NOGRAM ]] || { echo "ERROR: CONSTLR_ARK=1 is set without CONSTLR_NOGRAM=1: the Ark-teacher trunk is the no-Gram trunk with the cyclic Ark+ teacher, so set both" >&2; exit 1; }
[[ -z $BRANCH_ARK || -n $BRANCH ]] || { echo "ERROR: CONSTLR_BRANCH_ARK=1 is set without the CONSTLR_BRANCH_* variables: it picks the Ark-teacher trunk for a branch (run_d1_8_branch_gram_gaudi.sh sets the four)" >&2; exit 1; }
ARK_LOCK=${CONSTLR_ARK_LOCK:-}
[[ -z $ARK_LOCK || $ARK_LOCK == 0 || $ARK_LOCK == 1 ]] || { echo "ERROR: CONSTLR_ARK_LOCK holds '$ARK_LOCK', which is not empty, 0 or 1" >&2; exit 1; }
[[ -z $ARK_LOCK || -n $ARK || -n $BRANCH_ARK ]] || { echo "ERROR: CONSTLR_ARK_LOCK is set without CONSTLR_ARK=1 or CONSTLR_BRANCH_ARK=1: it picks the Ark-teacher trunk's lock arm, so set one of them" >&2; exit 1; }
ABLATE=${CONSTLR_ABLATE:-}
[[ -z $ABLATE || $ABLATE == drop_path || $ABLATE == rope || $ABLATE == local_norm ]] || { echo "ERROR: CONSTLR_ABLATE holds '$ABLATE', which is not empty, drop_path, rope or local_norm" >&2; exit 1; }
[[ -z $ABLATE || -z $PLAIN ]] || { echo "ERROR: CONSTLR_ABLATE and CONSTLR_PLAIN are both set: an ablation arm is the no-Gram trunk with one DINOv3 recipe flag removed, and the plain trunk has no recipe flag left to remove, so unset CONSTLR_PLAIN" >&2; exit 1; }
[[ -z $ABLATE || -z $ARK ]] || { echo "ERROR: CONSTLR_ABLATE and CONSTLR_ARK are both set: an ablation arm is the no-Gram trunk's line with one recipe flag removed and nothing added, so unset CONSTLR_ARK" >&2; exit 1; }
[[ -z $ABLATE || -z $BRANCH ]] || { echo "ERROR: CONSTLR_ABLATE and CONSTLR_BRANCH_* are both set: branches of an ablation arm are not supported, so unset CONSTLR_ABLATE" >&2; exit 1; }
[[ -z $ABLATE || -n $NOGRAM ]] || { echo "ERROR: CONSTLR_ABLATE=$ABLATE is set without CONSTLR_NOGRAM=1: an ablation arm is the no-Gram trunk with one recipe flag removed, so set both" >&2; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/ssl_common.sh"

EPOCHS=1000  # the --epochs ceiling (never reached): the stop epoch's upper limit and the launch line both use it, so they cannot drift
TRUNK=ssl_dinov3_cxr_constlr_nogram_seed_100  # the no-Gram trunk's run dir (CONSTLR_NOGRAM=1), which every branch starts from
RUN=ssl_dinov3_cxr_constlr_seed_100
STOP_FILE=chain/constlr_stop_after.txt
GRAM_ARGS=(--gram_refresh_steps 10000,20000,30000)
RECIPE=(--drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm)  # the trunk's and its branches' (DINOv3's training recipe), never the phase's
RECIPE_NOTE="DINOv3 recipe: ${RECIPE[*]}"  # what the trunk's run: line says about it
ARK_TOKEN=  ARK_LOCK_FLAG=--no_lock_half_pass  ARK_LOCK_JSON=false  ARK_LOCK_NOTE="no lock passes: the full-pass release per dataset alone (CONSTLR_ARK_LOCK=0)"
if [[ ${ARK_LOCK:-0} == 1 ]]; then  # the lock arm: its own run dir, stop file and line
  ARK_TOKEN=_lock  ARK_LOCK_FLAG=--lock_half_pass  ARK_LOCK_JSON=true  ARK_LOCK_NOTE="FX lock-release: half-pass lock + full-pass release per dataset (CONSTLR_ARK_LOCK=1)"
fi
ARK_TRUNK=ssl_dinov3_cxr_constlr_arkteacher${ARK_TOKEN}_seed_100  # the Ark-teacher trunk's run dir (CONSTLR_NOGRAM=1 CONSTLR_ARK=1), which every CONSTLR_BRANCH_ARK=1 branch of that arm starts from
ARK_FLAGS=(--cyclic_ark_teacher --heads per_dataset "$ARK_LOCK_FLAG" --keep_all_checkpoints)  # the Ark-teacher trunk's and its branches', after --no_gram / the branch pair
[[ -z $BRANCH_ARK ]] || TRUNK=$ARK_TRUNK
if [[ -n $NOGRAM ]]; then
  RUN=$TRUNK
  STOP_FILE=chain/constlr_nogram_stop_after.txt
  GRAM_ARGS=("${RECIPE[@]}" --no_gram)
  if [[ -n $PLAIN ]]; then  # no recipe flags at all (not an empty RECIPE: "${RECIPE[@]}" of an empty array is an unbound variable under set -u before bash 4.4); TRUNK is left alone
    RUN=ssl_dinov3_cxr_constlr_nogram_plain_seed_100
    STOP_FILE=chain/constlr_nogram_plain_stop_after.txt
    GRAM_ARGS=(--no_gram)
    RECIPE_NOTE="plain: no DINOv3 recipe flags, the one-variable Gram-off control of ssl_dinov3_cxr_constlr_seed_100"
  fi
  if [[ -n $ARK ]]; then
    RUN=$ARK_TRUNK
    STOP_FILE=chain/constlr_arkteacher${ARK_TOKEN}_stop_after.txt
    GRAM_ARGS=("${RECIPE[@]}" --no_gram "${ARK_FLAGS[@]}")
    RECIPE_NOTE="DINOv3 recipe: ${RECIPE[*]}; Ark teacher: ${ARK_FLAGS[*]} after --no_gram ($ARK_LOCK_NOTE; clean teacher views implied, --ark_teacher_momentum 0.8 by default; an epoch is a cycle over the four datasets)"
  fi
  if [[ -n $ABLATE ]]; then  # RECIPE minus one flag and its value (the rest in its order), the line otherwise the no-Gram trunk's; TRUNK is left alone (no branch of an arm: refused above)
    case $ABLATE in drop_path) OFF_FLAG=--drop_path_rate ;; rope) OFF_FLAG=--rope_rescale_coords ;; local_norm) OFF_FLAG=--untie_local_cls_norm ;; esac
    KEPT=() OFF= DROP=
    for word in "${RECIPE[@]}"; do
      [[ $word != --* ]] || DROP=  # a flag starts a new group, kept unless it is the removed one ...
      [[ $word != "$OFF_FLAG" ]] || DROP=1  # ... whose value (the words without --) then goes with it
      if [[ -n $DROP ]]; then OFF+=${OFF:+ }$word; else KEPT+=("$word"); fi
    done
    RUN=ssl_dinov3_cxr_constlr_nogram_no_${ABLATE}_seed_100
    STOP_FILE=chain/constlr_nogram_no_${ABLATE}_stop_after.txt
    GRAM_ARGS=("${KEPT[@]}" --no_gram)
    RECIPE_NOTE="recipe ablation: $ABLATE off ($OFF removed); DINOv3 recipe otherwise: ${KEPT[*]}"
  fi
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
  if [[ -n $BRANCH_ARK ]]; then  # the branch inherits its trunk's lock arm: its args.json must record the lock_half_pass this arm's run dir stands for
    TRUNK_LOCK=$(sed -nE 's/^ *"lock_half_pass": *(true|false),? *$/\1/p' "$TRUNK_ARGS" 2>/dev/null || true)
    [[ $TRUNK_LOCK == "$ARK_LOCK_JSON" ]] || { echo "ERROR: $TRUNK_ARGS holds lock_half_pass '$TRUNK_LOCK', not the $ARK_LOCK_JSON of CONSTLR_ARK_LOCK=${ARK_LOCK:-0} (whose run dir it is): refusing to branch with another lock arm than the trunk's" >&2; exit 1; }
  fi
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
  GRAM_ARGS=("${RECIPE[@]}" --branch_from "/workspace/$SOURCE" --gram_refresh_steps "$REFRESH")
  [[ -z $BRANCH_ARK ]] || GRAM_ARGS+=("${ARK_FLAGS[@]}")  # the Ark-teacher trunk's mode flags, where its line has them (no empty-array expansion: bash 3.2 under set -u)
  echo "run: $RUN (Gram-anchored branch of $TRUNK from its epoch $K: --branch_from /workspace/$SOURCE --gram_refresh_steps in place of --no_gram; stop after epoch $STOP_AFTER)"
  echo "gram refresh steps: $REFRESH (branch step $BRANCH_STEP = epoch $K x epoch_length $EPOCH_LENGTH from $TRUNK_ARGS, then every $EVERY steps, $COUNT times)"
fi
[[ -z $NOGRAM ]] || echo "run: $RUN (no Gram teacher: --no_gram replaces --gram_refresh_steps 10000,20000,30000; $RECIPE_NOTE; stop after epoch $STOP_AFTER from $STOP_FILE)"

ssl_launch "$RUN" --lr "$LR" --min_lr "$LR" --no_lr_scaling --epochs "$EPOCHS" --stop_after_epochs "$STOP_AFTER" "${GRAM_ARGS[@]}"
