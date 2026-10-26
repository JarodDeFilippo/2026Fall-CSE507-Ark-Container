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
# One env override, unset in the chain's phase (so the phase's run dir, launch line and messages are unchanged): CONSTLR_NOGRAM=1 makes this the no-Gram trunk,
# "DINO-X ConstLR 224, no Gram": the same launch line with --no_gram (no Gram teacher built, loaded or run) in place of --gram_refresh_steps 10000,20000,30000, into its own
# run dir ssl_dinov3_cxr_constlr_nogram_seed_100, with the stop epoch read from chain/constlr_nogram_stop_after.txt instead (the same rules, cluster-only too; chain/constlr_stop_after.txt
# is not read then), and one `run:` line printed before the launch. Any other non-empty value is refused before anything is read. run_d1_8_constlr_nogram_gaudi.sh sets it: its own
# 4-card job outside the chain, never a phase.
set -euo pipefail
NOGRAM=${CONSTLR_NOGRAM:-}  # first, before anything is read or launched
[[ -z $NOGRAM || $NOGRAM == 1 ]] || { echo "ERROR: CONSTLR_NOGRAM holds '$NOGRAM', which is not empty or 1" >&2; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/ssl_common.sh"

EPOCHS=1000  # the --epochs ceiling (never reached): the stop epoch's upper limit and the launch line both use it, so they cannot drift
RUN=ssl_dinov3_cxr_constlr_seed_100
STOP_FILE=chain/constlr_stop_after.txt
GRAM_ARGS=(--gram_refresh_steps 10000,20000,30000)
if [[ -n $NOGRAM ]]; then
  RUN=ssl_dinov3_cxr_constlr_nogram_seed_100
  STOP_FILE=chain/constlr_nogram_stop_after.txt
  GRAM_ARGS=(--no_gram)
fi
LR=$(read_setting chain/main_lr.txt '^([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$' 'a positive number such as 2e-3 or 0.0003')
awk -v v="$LR" 'BEGIN { exit !(v + 0 > 0) }' || { echo "ERROR: chain/main_lr.txt holds '$LR', which is not positive" >&2; exit 1; }
STOP_AFTER=$(read_setting "$STOP_FILE" '^[1-9][0-9]*$' 'a positive integer')
# awk, not (( )): bash arithmetic wraps silently past 2^63 (18446744073709551617 would read as 1)
awk -v v="$STOP_AFTER" -v cap="$EPOCHS" 'BEGIN { exit !(v + 0 <= cap + 0) }' || { echo "ERROR: $STOP_FILE holds '$STOP_AFTER', which is above the --epochs ceiling of $EPOCHS" >&2; exit 1; }
[[ -z $NOGRAM ]] || echo "run: $RUN (no Gram teacher: --no_gram replaces --gram_refresh_steps 10000,20000,30000; stop after epoch $STOP_AFTER from $STOP_FILE)"

ssl_launch "$RUN" --lr "$LR" --min_lr "$LR" --no_lr_scaling --epochs "$EPOCHS" --stop_after_epochs "$STOP_AFTER" "${GRAM_ARGS[@]}"
