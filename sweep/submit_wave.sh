#!/usr/bin/env bash
# Submits the arms of a sweep table to Slurm: one `sbatch --parsable --job-name=d1-8sw-<id> --dependency=singleton run_d1_8sw_gaudi.sh <id>` per sweep id,
# in the order given (default: every row of the table, in file order). Must run from the repo root, because sbatch finds run_d1_8sw_gaudi.sh relative to the
# current directory.
# usage: sweep/submit_wave.sh [--dry-run] sweep/wave1.tsv [sweep id ...]     e.g. sweep/submit_wave.sh sweep/wave1.tsv S2 S4
# A job is one 1-day link of an arm. The job name d1-8sw-<id> plus --dependency=singleton make Slurm run at most one job of that name per user at a time, so
# to continue the arms run this script again (all ids, or only the ones named): it queues the next link of each arm behind the one that is queued or running,
# which starts when that one has ended, however it ended, and resumes from the latest checkpoint. Two links never train on the same run dir, and a link that
# finds its run finished reports "already done" and exits 0.
# The table has to be sweep/wave1.tsv itself (the same file under any spelling; a copy of it does not count): chain/sweep_run.sh always reads that table, so any
# other one would silently run wave-1 parameters under its own ids. A wave 2 will need the table passed through to chain/sweep_run.sh first.
# Known limit: a queued job reads sweep/wave1.tsv when it starts, not when it is submitted, so do not edit the table while sweep jobs are queued.
# Every id is checked against the table before anything is submitted. Each sbatch command is printed (to stderr, or to stdout under --dry-run, which
# submits nothing), and stdout gets one line "<sweep id> <job id>" per submitted arm. The first failing sbatch stops the wave and names the ids
# that were not submitted: submit just those again (an id that did get submitted would only queue another link behind its first job).
# Exit 0: all submitted (or dry run). 1: sbatch failed. 2: usage error, not run from the repo root, a table other than sweep/wave1.tsv, or an unknown /
# repeated / malformed sweep id.
set -uo pipefail
set -f  # the ids are word-split below and must never be globbed

ID_RE='^[A-Za-z0-9][A-Za-z0-9_-]*$'  # the same as chain/sweep_run.sh
USAGE="usage: $0 [--dry-run] sweep/wave1.tsv [sweep id ...]"
die() { echo "ERROR: $*" >&2; exit 2; }

DRY=
TSV=
WANTED=
for arg in "$@"; do
  case $arg in
    --dry-run) DRY=1 ;;
    -*) die "unknown option '$arg' ($USAGE)" ;;
    *) if [[ -z $TSV ]]; then TSV=$arg; else WANTED="$WANTED $arg"; fi ;;
  esac
done
[[ -n $TSV ]] || die "$USAGE"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ $(pwd -P) == "$ROOT" ]] || die "run this from the repo root ($ROOT), not from $(pwd -P): sbatch finds run_d1_8sw_gaudi.sh relative to the current directory"
[[ -r $TSV ]] || die "$TSV is missing or unreadable"
# -ef: the same file (device and inode, links followed), which is what comparing realpaths decides: ./sweep/wave1.tsv, sweep/../sweep/wave1.tsv or a link to it
# pass, a byte-identical copy does not. Unlike a string compare of two realpath outputs it cannot pass by accident when realpath is missing (two empty strings).
[[ $TSV -ef $ROOT/sweep/wave1.tsv ]] || die "$TSV is not sweep/wave1.tsv: chain/sweep_run.sh always reads sweep/wave1.tsv, so any other table (a byte-identical copy included) would silently run wave-1 parameters under its own ids; a wave 2 will need the table passed through to chain/sweep_run.sh first"
IFS= read -r FIRST < "$TSV"
[[ $FIRST == id$'\t'* ]] || die "$TSV does not start with the header row (id, then the other columns, tab-separated)"

IN_TABLE=$(awk -F'\t' 'NR > 1 { print $1 }' "$TSV")  # one id per line, file order
[[ -n $IN_TABLE ]] || die "$TSV has no rows"
while IFS= read -r id; do
  [[ $id =~ $ID_RE ]] || die "$TSV: '$id' is not a valid sweep id (letters, digits, _ and -)"
done <<< "$IN_TABLE"
SELECTED=${WANTED# }
if [[ -z $SELECTED ]]; then
  SELECTED=${IN_TABLE//$'\n'/ }
fi
for id in $SELECTED; do
  [[ $id =~ $ID_RE ]] || die "'$id' is not a valid sweep id (letters, digits, _ and -)"
  grep -Fxq -- "$id" <<< "$IN_TABLE" || die "sweep id '$id' is not a row of $TSV"
done
# shellcheck disable=SC2086  # unquoted on purpose: SELECTED holds regex-checked ids and globbing is off (set -f)
REPEATED=$(printf '%s\n' $SELECTED | sort | uniq -d | tr '\n' ' ')
[[ -z $REPEATED ]] || die "repeated sweep id(s): $REPEATED(one link per id and call: run this script again to queue another link behind it)"

# shellcheck disable=SC2086
set -- $SELECTED
while (( $# > 0 )); do
  id=$1
  shift
  cmd=(sbatch --parsable "--job-name=d1-8sw-$id" --dependency=singleton run_d1_8sw_gaudi.sh "$id")
  if [[ -n $DRY ]]; then
    echo "${cmd[*]}"
    continue
  fi
  echo "+ ${cmd[*]}" >&2
  if ! out=$("${cmd[@]}" < /dev/null); then
    echo "ERROR: sbatch failed for $id; stopping. Not submitted: $id${*:+ $*}" >&2
    exit 1
  fi
  echo "$id ${out%%;*}"  # --parsable prints "<job id>" or "<job id>;<cluster>"
done
