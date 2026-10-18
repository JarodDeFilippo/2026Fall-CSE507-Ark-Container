#!/usr/bin/env bash
#SBATCH --job-name=d1-8-chain
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=256G
# Chain driver. The gaudi partition queues ~2 days per job, so every phase of the experiment runs back to back inside this ONE
# allocation (4 cards, up to 7 days). The phases are named in chain/phases.txt, which is re-read before each phase, so lines
# appended or edited after submission count as long as chain/<phase>.sh exists when the driver reaches them (chain/README.md).
# usage: sbatch run_d1_8_chain_gaudi.sh      a re-submit continues from the first phase with no chain/done/<phase> marker
#        ./run_d1_8_chain_gaudi.sh --dry-run print the plan only: no cd, no card checks, nothing run or written (for the Mac)
# No -e on purpose: the loop checks every exit code itself, and a failed phase stops the chain (exit 1), never continues past it.
set -uo pipefail

DRY=
case "${1:-}" in
  "") ;;
  --dry-run) DRY=1 ;;
  *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

export PT_HPU_LAZY_MODE=1
if [[ -z $DRY ]]; then
  cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
  # fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
  ./gaudi_check_cards.sh || exit 1
  # report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
  [[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
  ./gaudi_verify_busy.sh 900 &
fi

CHAIN=chain
NAME_RE='^[A-Za-z0-9][A-Za-z0-9_.-]*$'   # a phase name is a file stem under $CHAIN/: no path separators, no spaces
SIM=" "   # dry-run only: phases the plan has already "run" (nothing writes a done marker then)
ANN=" "   # phases already announced by this invocation (skipped or run), so a skip is logged once, not on every pass

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s%s\n' "$(ts)" "${DRY:+[dry-run] }" "$*"; }
is_done() { [[ -e $CHAIN/done/$1 || $SIM == *" $1 "* ]]; }

[[ -n $DRY ]] || mkdir -p "$CHAIN/done" "$CHAIN/log"
log "chain driver start (job ${SLURM_JOB_ID:-none}, host $(hostname))"

while :; do
  if [[ -e $CHAIN/STOP ]]; then
    log "STOP file present ($CHAIN/STOP), stopping"
    exit 0
  fi
  [[ -r $CHAIN/phases.txt ]] || { log "$CHAIN/phases.txt missing or unreadable, stopping"; exit 1; }
  phase=
  # the file is opened afresh on every pass, so phases appended or edited since the last pass are seen
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%%#*}
    read -r line <<< "${line//$'\r'/}"   # trims surrounding whitespace
    [[ -n $line ]] || continue
    [[ $line =~ $NAME_RE ]] || { log "$CHAIN/phases.txt: '$line' is not a valid phase name, stopping"; exit 1; }
    if is_done "$line"; then
      [[ $ANN == *" $line "* ]] || { log "phase $line: done marker found, skipping"; ANN="$ANN$line "; }
      continue
    fi
    phase=$line
    break
  done < "$CHAIN/phases.txt"
  [[ -n $phase ]] || break   # a full pass found nothing left to run

  script=$CHAIN/$phase.sh
  if [[ ! -f $script ]]; then
    log "phase $phase: script missing, stopping"
    exit 1
  fi
  if [[ -n $DRY ]]; then
    log "phase $phase: would run bash $script >> $CHAIN/log/$phase.out 2>&1"
    SIM="$SIM$phase "; ANN="$ANN$phase "
    continue
  fi

  start_iso=$(ts); start_s=$(date +%s)
  log "phase $phase: start"
  echo "=== $start_iso driver: start $phase ===" >> "$CHAIN/log/$phase.out"
  bash "$script" < /dev/null >> "$CHAIN/log/$phase.out" 2>&1
  rc=$?
  end_iso=$(ts); secs=$(( $(date +%s) - start_s ))
  [[ -s $CHAIN/times.csv ]] || echo "phase,start_iso,end_iso,seconds,exit_code" > "$CHAIN/times.csv"
  echo "$phase,$start_iso,$end_iso,$secs,$rc" >> "$CHAIN/times.csv"
  if [[ $rc -ne 0 ]]; then
    log "phase $phase: FAILED with exit code $rc after ${secs}s, stopping (see $CHAIN/log/$phase.out)"
    exit 1
  fi
  # without the marker the next pass would pick this phase again, so a failed touch must stop the chain
  touch "$CHAIN/done/$phase" || { log "phase $phase: cannot write $CHAIN/done/$phase, stopping"; exit 1; }
  ANN="$ANN$phase "
  log "phase $phase: done in ${secs}s (exit 0)"
done

log "no phase left to run in $CHAIN/phases.txt, chain complete"
exit 0
