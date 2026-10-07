#!/usr/bin/env bash
# usage: gaudi_verify_busy.sh <delay_seconds>
# REPORT-ONLY check, started in the background right before training: after <delay_seconds> it logs whether every card Slurm allocated to
# this job is busy according to the host's hl-smi. It never cancels or kills anything: that is Jarod's decision of 2026-10-04, taken after
# the card mapping was proven on gaudi003 (1 card) and gaudi005 (4 cards: the allocated 0-3 at 3219 MiB, the others at 768). The pre-launch
# checks in gaudi_check_cards.sh stay fail-closed; this leaves evidence in the job log, and a WARNING means: inspect, and scancel by hand.
# torch.hpu.device_count() ignores HABANA_VISIBLE_MODULES (measured, job 64607982), so which cards the ranks really took can only be seen
# by measurement. The argument: each rank acquires exactly one card and there are as many ranks as allocated cards; the allocated cards
# were idle at the pre-launch check (gaudi_check_cards.sh) and are busy at T; acquisition is exclusive (one process per card); so the
# transition is this job's ranks, and every rank sits on an allocated card. Residual window: a foreign process that grabs an allocated
# card between that check and our acquisition makes our rank fail to acquire it (HABANA_VISIBLE_MODULES restricts the rank to the
# allocated cards), not land elsewhere. Assumption (as in gaudi.sh): Slurm's gres index == hl-smi `index`.
# Exit: 0 all allocated cards busy, 1 WARNING printed (an allocated card idle, missing or unparseable; hl-smi or the helper missing or
# failing), 2 usage error (nothing checked).
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

delay=${1:-}
if [[ ! $delay =~ ^[0-9]+$ ]]; then
  echo "usage: gaudi_verify_busy.sh <delay_seconds>" >&2
  exit 2
fi
# the allocated indices, validated like gaudi.sh: trimmed, non-empty, each a canonical integer
if ! ids=$(printf '%s\n' "${SLURM_JOB_GPUS:-}" | tr ',' '\n' | awk '{ gsub(/[[:space:]]/, "") } $0 != "" { print; if ($0 !~ /^(0|[1-9][0-9]*)$/) bad = 1 } END { exit bad }') || [[ -z $ids ]]; then
  echo "ERROR: gaudi_verify_busy.sh needs SLURM_JOB_GPUS as a list of plain card indices (got '${SLURM_JOB_GPUS:-}'); nothing checked" >&2
  exit 2
fi
list=$(printf '%s\n' "$ids" | paste -sd, -)
# a missing helper is reported below, after the delay, together with the raw table
# shellcheck source=gaudi_hlsmi.sh
helper_ok=1; source "$ROOT/gaudi_hlsmi.sh" 2>/dev/null || helper_ok=

# the sleep is a background child so that a TERM (sent when training ends first) kills it too; sp is cleared once
# the sleep is reaped so the trap can never signal a stale pid
sp=
trap '[[ -n $sp ]] && kill "$sp" 2>/dev/null; exit 143' TERM
sleep "$delay" & sp=$!
wait "$sp"; sp=

echo "== card use after $delay s (allocated index: $SLURM_JOB_GPUS) =="
table=$(hl-smi -Q index,module_id,memory.used -f csv); hl_rc=$?
printf '%s\n' "$table"

# busy = memory.used above GAUDI_IDLE_MAX_MIB MiB (gaudi_hlsmi.sh, the same threshold as the idle baseline); an allocated index that
# is missing, idle or unparseable counts as a problem
reason=
if [[ -z $helper_ok ]]; then
  reason="cannot source $ROOT/gaudi_hlsmi.sh, so the table could not be parsed"
elif [[ $hl_rc -ne 0 ]]; then
  reason="hl-smi is missing or failed (rc=$hl_rc)"
elif ! allocmem=$(printf '%s\n' "$table" | gaudi_hlsmi_mem | gaudi_alloc_mem "$list") || [[ -z $allocmem ]]; then
  reason="the hl-smi table could not be parsed"
else
  while IFS='=' read -r i m; do
    case $m in
      MISSING) why="index $i is missing from the hl-smi table" ;;
      NA) why="index $i has an unparseable memory.used" ;;
      *) if (( m <= GAUDI_IDLE_MAX_MIB )); then why="index $i is idle ($m MiB used)"; else why=; fi ;;
    esac
    if [[ -n $why ]]; then reason="$reason${reason:+; }$why"; fi
  done <<< "$allocmem"
fi

if [[ -n $reason ]]; then
  warning="WARNING: card use check: $reason; a rank may be on another user's card; inspect and scancel ${SLURM_JOB_ID:-<SLURM_JOB_ID>} by hand if confirmed"
  echo "$warning"
  echo "$warning" >&2
  exit 1
fi
echo "card use OK: allocated index(es) $list busy (one process per card, so every rank is on an allocated card)"
