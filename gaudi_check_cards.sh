#!/usr/bin/env bash
# Fail-closed card check, run before anything trains. Slurm on Sol sets no HABANA_VISIBLE_*, so gaudi.sh maps
# SLURM_JOB_GPUS to HABANA_VISIBLE_MODULES through hl-smi. torch.hpu.device_count() ignores that restriction and
# counts every card on the node (measured, job 64607982), so a count proves nothing. Instead this exits 1 unless
# the restriction reached the container with exactly as many entries as Slurm allocated cards and, when a
# HABANA_VISIBLE_* variable is set on the host, that variable selects exactly the allocated ones. Before the container
# touches any card, every allocated card must also read idle on the host (the idle baseline). That the runtime really
# honours the restriction is reported later by gaudi_verify_busy.sh (report-only), which logs whether the allocated
# cards are busy during training.
set -uo pipefail  # no -e: the probe exit code is handled explicitly below

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=gaudi_hlsmi.sh
source "$ROOT/gaudi_hlsmi.sh" || { echo "ERROR: cannot source $ROOT/gaudi_hlsmi.sh; refusing to launch" >&2; exit 1; }

refuse() { echo "ERROR: $*" >&2; exit 1; }
# "a, b,,a" -> "a,b" (trimmed, non-empty, unique, numeric order); fails if an entry is not a plain integer
norm_set() { printf '%s\n' "$1" | tr ',' '\n' | awk '{ gsub(/[[:space:]]/, "") } $0 != "" { print; if ($0 !~ /^(0|[1-9][0-9]*)$/) bad = 1 } END { exit bad }' | sort -un | paste -sd, -; }

echo "== host environment =="
env | grep -E '^(HABANA|PT_HPU|SLURM_JOB_GPUS|SLURM_GPUS_ON_NODE|GPU_DEVICE_ORDINAL)' | sort

# cards Slurm allocated: the non-empty entries of SLURM_JOB_GPUS, else SLURM_GPUS_ON_NODE
expected=$(printf '%s\n' "${SLURM_JOB_GPUS:-}" | awk -F, '{for (i = 1; i <= NF; i++) if ($i ~ /[^[:space:]]/) n++} END {print n + 0}')
if [[ $expected -eq 0 && "${SLURM_GPUS_ON_NODE:-}" =~ ^[1-9][0-9]*$ ]]; then
  expected=$SLURM_GPUS_ON_NODE
fi
if [[ $expected -eq 0 ]]; then
  echo "ERROR: no Slurm card allocation found (SLURM_JOB_GPUS / SLURM_GPUS_ON_NODE unset); refusing to launch" >&2
  exit 1
fi

echo "== idle baseline =="
# Before anything of ours touches a card (the probe below is the first thing that does), every allocated card must read idle on the
# host: memory.used <= GAUDI_IDLE_MAX_MIB MiB (gaudi_hlsmi.sh; an idle card reads 768 MiB). Otherwise a card that looks busy later, to gaudi_verify_busy.sh, could be
# another process's rather than this job's ranks'. This is the idle half of that transition: allocated cards idle here, busy during
# training, and acquisition exclusive (one process per card), so the transition is this job's ranks.
# Residual window: a foreign process that grabs an allocated card between this check and our own acquisition makes our rank fail to
# acquire it (HABANA_VISIBLE_MODULES restricts the rank to the allocated cards, and acquisition is exclusive); it cannot land elsewhere.
if ! baseline_ids=$(norm_set "${SLURM_JOB_GPUS:-}"); then
  refuse "cannot verify the idle baseline: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} has entries that are not plain card indices; refusing to launch"
elif [[ -z $baseline_ids ]]; then
  echo "idle baseline skipped: SLURM_JOB_GPUS lists no indices"
else
  table=$(hl-smi -Q index,module_id,memory.used -f csv); hl_rc=$?
  if [[ $hl_rc -ne 0 ]]; then
    refuse "cannot verify the idle baseline: hl-smi is missing or failed (rc=$hl_rc); refusing to launch"
  fi
  if ! allocmem=$(printf '%s\n' "$table" | gaudi_hlsmi_mem | gaudi_alloc_mem "$baseline_ids") || [[ -z $allocmem ]]; then
    refuse "cannot verify the idle baseline: the hl-smi table could not be parsed; refusing to launch"
  fi
  while IFS='=' read -r i m; do
    case $m in
      MISSING) refuse "cannot verify the idle baseline: index $i is missing from the hl-smi table; refusing to launch" ;;
      NA) refuse "cannot verify the idle baseline: index $i has an unparseable memory.used; refusing to launch" ;;
    esac
    if (( m > GAUDI_IDLE_MAX_MIB )); then
      refuse "allocated card index $i already shows $m MiB in use before launch; another process is on it; refusing to launch"
    fi
  done <<< "$allocmem"
  echo "idle baseline OK: allocated index(es) $baseline_ids idle before launch (<= ${GAUDI_IDLE_MAX_MIB} MiB)"
fi

echo "== card identity =="
# A variable such as HABANA_VISIBLE_MODULES can select N cards that are not the N Slurm allocated, and the
# entry count checked below cannot tell. So when one is set it is checked against the Slurm allocation here,
# before any container starts. Assumption: Slurm's gres index for hl225 equals the /dev/accel index that hl-smi
# reports as `index` (gres.conf File order). If that is wrong the check refuses instead of passing, so it
# fails closed.
if [[ -n "${HABANA_VISIBLE_DEVICES:-}" || -n "${HABANA_VISIBLE_MODULES:-}" ]]; then
  if ! alloc=$(norm_set "${SLURM_JOB_GPUS:-}"); then
    refuse "cannot verify card identity: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} has entries that are not plain card indices"
  fi
  if [[ -z $alloc ]]; then
    refuse "cannot verify card identity: HABANA_VISIBLE_* is set but SLURM_JOB_GPUS lists no indices"
  fi
  if [[ -n "${HABANA_VISIBLE_DEVICES:-}" ]]; then  # device indices: the same index space as Slurm's gres index
    if [[ "${HABANA_VISIBLE_DEVICES//[[:space:]]/}" == all ]]; then
      refuse "card identity mismatch: HABANA_VISIBLE_DEVICES=all selects every card on the node, not just the Slurm-allocated indices {$alloc}"
    fi
    if ! vis=$(norm_set "$HABANA_VISIBLE_DEVICES") || [[ $vis != "$alloc" ]]; then
      refuse "card identity mismatch: HABANA_VISIBLE_DEVICES=$HABANA_VISIBLE_DEVICES selects {${vis:-nothing}} but Slurm allocated indices {$alloc}"
    fi
    echo "card identity OK: HABANA_VISIBLE_DEVICES=$HABANA_VISIBLE_DEVICES matches Slurm allocation $alloc"
  fi
  if [[ -n "${HABANA_VISIBLE_MODULES:-}" ]]; then  # Habana module IDs: map the allocated indices through the host's hl-smi
    # keep only rows whose first two comma-separated fields are integers, which drops the header if one is printed
    if ! map=$(hl-smi -Q index,module_id -f csv | awk -F, '{ gsub(/[[:space:]]/, "") } $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ { print $1 "=" $2 }'); then
      refuse "cannot verify card identity: hl-smi -Q index,module_id -f csv is missing or failed, so Slurm indices {$alloc} cannot be mapped to module IDs"
    fi
    map_line=$(printf '%s\n' "$map" | paste -sd' ' -)
    want=
    IFS=, read -r -a idx <<< "$alloc"
    for i in "${idx[@]}"; do
      m=$(printf '%s\n' "$map" | awk -F= -v i="$i" '$1 == i { print $2; exit }')
      if [[ -z $m ]]; then
        refuse "cannot verify card identity: hl-smi lists no module_id for allocated index $i (hl-smi index=module_id: ${map_line:-none})"
      fi
      want="${want:+$want,}$m"
    done
    want=$(norm_set "$want")
    if ! vis=$(norm_set "$HABANA_VISIBLE_MODULES") || [[ $vis != "$want" ]]; then
      refuse "card identity mismatch: HABANA_VISIBLE_MODULES=$HABANA_VISIBLE_MODULES selects modules {${vis:-nothing}} but Slurm allocated indices {$alloc}, which are modules {$want} (hl-smi index=module_id: $map_line)"
    fi
    echo "card identity OK: HABANA_VISIBLE_MODULES=$HABANA_VISIBLE_MODULES matches Slurm allocation $alloc"
  fi
else
  echo "card identity: no HABANA_VISIBLE_* set on the host; gaudi.sh maps SLURM_JOB_GPUS to HABANA_VISIBLE_MODULES through hl-smi, and the container check below verifies the restriction reached the container"
fi

echo "== container probe =="
probe=$("$ROOT/gaudi.sh" python -c '
import os
import torch
import torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__)
for k in sorted(os.environ):
    if k.startswith(("HABANA", "PT_HPU")):
        print("{}={}".format(k, os.environ[k]))
# the restriction as the container sees it: HABANA_VISIBLE_MODULES, else HABANA_VISIBLE_DEVICES
name = "HABANA_VISIBLE_MODULES" if os.environ.get("HABANA_VISIBLE_MODULES") else "HABANA_VISIBLE_DEVICES"
value = os.environ.get(name, "")
print("VISIBLE_ENTRIES={}".format(len([e for e in value.split(",") if e.strip()])))
print("VISIBLE={}={}".format(name, value) if value else "VISIBLE=")
import habana_frameworks.torch.core
print("HPU_NODE_COUNT={} (torch.hpu.device_count() counts every card on the node; not a restriction measure)".format(torch.hpu.device_count()))
')
probe_rc=$?
echo "$probe"
entries=$(printf '%s\n' "$probe" | awk -F= '/^VISIBLE_ENTRIES=/ {value=$2} END {print value}')
visible=$(printf '%s\n' "$probe" | awk '/^VISIBLE=/ {sub(/^VISIBLE=/, ""); value=$0} END {print value}')
if [[ $probe_rc -ne 0 ]]; then
  refuse "container probe failed (rc=$probe_rc), so the restriction cannot be verified; refusing to launch"
fi
if [[ -z $visible ]]; then
  refuse "no HABANA_VISIBLE_* reached the container; refusing to launch with every card visible"
fi
if [[ "$entries" != "$expected" ]]; then
  refuse "restriction mismatch: $visible has $entries entries in the container but Slurm allocated $expected; refusing to launch"
fi
echo "restriction OK: $visible in the container ($entries entries), $expected allocated by Slurm; card use is reported during training by gaudi_verify_busy.sh"
