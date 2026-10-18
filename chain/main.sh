#!/usr/bin/env bash
# Main run. chain/main_lr.txt holds the ABSOLUTE peak LR (the winning pilot's effective_peak_lr from its args.json), which is why
# this runs with --no_lr_scaling; chain/main_epochs.txt holds the schedule length. Both are read relative to the repo root
# (cwd), and a missing, empty or non-numeric file stops the phase before anything is launched.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/ssl_common.sh"

LR=$(read_setting chain/main_lr.txt '^([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$' 'a positive number such as 2e-3 or 0.0003')
awk -v v="$LR" 'BEGIN { exit !(v + 0 > 0) }' || { echo "ERROR: chain/main_lr.txt holds '$LR', which is not positive" >&2; exit 1; }
EPOCHS=$(read_setting chain/main_epochs.txt '^[1-9][0-9]*$' 'a positive integer')

ssl_launch ssl_dinov3_cxr_main_seed_100 --lr "$LR" --no_lr_scaling --epochs "$EPOCHS"
