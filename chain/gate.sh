#!/usr/bin/env bash
# Chain phase "gate": the Stage A gate (chain/gate.py) over arm D, probes A/B and the two SSL pilots.
# Runs inside an allocation like the other phases (the driver has already done the cd to the repo root and exported PT_HPU_LAZY_MODE=1).
# gate.py runs INSIDE the container, so it gets the container's Python: gaudi.sh ends in `exec apptainer exec ... "$@"` with the repo root
# bound at /workspace and the working directory at /workspace/base, hence the absolute /workspace paths.
# The exit status is gate.py's, unchanged (exec): 0 = decision written to chain/main_lr.txt + chain/main_epochs.txt, 2 = STOP written,
# 1 = parse error. Exit 1 with a message on stderr also when chain/gate.py is missing or empty.
set -uo pipefail
if [[ ! -s chain/gate.py ]]; then
  echo "ERROR: chain/gate.py is missing or empty (looked in $PWD/chain); the Stage A gate cannot run" >&2
  exit 1
fi
exec ./gaudi.sh python /workspace/chain/gate.py \
  --arm_d /workspace/all4_cyclic_dinov3_frozen_seed_100 \
  --probe_a /workspace/all4_cyclic_dinov3_probe_a_seed_100 \
  --probe_b /workspace/all4_cyclic_dinov3_probe_b_seed_100 \
  --pilot_a /workspace/ssl_dinov3_cxr_pilot_a_seed_100 \
  --pilot_b /workspace/ssl_dinov3_cxr_pilot_b_seed_100 \
  --out /workspace/chain
