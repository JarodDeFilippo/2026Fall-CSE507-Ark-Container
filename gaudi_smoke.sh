#!/usr/bin/env bash
#SBATCH --job-name=gaudi-smoke
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 00:45:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:1
#SBATCH -c 18
#SBATCH --mem=96G
# Gaudi smoke test (1 card, ~30 min): checks inside the container that exactly the allocated card is
# visible and the DINOv3 weights load, then trains arm B as a throwaway seed-1 run that the timeout stops.
set -uo pipefail  # no -e: the training exit code is handled explicitly below
export ARK_BATCH_SIZE=200
# Meta's distilled-student lr 2e-4 under its sqrt_wrt_1024 rule (lr *= 4*sqrt(batch/1024)) at batch 200 = 3.5e-4
export ARK_LR=0.00035
export ARK_TEST_AUGMENT=false
export ARK_EVAL_EVERY=5
export ARK_WORKERS=8
export ARK_MODEL=vit_base_dinov3
export ARK_INIT=dinov3
export ARK_PRETRAINED_WEIGHTS=/workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
RUN_DIR=all4_cyclic_dinov3_adamw_seed_1
IMAGE=${GAUDI_IMAGE:-containers/gaudi/gaudi.sif}
# fail fast (seconds, not a Gaudi slot) if the DINOv3 checkpoint is not in place yet;
# ARK_PRETRAINED_WEIGHTS is the in-container path (/workspace = this repo root)
HOST_WEIGHTS="${ARK_PRETRAINED_WEIGHTS/#\/workspace\//./}"
if [[ ! -s "$HOST_WEIGHTS" ]]; then
  echo "ERROR: DINOv3 weights not found at $HOST_WEIGHTS (host path for $ARK_PRETRAINED_WEIGHTS); requeue after the download lands" >&2
  exit 1
fi
if [[ ! -f "$IMAGE" ]]; then
  echo "ERROR: Gaudi image not found at $IMAGE; run build_gaudi_image.sh first (sbatch build_gaudi_image.sh)" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "ERROR: $RUN_DIR already exists and the engine refuses an existing run dir; delete it with rm -rf $RUN_DIR and resubmit" >&2
  exit 1
fi

echo "== host hl-smi =="
hl-smi || true
# the index -> module_id map (gaudi.sh maps Slurm's gres index through it) and the memory in use before training
hl-smi -Q index,module_id,bus_id,memory.used -f csv || true

./gaudi_check_cards.sh || exit 1

echo "== DINOv3 load check =="
if ! ./gaudi.sh python check_dinov3_load.py "$ARK_PRETRAINED_WEIGHTS"; then
  echo "ERROR: check_dinov3_load.py failed (see the output above)" >&2
  exit 1
fi

echo "== training (seed 1, stopped by a 30 min timeout) =="
# report-only: ~10 min into training, log whether every allocated card is busy (gaudi_verify_busy.sh prints the hl-smi
# table, to read in the .out file, and a WARNING if one is not); it cancels and kills nothing, training goes on
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 600 & snap_pid=$!
timeout --kill-after=60 1800 ./experiment_scripts/all4_cyclic_dinov3_adamw/start_all4_cyclic_dinov3_adamw.sh 1 gaudi
rc=$?
# signal the verifier only while bash still lists it as running: once it has exited and been reaped, its pid may be reused
if jobs -rp | grep -x "$snap_pid" > /dev/null; then kill "$snap_pid" 2>/dev/null; fi; wait "$snap_pid" 2>/dev/null

echo "== run log =="
if [[ -f "$RUN_DIR/train.log" ]]; then
  grep -E 'Backbone|World Size|Per-rank|Loaded with msg' "$RUN_DIR/train.log"
  grep 'BT=' "$RUN_DIR/train.log" | tail -n 20
else
  echo "no $RUN_DIR/train.log was written"
fi
progress=$(grep -c 'BT=' "$RUN_DIR/train.log" 2>/dev/null)  # BT= lines = logged training progress; empty if train.log is missing

echo "== result =="
echo "$RUN_DIR is throwaway: read it, then delete it (rm -rf $RUN_DIR)"
# a hung run also ends in 124, so a timeout only counts as success if training logged progress first
if [[ $rc -eq 124 && ${progress:-0} -eq 0 ]]; then
  echo "training STALLED: timed out with no logged step (no BT= line in train.log; BT= prints every 50 steps)" >&2
  exit 124
fi
if [[ $rc -eq 0 || $rc -eq 124 ]]; then
  echo "training OK (rc=$rc; 124 = stopped by the 30 min timeout, the intended outcome)"
  exit 0
fi
echo "training FAILED (rc=$rc; 137 = SIGKILL from the 60 s kill-after or the OOM killer, so read $RUN_DIR/train.log)" >&2
exit "$rc"
