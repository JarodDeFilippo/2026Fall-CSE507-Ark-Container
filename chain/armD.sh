#!/usr/bin/env bash
# Chain phase "armD": arm D, the frozen-backbone linear probe on the stock DINOv3 weights (5 cycles, seed 100, 4 Gaudi cards).
# Runs inside an allocation whose driver has already done cd /scratch/$USER/2026Fall-CSE507-Ark-Container, gaudi_check_cards.sh and
# gaudi_verify_busy.sh and exported PT_HPU_LAZY_MODE=1, so none of that is repeated here. Same ARK_* recipe as run_d1_8d_gaudi.sh.
# Starts the run, or resumes it from its latest checkpoint when an interrupted attempt left its run dir behind (the engine refuses to
# start into an existing run dir). Never deletes anything.
# Exit 0: finished now, or already done. 1: the stock weights are missing, or the run dir holds no resumable checkpoint.
# Otherwise the start / resume script's exit status.
set -uo pipefail
export ARK_BATCH_SIZE=200
export ARK_LR=7e-5
export ARK_TEST_AUGMENT=false
export ARK_EVAL_EVERY=1
export ARK_WARMUP_EPOCHS=2
export ARK_WORKERS=16
export ARK_MODEL=vit_base_dinov3
export ARK_INIT=dinov3
export ARK_PRETRAINED_WEIGHTS=/workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
# main_ark.py names the run dir <exp_name>_seed_<seed> in the repo root; the last cycle's student weights mark the phase done
RUN=all4_cyclic_dinov3_frozen_seed_100
if compgen -G "$RUN/models/weights/epoch_0005/student_cycle_0005_*.pth" > /dev/null; then
  echo "armD already done"
  exit 0
fi
# fail fast (seconds, not a whole slot) if the DINOv3 checkpoint is not in place yet (a resume builds the model from it too);
# ARK_PRETRAINED_WEIGHTS is the in-container path (/workspace = this repo root)
HOST_WEIGHTS="${ARK_PRETRAINED_WEIGHTS/#\/workspace\//./}"
if [[ ! -s "$HOST_WEIGHTS" ]]; then
  echo "ERROR: DINOv3 weights not found at $HOST_WEIGHTS (host path for $ARK_PRETRAINED_WEIGHTS); requeue after the download lands" >&2
  exit 1
fi
if [[ -d "$RUN" ]]; then
  # an interrupted attempt: continue it. A dir without a checkpoint can be neither resumed nor started over; leave it for a human.
  if ! compgen -G "$RUN/models/checkpoints/*.pth.tar" > /dev/null; then
    echo "ERROR: run dir ./$RUN exists but holds no resumable checkpoint ($RUN/models/checkpoints/*.pth.tar); neither resuming nor starting over it, nothing deleted: inspect or remove it, then requeue" >&2
    exit 1
  fi
  echo "armD: resuming ./$RUN from its latest checkpoint"
  ./experiment_scripts/all4_cyclic_dinov3_frozen/resume_all4_cyclic_dinov3_frozen.sh 100 gaudi
else
  ./experiment_scripts/all4_cyclic_dinov3_frozen/start_all4_cyclic_dinov3_frozen.sh 100 gaudi
fi
