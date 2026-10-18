#!/usr/bin/env bash
# Standalone phase script "armD_long" (D-long): arm D's frozen-backbone linear probe on the stock DINOv3 weights, run for 50 cycles instead of 5 (seed 100, ONE Gaudi card).
# NOT a chain phase: it is deliberately absent from chain/phases.txt, so the chain driver never runs it. It lives in chain/ only so that
# base/test_chain_scripts.py can test it like the chain phase scripts.
# run_d1_8dl_gaudi.sh runs it inside an allocation that has already done cd /scratch/$USER/2026Fall-CSE507-Ark-Container, gaudi_check_cards.sh and
# gaudi_verify_busy.sh and exported PT_HPU_LAZY_MODE=1, so none of that is repeated here. Same ARK_* recipe as chain/armD.sh, except ARK_EVAL_EVERY.
# Starts the run, or resumes it from its latest checkpoint when an interrupted attempt left its run dir behind (the engine refuses to
# start into an existing run dir). Never deletes anything.
# Exit 0: finished now, or already done. 1: the stock weights are missing, or the run dir holds no resumable checkpoint.
# Otherwise the start / resume script's exit status.
set -uo pipefail
export ARK_BATCH_SIZE=200
export ARK_LR=7e-5
export ARK_TEST_AUGMENT=false
# evaluate every 5 cycles plus the final one (engine.py always evaluates the last cycle), like arms B and C; arm D evaluated every cycle
export ARK_EVAL_EVERY=5
export ARK_WARMUP_EPOCHS=2
export ARK_WORKERS=16
export ARK_MODEL=vit_base_dinov3
export ARK_INIT=dinov3
export ARK_PRETRAINED_WEIGHTS=/workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
# main_ark.py names the run dir <exp_name>_seed_<seed> in the repo root; the last cycle's student weights mark the phase done
RUN=all4_cyclic_dinov3_frozen_long_seed_100
if compgen -G "$RUN/models/weights/epoch_0050/student_cycle_0050_*.pth" > /dev/null; then
  echo "armD_long already done"
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
  echo "armD_long: resuming ./$RUN from its latest checkpoint"
  ./experiment_scripts/all4_cyclic_dinov3_frozen_long/resume_all4_cyclic_dinov3_frozen_long.sh 100 gaudi
else
  ./experiment_scripts/all4_cyclic_dinov3_frozen_long/start_all4_cyclic_dinov3_frozen_long.sh 100 gaudi
fi
