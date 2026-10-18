#!/usr/bin/env bash
# Chain phase "probe_a": the 5-cycle frozen-backbone probe (all4_cyclic_dinov3_probe, arm D's recipe) on pilot A's exported teacher backbone.
# Runs inside an allocation whose driver has already done cd /scratch/$USER/2026Fall-CSE507-Ark-Container, gaudi_check_cards.sh and
# gaudi_verify_busy.sh and exported PT_HPU_LAZY_MODE=1, so none of that is repeated here. Same ARK_* recipe as run_d1_8d_gaudi.sh,
# except the weights (the pilot export, as seen from inside the container).
# Starts the run, or resumes it from its latest checkpoint when an interrupted attempt left its run dir behind (the engine refuses to
# start into an existing run dir). Never deletes anything.
# Exit 0: finished now, or already done. 1: the pilot A export is missing or empty, or the run dir holds no resumable checkpoint.
# Otherwise the start / resume script's exit status.
set -uo pipefail
# the trainer exports teacher_backbone_epoch_%04d.pth counting epochs from 1; pilot A stops after epoch 4, so that is its last export
EXPORT=ssl_dinov3_cxr_pilot_a_seed_100/export/teacher_backbone_epoch_0004.pth
export ARK_BATCH_SIZE=200
export ARK_LR=7e-5
export ARK_TEST_AUGMENT=false
export ARK_EVAL_EVERY=1
export ARK_WARMUP_EPOCHS=2
export ARK_WORKERS=16
export ARK_MODEL=vit_base_dinov3
export ARK_INIT=dinov3
export ARK_PRETRAINED_WEIGHTS=/workspace/$EXPORT
# main_ark.py names the run dir <exp_name>_seed_<seed> in the repo root; the last cycle's student weights mark the phase done
RUN=all4_cyclic_dinov3_probe_a_seed_100
if compgen -G "$RUN/models/weights/epoch_0005/student_cycle_0005_*.pth" > /dev/null; then
  echo "probe_a already done"
  exit 0
fi
# fail fast (seconds, not a whole slot) if pilot A's export is not there (a resume builds the model from it too);
# $EXPORT is relative to the repo root, ARK_PRETRAINED_WEIGHTS is in-container
if [[ ! -s "./$EXPORT" ]]; then
  echo "ERROR: pilot A export ./$EXPORT is missing or empty; the pilot A phase has to finish first" >&2
  exit 1
fi
if [[ -d "$RUN" ]]; then
  # an interrupted attempt: continue it. A dir without a checkpoint can be neither resumed nor started over; leave it for a human.
  if ! compgen -G "$RUN/models/checkpoints/*.pth.tar" > /dev/null; then
    echo "ERROR: run dir ./$RUN exists but holds no resumable checkpoint ($RUN/models/checkpoints/*.pth.tar); neither resuming nor starting over it, nothing deleted: inspect or remove it, then requeue" >&2
    exit 1
  fi
  echo "probe_a: resuming ./$RUN from its latest checkpoint"
  ./experiment_scripts/all4_cyclic_dinov3_probe/resume_all4_cyclic_dinov3_probe.sh 100 gaudi all4_cyclic_dinov3_probe_a "/workspace/$EXPORT"
else
  ./experiment_scripts/all4_cyclic_dinov3_probe/start_all4_cyclic_dinov3_probe.sh 100 gaudi all4_cyclic_dinov3_probe_a "/workspace/$EXPORT"
fi
