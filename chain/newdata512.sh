#!/usr/bin/env bash
# DINO-X Ark teacher at 512 px with three new datasets: a NEW run, ssl_dinov3_cxr_newdata512_arkteacher_seed_100, started from the finished 224 px Ark-teacher trunk's full checkpoint (INIT
# below: ssl_dinov3_cxr_constlr_arkteacher_seed_100 at epoch 200, student and EMA teacher, backbone and the heads of its four datasets) and trained on seven datasets, the trunk's four plus
# PadChest, TAIXRay and ReXGradient. It is the Ark-teacher trunk's launch line (chain/constlr.sh with CONSTLR_NOGRAM=1 CONSTLR_ARK=1 and no lock passes: DINOv3's recipe --drop_path_rate 0.3
# --rope_rescale_coords 2 --untie_local_cls_norm, --no_gram, --cyclic_ark_teacher --heads per_dataset --no_lock_half_pass --keep_all_checkpoints) with these changes:
#   crops: global 512, local 224 and --gram_teacher_size 512. --no_gram accepts a Gram size only when it equals the global size, which cuts the global crops at 512 with nothing to resize (the
#       224 trunk cut them at 448 and resized to 224).
#   data: SSL_DATASETS (ssl_common.sh) names the seven sets, in the --data_set order that --resume locks; the cycle visits them largest first whatever that order is. Their image lists and directories
#       are base/datasets_config.yaml's: PadChest's list is base/dataset/PadChest/padchest_train.csv, and TAIXRay and ReXGradient need their entries (data_dir, list) confirmed and their lists written first.
#   heads: the three new datasets' DINO and iBOT heads (student and EMA teacher) start as copies of MIMIC's pair from the checkpoint (--init_new_heads_from MIMIC); the trunk's four keep their own.
#   schedule: --warmup_epochs 2 --teacher_temp_warmup_epochs 2. --init_from_checkpoint starts the optimizer and the schedule afresh at step 0 (and the trainer's default freezes the backbone for the
#       first cycle). The checkpoint is read only when the run starts.
#   LR: chain/main_lr.txt (constlr's absolute LR at its global batch of 256) is converted to the trainer's sqrt_wrt_1024 base as chain/hires.sh does, base = lr / (4 x sqrt(256 / 1024)), and passed as --lr
#       and --min_lr WITHOUT --no_lr_scaling, so the trainer applies base x 4 x sqrt(G / 1024) at this run's global batch G = per-card batch x cards (ark_device_count, as ssl_launch counts them): the
#       main_lr.txt value exactly at G = 256, lower by sqrt(G / 256) below it and higher above, flat after the warmup. The base, the effective LR, G and the card count are printed before the launch.
# Settings, read relative to the repo root (cwd), cluster-only like main_lr.txt (never in the repo, so an rsync cannot overwrite a raised value); a missing, empty or invalid file stops the run before
# anything is launched:
#   chain/main_lr.txt                   constlr's absolute LR (its file), converted to the base as above
#   chain/newdata512_stop_after.txt     the epoch (= cycle) this invocation stops after: a positive integer, at most the --epochs ceiling of 1000 (never reached, not a plan); not locked on --resume
#   chain/newdata512_batch_per_gpu.txt  the per-card batch, 1 to 4 digits; locked on --resume. The SSL memory probe (run_probe_ssl_gaudi.sh) measures one head pair and a Gram teacher; this run holds
#       fourteen heads (12 x 23.1M parameters x 20 bytes = about 5.5 GB a card more) and no Gram teacher, so leave margin.
# The launch fails early, with an ERROR, if INIT is missing (see below). To stop, extend or resume: set the stop epoch in chain/newdata512_stop_after.txt and re-submit the wrapper,
# run_d1_8_newdata512_arkteacher_gaudi.sh; --resume (always passed by ssl_launch) continues from the run's last epoch checkpoint, and locks the data sets and their order, the crops, the batch, the LR, the
# ceiling, the mode and --init_new_heads_from, but not the stop epoch. A resume reads no INIT: once the run has an epoch checkpoint of its own the source may be moved or pruned.
# NOT a chain phase: it is never in chain/phases.txt and writes no chain/done marker.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/ssl_common.sh"

EPOCHS=1000  # the --epochs ceiling (never reached), as constlr.sh and hires.sh: the stop epoch's upper limit and the launch line both use it, so they cannot drift
RUN=ssl_dinov3_cxr_newdata512_arkteacher_seed_100
export SSL_DATASETS="MIMIC ReXGradient CheXpert TAIXRay PadChest ChestXray14 VinDrCXR"  # ssl_launch's --data_set flags, in this order (the trainer visits them largest first anyway)
RECIPE=(--drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm)  # DINOv3's training recipe, as constlr.sh's Ark-teacher trunk passes it
ARK_FLAGS=(--cyclic_ark_teacher --heads per_dataset --no_lock_half_pass --keep_all_checkpoints)  # the Ark-teacher trunk's mode flags (constlr.sh's ARK_FLAGS without the lock arm)
INIT=/workspace/ssl_dinov3_cxr_constlr_arkteacher_seed_100/ckpt/epoch_0200.pth  # in-container path (/workspace = this repo root), like ssl_common.sh's WEIGHTS
HOST_INIT="${INIT/#\/workspace\//./}"
# fail fast (seconds, not a 7-day slot) if the checkpoint to start from is not in place. The trainer reads it only when the run starts: once the run has an epoch checkpoint of its own it resumes from
# that one and never reads INIT again. The name test is the trainer's epoch_checkpoints(): epoch_<digits>.pth, a regular file
OWN=
for f in "$RUN"/ckpt/epoch_*.pth; do if [[ -f $f && ${f##*/} =~ ^epoch_[0-9]+\.pth$ ]]; then OWN=$f; fi; done  # if, not &&: a false test leaves status 0 under set -e
if [[ -z $OWN && ! -s $HOST_INIT ]]; then
  echo "ERROR: $HOST_INIT (host path for $INIT) not found and $RUN has no epoch checkpoint of its own yet: nothing to start from" >&2
  exit 1
fi

LR=$(read_setting chain/main_lr.txt '^([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$' 'a positive number such as 2e-3 or 0.0003')
awk -v v="$LR" 'BEGIN { exit !(v + 0 > 0) }' || { echo "ERROR: chain/main_lr.txt holds '$LR', which is not positive" >&2; exit 1; }
STOP_AFTER=$(read_setting chain/newdata512_stop_after.txt '^[1-9][0-9]*$' 'a positive integer')
# awk, not (( )): bash arithmetic wraps silently past 2^63 (18446744073709551617 would read as 1)
awk -v v="$STOP_AFTER" -v cap="$EPOCHS" 'BEGIN { exit !(v + 0 <= cap + 0) }' || { echo "ERROR: chain/newdata512_stop_after.txt holds '$STOP_AFTER', which is above the --epochs ceiling of $EPOCHS" >&2; exit 1; }
BATCH=$(read_setting chain/newdata512_batch_per_gpu.txt '^[1-9][0-9]{0,3}$' 'a positive integer of at most 4 digits, no leading zero')
REF_BATCH=256  # constlr's global batch: 64 per card on 4 cards (its args.json global_batch_size), the batch main_lr.txt's LR belongs to
CARDS=$(ark_device_count gaudi)  # the rank count ssl_launch passes as --nproc-per-node: the same function on the same environment
G=$(( BATCH * CARDS ))
# %.17g round-trips: base x the trainer's factor 2.0 at G = 256 gives back main_lr.txt's value bit for bit
BASE_LR=$(awk -v lr="$LR" -v ref="$REF_BATCH" 'BEGIN { printf "%.17g", lr / (4 * sqrt(ref / 1024)) }')
EFFECTIVE_LR=$(awk -v b="$BASE_LR" -v g="$G" 'BEGIN { printf "%g", b * 4 * sqrt(g / 1024) }')
echo "run: $RUN (Ark teacher at 512 px, no Gram teacher: global crops 512 cut at 512, local 224; data sets $SSL_DATASETS; from $INIT with the new heads copied from MIMIC's; DINOv3 recipe: ${RECIPE[*]}; Ark teacher: ${ARK_FLAGS[*]}; stop after epoch $STOP_AFTER from chain/newdata512_stop_after.txt)"
echo "LR: base $BASE_LR -> effective $EFFECTIVE_LR at global batch $G ($BATCH x $CARDS cards; the trainer's sqrt_wrt_1024 rule; constlr: $LR at $REF_BATCH)"

ssl_launch "$RUN" --global_crops_size 512 --local_crops_size 224 --gram_teacher_size 512 --batch_size_per_gpu "$BATCH" \
  --lr "$BASE_LR" --min_lr "$BASE_LR" --epochs "$EPOCHS" --stop_after_epochs "$STOP_AFTER" --warmup_epochs 2 --teacher_temp_warmup_epochs 2 \
  "${RECIPE[@]}" --no_gram "${ARK_FLAGS[@]}" --init_from_checkpoint "$INIT" --init_new_heads_from MIMIC
