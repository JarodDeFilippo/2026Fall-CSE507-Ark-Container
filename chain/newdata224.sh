#!/usr/bin/env bash
# DINO-X Ark teacher at 224 px with three new datasets: a NEW run, ssl_dinov3_cxr_newdata224_arkteacher_seed_100, started from the finished 224 px Ark-teacher trunk's full checkpoint (INIT below:
# ssl_dinov3_cxr_constlr_arkteacher_seed_100 at epoch 200, student and EMA teacher, backbone and the heads of its four datasets) and trained on seven datasets, the trunk's four plus PadChest, TAIXRay and
# ReXGradient. The recipe is the trunk's, argument for argument (chain/constlr.sh with CONSTLR_NOGRAM=1 CONSTLR_ARK=1 and no lock passes; see there for the reasons): --lr = --min_lr = chain/main_lr.txt with
# --no_lr_scaling, --epochs 1000, DINOv3's --drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm, --no_gram, --cyclic_ark_teacher --heads per_dataset --no_lock_half_pass --keep_all_checkpoints,
# crops 224 / 96 with the Gram size left at its 448 default (the global crops are cut at 448 and resized to 224, and the JPEG decode side is 1024: these are the trainer's STAGE_A_CROPS), no --pre_resize, the default
# warmups (4 cycles of LR, 12 of teacher temperature), 64 per card on 4 cards = global batch 256. Only these differ from the trunk's launch line:
#   data: SSL_DATASETS (ssl_common.sh) names the seven sets, in chain/newdata512.sh's order (the cycle visits them largest first whatever that order is; --resume locks it). The cycle is longer: the trainer's
#       epoch_length is derived from the lists, not set here: per dataset floor(floor(images / 4 cards) / 64) steps, MIMIC 1441 + ReXGradient 933 + CheXpert 872 + PadChest 628 + TAIXRay 537 + ChestXray14 294 + VinDrCXR 58
#       = 4763 steps for 1,220,075 images (the trunk's four: 2665 for 682,686), if the TAIXRay list has the paper's 137,593 train rows (taix_unpack_original.sh prints the count it found). The trainer prints it on its
#       `cycle (--cyclic_ark_teacher)` line.
#   TAIXRay: read from the authors' native-resolution files, not png512/: --datasets_config datasets_config_taix_original.yaml (in base/) is datasets_config.yaml with only TAIXRay's data_dir and train_list changed, to
#       png_original/ and taix_original_train.csv (written on Sol by base/dataset/TAIXRay/taix_unpack_original.sh train, sbatch once before the first launch). The dataset keeps its NAME, so its heads are keyed as in every
#       other run and checkpoint of the seven sets. As in the trunk, the crops are cut from the images as decoded (a PNG at its own size; 16-bit PadChest, TAIXRay and ReXGradient rescaled per image to 8 bit).
#   init: --init_from_checkpoint INIT --init_new_heads_from MIMIC, as chain/newdata512.sh: the checkpoint's student and EMA teacher load strictly but for the three new datasets' DINO and iBOT heads, which start as copies
#       of MIMIC's pair (student's from its student, teacher's from its teacher); the trunk's four keep their own. The checkpoint is read only when the run starts.
# What a fresh start cannot keep (--init_from_checkpoint starts the optimizer and the schedules afresh, at step 0, in cycle 0; the trunk at its epoch 200 was past all of it):
#   LR: warmup from 0 to 0.002 over the first 4 cycles (4 x epoch_length steps), then flat at 0.002; the trunk was flat at 0.002. (chain/newdata512.sh warmed up over 2 cycles to an LR scaled to its batch.)
#   teacher temperature: 0.04 rising linearly to 0.07 over the first 12 cycles, then 0.07; the trunk was at 0.07. (newdata512: over 2 cycles.)
#   cycle 0: the backbone at LR 0 and the heads' last layers at LR 0 (--freeze_backbone_epochs 1, the default, and the trainer's one frozen cycle for the last layers); the trunk had neither frozen.
#   optimizer: AdamW's moments start empty (the trunk's were 533,000 steps old); step 0, not 533,000. Weight decay stays 0.04, constant.
# Settings, read relative to the repo root (cwd), cluster-only like main_lr.txt (never in the repo, so an rsync cannot overwrite a raised value); a missing, empty or invalid file stops the run before anything is launched:
#   chain/main_lr.txt                  the trunk's absolute LR (its file), 0.002
#   chain/newdata224_stop_after.txt    the epoch (= cycle) this invocation stops after: a positive integer, at most the --epochs ceiling of 1000 (never reached, not a plan); not locked on --resume
# The launch also fails early, with an ERROR, if INIT is missing while the run has no epoch checkpoint of its own yet, and if the allocation is not 4 cards (64 x 4 = 256 and the absolute LR are the trunk's; another card
# count would change both). To stop, extend or resume: set the stop epoch in chain/newdata224_stop_after.txt and re-submit the wrapper, run_d1_8_newdata224_arkteacher_gaudi.sh; --resume (always passed by ssl_launch)
# continues from the run's last epoch checkpoint, and locks the data sets and their order, the crops, the batch, the LR, the ceiling, the mode, --init_from_checkpoint and --init_new_heads_from, but not the stop epoch.
# A resume reads no INIT: once the run has an epoch checkpoint of its own the source may be moved or pruned.
# NOT a chain phase: it is never in chain/phases.txt and writes no chain/done marker.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/ssl_common.sh"

EPOCHS=1000  # the --epochs ceiling (never reached), as constlr.sh: the stop epoch's upper limit and the launch line both use it, so they cannot drift
RUN=ssl_dinov3_cxr_newdata224_arkteacher_seed_100
export SSL_DATASETS="MIMIC ReXGradient CheXpert TAIXRay PadChest ChestXray14 VinDrCXR"  # ssl_launch's --data_set flags, in chain/newdata512.sh's order (the trainer visits them largest first anyway)
RECIPE=(--drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm)  # DINOv3's training recipe, as constlr.sh
ARK_FLAGS=(--cyclic_ark_teacher --heads per_dataset --no_lock_half_pass --keep_all_checkpoints)  # the Ark-teacher trunk's mode flags, as constlr.sh's ARK_FLAGS without the lock arm
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

CARDS=$(ark_device_count gaudi)  # the rank count ssl_launch passes as --nproc-per-node: the same function on the same environment
[[ $CARDS == 4 ]] || { echo "ERROR: $CARDS cards in the allocation, this run needs 4: 64 per card x 4 = global batch 256 and the absolute LR of ssl_dinov3_cxr_constlr_arkteacher_seed_100; another card count changes both" >&2; exit 1; }
LR=$(read_setting chain/main_lr.txt '^([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$' 'a positive number such as 2e-3 or 0.0003')
awk -v v="$LR" 'BEGIN { exit !(v + 0 > 0) }' || { echo "ERROR: chain/main_lr.txt holds '$LR', which is not positive" >&2; exit 1; }
STOP_AFTER=$(read_setting chain/newdata224_stop_after.txt '^[1-9][0-9]*$' 'a positive integer')
# awk, not (( )): bash arithmetic wraps silently past 2^63 (18446744073709551617 would read as 1)
awk -v v="$STOP_AFTER" -v cap="$EPOCHS" 'BEGIN { exit !(v + 0 <= cap + 0) }' || { echo "ERROR: chain/newdata224_stop_after.txt holds '$STOP_AFTER', which is above the --epochs ceiling of $EPOCHS" >&2; exit 1; }
echo "run: $RUN (Ark teacher at 224 px as ssl_dinov3_cxr_constlr_arkteacher_seed_100, no Gram teacher: global crops cut at 448 and resized to 224, local 96; data sets $SSL_DATASETS, TAIXRay from the native files (datasets_config_taix_original.yaml); from $INIT with the new heads copied from MIMIC's; DINOv3 recipe: ${RECIPE[*]}; Ark teacher: ${ARK_FLAGS[*]}; stop after epoch $STOP_AFTER from chain/newdata224_stop_after.txt)"
echo "LR: $LR absolute (--no_lr_scaling) at global batch $((64 * CARDS)) (64 x $CARDS cards), flat after a 4-cycle warmup"

ssl_launch "$RUN" --lr "$LR" --min_lr "$LR" --no_lr_scaling --epochs "$EPOCHS" --stop_after_epochs "$STOP_AFTER" \
  "${RECIPE[@]}" --no_gram "${ARK_FLAGS[@]}" --init_from_checkpoint "$INIT" --init_new_heads_from MIMIC --datasets_config datasets_config_taix_original.yaml
