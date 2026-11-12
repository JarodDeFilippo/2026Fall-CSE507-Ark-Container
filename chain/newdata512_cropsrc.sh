#!/usr/bin/env bash
# DINO-X Ark teacher at 512 px with three new datasets, the crops cut from the source image: a NEW run, ssl_dinov3_cxr_newdata512_cropsrc_arkteacher_seed_100, that is chain/newdata512.sh's launch line with the changes below and
# nothing else (its header has the recipe, the settings and the checks, none of which is repeated here): the same seven datasets in the same order, the same start (INIT, the finished 224 px Ark-teacher trunk's
# epoch-200 checkpoint, with the three new datasets' heads copied from MIMIC's), recipe, warmups and per-card batch (16 on 8 cards in the run it copies; the wrapper requests 8), and the same 7-day job plus a dependent
# continuation link.
#   LR: a constant base LR of 1e-4 (LR_BASE below, the one place to change it), where newdata512.sh's base is 0.001 (chain/main_lr.txt's 0.002 at batch 256, converted to the base). Set as newdata512.sh sets its LR:
#       --lr = --min_lr = the base, passed without --no_lr_scaling, so the trainer multiplies it by 4 x sqrt(G / 1024) (1.4142135623730951 at G = 128) into the effective peak LR = the effective end LR =
#       1.4142135623730951e-4 at G = 128 (newdata512.sh: 1.4142e-3), flat after the 2-cycle warmup (at another batch G the same rule gives 1e-4 x 4 x sqrt(G / 1024)). chain/main_lr.txt is not read.
#   crops from the source: no --pre_resize. newdata512.sh squashes every image to 512 x 512 first (the Ark loader's convention) and cuts the crops from that; here the random crops are cut from the decoded image itself, as the
#       224 px runs did, so a PNG (PadChest, TAIXRay, ReXGradient, ChestXray14) is cut at its own size, whatever it is.
#   JPEG decode side: --decode_min_side 1024. Without --pre_resize the trainer's rule for these crops (512 global, 224 local, Gram size 512) gives 1344, the local crops' (ceil(224 / (sqrt(0.04) x sqrt(0.75)) / 64) x 64; the global
#       ones need 1216), where the 224 px runs decoded at 1024. This run keeps 1024 by hand, so a MIMIC, CheXpert or VinDrCXR JPEG is decoded at half size (PIL draft, both sides >= 1024) as in those runs and not at its full 2-3k px;
#       a PNG has no draft mode and is decoded whole at either. The price: the smallest local crops (area 0.04, aspect 3/4) hold 177 px of a 1024 px decode for 224 output px (up to 1.26x upsampled; 233 px at 1344).
#   TAIXRay: read from the authors' native-resolution files: --datasets_config datasets_config_taix_original.yaml (in base/) is datasets_config.yaml with only TAIXRay's data_dir and train_list changed, to png_original/ and
#       taix_original_train.csv (written on Sol by base/dataset/TAIXRay/taix_unpack_original.sh train, sbatch once before the first launch). The dataset keeps its NAME, so the heads keyed by it load as before.
#   run: ssl_dinov3_cxr_newdata512_cropsrc_arkteacher_seed_100, with its own settings files chain/newdata512_cropsrc_stop_after.txt and chain/newdata512_cropsrc_batch_per_gpu.txt (cluster-only, as newdata512.sh's).
#   start: CROPSRC_INIT (an in-container path; default the 224 px trunk's epoch-200 checkpoint, as newdata512.sh) and CROPSRC_NEW_HEADS_FROM (default MIMIC: the dataset whose heads the three new datasets' start as copies of) pick the
#       checkpoint to start from. To continue instead from a checkpoint of the seven-dataset 224 px run (ssl_dinov3_cxr_newdata224_arkteacher_seed_100, chain/newdata224.sh), which already has all seven datasets' heads, set
#       CROPSRC_INIT=/workspace/ssl_dinov3_cxr_newdata224_arkteacher_seed_100/ckpt/epoch_<N, four digits>.pth and CROPSRC_NEW_HEADS_FROM=none: none leaves --init_new_heads_from out, which the trainer refuses when no dataset
#       is new (all seven heads then load strictly). Set both on every submission of the run: --resume compares them with the checkpoint's.
# What a fresh start cannot keep, as in newdata512.sh (--init_from_checkpoint starts the optimizer and the schedules afresh, at step 0, in cycle 0): the LR warms up from 0 over 2 cycles to 1.4142e-4 (effective, at batch 128),
# the teacher temperature rises from 0.04 to 0.07 over 2 cycles, cycle 0 trains the heads only (backbone and heads' last layers at LR 0), AdamW's moments start empty. From the 224 px trunk's epoch 200 (LR flat at 0.002 at
# batch 256, temperature 0.07, nothing frozen) this restarts all of it, at a fourteenth of that LR; from a newdata224 checkpoint it restarts what that run had reached.
# The launch fails early, with an ERROR, if INIT is missing while the run has no epoch checkpoint of its own yet, or a settings file is missing or invalid. To stop, extend or resume: set the stop epoch in
# chain/newdata512_cropsrc_stop_after.txt and re-submit the wrapper, run_d1_8_newdata512_cropsrc_arkteacher_gaudi.sh; --resume (always passed by ssl_launch) continues from the run's last epoch checkpoint, and locks the data sets
# and their order, the crops, the decode side, the batch, the LR, the ceiling, the mode, --init_from_checkpoint and --init_new_heads_from, but not the stop epoch. A resume reads no INIT.
# NOT a chain phase: it is never in chain/phases.txt and writes no chain/done marker.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/ssl_common.sh"

LR_BASE=0.0001  # base rate, before the trainer's 4*sqrt(batch/1024) scaling: effective 1.4142e-4 at global batch 128 (16 x 8 cards). One tenth of the base 0.001 that newdata512 and the 224 px trunk used (their effective rates: 1.4142e-3 at batch 128, 2e-3 at batch 256); Meta's constant-rate DINOv3 ViT-7B recipe uses base 5e-5
EPOCHS=1000  # the --epochs ceiling (never reached, not a plan), as constlr.sh and hires.sh: the stop epoch's upper limit and the launch line both use it, so they cannot drift
RUN=ssl_dinov3_cxr_newdata512_cropsrc_arkteacher_seed_100
export SSL_DATASETS="MIMIC ReXGradient CheXpert TAIXRay PadChest ChestXray14 VinDrCXR"  # ssl_launch's --data_set flags, in this order (the trainer visits them largest first anyway), as newdata512.sh
RECIPE=(--drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm)  # DINOv3's training recipe, as constlr.sh's Ark-teacher trunk passes it
ARK_FLAGS=(--cyclic_ark_teacher --heads per_dataset --no_lock_half_pass --keep_all_checkpoints)  # the Ark-teacher trunk's mode flags (constlr.sh's ARK_FLAGS without the lock arm)
INIT=${CROPSRC_INIT:-/workspace/ssl_dinov3_cxr_constlr_arkteacher_seed_100/ckpt/epoch_0200.pth}  # in-container path (/workspace = this repo root), like ssl_common.sh's WEIGHTS
HEADS_FROM=${CROPSRC_NEW_HEADS_FROM:-MIMIC}  # none: INIT has all seven datasets' heads already
INIT_ARGS=(--init_from_checkpoint "$INIT")
[[ $HEADS_FROM == none ]] || INIT_ARGS+=(--init_new_heads_from "$HEADS_FROM")
HOST_INIT="${INIT/#\/workspace\//./}"
# fail fast (seconds, not a 7-day slot) if the checkpoint to start from is not in place. The trainer reads it only when the run starts: once the run has an epoch checkpoint of its own it resumes from
# that one and never reads INIT again. The name test is the trainer's epoch_checkpoints(): epoch_<digits>.pth, a regular file
OWN=
for f in "$RUN"/ckpt/epoch_*.pth; do if [[ -f $f && ${f##*/} =~ ^epoch_[0-9]+\.pth$ ]]; then OWN=$f; fi; done  # if, not &&: a false test leaves status 0 under set -e
if [[ -z $OWN && ! -s $HOST_INIT ]]; then
  echo "ERROR: $HOST_INIT (host path for $INIT) not found and $RUN has no epoch checkpoint of its own yet: nothing to start from" >&2
  exit 1
fi

awk -v v="$LR_BASE" 'BEGIN { exit !(v + 0 > 0) }' || { echo "ERROR: LR_BASE holds '$LR_BASE', which is not positive" >&2; exit 1; }
STOP_AFTER=$(read_setting chain/newdata512_cropsrc_stop_after.txt '^[1-9][0-9]*$' 'a positive integer')
# awk, not (( )): bash arithmetic wraps silently past 2^63 (18446744073709551617 would read as 1)
awk -v v="$STOP_AFTER" -v cap="$EPOCHS" 'BEGIN { exit !(v + 0 <= cap + 0) }' || { echo "ERROR: chain/newdata512_cropsrc_stop_after.txt holds '$STOP_AFTER', which is above the --epochs ceiling of $EPOCHS" >&2; exit 1; }
BATCH=$(read_setting chain/newdata512_cropsrc_batch_per_gpu.txt '^[1-9][0-9]{0,3}$' 'a positive integer of at most 4 digits, no leading zero')
CARDS=$(ark_device_count gaudi)  # the rank count ssl_launch passes as --nproc-per-node: the same function on the same environment
G=$(( BATCH * CARDS ))
EFFECTIVE_LR=$(awk -v b="$LR_BASE" -v g="$G" 'BEGIN { printf "%g", b * 4 * sqrt(g / 1024) }')
HEADS_NOTE="with the new heads copied from $HEADS_FROM's"
[[ $HEADS_FROM != none ]] || HEADS_NOTE="with all its datasets' heads already in it (no --init_new_heads_from)"
echo "run: $RUN (Ark teacher at 512 px, crops cut from the source: no --pre_resize, JPEG decode side 1024 by --decode_min_side; no Gram teacher: global crops 512 cut at 512, local 224; data sets $SSL_DATASETS, TAIXRay from the native files (datasets_config_taix_original.yaml); from $INIT $HEADS_NOTE; DINOv3 recipe: ${RECIPE[*]}; Ark teacher: ${ARK_FLAGS[*]}; stop after epoch $STOP_AFTER from chain/newdata512_cropsrc_stop_after.txt)"
echo "LR: base $LR_BASE -> effective $EFFECTIVE_LR at global batch $G ($BATCH x $CARDS cards; the trainer's sqrt_wrt_1024 rule)"

ssl_launch "$RUN" --global_crops_size 512 --local_crops_size 224 --gram_teacher_size 512 --decode_min_side 1024 --batch_size_per_gpu "$BATCH" \
  --lr "$LR_BASE" --min_lr "$LR_BASE" --epochs "$EPOCHS" --stop_after_epochs "$STOP_AFTER" --warmup_epochs 2 --teacher_temp_warmup_epochs 2 \
  "${RECIPE[@]}" --no_gram "${ARK_FLAGS[@]}" "${INIT_ARGS[@]}" --datasets_config datasets_config_taix_original.yaml
