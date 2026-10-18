#!/usr/bin/env bash
# High-resolution constant-LR run, "DINO-X ConstLR 512 px" in the ledger naming: DINOv3's constant-schedule recipe at 512 px, a SEPARATE run from the stock
# DINOv3 weights into ssl_dinov3_cxr_constlr512_seed_100. It is chain/constlr.sh with only the crop sizes and the per-card batch changed: global crops 512,
# local crops 224, the Gram teacher at chain/hires_gram_size.txt (1024 = DINOv3's 2x rule, or 512 if the SSL memory probe says 1024 does not fit), the batch
# from chain/hires_batch_per_gpu.txt. Everything else is what constlr.sh passes or relies on: the same stock weights (ssl_common.sh's WEIGHTS), --lr = --min_lr
# (so the LR is flat after the trainer's 4-epoch warmup), the --epochs 1000 ceiling (never reached, not a plan), and the trainer defaults constlr runs with
# (backbone frozen for epoch 1, teacher temperature warmup 12 epochs, teacher momentum 0.999, weight decay 0.04, 8 local crops); the four Stage A train lists
# (ssl_launch). Two settings are carried over to this run's global batch G = per-card batch x cards (ark_device_count, as ssl_launch counts them) instead of
# copied, against constlr's G of 256 (64 per card on 4 cards):
#   LR: constlr's absolute chain/main_lr.txt (the very file constlr.sh reads, referenced, not copied) is converted to the trainer's sqrt_wrt_1024 base,
#       base = lr / (4 x sqrt(256 / 1024)), and passed WITHOUT --no_lr_scaling, so the trainer applies base x 4 x sqrt(G / 1024) to --lr and --min_lr alike:
#       at G = 256 that is constlr's LR exactly, at a smaller G it is lower by sqrt(G / 256).
#   Gram refreshes: constlr's 10000 / 20000 / 30000 global steps x 256 / G, rounded, so they fall at constlr's epoch fractions (about 3.75 / 7.5 / 11.25)
#       although a smaller batch makes an epoch more steps (at 8 per card on 4 cards: 80000 / 160000 / 240000).
# Both lines are printed before the launch; the trainer records the effective LR and its epoch_length / total_steps in args.json.
# JPEG decode: the trainer raises the decode side with the crops (decode_min_side 2368 here, 1344 with a 512 Gram, 1024 in the 224 runs; printed and in args.json), so a 2-3k px
# MIMIC / CheXpert / VinDr file decodes at full size instead of at 1/2: about 4x the decode cost per image of the 224 runs, on top of the larger crops,
# and the data loader (--workers 16 per rank, ssl_launch) may bound throughput; the trainer's data_time column shows it. ChestXray14's 1024 px PNGs
# are what they are: their 1024 Gram crops below full scale are upsampled whatever the decode side.
# Later data expansions start NEW stages from this run's full checkpoint (ssl_dinov3_cxr_constlr512_seed_100/ckpt/epoch_NNNN.pth) via the trainer's
# --init_from_checkpoint, which this run does not use.
# Settings, read relative to the repo root (cwd); a missing, empty or invalid file stops the phase before anything is launched:
#   chain/main_lr.txt              constlr's absolute LR (its file, cluster-only), converted to the base as above
#   chain/hires_stop_after.txt     the epoch this invocation stops after (30 to start), at most the ceiling; not locked on --resume
#   chain/hires_gram_size.txt      the Gram teacher crop size: a multiple of 16 from 512 to 1024 (global <= Gram <= 2 x global)
#   chain/hires_batch_per_gpu.txt  the per-card batch, set from the SSL memory probe (run_probe_ssl_gaudi.sh, its sizes=512:224:<gram> rows); no file, no launch
# --resume (always passed by ssl_launch) locks the crop sizes, the batch, the LR, the ceiling and the refresh list, not --stop_after_epochs. To extend, as
# constlr: raise chain/hires_stop_after.txt, rm chain/done/hires, re-submit; it resumes from its last epoch checkpoint.
# Launch: the phase is the line `hires` in chain/phases.txt, after constlr; with chain/done/constlr in place: sbatch -t 7-00:00:00 run_d1_8_chain_gaudi.sh
# (the constlr extension was sbatch -t 14:00:00 run_d1_8_chain_gaudi.sh; size -t from the probe's step_s x the steps per epoch).
# Two env overrides, both unset in the chain's phase (so its run is unchanged): HIRES_GRAM=<size> replaces chain/hires_gram_size.txt (the same check: a multiple of 16
# from 512 to 1024) and HIRES_RUN_SUFFIX=<_name> (empty, or _ and lowercase letters/digits) goes into the run dir, ssl_dinov3_cxr_constlr512<suffix>_seed_100; --resume locks
# the Gram size, so a different size needs its own suffix. run_d1_8hires_gram512_gaudi.sh sets HIRES_GRAM=512 HIRES_RUN_SUFFIX=_gram512: the twin with the Gram teacher at
# 512 px, its own 4-card job outside the chain, sharing every settings file above with this run. The run name and the source of the Gram size are printed before the launch.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/ssl_common.sh"

EPOCHS=1000  # constlr.sh's --epochs ceiling (never reached): the stop epoch's upper limit and the launch line both use it, so they cannot drift
LR=$(read_setting chain/main_lr.txt '^([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$' 'a positive number such as 2e-3 or 0.0003')
awk -v v="$LR" 'BEGIN { exit !(v + 0 > 0) }' || { echo "ERROR: chain/main_lr.txt holds '$LR', which is not positive" >&2; exit 1; }
STOP_AFTER=$(read_setting chain/hires_stop_after.txt '^[1-9][0-9]*$' 'a positive integer')
# awk, not (( )): bash arithmetic wraps silently past 2^63 (18446744073709551617 would read as 1)
awk -v v="$STOP_AFTER" -v cap="$EPOCHS" 'BEGIN { exit !(v + 0 <= cap + 0) }' || { echo "ERROR: chain/hires_stop_after.txt holds '$STOP_AFTER', which is above the --epochs ceiling of $EPOCHS" >&2; exit 1; }
# (( )) evaluates what it is given as an expression: the pattern first keeps anything but 1-4 digits (an override's 'abc', 'a[$(cmd)]') out of it
GRAM_FROM=chain/hires_gram_size.txt; [[ -z ${HIRES_GRAM:-} ]] || GRAM_FROM=HIRES_GRAM
GRAM=${HIRES_GRAM:-$(read_setting chain/hires_gram_size.txt '^[1-9][0-9]{0,3}$' 'a positive integer of at most 4 digits')}
[[ $GRAM =~ ^[1-9][0-9]{0,3}$ ]] && (( GRAM % 16 == 0 && GRAM >= 512 && GRAM <= 1024 )) || { echo "ERROR: $GRAM_FROM holds '$GRAM', which is not a multiple of 16 from 512 to 1024 (global 512 <= Gram <= 2 x 512)" >&2; exit 1; }
SUFFIX=${HIRES_RUN_SUFFIX:-}
[[ -z $SUFFIX || $SUFFIX =~ ^_[a-z0-9]+$ ]] || { echo "ERROR: HIRES_RUN_SUFFIX holds '$SUFFIX', which is not empty or an underscore and lowercase letters and digits (such as _gram512)" >&2; exit 1; }
RUN=ssl_dinov3_cxr_constlr512${SUFFIX}_seed_100
BATCH=$(read_setting chain/hires_batch_per_gpu.txt '^[1-9][0-9]{0,3}$' 'a positive integer of at most 4 digits (the per-card batch the SSL memory probe found)')
REF_BATCH=256  # constlr's global batch: ssl_launch's 64 per card on the chain's 4 cards (its args.json global_batch_size)
CARDS=$(ark_device_count gaudi)  # the rank count ssl_launch passes as --nproc-per-node: the same function on the same environment
G=$(( BATCH * CARDS ))
# %.17g round-trips: base x the trainer's factor 2.0 at G = 256 gives back main_lr.txt's value bit for bit
BASE_LR=$(awk -v lr="$LR" -v ref="$REF_BATCH" 'BEGIN { printf "%.17g", lr / (4 * sqrt(ref / 1024)) }')
EFFECTIVE_LR=$(awk -v b="$BASE_LR" -v g="$G" 'BEGIN { printf "%g", b * 4 * sqrt(g / 1024) }')
REFRESH=$(awk -v ref="$REF_BATCH" -v g="$G" 'BEGIN { for (i = 1; i <= 3; i++) printf "%s%d", (i > 1 ? "," : ""), int(i * 10000 * ref / g + 0.5) }')
echo "run: $RUN (Gram $GRAM from $GRAM_FROM)"
echo "LR: base $BASE_LR -> effective $EFFECTIVE_LR at global batch $G ($BATCH x $CARDS cards; the trainer's sqrt_wrt_1024 rule; constlr: $LR at $REF_BATCH)"
echo "gram refresh steps: $REFRESH (constlr's 10k/20k/30k at batch $REF_BATCH scaled by $REF_BATCH/$G)"

ssl_launch "$RUN" --pretrained_weights "$WEIGHTS" --global_crops_size 512 --local_crops_size 224 --gram_teacher_size "$GRAM" \
  --lr "$BASE_LR" --min_lr "$BASE_LR" --epochs "$EPOCHS" --stop_after_epochs "$STOP_AFTER" --gram_refresh_steps "$REFRESH" \
  --batch_size_per_gpu "$BATCH"
