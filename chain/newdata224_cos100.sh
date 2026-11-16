#!/usr/bin/env bash
# DINO-X Ark teacher with a cosine schedule: a NEW run, ssl_dinov3_cxr_newdata224_cos100_tiedhead_arkteacher_seed_100 (DINO-X registry id ssl-newdata224_cos100_tiedhead_arkteacher, proposed: not yet confirmed), started from the STOCK DINOv3
# ViT-B/16 (the weights ssl_launch loads from ssl_common.sh, --pretrained_weights; no --init_from_checkpoint, so there is no start checkpoint to find and no earlier run's settings to carry) and trained on seven datasets, MIMIC, ReXGradient, CheXpert,
# TAIXRay, PadChest, ChestXray14 and VinDrCXR, cyclically, one dataset to a batch (--cyclic_ark_teacher --no_lock_half_pass: a cycle = one epoch = one release pass over each dataset in turn, largest first, the order and the cycle length derived by the
# trainer from the lists). The Ark teacher has no per-step EMA: at the end of each dataset's pass teacher <- 0.8 x teacher + 0.2 x student (the trainer's default --ark_teacher_momentum), backbone and head. ONE tied DINO + iBOT head (--no_ibot_separate_head:
# the iBOT loss uses the DINO head; --heads stays at its default, shared) is shared by all seven datasets. Images: each is squashed to 512 x 512 (--pre_resize 512; the squash is made from the image as decoded, a PNG at its own size, 16-bit PadChest, TAIXRay and
# ReXGradient rescaled per image to 8 bit first). The student's two global crops are 224 (scale 0.15 to 1) and are cut at 224 straight from the squash (--gram_teacher_size 224: with --no_gram it only sets the size the global crops are cut at before the resize to
# --global_crops_size, so equal to it nothing is resized; the trainer's default of 2 x global, 448, would cut at 448 and resize to 224, and --no_gram refuses any explicit value but the global size: to cut at 448 leave the flag out), its 8 local crops 96 (scale 0.05 to
# 0.15). The teacher (--cyclic_ark_teacher implies --teacher_clean_view) scores the two clean global crops (the student's geometry, crop and rotation, with no intensity change and no blur) and, as an extra DINO target for every student crop, the whole squash
# resized to 448 (--teacher_whole_image_size 448, --whole_image_dino_weight 1.0). Student recipe: RoPE coordinate jitter 2 (--rope_rescale_coords 2: a log-uniform factor in [1/2, 2] on the student's coordinates), the local crops' CLS norm untied (--untie_local_cls_norm), drop path 0.1, no Gram teacher (--no_gram). Schedules, per cycle: the LR warms up from 0 over 10 cycles to
# 2e-3 then follows a cosine down to 1e-6 at the last step of the last cycle, absolute (--no_lr_scaling); the weight decay follows a cosine from 0.04 to 0.4 over the whole schedule (--weight_decay_end); the teacher temperature warms up over 30 cycles (0.04 to 0.07, then
# flat). Augmentation is Ark+'s: rotation up to 10 degrees at p 0.5 (--rotate_limit 10), Ark's intensity step (--intensity_aug ark), no horizontal flip, no blur. NO frozen-backbone first cycle: --freeze_backbone_epochs 0 (Jarod 2026-11-16; the trainer's default of 1 holds the
# backbone at LR 0 through cycle 0); the heads' last layers stay at LR 0 through cycle 0 all the same (FREEZE_LAST_LAYER_EPOCHS = 1 is a trainer constant, not a flag).
# TAIXRay is read from the authors' native-resolution files, not png512/: --datasets_config datasets_config_taix_original.yaml (in base/) is datasets_config.yaml with only TAIXRay's data_dir and train_list changed, to png_original/ and taix_original_train.csv (written
# on Sol by base/dataset/TAIXRay/taix_unpack_original.sh train, sbatch once before the first launch); the dataset keeps its NAME.
# 8 cards or nothing: 64 per card x 8 = global batch 512 (Sinkhorn-Knopp centering and the DINO / iBOT statistics are computed over all ranks), and the absolute LR and the cycle of 2380 steps are set for it (per dataset floor(floor(images / 8 cards) / 64): MIMIC 720 + ReXGradient 466
# + CheXpert 436 + PadChest 314 + TAIXRay 268 + ChestXray14 147 + VinDrCXR 29, if the lists have the plan's sizes, 368,960 + 238,968 + 223,414 + 160,828 + 137,593 + 75,312 + 15,000 images; base/test_ssl_pretrain.py counts it so; the trainer prints the real one on its `cycle
# (--cyclic_ark_teacher)` line), so another card count would change all three: this script refuses one, with an ERROR, before anything is launched.
# No settings files (no chain/main_lr.txt, no stop file): every setting is in this script, and --stop_after_epochs is not passed, so the run trains all 100 cycles of its schedule. To stop earlier, scancel: checkpoints are written at cycle ends only, so a kill mid-cycle loses at most that cycle.
# --resume (always passed by ssl_launch) continues from the run's last epoch checkpoint and locks every setting (the data sets and their order, the batch, the crops, the schedules, every flag below, the seed), so re-submitting run_d1_8_newdata224_cos100_gaudi.sh, the same
# script, is how a killed or timed-out run continues; a run that finished all 100 cycles finds nothing to do and ends at once.
# NOT a chain phase: it is never in chain/phases.txt and writes no chain/done marker.
# Disk: a checkpoint holds the student and the teacher (the backbone, about 86M parameters, and one head of 23.1M each) plus AdamW's two moments, about 1.7 GB by parameter count (an estimate, not measured), and none is pruned (--keep_all_checkpoints).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/ssl_common.sh"

RUN=ssl_dinov3_cxr_newdata224_cos100_tiedhead_arkteacher_seed_100  # confirmed by Jarod 2026-11-16
REGISTRY_ID=ssl-newdata224_cos100_tiedhead_arkteacher  # the DINO-X registry id, proposed (not yet confirmed); printed on the run: line, the trainer does not read it
GRAM_TEACHER_SIZE=224  # confirmed 2026-11-16: the global crops are cut at 224 straight from the 512 squash; 448 (the trainer default 2 x global) would cut at 448 and resize to 224
export SSL_DATASETS="MIMIC ReXGradient CheXpert TAIXRay PadChest ChestXray14 VinDrCXR"  # ssl_launch's --data_set flags, in chain/newdata512.sh's order (the cycle visits them largest first whatever that order is; --resume locks it)
FLAGS=(--no_gram --cyclic_ark_teacher --no_lock_half_pass --keep_all_checkpoints --no_ibot_separate_head
  --drop_path_rate 0.1 --rope_rescale_coords 2 --untie_local_cls_norm
  --lr 2e-3 --min_lr 1e-6 --no_lr_scaling --epochs 100 --warmup_epochs 10 --teacher_temp_warmup_epochs 30 --weight_decay 0.04 --weight_decay_end 0.4
  --pre_resize 512 --global_crops_scale 0.15 1.0 --local_crops_scale 0.05 0.15 --intensity_aug ark --no_blur --no_horizontal_flips --rotate_limit 10
  --freeze_backbone_epochs 0)  # the confirmed flag line (handoff 2026-11-16 section 4) unchanged, plus Jarod's no-frozen-first-cycle
WHOLE=(--teacher_whole_image_size 448 --whole_image_dino_weight 1.0)  # the whole 512 squash resized to 448 as an extra DINO target, at weight 1.0 (the trainer's default, given)

CARDS=$(ark_device_count gaudi)  # the rank count ssl_launch passes as --nproc-per-node: the same function on the same environment
[[ $CARDS == 8 ]] || { echo "ERROR: $CARDS cards in the allocation, this run needs 8: 64 per card x 8 = global batch 512, and the absolute LR and the 2380-step cycle are set for it; another card count changes all three" >&2; exit 1; }
echo "run: $RUN (registry id $REGISTRY_ID; stock DINOv3 ViT-B/16, no start checkpoint; data sets $SSL_DATASETS, TAIXRay from the native files (datasets_config_taix_original.yaml); $CARDS cards x 64 = global batch $((64 * CARDS)); global crops cut at $GRAM_TEACHER_SIZE straight from the squash; no --stop_after_epochs: every cycle of the schedule trains; flags: ${FLAGS[*]} ${WHOLE[*]})"

ssl_launch "$RUN" "${FLAGS[@]}" "${WHOLE[@]}" --gram_teacher_size "$GRAM_TEACHER_SIZE" --datasets_config datasets_config_taix_original.yaml
