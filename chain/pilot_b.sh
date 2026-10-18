#!/usr/bin/env bash
# Pilot B: the first 4 epochs of the 40-epoch schedule at an absolute peak LR of 3e-4 (no sqrt_wrt_1024 scaling). 4 = the end of the
# 4-epoch LR warmup, as in pilot_a.sh (a 2-epoch pilot trains the backbone only in epoch 2, at <= 50% of the peak LR).
source "$(dirname "${BASH_SOURCE[0]}")/ssl_common.sh"
ssl_launch ssl_dinov3_cxr_pilot_b_seed_100 --lr 3e-4 --no_lr_scaling --epochs 40 --stop_after_epochs 4
