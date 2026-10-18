#!/usr/bin/env bash
# Pilot A: the first 4 epochs of the 40-epoch schedule at --lr 1e-3. 4 = the end of the 4-epoch LR warmup: epoch 1 trains only the heads
# (backbone frozen), so a 2-epoch pilot would train the backbone in epoch 2 alone, at <= 50% of the peak LR. Warmup (4 epochs) and the
# teacher-temperature warmup (12 epochs) do not depend on the total length, so this is also a prefix of a 30-epoch run. The LR is scaled
# by dinov3's sqrt_wrt_1024 rule: at the global batch 4 x 64 = 256 the peak is 1e-3 * 4 * sqrt(256/1024) = 2e-3.
source "$(dirname "${BASH_SOURCE[0]}")/ssl_common.sh"
ssl_launch ssl_dinov3_cxr_pilot_a_seed_100 --lr 1e-3 --epochs 40 --stop_after_epochs 4
