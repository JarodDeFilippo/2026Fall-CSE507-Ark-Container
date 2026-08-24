#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SCRIPTS_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$EXPERIMENT_SCRIPTS_ROOT/.." && pwd)

ark_visible_device_count() {
    local visible_devices="$1"
    local device
    local count=0
    local devices=()

    if [[ -z "$visible_devices" || "$visible_devices" == "all" ]]; then
        return 1
    fi
    IFS=',' read -r -a devices <<< "$visible_devices"
    for device in "${devices[@]}"; do
        device="${device//[[:space:]]/}"
        if [[ -n "$device" && "$device" != "-1" && "$device" != "NoDevFiles" ]]; then
            count=$((count + 1))
        fi
    done
    if (( count == 0 )); then
        return 1
    fi
    printf '%s\n' "$count"
}

ark_device_count() {
    local container="$1"
    local wrapper
    local probe
    local output
    local count
    local visible_devices

    case "$container" in
        nvidia)
            wrapper="$REPO_ROOT/nvidia.sh"
            probe='import torch; print("DEVICE_COUNT={}".format(torch.cuda.device_count()))'
            visible_devices="${CUDA_VISIBLE_DEVICES:-}"
            ;;
        gaudi)
            wrapper="$REPO_ROOT/gaudi.sh"
            probe='import habana_frameworks.torch.core; import torch; print("DEVICE_COUNT={}".format(torch.hpu.device_count()))'
            visible_devices="${HABANA_VISIBLE_MODULES:-${HABANA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}}"
            ;;
        *)
            echo "Container must be nvidia or gaudi" >&2
            return 2
            ;;
    esac

    if count=$(ark_visible_device_count "$visible_devices"); then
        printf '%s\n' "$count"
        return 0
    fi

    if ! output=$("$wrapper" python -c "$probe"); then
        echo "Unable to query devices in the $container container" >&2
        return 1
    fi
    count=$(printf '%s\n' "$output" | awk -F= '/^DEVICE_COUNT=/ {value=$2} END {print value}')
    if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
        echo "The $container container reported no usable devices" >&2
        return 1
    fi
    printf '%s\n' "$count"
}

ark_launch_experiment() {
    local container="$1"
    local experiment_name="$2"
    local seed="$3"
    local resume="$4"
    shift 4

    if [[ $# -eq 0 ]]; then
        echo "No datasets were provided" >&2
        return 2
    fi
    if [[ ! "$seed" =~ ^-?[0-9]+$ ]]; then
        echo "Seed must be an integer: $seed" >&2
        return 2
    fi

    local wrapper
    local device
    case "$container" in
        nvidia)
            wrapper="$REPO_ROOT/nvidia.sh"
            device="cuda"
            ;;
        gaudi)
            wrapper="$REPO_ROOT/gaudi.sh"
            device="hpu"
            ;;
        *)
            echo "Container must be nvidia or gaudi" >&2
            return 2
            ;;
    esac

    local device_count
    device_count=$(ark_device_count "$container")

    local training_args=(
        --opt sgd
        --warmup-epochs 20
        --batch_size 200
        --model swin_base
        --init imagenet
        --pretrain_epochs 200
        --pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth
        --momentum_teacher 0.9
        --projector_features 1376
        --ema_mode epoch
        --device "$device"
        --exp_name "$experiment_name"
        --seed "$seed"
    )
    if [[ "$resume" == true ]]; then
        training_args+=(--resume true)
    fi
    training_args+=("$@")

    exec "$wrapper" \
        python -m torch.distributed.run \
        --standalone \
        --nproc-per-node "$device_count" \
        main_ark.py \
        "${training_args[@]}"
}
