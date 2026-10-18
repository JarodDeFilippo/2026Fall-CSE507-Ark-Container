#!/usr/bin/env bash
#SBATCH --job-name=d1-6-eval
#SBATCH -A grp_jliang12
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 01:30:00
# measured 2026-10-17: a ten-crop eval of one cycle takes ~26 min (sacct 65964098/99); the class QOS MaxGRESRunMinsPerUser counts the REQUESTED limit, so 4 h limits allowed only 4 evals at a time.
#SBATCH -G a100:1
#SBATCH --constraint=a100_80
#SBATCH -c 16
#SBATCH --mem=64G
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <run_dir_name> <cycle> [dataset ...]  (default datasets: VinDrCXR ChestXray14 MIMIC CheXpert; cycle is zero-padded for you)" >&2
    echo "ARK_TEST_AUGMENT=true (default): ten-crop, results in cycle_<NNNN>_<run_dir_name>.csv; false: center crop, results in cycle_<NNNN>_<run_dir_name>.centercrop.csv" >&2
    exit 2
fi
run="$1"
cycle=$(printf '%04d' "$((10#$2))")
shift 2
if [[ $# -eq 0 ]]; then
    set -- VinDrCXR ChestXray14 MIMIC CheXpert
fi
dataset_args=()
for dataset in "$@"; do
    dataset_args+=(--data_set "$dataset")
done
# ARK_TEST_AUGMENT picks the protocol: true (default) = ten-crop, whose result file keeps the name cycle_<NNNN>_<run dir name>.csv; false = center crop, whose
# file is cycle_<NNNN>_<run dir name>.centercrop.csv (main_ark.py --test_output_suffix), so the two never collide. Only those two words are accepted: main_ark.py
# reads 0 / no / f / n as false and any other word as false too, which would write center-crop results into the ten-crop file.
test_augment="${ARK_TEST_AUGMENT:-true}"
suffix_args=()
case "$test_augment" in
    true) ;;
    false) suffix_args=(--test_output_suffix .centercrop) ;;
    *) echo "ARK_TEST_AUGMENT must be true (ten-crop, the default) or false (center crop), not '$test_augment'" >&2; exit 2 ;;
esac
cd /scratch/$USER/2026Fall-CSE507-Ark-Container
exec ./nvidia.sh python main_ark.py \
    --mode test \
    --test_augment "$test_augment" \
    ${suffix_args[@]+"${suffix_args[@]}"} \
    --pretrained_weights "/workspace/${run}/models/weights/epoch_${cycle}" \
    "${dataset_args[@]}" \
    --model "${ARK_MODEL:-swin_base}" \
    --projector_features 1376 \
    --device "${ARK_DEVICE:-cuda}"
