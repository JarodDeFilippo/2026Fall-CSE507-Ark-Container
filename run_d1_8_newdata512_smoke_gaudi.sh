#!/usr/bin/env bash
#SBATCH --job-name=d1-8-newdata512-smoke
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=256G
# Multi-card MEMORY SMOKE for the 512 px Ark-teacher run (chain/newdata512_cropsrc.sh): the real trainer (base/ssl_pretrain.py) through the real launch path (ssl_launch -> gaudi.sh -> torch.distributed.run, one rank per
# allocated card, DDP and HCCL as in the run) with the run's own argument line, for STEPS steps at the per-card batch given as the first argument, into a throwaway run dir. Question answered: does this per-card batch fit on
# this many cards? The card count is the allocation's (ark_device_count, as the run counts it): the header asks for 4; 2 and 8 work the same.
# usage (the repo root on the cluster is /scratch/$USER/2026Fall-CSE507-Ark-Container; this script cd's there):
#   sbatch run_d1_8_newdata512_smoke_gaudi.sh <per-card batch> [steps]            4 cards (this header): sbatch ... run_d1_8_newdata512_smoke_gaudi.sh 32
#   2 cards: sbatch --gres=gpu:hl225:2 -c 36 --mem=128G run_d1_8_newdata512_smoke_gaudi.sh 32          8 cards: sbatch --gres=gpu:hl225:8 -c 144 --mem=480G run_d1_8_newdata512_smoke_gaudi.sh 32
#   interactive (the same script, on the allocated node; the output goes to the terminal): srun -A grp_jliang12 -p gaudi -q public -N 1 --gres=gpu:hl225:4 -c 72 --mem=256G -t 00:30:00 ./run_d1_8_newdata512_smoke_gaudi.sh 32
# One batch per job: a trial is the container start, the model build and the c200 checkpoint load (under 2 min together on 1 card: the 4 h smoke 697293 ran 19,100 steps at 0.747 s after a 30 s first row), the first step's
# graph compile (that 30 s row on 1 card, 53 s on 8, in the CSVs of the 512 px runs), STEPS steps (0.75 s a step at batch 16 on 1 card, 1.35 s on 8 cards; at 32 about twice the compute, an estimate) and a checkpoint save (below):
# about 10 to 15 min by that arithmetic, so three sizes do not fit in the 30 min. Try 32, then 24, then 16.
# The launch line is chain/newdata512_cropsrc.sh's, flag for flag (base/test_smoke_newdata512.py runs both in a fake repo and compares them); only these differ:
#   --batch_size_per_gpu   the argument (the run reads chain/newdata512_cropsrc_batch_per_gpu.txt; neither that file nor chain/newdata512_cropsrc_stop_after.txt is read or needed here)
#   --output_dir           /workspace/smoke_newdata512_b<batch>_x<cards>_<job id>: a name no run uses, new every job, so it can neither collide with ssl_dinov3_cxr_newdata512_cropsrc_arkteacher_seed_100 nor resume
#                          into anything (the script also refuses a directory that exists)
#   --stop_after_epochs    dropped, --max_steps STEPS in its place (the trainer saves one checkpoint at the end; this script deletes that run dir's ckpt/ and export/ after the summary)
#   --datasets_config      dropped: the default datasets_config.yaml (TAIXRay from png512/, the lists the running 512 px run reads). GPU memory does not depend on which TAIX-Ray files are read, so the native list
#                          (base/datasets_config_taix_original.yaml, taix_original_train.csv: not written yet) is not needed. Same seven datasets, same order, same image sizes: the crops are 512 / 224 either way
#   --freeze_backbone_epochs 0 added, --log_peak_memory added (each rank prints its own card's peak; the trainer logs no memory otherwise)
# The backbone is training in every measured step: the trainer's freezes are LR-0 freezes, never requires_grad_(False): the first cycle's backbone (--freeze_backbone_epochs 1, the default; here 0) and the heads' last
# layers (FREEZE_LAST_LAYER_EPOCHS = 1, a constant, which no flag turns off: all STEPS steps are inside cycle 0, so those stay at LR 0 here as in a real first cycle) still get gradients, are clipped, and take an AdamW step that
# allocates their moments, exactly as in later cycles; gradients are dropped only in a lock pass (--lock_half_pass), which the run does not use (--no_lock_half_pass, also here). base/test_smoke_newdata512.py checks it on a tiny run.
# Held in the smoke and in the run alike: the student, the teacher, the 14 heads, DDP's buckets (every head is its own DDP module), HCCL, the activations at this batch, the graph compile.
# NOT held in the smoke: the cycle visits MIMIC first (the largest set) and STEPS steps stay inside its pass, so AdamW has moments for MIMIC's DINO and iBOT heads only; the other 12 heads (6 datasets x 2 x 23,073,024
# parameters x 8 bytes of exp_avg and exp_avg_sq) get theirs in the run's first cycle: ADD 2.06 GiB per card to what is measured here. The teacher blend (end of each dataset's pass) does not run either.
# That holds only while STEPS stays below MIMIC's pass, so a STEPS at or past it is refused (exit 2, before the run dir is made or anything launches; the pass's last step is the blend, hence below, not up to). The pass is counted as the trainer
# counts it, from the list it reads (base/dataset/MIMIC/mimic-cxr-2.0.0-train.csv: a header row, then an image a row; read here at run time): each card gets rows / cards images (DistributedSampler, drop_last) and the loader
# (drop_last) cuts batches of the per-card batch from them, both rounded down: 1441 steps at batch 32 on 8 cards (368,960 rows), 2882 on 4.
# Output (grep -E '^(SMOKE|peak_memory)' on the job's output):
#   peak_memory rank=R step=S GiB=G     the trainer's, every rank at each logged step: the allocator's high-water mark on that rank's card since the start (compile included); GiB=unavailable when the stack answers no memory counter
#                                       (neither torch.hpu.max_memory_allocated nor Habana's memory_stats MaxInUse): the summary says so, and the hl-smi peaks, the speed and the verdict are then all there is
#   SMOKE ... hl-smi card index I       this script's: memory.used of each allocated card as the host's hl-smi reads it every 5 s, peak over the run; it can count what the allocator's figure may not (HCCL's own buffers, the runtime).
#                                       A card shows 98,304 MiB in hl-smi (docs/sol-gaudi-guide.md); how much of it the runtime lets the allocator use is not recorded, so read the peaks against that with the 2.06 GiB added
#   SMOKE speed / SMOKE verdict         s/step and steps/s from the last row of the run dir's ssl_losses.csv; FITS (exit 0), OUT OF MEMORY, or FAILED (anything else: read the log)
# Fail closed as the other wrappers do: ./gaudi_check_cards.sh refuses to launch unless the container sees exactly the allocated cards.
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
BATCH=${1:-} STEPS=${2:-200}
if [[ ! $BATCH =~ ^[1-9][0-9]{0,3}$ || ! $STEPS =~ ^[1-9][0-9]{0,5}$ ]]; then
  echo "usage: run_d1_8_newdata512_smoke_gaudi.sh <per-card batch, 1-4 digits> [steps, default 200]" >&2
  exit 2
fi
source ./chain/ssl_common.sh
source ./gaudi_hlsmi.sh  # gaudi_hlsmi_mem, gaudi_alloc_mem: the busy check's own readers of hl-smi
LR_BASE=0.0001  # as chain/newdata512_cropsrc.sh: the memory does not depend on it, the line stays alike
EPOCHS=1000
export SSL_DATASETS="MIMIC ReXGradient CheXpert TAIXRay PadChest ChestXray14 VinDrCXR"  # as chain/newdata512_cropsrc.sh
RECIPE=(--drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm)
ARK_FLAGS=(--cyclic_ark_teacher --heads per_dataset --no_lock_half_pass --keep_all_checkpoints)
INIT=/workspace/ssl_dinov3_cxr_constlr_arkteacher_seed_100/ckpt/epoch_0200.pth  # in-container path; the head shapes of the run's start
INIT_ARGS=(--init_from_checkpoint "$INIT" --init_new_heads_from MIMIC)
HOST_INIT="${INIT/#\/workspace\//./}"
# fail fast (seconds, not a queue wait): the start checkpoint, the datasets mount, MIMIC's list, and a trainer that knows the flags used below
[[ -s $HOST_INIT ]] || { echo "ERROR: $HOST_INIT (host path for $INIT) not found: nothing to start from" >&2; exit 1; }
[[ -d /scratch/$USER/datasets-new ]] || { echo "ERROR: /scratch/$USER/datasets-new not found: TAIXRay and ReXGradient live there" >&2; exit 1; }
MIMIC_LIST=base/dataset/MIMIC/mimic-cxr-2.0.0-train.csv  # MIMIC's train list in the default datasets config, from the trainer's cwd (base/): the first pass of the cycle
[[ -s $MIMIC_LIST ]] || { echo "ERROR: $MIMIC_LIST not found: its row count sets how many steps stay inside MIMIC's pass" >&2; exit 1; }
for flag in --log_peak_memory --decode_min_side; do
  grep -q -- "\"$flag\"" base/ssl_pretrain.py || { echo "ERROR: base/ssl_pretrain.py here has no $flag: copy the current one (see the file list in the smoke's header)" >&2; exit 1; }
done
./gaudi_check_cards.sh || exit 1
CARDS=$(ark_device_count gaudi)  # the rank count ssl_launch passes as --nproc-per-node: the same function on the same environment
MIMIC_ROWS=$(awk 'END { print NR - 1 }' "$MIMIC_LIST")  # images: one a row after the header
PASS_LEN=$(( MIMIC_ROWS / CARDS / BATCH ))  # MIMIC's pass in steps, rounded down at each division as the sampler and the loader do (header). STEPS stays below it: step PASS_LEN - 1 blends the teacher, step PASS_LEN is the next dataset's
(( STEPS < PASS_LEN )) || { echo "ERROR: $STEPS steps do not stay below MIMIC's pass of $PASS_LEN steps ($MIMIC_ROWS images / $CARDS cards / per-card batch $BATCH, rounded down at each division): its last step blends the teacher, which this smoke leaves out, and past it other heads get AdamW moments that the summary's +2.06 GiB would count twice; use at most $(( PASS_LEN - 1 )) steps" >&2; exit 2; }
RUN=smoke_newdata512_b${BATCH}_x${CARDS}_${SLURM_JOB_ID:-$$}
[[ ! -e $RUN ]] || { echo "ERROR: $RUN exists: this run dir is new for every job" >&2; exit 1; }
mkdir "$RUN"
echo "SMOKE plan: $CARDS cards x per-card batch $BATCH (global batch $(( BATCH * CARDS ))), $STEPS steps, run dir $RUN (throwaway: its ckpt/ and export/ are deleted at the end)"
# the host's view of each allocated card (hl-smi memory.used in MiB, one index=MiB line per card and sample), from the start: the first sample is taken at once
IDS=${SLURM_JOB_GPUS:-}; IDS=${IDS//[[:space:]]/}
( while :; do { hl-smi -Q index,module_id,memory.used -f csv | gaudi_hlsmi_mem | gaudi_alloc_mem "$IDS"; } || true; sleep 5; done ) > "$RUN/hlsmi.txt" 2>/dev/null &
SAMPLER=$!
set +e
( export EXTRA_BINDS=/scratch/$USER/datasets-new:/scratch/$USER/datasets-new:ro  # TAIXRay and ReXGradient, as run_d1_8_newdata512_cropsrc_arkteacher_gaudi.sh mounts them
  ssl_launch "$RUN" --global_crops_size 512 --local_crops_size 224 --gram_teacher_size 512 --decode_min_side 1024 --batch_size_per_gpu "$BATCH" \
    --lr "$LR_BASE" --min_lr "$LR_BASE" --epochs "$EPOCHS" --max_steps "$STEPS" --warmup_epochs 2 --teacher_temp_warmup_epochs 2 \
    "${RECIPE[@]}" --no_gram "${ARK_FLAGS[@]}" "${INIT_ARGS[@]}" --freeze_backbone_epochs 0 --log_peak_memory ) 2>&1 | tee "$RUN/train.log"
RC=${PIPESTATUS[0]}
set -e
kill "$SAMPLER" 2>/dev/null || true; wait "$SAMPLER" 2>/dev/null || true
# the summary, after the run
awk '$1 == "peak_memory" { split($2, r, "="); split($4, g, "="); seen[r[2]] = 1; if (g[2] ~ /^[0-9.]+$/ && (!(r[2] in m) || g[2] + 0 > m[r[2]])) m[r[2]] = g[2] + 0 } END { for (k in seen) if (k in m) printf "SMOKE allocator peak, rank %s: %.2f GiB\n", k, m[k]; else printf "SMOKE allocator peak, rank %s: unavailable (no allocator counter on this stack: the hl-smi peaks below are the measure)\n", k }' "$RUN/train.log" | sort
awk -F= '$2 ~ /^[0-9]+$/ { if (!($1 in m) || $2 + 0 > m[$1]) m[$1] = $2 + 0 } END { for (k in m) printf "SMOKE hl-smi card index %s: peak %d MiB = %.2f GiB of memory.used (an idle card reads 768 MiB)\n", k, m[k], m[k] / 1024 }' "$RUN/hlsmi.txt" | sort
echo "SMOKE not held here: AdamW moments of the other 12 heads, +2.06 GiB per card in the run (this script's header)"
if [[ -s $RUN/ssl_losses.csv ]]; then
  awk -F, -v b="$BATCH" 'NR > 1 { s = $1; t = $14 } END { if (NR > 1 && t > 0) printf "SMOKE speed: row at step %d: %.3f s/step = %.3f steps/s, %.1f images/s per card (the mean over the steps since the row before; the first row, step 0, holds the compile)\n", s, t, 1 / t, b / t }' "$RUN/ssl_losses.csv"
fi
if [[ $RC -eq 0 ]]; then
  echo "SMOKE verdict: FITS: $CARDS cards x per-card batch $BATCH ran $STEPS steps (exit 0)"
elif grep -qiE 'no storage tensor attached to lazy tensor|out of memory|allocation failed|PT_DEVMEM|\boom\b' "$RUN/train.log"; then
  echo "SMOKE verdict: OUT OF MEMORY: $CARDS cards x per-card batch $BATCH (exit $RC); the log above has the error"
else
  echo "SMOKE verdict: FAILED (exit $RC) and not obviously memory: read the log above"
fi
[[ $RUN == smoke_newdata512_* ]] && rm -rf -- "./$RUN/ckpt" "./$RUN/export"  # the end-of-run save: GBs nobody needs
exit "$RC"
