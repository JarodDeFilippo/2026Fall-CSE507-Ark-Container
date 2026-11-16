#!/usr/bin/env bash
#SBATCH --job-name=d1-8-newdata224-cos100-smoke
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:8
#SBATCH -c 144
#SBATCH --mem=480G
# Pre-run SMOKE for the cos100 run (chain/newdata224_cos100.sh, submitted by run_d1_8_newdata224_cos100_gaudi.sh): the real trainer (base/ssl_pretrain.py) through the real launch path (ssl_launch -> gaudi.sh -> torch.distributed.run, one rank per allocated card, DDP and HCCL as in
# the run) with the run's own launch line, for ONE cycle over truncated lists, into a throwaway smoke dir. Questions answered: does the run's line fit at 64 per card on 8 cards with the 448 px whole-image pass (peak memory per rank and per card); how long a step takes and how long the
# ranks wait for the loader, per dataset; and what the loss terms are per dataset, dino_whole read beside dino_global and dino_local. 8 cards and no other count, as the run (64 x 8 = global batch 512): the card count is the allocation's (ark_device_count, as the run counts it) and any other one is refused.
# usage (the repo root on the cluster is /scratch/$USER/2026Fall-CSE507-Ark-Container; this script cd's there), no arguments:
#   sbatch run_d1_8_newdata224_cos100_smoke_gaudi.sh
# Two lines in one job, each the run's launch line over the same truncated lists, one cycle each, into <smoke dir>/fp32/ and <smoke dir>/bf16/:
#   fp32   the run's precision: its verdict is this script's exit code. The one line with --export_pass_end_every 1 (the run's value): it writes the run's pass-end exports, after every pass of its cycle but the last (6),
#          whose names and sizes it reports before the cleanup deletes them with export/.
#   bf16   the same line plus --bf16, REPORT ONLY: it always runs, even when the fp32 line failed, and its result never changes the exit code. The run is fp32, so the fp32 numbers are the run's and the bf16 ones are for reference: under --bf16 the loss math, KoLeo included, also runs in bf16 under autocast. No pass-end exports.
# The launch line is chain/newdata224_cos100.sh's, flag for flag (its FLAGS, WHOLE, GRAM_TEACHER_SIZE and SSL_DATASETS are repeated below; base/test_newdata.py runs both in a fake repo and compares them); only these differ:
#   --output_dir          /workspace/<smoke dir>/fp32 and /workspace/<smoke dir>/bf16, the smoke dir being smoke_newdata224_cos100_x8_<job id>: a name no run uses, new every job, so it can neither collide with ssl_dinov3_cxr_newdata224_cos100_tiedhead_arkteacher_seed_100
#                         nor resume into anything (the script also refuses a smoke dir that exists)
#   --datasets_config     /workspace/<smoke dir>/datasets_config_smoke.yaml in place of datasets_config_taix_original.yaml: the same seven entries, each data_dir unchanged and each train_list the truncated copy in <smoke dir>/lists/ (in-container paths: /workspace is the repo root)
#   --stop_after_epochs 1 --print_freq 5 --log_peak_memory   added: one cycle, a log row every 5 steps, and each rank prints its own card's peak (the trainer logs no memory otherwise)
#   --bf16                added to the second line only
#   --export_pass_end_every 1   the run's own flag and value, given to the first line only (the second writes no pass-end export)
# Truncated lists: each dataset's pass gets STEPS_PER_DATASET = 60 steps (a setting below, with BATCH = 64, ssl_launch's per-card batch): its train list is cut to its first ROWS = STEPS_PER_DATASET x cards x BATCH = 60 x 8 x 64 = 30,720 images (plus its header row, which every list has but
# ChestXray14's and VinDrCXR's: ssl_dinov3/data.py image_paths), so each of the 8 ranks gets 3,840 images (DistributedSampler, drop_last) and the loader (drop_last) cuts 60 batches of 64 from them. A list shorter than ROWS is kept whole: VinDrCXR has 15,000 images, so it keeps its whole list,
# 29 steps on 8 cards, as in the run. The cycle is 6 x 60 + 29 = 389 steps. The first rows of a list are not a random sample of it, so a dataset's speed and loss here are indicative of its pass, not a measurement of it.
# What the cycle holds: all seven datasets in turn, largest first by image count (a tie keeps the --data_set order: the six cut lists tie, so TAIXRay's pass comes before PadChest's here, as the run's sizes do not), a teacher blend (teacher <- 0.8 x teacher + 0.2 x student) at the end of each pass,
# and a checkpoint save at the cycle's end (ckpt/ and export/ in each line's dir, which this script deletes after the line's summary: GBs nobody needs; train.log, hlsmi.txt, ssl_losses.csv and args.json stay). With ONE shared head every optimizer moment exists after the first step: the trainer's freezes are
# LR-0 freezes, never requires_grad_(False), so the head's last layer (LR 0 through cycle 0, FREEZE_LAST_LAYER_EPOCHS = 1) still gets gradients, is clipped and takes an AdamW step that allocates its moments, exactly as in later cycles, and the backbone trains from step 0 (--freeze_backbone_epochs 0). So the smoke's memory
# is the run's: no per-dataset-head caveat as in run_d1_8_newdata512_smoke_gaudi.sh, whose 14 heads got their moments dataset by dataset. Held in the smoke and in the run alike: the student, the teacher, the one head, DDP's buckets, HCCL, the activations at 64 per card with the 448 px whole-image forward, the graph compiles, the blends and the save.
# Output (grep -E '^(SMOKE|peak_memory)' on the job's output):
#   peak_memory rank=R step=S GiB=G         the trainer's, every rank at each logged step: the allocator's high-water mark on that rank's card since the start (compile included); GiB=unavailable when the stack answers no memory counter (neither torch.hpu.max_memory_allocated nor Habana's memory_stats MaxInUse):
#                                           the summary says so, and the hl-smi peaks, the speed and the verdict are then all there is
#   SMOKE <tag> allocator peak, rank R      the largest of those values per rank, <tag> being fp32 or bf16 on every SMOKE line below
#   SMOKE <tag> hl-smi card index I         memory.used of each allocated card as the host's hl-smi reads it every 5 s, peak over the line; it can count what the allocator's figure may not (HCCL's own buffers, the runtime). A card shows 98,304 MiB in hl-smi (see
#                                           run_d1_8_newdata512_smoke_gaudi.sh); how much of it the runtime lets the allocator use is not recorded, so read the peaks against that
#   SMOKE <tag> speed, projection           the mean step_time and data_time (the wait for the loader) over the rows of ssl_losses.csv after the first, step 0, which holds the compile; each row is itself the mean over the steps since the row before. The projection is that mean x 238000 steps (100 cycles x 2380)
#                                           in hours: arithmetic on this smoke's mean, not a measurement of the run
#   SMOKE <tag> dataset <name>:release      one line per dataset, in the cycle's order: its rows (step 0's row not counted), mean s/step and data s/step, and its last row's dino_global, dino_local, dino_whole and total
#   SMOKE <tag> pass-end export <file>: B bytes   each pass-end export the line wrote (the fp32 line only), before the cleanup deletes it
#   SMOKE <tag> pass-end export time        from the trainer's lines (each save's seconds): the mean per file, per exporting cycle (6 files) and over 100 cycles, arithmetic on this smoke's mean. The saves
#                                           (pass end and cycle end) are in no row's step_time, so the speed and its projection above are the training's alone
#   SMOKE <tag> verdict                     FITS (the line exited 0), OUT OF MEMORY, or FAILED (anything else: read the log); the bf16 one says (report only)
# The 1 h limit is a guess with margin, not measured: two lines, each a container start, the model build and the stock weights load, the first step's graph compile, 389 steps and the cycle-end save. The fp32 line runs first, so a time-out costs the bf16 report, not the verdict.
# Fail closed as the other wrappers do: ./gaudi_check_cards.sh refuses to launch unless the container sees exactly the allocated cards.
export PT_HPU_LAZY_MODE=1
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
source ./chain/ssl_common.sh
source ./gaudi_hlsmi.sh  # gaudi_hlsmi_mem, gaudi_alloc_mem: the busy check's own readers of hl-smi
STEPS_PER_DATASET=60  # steps each dataset's pass gets: its list is cut to STEPS_PER_DATASET x cards x BATCH rows
BATCH=64  # ssl_launch's per-card batch (chain/ssl_common.sh): the run's
RUN_STEPS=238000  # the run's steps, for the projection: 100 cycles x 2380 steps (chain/newdata224_cos100.sh's header)
# the run's launch line, repeated from chain/newdata224_cos100.sh
export SSL_DATASETS="MIMIC ReXGradient CheXpert TAIXRay PadChest ChestXray14 VinDrCXR"
GRAM_TEACHER_SIZE=224
FLAGS=(--no_gram --cyclic_ark_teacher --no_lock_half_pass --keep_all_checkpoints --no_ibot_separate_head
  --drop_path_rate 0.1 --rope_rescale_coords 2 --untie_local_cls_norm
  --lr 2e-3 --min_lr 1e-6 --no_lr_scaling --epochs 100 --warmup_epochs 10 --teacher_temp_warmup_epochs 30 --weight_decay 0.04 --weight_decay_end 0.4
  --pre_resize 512 --global_crops_scale 0.15 1.0 --local_crops_scale 0.05 0.15 --intensity_aug ark --no_blur --no_horizontal_flips --rotate_limit 10
  --freeze_backbone_epochs 0)
WHOLE=(--teacher_whole_image_size 448 --whole_image_dino_weight 1.0)
CONFIG=base/datasets_config_taix_original.yaml  # the run's --datasets_config, which the trainer reads from its cwd, base/
# fail fast (seconds, not a queue wait): the datasets mount, a trainer that knows the flags used below, the cards
[[ -d /scratch/$USER/datasets-new ]] || { echo "ERROR: /scratch/$USER/datasets-new not found: TAIXRay and ReXGradient live there" >&2; exit 1; }
for flag in --teacher_whole_image_size --log_peak_memory --export_pass_end_every; do
  grep -q -- "\"$flag\"" base/ssl_pretrain.py || { echo "ERROR: base/ssl_pretrain.py here has no $flag: copy the current one" >&2; exit 1; }
done
./gaudi_check_cards.sh || exit 1
CARDS=$(ark_device_count gaudi)  # the rank count ssl_launch passes as --nproc-per-node: the same function on the same environment
[[ $CARDS == 8 ]] || { echo "ERROR: $CARDS cards in the allocation, this smoke needs 8: it is the run's launch line, 64 per card x 8 = global batch 512, and the lists are cut for 8 ranks" >&2; exit 1; }
ROWS=$(( STEPS_PER_DATASET * CARDS * BATCH ))  # each of the CARDS ranks gets ROWS / CARDS images and the loader cuts STEPS_PER_DATASET batches of BATCH from them, both rounded down
SMOKE=smoke_newdata224_cos100_x${CARDS}_${SLURM_JOB_ID:-$$}
[[ ! -e $SMOKE ]] || { echo "ERROR: $SMOKE exists: this smoke dir is new for every job" >&2; exit 1; }
# the run's config, parsed by awk: an entry is "Name:", then "  data_dir: X" and "  train_list: Y" (a relative train_list is relative to base/). Every dataset's list is checked before anything is made
[[ -s $CONFIG ]] || { echo "ERROR: $CONFIG not found: the run's config, which holds the seven datasets' lists" >&2; exit 1; }
NAMES=() DIRS=() SRCS=()
read -ra WANTED <<< "$SSL_DATASETS"
for name in "${WANTED[@]}"; do
  entry=$(awk -v want="$name" '/^[^[:space:]#][^:]*:[[:space:]]*$/ { sec = $1; sub(/:$/, "", sec); next } sec == want && $1 == "data_dir:" { dir = $2 } sec == want && $1 == "train_list:" { list = $2 } END { if (dir != "" && list != "") print dir "\t" list }' "$CONFIG")
  [[ -n $entry ]] || { echo "ERROR: $CONFIG has no entry with a data_dir and a train_list for $name" >&2; exit 1; }
  IFS=$'\t' read -r dir list <<< "$entry"
  if [[ $list == /* ]]; then src=$list; else src=base/$list; fi
  [[ -s $src ]] || { echo "ERROR: $src (the train list of $name in $CONFIG) is missing or empty" >&2; exit 1; }
  NAMES+=("$name"); DIRS+=("$dir"); SRCS+=("$src")
done
mkdir "$SMOKE" "$SMOKE/lists"
echo "# $CONFIG's entries for $SSL_DATASETS, each data_dir unchanged and each train_list cut to its first $ROWS rows (plus its header row, if it has one) by run_d1_8_newdata224_cos100_smoke_gaudi.sh; /workspace is the repo root" > "$SMOKE/datasets_config_smoke.yaml"
for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  if [[ $name == ChestXray14 || $name == VinDrCXR ]]; then keep=$ROWS; else keep=$(( ROWS + 1 )); fi  # these two lists have no header row (ssl_dinov3/data.py image_paths), the other five have one
  head -n "$keep" "${SRCS[$i]}" > "$SMOKE/lists/$name.list"
  printf '%s:\n  data_dir: %s\n  train_list: /workspace/%s/lists/%s.list\n\n' "$name" "${DIRS[$i]}" "$SMOKE" "$name" >> "$SMOKE/datasets_config_smoke.yaml"
done
echo "SMOKE plan: $CARDS cards x per-card batch $BATCH (global batch $(( BATCH * CARDS ))), one cycle a line over lists cut to $ROWS rows ($STEPS_PER_DATASET steps a dataset; a shorter list is kept whole), smoke dir $SMOKE (throwaway: each line's ckpt/ and export/ are deleted at its end); lines: fp32, then bf16 (report only)"

# smoke_line <tag> [more trainer flags]: one line of the smoke, the run's launch line over the cut lists for one cycle into $SMOKE/<tag>, then its summary. Sets RC to the trainer's exit code and always returns 0, so the next line runs
smoke_line() {
  local tag=$1 ids sampler note=""
  shift
  [[ $tag == fp32 ]] || note=" (report only)"  # fp32 is the run's precision: its verdict is this script's. Any other line is for reference
  mkdir "$SMOKE/$tag"
  # the host's view of each allocated card (hl-smi memory.used in MiB, one index=MiB line per card and sample), from the start: the first sample is taken at once
  ids=${SLURM_JOB_GPUS:-}; ids=${ids//[[:space:]]/}
  ( while :; do { hl-smi -Q index,module_id,memory.used -f csv | gaudi_hlsmi_mem | gaudi_alloc_mem "$ids"; } || true; sleep 5; done ) > "$SMOKE/$tag/hlsmi.txt" 2>/dev/null &
  sampler=$!
  set +e
  ( export EXTRA_BINDS=/scratch/$USER/datasets-new:/scratch/$USER/datasets-new:ro  # TAIXRay and ReXGradient, as run_d1_8_newdata224_cos100_gaudi.sh mounts them
    ssl_launch "$SMOKE/$tag" "${FLAGS[@]}" "${WHOLE[@]}" --gram_teacher_size "$GRAM_TEACHER_SIZE" --datasets_config "/workspace/$SMOKE/datasets_config_smoke.yaml" \
      --stop_after_epochs 1 --print_freq 5 --log_peak_memory "$@" ) 2>&1 | tee "$SMOKE/$tag/train.log"
  RC=${PIPESTATUS[0]}
  set -e
  kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true
  # the summary, after the line
  awk -v t="$tag" '$1 == "peak_memory" { split($2, r, "="); split($4, g, "="); seen[r[2]] = 1; if (g[2] ~ /^[0-9.]+$/ && (!(r[2] in m) || g[2] + 0 > m[r[2]])) m[r[2]] = g[2] + 0 } END { for (k in seen) if (k in m) printf "SMOKE %s allocator peak, rank %s: %.2f GiB\n", t, k, m[k]; else printf "SMOKE %s allocator peak, rank %s: unavailable (no allocator counter on this stack: the hl-smi peaks below are the measure)\n", t, k }' "$SMOKE/$tag/train.log" | sort
  awk -F= -v t="$tag" '$2 ~ /^[0-9]+$/ { if (!($1 in m) || $2 + 0 > m[$1]) m[$1] = $2 + 0 } END { for (k in m) printf "SMOKE %s hl-smi card index %s: peak %d MiB = %.2f GiB of memory.used (an idle card reads 768 MiB)\n", t, k, m[k], m[k] / 1024 }' "$SMOKE/$tag/hlsmi.txt" | sort
  if [[ -s $SMOKE/$tag/ssl_losses.csv ]]; then  # columns: 3 dino_global, 4 dino_local, 8 total, 14 step_time, 15 data_time, 16 dataset, 17 dino_whole; the means leave out the first data row, step 0, which holds the compile
    awk -F, -v t="$tag" -v steps="$RUN_STEPS" '
      NR == 1 { next }
      { d = $16; if (!(d in seen)) { seen[d] = 1; order[++k] = d }
        if (NR > 2) { n++; st += $14; dt += $15; rows[d]++; dst[d] += $14; ddt[d] += $15 }
        dg[d] = $3; dl[d] = $4; dw[d] = $17; tot[d] = $8 }
      END {
        if (n) {
          printf "SMOKE %s speed: %.3f s/step, data wait %.3f s/step (the mean of the %d rows after step 0, which holds the compile; each row is the mean over the steps since the row before)\n", t, st / n, dt / n, n
          printf "SMOKE %s projection: %.1f h = %.3f s/step x %d steps / 3600: arithmetic on the mean of this smoke, not a measurement of the run\n", t, st / n * steps / 3600, st / n, steps
        } else printf "SMOKE %s speed: no row after step 0 in ssl_losses.csv\n", t
        for (i = 1; i <= k; i++) {
          d = order[i]
          printf "SMOKE %s dataset %s: ", t, d
          if (rows[d]) printf "%d rows, mean %.3f s/step, data wait %.3f s/step; ", rows[d], dst[d] / rows[d], ddt[d] / rows[d]; else printf "no row after step 0; "
          printf "last row: dino_global %.4f, dino_local %.4f, dino_whole %s, total %.4f\n", dg[d], dl[d], (dw[d] == "" ? "n/a" : sprintf("%.4f", dw[d])), tot[d]
        }
      }' "$SMOKE/$tag/ssl_losses.csv"
  fi
  if [[ $RC -eq 0 ]]; then
    echo "SMOKE $tag verdict$note: FITS: $CARDS cards x per-card batch $BATCH ran one cycle over the cut lists (exit 0)"
  elif grep -qiE 'no storage tensor attached to lazy tensor|out of memory|allocation failed|PT_DEVMEM|\boom\b' "$SMOKE/$tag/train.log"; then
    echo "SMOKE $tag verdict$note: OUT OF MEMORY: $CARDS cards x per-card batch $BATCH (exit $RC); the log above has the error"
  else
    echo "SMOKE $tag verdict$note: FAILED (exit $RC) and not obviously memory: read the log above"
  fi
  awk -v t="$tag" '$1 == "pass-end" && $2 == "export" && $3 == "after" && $NF == "write)" { s = $(NF - 3); sub(/^\(/, "", s); n++; sum += s }
    END { if (n) printf "SMOKE %s pass-end export time: %d files, mean %.1f s to write each, %.1f min per exporting cycle (6 files): %.2f h over 100 cycles at --export_pass_end_every 1, half that at 2 (arithmetic on this smoke'"'"'s mean)\n", t, n, sum / n, sum / n * 6 / 60, sum / n * 600 / 3600 }' "$SMOKE/$tag/train.log"
  for f in "$SMOKE/$tag"/export/pass_end/*.pth; do  # --export_pass_end_every: what the run writes after each pass but the last, as written here
    if [[ -f $f ]]; then echo "SMOKE $tag pass-end export ${f##*/}: $(wc -c < "$f" | tr -d '[:space:]') bytes"; fi
  done
  if [[ $SMOKE == smoke_newdata224_cos100_* ]]; then rm -rf -- "./$SMOKE/$tag/ckpt" "./$SMOKE/$tag/export"; fi  # the cycle-end save: GBs nobody needs
  return 0
}

smoke_line fp32 --export_pass_end_every 1
RC_FP32=$RC
smoke_line bf16 --bf16
exit "$RC_FP32"
