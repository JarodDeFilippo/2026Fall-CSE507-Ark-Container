#!/usr/bin/env bash
# The STUDENT backbone of a Stage A epoch checkpoint as a flat backbone file the probes can take (ssl_pretrain.py exports only the EMA teacher's, export/teacher_backbone_epoch_NNNN.pth): base/ssl_dinov3/export_student.py as ONE short
# CPU job on the class account, so the login node does no load. NOT an sbatch script: run it on the login node. It submits, prints one line with the job id and exits; nothing is left running here.
# usage: ./export_student.sh <run dir name under the repo root> <epoch: 1 to 4 digits, zero-padded or not>
#   e.g. ./export_student.sh ssl_dinov3_cxr_newdata512_arkteacher_seed_100 20
# The job reads <run>/ckpt/epoch_NNNN.pth and <run>/export/teacher_backbone_epoch_NNNN.pth and writes <run>/export/student_backbone_epoch_NNNN.pth, which submit_probe_class.sh and launch_dense.sh take as
# /workspace/<run>/export/student_backbone_epoch_NNNN.pth, as they take the teacher's. export_student.py first checks that its mapping, applied to the checkpoint's teacher, reproduces the teacher export tensor for tensor,
# and it never overwrites a file. This wrapper checks what needs no torch (both inputs are there, the output is not), so a typo fails in a second and not after the queue. The job's log is slurm-<id>.out in the repo root.
# The header is run_emb_fit_class.sh's (CPU only, no --gres): -A class_cse49478170fall2026 -p public,htc -q class -N 1, then -c 2 --mem=16G -t 00:10:00. The memory is sized from the checkpoint: torch.load reads the whole file into
# memory, and the largest is the 7-dataset 512 px run's, about 6.5 GB (run_d1_8_newdata512_arkteacher_gaudi.sh; the 224 px Ark-teacher trunk's is about 4.3 GB). The exporter drops the checkpoint once it holds the two backbones, so the
# peak is the load: the file plus about 2 GB (three 343 MB backbone copies, the interpreter, torch). On a Mac, with a full-size ViT-B student and teacher in a 5.0 GB stand-in file, it peaked at 7.0 GB (macOS peak memory footprint)
# and took 6 s (2026-11-12), so 6.5 GB comes to about 8.5 GB and 16G is twice that: an extrapolation, not a measurement on Sol. Ten minutes is for reading the file from /scratch and writing 343 MB.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1   # the repo root: the job runs here, and sbatch writes slurm-<id>.out in the directory it is run from
RUN_RE='^[A-Za-z0-9][A-Za-z0-9_.-]*$'   # the run dir is a directory name under the repo root: no path separators, no spaces
EPOCH_RE='^[0-9]{1,4}$'
if [[ $# -ne 2 || ! $1 =~ $RUN_RE || ! $2 =~ $EPOCH_RE ]]; then
  echo "Usage: ./export_student.sh <run dir name under the repo root: letters, digits, . _ - only> <epoch: 1 to 4 digits>" >&2
  exit 2
fi
printf -v epoch %04d "$((10#$2))"   # 10#: base 10, so 020 is twenty and not octal sixteen
ckpt=$1/ckpt/epoch_$epoch.pth
teacher=$1/export/teacher_backbone_epoch_$epoch.pth
out=$1/export/student_backbone_epoch_$epoch.pth
for file in "$ckpt" "$teacher"; do
  [[ -f $file ]] || { echo "ERROR: $file not found in $PWD; nothing was submitted" >&2; exit 1; }
done
[[ ! -e $out ]] || { echo "ERROR: $out exists and export_student.py will not overwrite it; nothing was submitted" >&2; exit 1; }
# the paths are the container's: nvidia.sh binds the repo root as /workspace and starts in /workspace/base, where python -m finds ssl_dinov3
job=$(sbatch --parsable -J "student-export-$epoch" -A class_cse49478170fall2026 -p public,htc -q class -N 1 -c 2 --mem=16G -t 00:10:00 \
  --wrap "cd $(printf %q "$PWD") && ./nvidia.sh python -m ssl_dinov3.export_student --ckpt /workspace/$ckpt --check /workspace/$teacher --out /workspace/$out") \
  || { echo "ERROR: sbatch refused the job; nothing was submitted" >&2; exit 1; }
job=${job%%;*}   # --parsable prints "<id>" or "<id>;<cluster>"
echo "student export of $1, epoch $epoch: job $job (log slurm-$job.out here; writes $out)"
