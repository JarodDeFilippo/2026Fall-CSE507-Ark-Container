#!/usr/bin/env bash
#SBATCH --job-name=d1-8-nogram-no
#SBATCH -A grp_jliang12
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:hl225:4
#SBATCH -c 72
#SBATCH --mem=256G
# DINO-X ConstLR 224, no Gram, one recipe flag off: the one-flag-off ablations of the no-Gram trunk, for group members to submit, one arm each (Jarod, 2026-10-26). The no-Gram trunk (run_d1_8_constlr_nogram_gaudi.sh, run dir
# ssl_dinov3_cxr_constlr_nogram_seed_100) passes all three settings of DINOv3's training recipe, --drop_path_rate 0.3 --rope_rescale_coords 2 --untie_local_cls_norm; the plain trunk (run_d1_8_constlr_nogram_plain_gaudi.sh, run dir
# ssl_dinov3_cxr_constlr_nogram_plain_seed_100) passes none. Each arm here is the no-Gram trunk's launch line with exactly ONE of the three removed and nothing else changed (ssl_launch's stock DINOv3 ViT-B/16 init, chain/main_lr.txt as
# the absolute peak LR, --min_lr = --lr, --no_lr_scaling, the --epochs 1000 ceiling, 64 per card on 4 cards = global batch 256, seed 100, 224 / 96 crops, --no_gram), so each flag's effect on the frozen probe and the cosine maps reads on its
# own: the plain trunk is the all-off control and the no-Gram trunk the all-on one. chain/constlr.sh owns the recipe, so none of it is repeated here; the setting on the last line is its env override (CONSTLR_NOGRAM=1), and the arm is picked by
# CONSTLR_ABLATE, which the sbatch line puts in the job's environment.
# The three arms, from the checkout (cd /scratch/$USER/2026Fall-CSE507-Ark-Container), each with its stop file created first (cluster-only, never in the repo, so an rsync cannot overwrite it; 100 as for the no-Gram trunk: ConstLR 224 took about
# 58 min per epoch on 4 cards WITH the Gram teacher, so 100 epochs is at most ~97 h of the 168 h) and its own job name (sbatch does not expand variables in the #SBATCH lines above, so --job-name goes on the sbatch line):
#   echo 100 > chain/constlr_nogram_no_drop_path_stop_after.txt && sbatch --job-name=d1-8-nogram-no-drop_path --export=ALL,CONSTLR_ABLATE=drop_path run_d1_8_constlr_nogram_ablate_gaudi.sh
#   echo 100 > chain/constlr_nogram_no_rope_stop_after.txt && sbatch --job-name=d1-8-nogram-no-rope --export=ALL,CONSTLR_ABLATE=rope run_d1_8_constlr_nogram_ablate_gaudi.sh
#   echo 100 > chain/constlr_nogram_no_local_norm_stop_after.txt && sbatch --job-name=d1-8-nogram-no-local_norm --export=ALL,CONSTLR_ABLATE=local_norm run_d1_8_constlr_nogram_ablate_gaudi.sh
# drop_path removes --drop_path_rate 0.3, rope removes --rope_rescale_coords 2, local_norm removes --untie_local_cls_norm; the other two stay. Arm <value> runs in ssl_dinov3_cxr_constlr_nogram_no_<value>_seed_100 and reads only
# chain/constlr_nogram_no_<value>_stop_after.txt, so the arms share no run dir or stop file with each other or with the trunks and can run side by side. CONSTLR_ABLATE is required: without it chain/constlr.sh would take CONSTLR_NOGRAM=1 alone
# for the no-Gram trunk and launch its run dir, so this script refuses to start (an ERROR naming the three values), and chain/constlr.sh refuses any value but those three. Pass it on the sbatch line only, never export it in the shell: sbatch hands
# the submitting shell's environment to every job, so the no-Gram trunk's launcher would start that arm in place of the trunk.
# Re-submit (to continue or extend an arm): pass the SAME CONSTLR_ABLATE, the sbatch part of the arm's line only (the echo would put the stop epoch back to 100). The value picks the run dir and the stop file, so another value launches a
# different run, in a different run dir. To extend an arm: raise its own stop file and re-submit; it resumes from its last epoch checkpoint (--resume locks no_gram, the LR, the ceiling and the three recipe settings, not the stop epoch).
# 4 cards, not 8: the global batch (Sinkhorn-Knopp centering and the DINO / iBOT statistics are computed over all ranks) and the per-card batch (KoLeo is computed per card) both match the no-Gram trunk.
# No smoke launcher: the no-Gram trunk's smoke (run_d1_8_nogram_smoke_gaudi.sh: 4 cards, all three recipe flags, 20 steps and the --max_steps checkpoint save) already covers these flags on HCCL, and an arm runs a subset of them.
# NOT a chain phase: it is submitted on its own and must never be added to chain/phases.txt. The log is the Slurm .out (no chain/log file, no chain/done marker); its first lines must show the
# `run: ssl_dinov3_cxr_constlr_nogram_no_<value>_seed_100 (no Gram teacher: --no_gram replaces --gram_refresh_steps 10000,20000,30000; <note>; stop after epoch 100 from chain/constlr_nogram_no_<value>_stop_after.txt)` line with the note
# below, the trainer's `Gram: OFF (--no_gram)` line and its `recipe (...)` line (the removed setting at its default, the other two on), and no `Gram refresh scheduled` line:
#   drop_path:  recipe ablation: drop_path off (--drop_path_rate 0.3 removed); DINOv3 recipe otherwise: --rope_rescale_coords 2 --untie_local_cls_norm
#               recipe (...): drop path 0, RoPE coordinate rescale 2, local-crop CLS norm untied
#   rope:       recipe ablation: rope off (--rope_rescale_coords 2 removed); DINOv3 recipe otherwise: --drop_path_rate 0.3 --untie_local_cls_norm
#               recipe (...): drop path 0.3, RoPE coordinate rescale none, local-crop CLS norm untied
#   local_norm: recipe ablation: local_norm off (--untie_local_cls_norm removed); DINOv3 recipe otherwise: --drop_path_rate 0.3 --rope_rescale_coords 2
#               recipe (...): drop path 0.3, RoPE coordinate rescale 2, local-crop CLS norm tied (and no `local_cls_norm: initialized as a copy ...` line, which the other two arms print on a first start)
export PT_HPU_LAZY_MODE=1
[[ -n ${CONSTLR_ABLATE:-} ]] || { echo "ERROR: CONSTLR_ABLATE is unset or empty: it picks the arm, one of drop_path, rope or local_norm (sbatch --export=ALL,CONSTLR_ABLATE=drop_path run_d1_8_constlr_nogram_ablate_gaudi.sh); without it chain/constlr.sh would launch the no-Gram trunk's own run dir" >&2; exit 1; }
cd /scratch/$USER/2026Fall-CSE507-Ark-Container || exit 1
# fail closed: --cleanenv drops SLURM_JOB_GPUS, so check the container sees exactly the allocated cards before the ranks start
./gaudi_check_cards.sh || exit 1
# report-only: at 15 min, log whether every allocated card is busy (gaudi_verify_busy.sh)
[[ -x ./gaudi_verify_busy.sh ]] || { echo "ERROR: gaudi_verify_busy.sh missing or not executable; refusing to train unverified" >&2; exit 1; }
./gaudi_verify_busy.sh 900 &
CONSTLR_NOGRAM=1 exec ./chain/constlr.sh
