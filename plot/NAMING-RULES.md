# Naming rules for every Step VIII deliverable (Jarod, 2026-10-17 ~13:00; standing)
Use the names things already have. Never coin a token for a variant; describe the variant in words. (The professor made a lab member fix "VINDRCXR"
and another fix "RICORD30PCT": the published name plus a plain description of the deviation is the convention.)
- Datasets, display form: VinDr-CXR, ChestX-ray14, MIMIC-CXR, CheXpert. Machine ids (directory names VinDrCXR / ChestXray14 / MIMIC / CheXpert,
  workbook column ids like T_CXR14_end, CSV paths) stay unchanged; only human-facing strings (titles, legends, table headers, covers, prose) change.
- Backbones: DINOv3 ViT-B/16 (Meta), DINOv2 ViT-B/14 (Meta), RAD-DINO ViT-B/14 (Microsoft), Swin-B (Ark+). "stock" = the published DINOv3 weights.
- Our self-supervised backbone (the DINOv3 ViT-B/16 continued-pretrained on the four chest X-ray datasets, previously written "Stage A", "main",
  "constlr"): **DINO-X**. Variants in words: "DINO-X, cosine schedule, epoch 30" (the run whose directory is ssl_dinov3_cxr_main_seed_100),
  "DINO-X, constant LR, epoch 30" (ssl_dinov3_cxr_constlr_seed_100). Short legend forms: "DINO-X Cosine e30", "DINO-X ConstLR e30".
  Fine-tune arms initialised from it: "init DINO-X Cosine e30" (S13), "init DINO-X ConstLR e30" (S16).
- Internal run tags (S2, armD, D-long, constlr, main, pilot_a) may appear as identifiers NEXT TO the real name, never instead of it; the first mention
  in a document says what the tag is.
- Ten-crop / center crop as already defined. "Stage A" may remain once, in parentheses, as the internal name of the DINO-X pretraining phase.

## Consistency (Jarod, 13:20: "imagine these graphs and numbers go into a journal or conference submission")
One convention per thing, across workbook, deck, PDFs, prose:
- Metric name: **mAUC** (mean AUROC over the dataset's classes), written "mAUC" everywhere; never "Mean AUROC", "AUROC", "mean AUC" as the label.
- Unit: **percent** everywhere, axis label "Test mAUC (%)" / "Validation mAUC (%)"; tables and prose one decimal (95.4), workbook two decimals, probe
  s.d. two decimals in parentheses. Figures that today plot fractions (0.50–0.85) switch to percent (×100) with the same label.
- Teacher / Student capitalised as words in legends ("Teacher", "Student"); the headline is the teacher at the end of the cycle (after the CheXpert pass),
  stated once per document, not re-explained in every caption.
- Dataset order everywhere: VinDr-CXR, ChestX-ray14, MIMIC-CXR, CheXpert (the training order inside a cycle).
- Colors: one color per arm, the same in every figure that shows that arm (per-run, comparison, sweep-wide ten-crop and center-crop). Define the palette
  once (SWEEP order → tab20 as in build_step8_report.py's sweep_all figures; early arms B, C, D-long get the next tab20 entries) and make
  compare_cyclic_runs.py accept `--colors RUNDIR=HEX …` (or a JSON map) so the builder passes the same mapping; per-run figures keep student/teacher
  blue/red since no arm comparison happens inside them.
- Captions/titles: "<Figure subject> — <dataset display name> — <split> — Teacher" pattern; no internal tags in titles except the arm id (S2, …).
- Cycle axis: integer cycles, label "Cycle"; evaluation points inside a cycle at quarter positions are allowed but the ticks are integers.

## Run names (Jarod, 2026-10-17 ~13:55)
The two DINO-X runs are written **DINO-X Cosine** and **DINO-X ConstLR** (exactly this casing), with the epoch after: "DINO-X Cosine e30", "DINO-X ConstLR e40". Init tags: "init DINO-X Cosine e30" (S13), "init DINO-X ConstLR e30" (S16).
