# Frozen-backbone TBX11K and CANDID-PTX box localization

Experimental detector: torchvision Faster R-CNN, a trainable 1x1 projection from the frozen ViT patch grid to 256 channels, a trainable region proposal network and box head, anchors {16,32,64,128,256} with aspect ratios {0.5,1,2}. All compared models use this same head. The 1x1 projection supports both ViT-B (768 channels) and RayDINO ViT-L (1024 channels). This 512-pixel head is an updated architecture, not the exact 768-channel head used for the earlier 224-pixel pilot. It requires torchvision detection ops, pycocotools 2.0.10 and pydicom.

TBX11K: `/data/jliang12/nuislam/data_files_splits/tbx11k/all_train_2.json` is split 85/15 by stable SHA256 filename ranking, stratified by folder and annotation presence. `splits/tbx11k_train_selection_seed100.json` records the exact image filenames and source file hash; split is fixed across seeds/models. Full `all_val_2.json` is evaluated once after model selection. Jarod reports 46 val/train image duplicates and 28 val images matching ChestX-ray14 pretraining; all negative for TB. Do not claim independent test data. Split is image-level; patient-level independence is not established.

CANDID-PTX uses `/data/jliang12/nuislam/FoundationX_localization_bbox_annotation_collections/CANDID_PTX/CANDID_PTX_{train,valid,test}_1.json`. The PNG label filenames are resolved to raw DICOMs in `/data/jliang12/shared/dataset/CANDID-PTX/dataset`, min-max converted to uint8 and resized to the annotated geometry. This matches the lab dense segmentation loader approach; verify DICOM inversion and annotation geometry on representative samples before treating results as final.

Inputs are 512 px (RAD-DINO 518 px), AdamW lr 1e-4 wd 1e-4, batch 4, 10 epochs, seeds 0-9, no augmentation. The best epoch is selected by COCO AP@[.50:.95] on the selection set. The full labeled test split is evaluated once per seed using pycocotools COCOeval; result.json records AP50:95, AP50, file hashes, configuration and epoch history. One seed per job; identical parameters and split for each model. No threshold was tuned on the test set.

Original 224-pixel pilot command (job 863402):

```
cd /scratch/jcuesta/BME/DINO-X-plain-control
NVIDIA_IMAGE=/scratch/jcuesta/BME/2026Fall-CSE507-Ark-Container/containers/nvidia/nvidia.sif ./nvidia.sh python -m tbx_localization.train_detector --weights /workspace/ssl_dinov3_cxr_constlr_nogram_plain_seed_100/export/teacher_backbone_epoch_0090.pth --out /workspace/results/tbx_localization/plain_e0090_seed100 --epochs 10
```

The original pilot used the old train_detector.py and split the lab labeled validation set in half; its custom AP50 is preliminary and not directly comparable to the 512-pixel protocol. The new path compare_detectors.py requires pycocotools 2.0.10 on Python's path or installed to `base/tbx_localization/requirements/` (that directory is not part of the source).

Example for seed 0 stock DINOv3, on an allocated GPU from repository root:

```
NVIDIA_IMAGE=/scratch/jcuesta/BME/2026Fall-CSE507-Ark-Container/containers/nvidia/nvidia.sif ./nvidia.sh python -m tbx_localization.compare_detectors --dataset tbx11k --model vit_base_dinov3 --weights /workspace/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth --normalization imagenet --seed 0 --epochs 10 --batch 4 --manifest /workspace/base/tbx_localization/splits/tbx11k_train_selection_seed100.json --out /workspace/results/bbox512/stock/tbx11k/seed0
```

To expose Jarod's shared weights without symlinks, set `DATA_BIND=/data:/data:ro,/scratch/jdefili4/2026Fall-CSE507-Ark-Container:/jarod:ro` for nvidia.sh and use the matching `/jarod/...` weight path. Access to licensed RayDINO requires each user to obtain its weights independently. Do not copy licensed model weights, image data, or generated predictions to Git.

For the 10-seed comparisons, submit one job array per model and dataset. Override the account with `-A` as needed and ensure `ARK_ROOT` and `NVIDIA_IMAGE` point to your own checkout and container. Example (place all `sbatch` options before the script):

```bash
sbatch -A <class-account> --array=0-9%2 run_localization_compare.sh tbx11k ark_c200 vit_base_dinov3 /jarod/ssl_dinov3_cxr_constlr_arkteacher_seed_100/export/teacher_backbone_epoch_0200.pth imagenet
sbatch -A <class-account> --array=0-9%2 run_localization_compare.sh candidptx rad_dino vit_base_raddino /jarod/weights/rad_dino_vitb14_pretrain.pth raddino
```

The array task ID is the seed, and each seed writes to a distinct directory. The `%2` limits concurrent tasks in each array. For the stock baseline, seed 0 has already been submitted by Jan for both datasets; use `--array=1-9%2` for that baseline to avoid duplicate outputs. Confirm the pilot jobs completed successfully before scheduling arrays. Set `DATA_BIND` if `/jarod` or `/data` differs, and obtain gated model weights independently. `run_localization_smoke.sh` and `run_tbx_localization.sh` record Jan's original site-specific commands and should be adapted before use elsewhere.

Important: the earlier pilot used one half of the same labeled TBX11K validation set for epoch selection and scored the other half. Consequently, this data has already informed development and must not be described as an untouched test set for the project, even though the new training script does not use it for model selection. Obtain a genuinely untouched evaluation set for a final unbiased claim.
