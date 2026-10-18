# Vendored DINOv3 ViT
Source: https://github.com/facebookresearch/dinov3 at commit 6876159a11b4df116f30f667f8c9888617df0751, vendored 2026-09-30.
Verbatim copies: layers/{attention,block,ffn_layers,layer_scale,patch_embed,rms_norm,rope_position_encoding}.py, models/vision_transformer.py.
Edited: layers/__init__.py (the fp8_linear import is removed); utils/__init__.py (only cat_keep_shapes, uncat_with_shapes, named_apply from utils/utils.py); __init__.py and models/__init__.py are empty.
fp8_linear.py omitted, along with the rest of the upstream package.
Licensed under the DINOv3 License Agreement (see LICENSE.md, copied from the upstream repo root).
Used by ../models.py (ArkViT, model name vit_base_dinov3). Pure torch, no xformers/triton/torchao.

## Stage A SSL modules (vendored 2026-10-08, same commit)
Verbatim copies (byte-identical, imports untouched): loss/{__init__,dino_clstoken_loss,ibot_patch_loss,koleo_loss,gram_loss}.py,
layers/dino_head.py, data/{masking,collate}.py, train/{param_groups,cosine_lr_scheduler}.py.
Ours, not Meta's: distributed/__init__.py, a five-function stand-in for upstream dinov3/distributed (Slurm/CUDA process-group
setup) that the loss modules import: one default process group, so the "subgroup" is the default group (None) and its size
is the world size. data/__init__.py and train/__init__.py are empty (upstream's import non-vendored modules).
Not vendored: data/augmentations.py, train/{train,ssl_meta_arch}.py (omegaconf, FSDP2, torch.cuda); the few lines used are
reimplemented in ../ssl_dinov3/ and ../ssl_pretrain.py, which say where each one comes from.
Upstream behaviour kept on purpose: iBOTPatchLoss.__init__ calls SinkhornKnoppTeacher.compile(); ssl_dinov3/meta.py calls
its .forward directly, so torch.compile never runs (no inductor on HPU lazy mode). KoLeoLoss disables autocast for "cuda"
only. The iBOT Sinkhorn-Knopp all_reduces unconditionally, so a process group must exist even at world size 1.
