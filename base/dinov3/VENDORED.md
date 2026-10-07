# Vendored DINOv3 ViT
Source: https://github.com/facebookresearch/dinov3 at commit 6876159a11b4df116f30f667f8c9888617df0751, vendored 2026-09-30.
Verbatim copies: layers/{attention,block,ffn_layers,layer_scale,patch_embed,rms_norm,rope_position_encoding}.py, models/vision_transformer.py.
Edited: layers/__init__.py (the fp8_linear import is removed); utils/__init__.py (only cat_keep_shapes, uncat_with_shapes, named_apply from utils/utils.py); __init__.py and models/__init__.py are empty.
fp8_linear.py omitted, along with the rest of the upstream package.
Licensed under the DINOv3 License Agreement (see LICENSE.md, copied from the upstream repo root).
Used by ../models.py (ArkViT, model name vit_base_dinov3). Pure torch, no xformers/triton/torchao.
