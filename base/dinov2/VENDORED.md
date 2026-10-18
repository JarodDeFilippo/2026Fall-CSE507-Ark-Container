# Vendored DINOv2 ViT
Source: https://github.com/facebookresearch/dinov2, main-branch archive unpacked as `dinov2-main/` (files dated 2026-06-03; the
archive has no git metadata, so no commit hash: the sha256 of each upstream file is listed below), vendored 2026-10-10.
Verbatim copies: __init__.py, models/{__init__,vision_transformer}.py, layers/{attention,block,drop_path,layer_scale,mlp,patch_embed,swiglu_ffn}.py.
Edited: layers/__init__.py (the DINOHead import is removed, with a notice in the file); dino_head.py is omitted, along with the rest of the upstream package
(hub, train, eval, loss, data, utils, fsdp ...).
Licensed under the Apache License 2.0 (LICENSE, copied from the upstream repo root; upstream has no NOTICE file).
Used by ../models.py (ArkViTDinoV2, model names vit_base_dinov2 and vit_base_raddino: the hub's dinov2_vitb14 architecture; vit_large_raydino: the same code as vit_large, patch 16, 512 px pos_embed).
Pure torch: the xFormers imports are optional upstream and fall back to torch's scaled_dot_product_attention when xformers is absent
(it is in neither container); each layer module then warns "xFormers is not available" once at import, as upstream does.

Upstream sha256 (dinov2-main/):
600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7  LICENSE
0d2b87b7e71c7f7279ccc4b4f8b541d26071425284a0ea01a5a435fac4511cfe  dinov2/__init__.py
b93329ce676f3eb0c9a7790d1b3b2b114a0855ea8dc76e32e262d1a7dc6c55ae  dinov2/models/__init__.py
7799a260f2d7d0fe197331d08502fb8c542f9b7424723650f6a39b64fa2639ea  dinov2/models/vision_transformer.py
1b55deed39d5ab0b589bef421bbfe06f24a10cd590a0f8403234c9ee4e34109d  dinov2/layers/__init__.py (before the edit)
79c7be7a452b3aad96698ec38d5d5150b9f4d8ac084fa93324510dc9f624775d  dinov2/layers/attention.py
60c0ac7dfa4474be313fabfa5a23d82faf6f0cecd4e720a88be35de9788cb636  dinov2/layers/block.py
b9f8236e86054b9d9a71275efcad2a9ecaa1f86b529d4b8d6109ddb5e806f67a  dinov2/layers/drop_path.py
dadd5aafe178f1bf72a205a02a6645c7e635cacbad585d4a7369c200c6e89135  dinov2/layers/layer_scale.py
255825c73b60a916dd00eb1e38aacbcdbf316e40d6a005efb46e245b7edb43aa  dinov2/layers/mlp.py
40da6add3d811198ea3e17cb99cdd4e5cda59e369efbbe3d18d89308618cf142  dinov2/layers/patch_embed.py
e46d2948fb97e497cf991ff82ce30cb59b59fdf2e12a19568417113716b4f119  dinov2/layers/swiglu_ffn.py
