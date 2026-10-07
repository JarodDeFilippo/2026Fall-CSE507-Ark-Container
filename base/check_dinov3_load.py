"""CPU smoke test for the vit_base_dinov3 backbone. Run from base/: python check_dinov3_load.py [weights_path]"""
import logging
import sys
from types import SimpleNamespace

import torch

from models import build_omni_model

weights = sys.argv[1] if len(sys.argv) > 1 else None

# build_omni_model logs "Loaded with msg: ..." on the "ark" logger; show it on stdout. With weights, it also raises
# if anything but the new projector/heads is missing, or any key is unexpected.
ark_log = logging.getLogger("ark")
ark_log.addHandler(logging.StreamHandler(sys.stdout))
ark_log.setLevel(logging.INFO)

args = SimpleNamespace(model_name="vit_base_dinov3", projector_features=1376, use_mlp=False, pretrained_weights=weights)
model = build_omni_model(args, [6, 14, 14, 14]).eval()

x = torch.zeros(2, 3, 224, 224)
with torch.no_grad():
    assert model(x, return_features=True).shape == (2, 1376)
    outs = model(x)
    assert isinstance(outs, list) and [tuple(o.shape) for o in outs] == [(2, 6), (2, 14), (2, 14), (2, 14)]
    assert all(torch.isfinite(o).all() for o in outs)  # NaN here means the k-bias masks were never initialised
    feats, logits = model(x, head_n=1)
    assert feats.shape == (2, 1376) and logits.shape == (2, 14)

n_backbone = sum(p.numel() for k, p in model.named_parameters() if not k.startswith(("projector", "omni_heads")))
print("backbone parameters: {:,}".format(n_backbone))
assert 85e6 < n_backbone < 88e6, n_backbone
print("OK: vit_base_dinov3 {}".format("loaded from " + weights if weights else "built without weights"))
