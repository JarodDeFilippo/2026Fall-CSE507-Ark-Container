import os

import torch


class Accelerator:
    """Resolve the training accelerator and its device-specific settings."""

    def __init__(self, accelerator=None):
        self.name = (accelerator or os.environ.get("PROJECT_ACCELERATOR", "cuda")).lower()
        if self.name not in {"cpu", "cuda", "hpu"}:
            raise ValueError("Unsupported accelerator '{}'; use cpu, cuda, or hpu".format(self.name))

        if self.name == "hpu":
            os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
            try:
                import habana_frameworks.torch.core  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("The Gaudi runtime is not available") from exc

        self.device = torch.device(self.name)

    @property
    def is_cuda(self):
        return self.name == "cuda"

    @property
    def pin_memory(self):
        return self.is_cuda
