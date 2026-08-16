import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class Accelerator:
    """Resolve the training accelerator and its device-specific settings."""

    def __init__(self, accelerator=None):
        self.name = (accelerator or os.environ.get("PROJECT_ACCELERATOR", "cuda")).lower()
        if self.name not in {"cpu", "cuda", "hpu"}:
            raise ValueError("Unsupported accelerator '{}'; use cpu, cuda, or hpu".format(self.name))

        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = self.world_size > 1

        if self.name == "hpu":
            os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
            try:
                import habana_frameworks.torch.core  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("The Gaudi runtime is not available") from exc

        if self.name == "cuda":
            device_index = self.local_rank if self.distributed else 0
            torch.cuda.set_device(device_index)
            self.device = torch.device("cuda", device_index)
        else:
            self.device = torch.device(self.name)

    def initialize_distributed(self):
        if not self.distributed or dist.is_initialized():
            return

        if self.name == "cuda":
            backend = "nccl"
        elif self.name == "hpu":
            import habana_frameworks.torch.distributed.hccl  # noqa: F401
            backend = "hccl"
        else:
            backend = "gloo"

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=self.rank,
            world_size=self.world_size,
        )

    def wrap_model(self, model):
        if not self.distributed:
            return model

        if self.is_cuda:
            return DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )

        return DistributedDataParallel(
            model,
            device_ids=None,
            find_unused_parameters=True,
        )

    @staticmethod
    def unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def strip_module_prefix(state_dict):
        if any(key.startswith("module.") for key in state_dict):
            return {
                key[len("module."):] if key.startswith("module.") else key: value
                for key, value in state_dict.items()
            }
        return state_dict

    @property
    def is_main_process(self):
        return self.rank == 0

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def broadcast(self, tensor):
        if self.distributed:
            dist.broadcast(tensor, src=0)

    def destroy_distributed(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()

    @property
    def is_cuda(self):
        return self.name == "cuda"

    @property
    def pin_memory(self):
        return self.is_cuda
