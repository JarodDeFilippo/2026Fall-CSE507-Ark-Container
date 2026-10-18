# Not Meta's code: a stand-in for dinov3/distributed, which the vendored loss modules import. Upstream's package sets
# up Slurm/CUDA process groups and subgroups; this trainer has one default process group (Ark's Accelerator), so the
# "subgroup" is the default group and its size is the world size.
import torch.distributed as dist


def is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank(group=None) -> int:
    return dist.get_rank(group=group) if is_enabled() else 0


def get_world_size(group=None) -> int:
    return dist.get_world_size(group=group) if is_enabled() else 1


def get_process_subgroup():
    return None  # None selects the default process group in every torch.distributed call


def get_subgroup_size() -> int:
    return get_world_size()
