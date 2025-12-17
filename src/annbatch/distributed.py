from __future__ import annotations

from importlib.util import find_spec


def get_rank_and_world_size() -> tuple[int, int]:
    """Return (rank, world_size) if torch.distributed is initialized, else (0, 1)."""
    if find_spec("torch") is None:
        return 0, 1

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


