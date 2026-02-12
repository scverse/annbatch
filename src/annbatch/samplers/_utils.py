from __future__ import annotations

import importlib.util
from typing import NamedTuple


class WorkerInfo(NamedTuple):
    """Minimal worker info for RNG handling."""

    id: int
    num_workers: int


def get_torch_worker_info() -> WorkerInfo | None:
    """Get torch DataLoader worker info if available.

    Returns None if torch is not installed or not in a worker process.
    """
    if importlib.util.find_spec("torch"):
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        if info is not None:
            return WorkerInfo(id=info.id, num_workers=info.num_workers)
    return None
