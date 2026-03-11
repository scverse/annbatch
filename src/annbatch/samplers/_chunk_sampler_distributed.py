"""Sampler classes for efficient chunk-based data access."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from annbatch.abc import Sampler
from annbatch.utils import _spawn_worker_rng

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy as np

    from annbatch.types import LoadRequest


def _get_dist_info_torch() -> tuple[int, int]:
    """Get rank and world_size from ``torch.distributed``."""
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Initialize it before creating a ChunkSamplerDistributed with dist_info='torch'."
        )
    return dist.get_rank(), dist.get_world_size()


def _get_dist_info_jax() -> tuple[int, int]:
    """Get rank and world_size from JAX multi-process API."""
    import jax

    if not jax.distributed.is_initialized():
        raise RuntimeError(
            "JAX distributed is not initialized. "
            "Call jax.distributed.initialize() before creating a ChunkSamplerDistributed with dist_info='jax'."
        )
    return jax.process_index(), jax.process_count()


DISTRIBUTED_BACKENDS: dict[str, Callable[[], tuple[int, int]]] = {
    "torch": _get_dist_info_torch,
    "jax": _get_dist_info_jax,
}


class MaskableSampler(Sampler):
    """A sampler whose observation range can be restricted via a mask.

    Subclass this to create chunk-based samplers that can be wrapped
    by :class:`ChunkSamplerDistributed`.
    """

    _mask: slice
    _rng: np.random.Generator

    @property
    def mask(self) -> slice:
        """The observation range this sampler operates on."""
        return self._mask

    @mask.setter
    def mask(self, value: slice) -> None:
        self._mask = value

    @property
    def rng(self) -> np.random.Generator:
        """The random number generator used by this sampler."""
        return self._rng

    @rng.setter
    def rng(self, value: np.random.Generator) -> None:
        self._rng = value


class ChunkSamplerDistributed(Sampler):
    """Distributed chunk-based sampler that shards data across distributed processes.

    Wraps any chunk-based sampler (e.g. :class:`ChunkSampler` or
    :class:`ChunkSamplerWithReplacement`) and partitions the observation
    range across ``world_size`` processes.  Each rank receives a
    non-overlapping slice of the data.

    When ``enforce_equal_batches`` is *True* (the default), the per-rank observation
    count is rounded down to the nearest multiple of ``batch_size``,
    guaranteeing that every rank yields exactly the same number of complete
    batches.

    Rank and world size are obtained from ``dist_info`` at construction time.
    The corresponding distributed framework must already be initialized.

    Parameters
    ----------
    sampler
        The base chunk sampler to distribute.
    dist_info
        How to obtain rank and world size.
        Either a string naming a distributed backend (``"torch"`` or ``"jax"``),
        or a callable that returns ``(rank, world_size)``.
    enforce_equal_batches
        If *True*, round each rank's observation count down to a multiple of ``batch_size`` so that all workers (ranks) yield the same number of batches.
        Set to *False* to use the raw ``n_obs // world_size`` split, which may result in an uneven number of batches per worker.
    """

    _rank: int
    _world_size: int
    _enforce_equal_batches: bool
    _sampler: MaskableSampler

    def __init__(
        self,
        sampler: MaskableSampler,
        *,
        dist_info: Literal["torch", "jax"] | Callable[[], tuple[int, int]],
        enforce_equal_batches: bool = True,
    ):
        if callable(dist_info):
            self._rank, self._world_size = dist_info()
        elif dist_info in DISTRIBUTED_BACKENDS:
            self._rank, self._world_size = DISTRIBUTED_BACKENDS[dist_info]()
        else:
            raise ValueError(f"Unknown dist_info {dist_info!r}. Supported backends: {sorted(DISTRIBUTED_BACKENDS)}")
        self._enforce_equal_batches = enforce_equal_batches
        self._sampler = sampler
        sampler.rng = _spawn_worker_rng(sampler.rng, self._rank)

    @property
    def batch_size(self) -> int:
        return self._sampler.batch_size

    @property
    def shuffle(self) -> bool:
        return self._sampler.shuffle

    def _shard_mask(self, n_obs: int) -> slice:
        """Return the contiguous observation slice for this rank."""
        per_rank = n_obs // self._world_size
        if self._enforce_equal_batches:
            per_rank = per_rank // self._sampler.batch_size * self._sampler.batch_size
        rank_start = self._rank * per_rank
        rank_stop = rank_start + per_rank
        return slice(rank_start, rank_stop)

    def n_iters(self, n_obs: int) -> int:
        self._sampler.mask = self._shard_mask(n_obs)
        return self._sampler.n_iters(n_obs)

    def validate(self, n_obs: int) -> None:
        self._sampler.mask = self._shard_mask(n_obs)
        self._sampler.validate(n_obs)

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        self._sampler.mask = self._shard_mask(n_obs)
        yield from self._sampler._sample(n_obs)
