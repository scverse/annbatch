"""DistributedSampler -- distributed sampler."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Literal

from annbatch.abc import Sampler
from annbatch.utils import _spawn_worker_rng

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from annbatch.types import LoadRequest


def _get_dist_info_torch() -> tuple[int, int]:
    """Get rank and world_size from ``torch.distributed``."""
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Initialize it before creating a DistributedSampler with dist_info='torch'."
        )
    return dist.get_rank(), dist.get_world_size()


def _get_dist_info_jax() -> tuple[int, int]:
    """Get rank and world_size from JAX multi-process API."""
    import jax

    if not jax.distributed.is_initialized():
        raise RuntimeError(
            "JAX distributed is not initialized. "
            "Call jax.distributed.initialize() before creating a DistributedSampler with dist_info='jax'."
        )
    return jax.process_index(), jax.process_count()


DISTRIBUTED_BACKENDS: dict[str, Callable[[], tuple[int, int]]] = {
    "torch": _get_dist_info_torch,
    "jax": _get_dist_info_jax,
}


class DistributedSampler(Sampler):
    """Distributed chunk-based sampler that shards data across distributed processes.

    Partitions the full observation range into ``world_size`` contiguous shards
    using the ``mask`` mechanism of :class:`~annbatch.abc.Sampler`. Each rank receives a
    non-overlapping slice of the data. The shard boundaries are computed lazily
    when ``n_obs`` becomes known.

    When ``enforce_equal_batches`` is *True* (the default), the per-rank observation
    count is rounded down to the nearest multiple of ``batch_size``,
    guaranteeing that every rank yields exactly the same number of complete
    batches.

    Rank and world size are obtained from ``dist_info`` at construction time.
    The corresponding distributed framework must already be initialized.

    Example
    -------
    >>> from annbatch.samplers import DistributedSampler, RandomSampler
    >>> sampler = RandomSampler(
    ...     chunk_size=256,
    ...     preload_nchunks=4,
    ...     batch_size=32,
    ... )

    Using PyTorch distributed

    >>> dist_sampler = DistributedSampler(sampler, dist_info="torch")

    Using JAX

    >>> dist_sampler = DistributedSampler(sampler, dist_info="jax")

    Using a custom callable

    >>> dist_sampler = DistributedSampler(
    ...     sampler,
    ...     dist_info=lambda: (rank, world_size),
    ... )

    Parameters
    ----------
    sampler
        The :class:`~annbatch.abc.Sampler` to distribute. It is copied, so changes made to it
        after wrapping (``sampler.rng``, say) do not reach the copy. Its own ``mask`` is ignored:
        the copy spans the whole range and is restricted to this rank's shard only while a pass runs.
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
    _sampler: Sampler
    _num_open_passes: int

    def __init__(
        self,
        sampler: Sampler,
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
        # a copy: _sample moves this sampler's mask onto the shard for each pass
        self._sampler = copy.deepcopy(sampler)
        self._sampler.mask = slice(0, None)
        self._num_open_passes = 0
        if self._sampler.rng is not None:
            self._sampler.rng = _spawn_worker_rng(self._sampler.rng, self._rank)

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

    def n_batches(self, n_obs: int) -> int:
        # Count this rank's shard. How to ask depends on where the wrapped sampler's mask is:
        if self._num_open_passes > 0:
            # during a pass the mask *is* the shard, so count it against the real n_obs
            # (asking with the shard size would put the shard out of bounds on ranks > 0)
            return self._sampler.n_batches(n_obs)
        # between passes the mask spans everything, so a shard-sized n_obs gives the shard's count
        shard = self._shard_mask(n_obs)
        return self._sampler.n_batches(shard.stop - shard.start)

    def validate(self, n_obs: int) -> None:
        # checks that don't depend on the shard; _sample validates the shard itself when the pass starts
        self._sampler.validate(n_obs)

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        self._sampler.mask = self._shard_mask(n_obs)
        self._num_open_passes += 1
        try:
            self._sampler.validate(n_obs)
            yield from self._sampler._sample(n_obs)
        finally:
            self._num_open_passes -= 1
            self._sampler.mask = slice(0, None)
