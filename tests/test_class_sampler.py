"""Tests for ClassSampler.

The passing tests check the sampler does what it promises: every chunk is
class-coherent, classes are drawn with the requested weights (a zero weight
excludes a class), masks restrict and renormalize correctly, and the
bookkeeping (``num_samples`` / ``n_batches`` / validation) is correct.

The final test (``test_pure_class_batches_unsupported``) is expected to
**fail**. It is deliberately not marked ``xfail``: it documents, with the real
:class:`~annbatch.Loader` ordering contract, why the sampler cannot currently
yield *class-pure batches*.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from annbatch.samplers import ClassSampler
from annbatch.samplers._utils import WorkerInfo
from tests.conftest import load_x_obs_var


def make_sampler(
    classes: pd.Categorical,
    *,
    num_samples: int = 1000,
    chunk_size: int = 10,
    preload_nchunks: int = 4,
    batch_size: int = 10,
    seed: int = 0,
    **kwargs,
) -> ClassSampler:
    """Build a sampler with sane defaults so each test only states what matters."""
    return ClassSampler(
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        batch_size=batch_size,
        classes=classes,
        num_samples=num_samples,
        rng=np.random.default_rng(seed),
        **kwargs,
    )


def _chunk_classes(chunks: list[slice], codes: np.ndarray) -> list[int]:
    """Class of each chunk, or -1 if the chunk straddles more than one class."""
    return [int(u[0]) if (u := np.unique(codes[c])).size == 1 else -1 for c in chunks]


def _collect_chunks(sampler: ClassSampler, n_obs: int) -> list[slice]:
    return [c for load_request in sampler.sample(n_obs) for c in load_request["requests"]]


def _draw_shares(sampler: ClassSampler, codes: np.ndarray) -> dict[int, float]:
    """Fraction of drawn chunks belonging to each class."""
    classes = np.array(_chunk_classes(_collect_chunks(sampler, len(codes)), codes))
    vals, counts = np.unique(classes, return_counts=True)
    return {int(v): cnt / counts.sum() for v, cnt in zip(vals, counts, strict=True)}


def _assert_shares(sampler: ClassSampler, codes: np.ndarray, expected: dict[int, float], atol: float = 0.02):
    shares = _draw_shares(sampler, codes)
    assert set(shares) == set(expected), f"sampled classes {sorted(shares)} != {sorted(expected)}"
    for cat, exp in expected.items():
        assert abs(shares[cat] - exp) <= atol, f"class {cat}: share {shares[cat]:.3f} vs expected {exp}"


# =============================================================================
# Construction / validation
# =============================================================================


@pytest.mark.parametrize(
    ("classes", "kwargs", "error_type", "match"),
    [
        pytest.param(
            pd.Categorical([0, 0, 1, 1, 2, 2]), {}, ValueError, "at least chunk_size", id="all_runs_too_short"
        ),
        pytest.param(
            pd.Categorical([0] * 30 + [1] * 30 + [0] * 3),
            {},
            ValueError,
            r"at least chunk_size.*\[0\]",
            id="one_run_too_short",
        ),
        pytest.param(
            pd.Categorical(np.repeat([0, 1], 50)),
            {"num_samples": 0},
            ValueError,
            "num_samples must be greater than 1",
            id="num_samples",
        ),
        pytest.param(
            pd.Categorical(np.repeat([0, 1], 50)),
            {"class_weights": np.ones(3)},
            ValueError,
            "one weight per class",
            id="weights_shape",
        ),
        pytest.param(
            pd.Categorical(np.repeat([0, 1], 50)),
            {"class_weights": np.zeros(2)},
            ValueError,
            "at least one positive",
            id="weights_zero",
        ),
        pytest.param(
            pd.Categorical(np.repeat([0, 1], 50)),
            {"mask": slice(0, 500)},
            ValueError,
            "exceeds loader n_obs",
            id="mask_out_of_range",
        ),
        # chunk_size(10) * preload_nchunks(4) = 40 < batch_size
        pytest.param(
            pd.Categorical(np.repeat([0, 1], 50)),
            {"batch_size": 50},
            ValueError,
            "batch_size cannot exceed",
            id="batch_gt_preload",
        ),
        # 40 % 30 != 0
        pytest.param(
            pd.Categorical(np.repeat([0, 1], 50)),
            {"batch_size": 30},
            ValueError,
            "must be divisible",
            id="batch_not_divisible",
        ),
        pytest.param(np.repeat([0, 1], 50), {}, TypeError, "pandas.Categorical", id="not_categorical"),
        pytest.param(
            pd.Categorical.from_codes([-1, 0, 0, 1, 1] * 20, categories=[0, 1]),
            {},
            ValueError,
            "NA values",
            id="na_values",
        ),
    ],
)
def test_invalid_construction(
    classes: pd.Categorical | np.ndarray, kwargs: dict, error_type: type[Exception], match: str
):
    with pytest.raises(error_type, match=match):
        make_sampler(classes, **kwargs)


def test_validate_rejects_n_obs_mismatch():
    sampler = make_sampler(pd.Categorical(np.repeat([0, 1], 50)), num_samples=50)
    with pytest.raises(ValueError, match="does not match loader n_obs"):
        sampler.validate(n_obs=999)


def test_multiple_workers_not_supported():
    sampler = make_sampler(pd.Categorical(np.repeat([0, 1], 50)), num_samples=50)
    with (
        patch(
            "annbatch.samplers._class_sampler.get_torch_worker_info",
            return_value=WorkerInfo(id=0, num_workers=2),
        ),
        pytest.raises(NotImplementedError, match="Multiple workers"),
    ):
        list(sampler.sample(100))


# =============================================================================
# Core behavior
# =============================================================================


@pytest.mark.parametrize(
    "codes",
    [
        pytest.param(np.repeat([0, 1, 2, 3], 100), id="contiguous"),
        # each class split into two runs interleaved with the others
        pytest.param(np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50), id="non_contiguous"),
        pytest.param(np.array([2] * 40 + [0] * 40 + [1] * 40 + [0] * 40 + [2] * 40), id="non_contiguous_3cat"),
    ],
)
def test_chunks_are_class_coherent(codes: np.ndarray):
    chunks = _collect_chunks(make_sampler(pd.Categorical(codes), num_samples=1000), len(codes))
    assert -1 not in _chunk_classes(chunks, codes), "every chunk must lie within a single class"
    # chunks stay in-bounds and are full size (num_samples is a multiple of chunk_size here)
    assert all(0 <= c.start and c.stop <= len(codes) and c.stop - c.start == 10 for c in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "batch_size", "preload_nchunks"),
    [
        pytest.param(10, 10, 4, id="batch_eq_chunk"),
        pytest.param(10, 5, 4, id="batch_lt_chunk"),
        pytest.param(20, 5, 3, id="many_batches_per_chunk"),
    ],
)
def test_batches_are_class_coherent(chunk_size: int, batch_size: int, preload_nchunks: int):
    # the preload window mixes several classes, but each *batch* (split) must not.
    codes = np.repeat([0, 1, 2, 3], 100)
    sampler = make_sampler(
        pd.Categorical(codes),
        num_samples=400,
        chunk_size=chunk_size,
        batch_size=batch_size,
        preload_nchunks=preload_nchunks,
    )
    for load_request in sampler.sample(len(codes)):
        concat = np.concatenate([codes[s.start : s.stop] for s in load_request["requests"]])
        for split in load_request["splits"]:
            assert np.unique(concat[split]).size == 1, "every batch must lie within a single class"


def test_shuffle_is_true():
    assert make_sampler(pd.Categorical(np.repeat([0, 1], 50))).shuffle is True


def test_noncontiguous_class_samples_all_runs():
    # class 0 lives in two separate runs; over many draws both should be hit.
    codes = np.array([0] * 50 + [1] * 50 + [0] * 50, dtype=np.int64)
    starts = [
        c.start
        for c in _collect_chunks(make_sampler(pd.Categorical(codes), num_samples=5000), len(codes))
        if codes[c.start] == 0
    ]
    assert any(s < 50 for s in starts) and any(s >= 100 for s in starts), "both runs of class 0 should be sampled"


@pytest.mark.parametrize(
    ("codes", "weights", "expected"),
    [
        pytest.param(np.repeat([0, 1, 2, 3], 100), None, {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}, id="uniform"),
        pytest.param(np.array([0] * 300 + [1] * 100), np.array([300.0, 100.0]), {0: 0.75, 1: 0.25}, id="proportional"),
        pytest.param(np.repeat([0, 1, 2], 100), np.array([6.0, 3.0, 1.0]), {0: 0.6, 1: 0.3, 2: 0.1}, id="explicit"),
        pytest.param(
            np.repeat([0, 1, 2, 3], 100), np.array([1.0, 0.0, 1.0, 0.0]), {0: 0.5, 2: 0.5}, id="zero_excludes"
        ),
        pytest.param(np.repeat([0, 1, 2], 100), np.array([1.0, -1.0, 1.0]), {0: 0.5, 2: 0.5}, id="negative_excludes"),
    ],
)
def test_class_draw_shares(codes: np.ndarray, weights: np.ndarray | None, expected: dict[int, float]):
    _assert_shares(make_sampler(pd.Categorical(codes), num_samples=40_000, class_weights=weights), codes, expected)


def test_class_draw_shares_batch_exceeds_chunk():
    # batch_size > chunk_size draws one class per 2-chunk batch; shares must still track weights
    codes = np.repeat([0, 1, 2], 100)
    sampler = make_sampler(
        pd.Categorical(codes), num_samples=40_000, batch_size=20, class_weights=np.array([6.0, 3.0, 1.0])
    )
    _assert_shares(sampler, codes, {0: 0.6, 1: 0.3, 2: 0.1})


def test_zero_weight_class_exempt_from_run_length_rule():
    codes = np.array([0] * 30 + [1] * 3 + [2] * 30, dtype=np.int64)  # class 1 has a 3-row run
    # excluding class 1 with a zero weight -> its short run is exempt, no error
    make_sampler(pd.Categorical(codes), class_weights=np.array([1.0, 0.0, 1.0]))
    # giving it a positive weight -> the short run violates the run-length rule
    with pytest.raises(ValueError, match="at least chunk_size"):
        make_sampler(pd.Categorical(codes), class_weights=np.array([1.0, 1.0, 1.0]))


def test_run_length_error_names_class_labels():
    # codes are alphabetical (B=0, NK=1, T=2), so matching 'B' proves the error prints
    # this is expected to fail because the min chunk size is 10 and B has a 3-row run
    # the label, not the raw code -- the whole point of taking a pd.Categorical.
    cat = pd.Categorical(["T"] * 30 + ["B"] * 3 + ["NK"] * 30)  # "B" has a 3-row run
    with pytest.raises(ValueError, match=r"classes \['B'\]"):
        make_sampler(cat, chunk_size=10)


def test_absent_class_weight_is_ignored():
    # "c" is declared but has no observations; its weight must be silently dropped and
    # the present classes renormalize among themselves (here -> 50/50).
    cat = pd.Categorical(["a"] * 100 + ["b"] * 100, categories=["a", "b", "c"])
    codes = np.asarray(cat.codes)
    sampler = make_sampler(cat, num_samples=40_000, class_weights=np.array([1.0, 1.0, 5.0]))
    _assert_shares(sampler, codes, {0: 0.5, 1: 0.5})


# =============================================================================
# Mask
# =============================================================================


@pytest.mark.parametrize("via", ["constructor", "setter"])
def test_mask_restricts_range(via: str):
    codes = np.array([0] * 100 + [1] * 100, dtype=np.int64)
    if via == "constructor":
        sampler = make_sampler(pd.Categorical(codes), num_samples=500, mask=slice(0, 100))
    else:
        sampler = make_sampler(pd.Categorical(codes), num_samples=500)
        sampler.mask = slice(0, 100)
    chunks = _collect_chunks(sampler, len(codes))
    assert all(0 <= c.start and c.stop <= 100 for c in chunks), "chunks must stay within the mask range"
    assert {int(np.unique(codes[c])[0]) for c in chunks} == {0}


def test_mask_renormalizes_from_original_weights():
    codes = np.concatenate([np.full(100, 0), np.full(100, 1), np.full(100, 2)]).astype(np.int64)
    sampler = make_sampler(pd.Categorical(codes), num_samples=40_000, class_weights=np.array([3.0, 1.0, 6.0]))
    _assert_shares(sampler, codes, {0: 0.3, 1: 0.1, 2: 0.6})  # full range
    sampler.mask = slice(0, 200)  # only classes 0 and 1 -> renormalize [3, 1] from the originals
    _assert_shares(sampler, codes, {0: 0.75, 1: 0.25})
    sampler.mask = slice(0, None)  # back to the full range -> original weights restored
    _assert_shares(sampler, codes, {0: 0.3, 1: 0.1, 2: 0.6})


@pytest.mark.parametrize("via", ["constructor", "setter"])
def test_mask_with_no_positive_weight_in_range_raises(via: str):
    codes = np.array([0] * 50 + [1] * 50, dtype=np.int64)
    weights = np.array([1.0, 0.0])  # class 1 excluded -> the [50, 100) range has no sampleable class
    if via == "constructor":
        with pytest.raises(ValueError, match="positive weight is present"):
            make_sampler(pd.Categorical(codes), class_weights=weights, mask=slice(50, 100))
    else:
        sampler = make_sampler(pd.Categorical(codes), class_weights=weights)
        with pytest.raises(ValueError, match="positive weight is present"):
            sampler.mask = slice(50, 100)


# =============================================================================
# Bookkeeping
# =============================================================================


@pytest.mark.parametrize(
    ("num_samples", "batch_size", "drop_last", "expected_iters"),
    [
        pytest.param(100, 10, False, 10, id="exact"),
        pytest.param(105, 10, False, 11, id="partial_kept"),
        pytest.param(105, 10, True, 10, id="partial_dropped"),
    ],
)
def test_n_batches(num_samples: int, batch_size: int, drop_last: bool, expected_iters: int):
    sampler = make_sampler(
        pd.Categorical(np.repeat([0, 1], 100)),
        num_samples=num_samples,
        preload_nchunks=2,
        batch_size=batch_size,
        drop_last=drop_last,
    )
    assert sampler.n_batches(200) == expected_iters


@pytest.mark.parametrize(
    ("chunk_size", "batch_size", "preload_nchunks", "num_samples", "drop_last"),
    [
        # batch_size <= chunk_size: a chunk holds one or more batches
        pytest.param(10, 10, 4, 400, False, id="exact_windows"),  # num_samples a multiple of the window
        pytest.param(10, 10, 4, 410, False, id="partial_window"),  # trailing window has fewer slices
        pytest.param(10, 10, 4, 405, False, id="remainder_slice"),  # trailing slice shorter than chunk_size
        pytest.param(10, 5, 4, 405, False, id="multi_batch_remainder"),  # >1 batch/chunk + short last batch
        pytest.param(20, 5, 2, 410, False, id="many_batches_per_chunk"),
        pytest.param(10, 10, 1, 355, False, id="preload_one"),  # one slice per window
        pytest.param(10, 5, 3, 302, True, id="drop_last_partial_batch"),  # final 2-row batch dropped
        pytest.param(10, 10, 4, 400, True, id="drop_last_exact"),  # drop_last is a no-op when it divides evenly
        pytest.param(10, 10, 2, 95, True, id="drop_last_remainder_chunk"),  # batch==chunk, drops trailing chunk
        pytest.param(10, 10, 2, 95, False, id="keep_remainder_chunk"),  # same, partial last batch kept
        # batch_size >= chunk_size: a batch spans several same-class chunks
        pytest.param(10, 20, 4, 400, False, id="batch_two_chunks"),
        pytest.param(10, 20, 4, 410, False, id="batch_two_chunks_partial"),  # final 10-row batch kept
        pytest.param(10, 40, 4, 400, False, id="batch_whole_window"),  # one batch per window
        pytest.param(10, 20, 2, 400, False, id="batch_two_chunks_preload2"),
        pytest.param(10, 20, 4, 410, True, id="batch_two_chunks_drop"),  # final 10-row batch dropped
        # chunk_size and batch_size do not divide each other: groups of lcm rows
        pytest.param(9, 6, 2, 540, False, id="indivisible_window_one_group"),  # gcd 3, group=2, pn==group
        pytest.param(9, 6, 4, 540, False, id="indivisible_two_groups"),  # group=2, 2 groups/window
        pytest.param(9, 6, 4, 545, False, id="indivisible_remainder"),  # short final slice + partial batch
        pytest.param(9, 6, 4, 545, True, id="indivisible_drop_last"),  # final partial batch dropped
        pytest.param(10, 4, 4, 400, False, id="indivisible_gcd2"),  # gcd 2, group=2
        pytest.param(6, 5, 10, 600, False, id="coprime_group_five"),  # gcd 1, group=5
    ],
)
def test_sampling_invariants(chunk_size: int, batch_size: int, preload_nchunks: int, num_samples: int, drop_last: bool):
    codes = np.concatenate(
        [np.repeat([0, 1, 2, 3], 250), np.repeat([3, 2, 1, 0], 250)]
    )  # 4 classes, every run >> chunk_size
    n = len(codes)
    sampler = make_sampler(
        pd.Categorical(codes),
        num_samples=num_samples,
        chunk_size=chunk_size,
        batch_size=batch_size,
        preload_nchunks=preload_nchunks,
        drop_last=drop_last,
    )

    requests: list[slice] = []
    batches_seen = total_obs = 0
    for lr in sampler.sample(n):
        requests.extend(lr["requests"])
        concat = np.concatenate([codes[s.start : s.stop] for s in lr["requests"]])
        for split in lr["splits"]:
            batches_seen += 1
            total_obs += split.size
            assert np.unique(concat[split]).size == 1, "every batch must lie within a single class"

    # drop_last drops only the final incomplete batch; otherwise every requested observation is yielded
    expected_obs = (num_samples // batch_size) * batch_size if drop_last else num_samples
    assert total_obs == expected_obs
    assert sampler.n_batches(n) == batches_seen, "n_batches must match the batches actually yielded"
    assert all(0 <= s.start and s.stop <= n for s in requests), "every chunk must stay in-bounds"
    assert -1 not in _chunk_classes(requests, codes), "every chunk must lie within a single class"


@pytest.mark.parametrize("preload_nchunks", [2, 4], ids=["pn2_one_category", "pn4_two_categories"])
def test_max_classes_per_window(preload_nchunks: int):
    # chunk_size=9, batch_size=6: gcd=3, lcm=18, group_chunks=lcm/cs=2. A window holds
    # preload_nchunks // group_chunks classes, so pn=2 -> 1 per window, pn=4 -> 2 per window.
    codes = np.repeat([0, 1, 2, 3], 250)
    n = len(codes)
    expected_max = preload_nchunks // 2
    sampler = make_sampler(
        pd.Categorical(codes),
        num_samples=9 * preload_nchunks * 40,
        chunk_size=9,
        batch_size=6,
        preload_nchunks=preload_nchunks,
    )
    classes_per_window = [
        len(np.unique(np.concatenate([codes[s.start : s.stop] for s in lr["requests"]]))) for lr in sampler.sample(n)
    ]
    assert max(classes_per_window) == expected_max, f"expected up to {expected_max} classes per window"
    assert min(classes_per_window) >= 1


def test_class_sampler_from_collection(simple_collection):
    from annbatch import Loader

    _, collection = simple_collection

    # Get categories from the collection
    classes = collection.obs(columns=["src_path"])["src_path"].values

    # Create ClassSampler with categories
    sampler = ClassSampler(
        chunk_size=1,
        preload_nchunks=4,
        batch_size=4,
        classes=classes,
        num_samples=100,
    )

    # Load it to the Loader (X/obs/var only; this test doesn't exercise obsm/layers)
    loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to=None)
    loader.use_collection(collection, load_adata=load_x_obs_var)

    # Iterate through the loader and verify class-coherence of each batch
    batches = list(loader)
    assert len(batches) == 25  # 100 num_samples / 4 batch_size
    for batch in batches:
        assert batch["X"].shape == (4, 100)
        labels = batch["obs"]["src_path"]
        assert len(np.unique(labels)) == 1
