"""ClassSampler -- class-based chunk sampler."""

from __future__ import annotations

import numpy as np
import pandas as pd

from annbatch.samplers._utils import validate_mask_n_obs_and_resolve


class RLEManager:
    """A class for creating the RLE encoding and then generating samples from it.

    Parameters
    ----------
    mask
        The mask applied to classes.
    n_obs
        The number of observations expected.
    classes
        The class labels.
    weights
        Weights per-class.
    chunk_size
        The desired chunk size of run. Each class must be present in `classes` with at least `chunk_size` number of consecutive observations.
    """

    _mask: slice
    _classes: pd.Categorical
    _weights: np.typing.NDArray[np.floating]
    _chunk_size: int
    _class_runs: pd.DataFrame
    _per_class_sampling_info: pd.DataFrame

    def __init__(
        self,
        *,
        mask: slice,
        classes: pd.Categorical,
        weights: np.typing.NDArray[np.floating],
        chunk_size: int,
    ):
        start, stop = validate_mask_n_obs_and_resolve(mask, len(classes))
        self._mask = slice(start, stop)
        self._chunk_size = chunk_size
        self._classes = classes
        self._weights = self._build_class_weights(weights)

        self._build_rle(self._mask)

    def _build_class_weights(self, class_weights: np.ndarray | None) -> np.ndarray:
        """Resolve the (non-excluded) classes and their renormalizable weights."""
        n_classes = len(self._classes.categories)
        if class_weights is None:
            weights = np.ones(n_classes, dtype=float)
        else:
            weights = np.array(class_weights, dtype=float)
            if weights.shape != (n_classes,):
                raise ValueError(
                    f"class_weights must have one weight per class in classes.categories "
                    f"(expected shape ({n_classes},), got {weights.shape})."
                )
        if not (weights > 0).any():
            raise ValueError("class_weights must have at least one positive weight.")

        return weights  # full array (0 for excluded); codes are 0..N-1 so direct indexing works

    def _build_rle(self, mask: slice):
        start, stop = mask.start, mask.stop
        masked = self._classes.codes[mask]

        # Boundaries of where the class changes including the startings/stopping points
        edges = np.concatenate([np.array([0]), np.flatnonzero(np.diff(masked)) + 1, np.array([masked.shape[0]])])
        # Per-run table: start/end in global coordinates, length, and class
        # Keep only runs of non-excluded classes; excluded (weight 0) runs are exempt from every check
        runs = pd.DataFrame(
            {
                "start": edges[:-1] + start,
                "end": edges[1:] + start,
                "len": np.diff(edges),
                "cat": masked[edges[:-1]],
            }
        )
        runs = runs.loc[self._weights[runs["cat"].to_numpy()] > 0].reset_index(drop=True)
        if runs.empty:
            raise ValueError(
                "No class with positive weight is present in the current mask range "
                f"[{start}, {stop}); its renormalized weights would sum to zero."
            )

        # run-length rule: every kept run must hold at least one full chunk
        too_short_mask = runs["len"].to_numpy() < self._chunk_size
        if np.any(too_short_mask):
            bad = np.unique(runs.loc[too_short_mask, "cat"].to_numpy())
            bad_labels = self._classes.categories[bad].tolist()
            raise ValueError(
                f"Every contiguous run must be at least chunk_size ({self._chunk_size}) observations long, "
                f"but {int(too_short_mask.sum())} run(s) are shorter (classes {bad_labels}). "
                "Re-chunk the data so each class's runs are large enough, lower chunk_size, "
                "or exclude these classes with a zero weight."
            )

        # Sort runs by class so each class's runs are contiguous in the table;
        # `first_row_in_runs_of_class` then indexes directly into the sorted run table.
        self._class_runs = runs.sort_values("cat", kind="stable").reset_index(drop=True)

        # Per-class table: probability, number of runs, and offset into the sorted run table
        classes_to_sample, n_runs_per_class = np.unique(self._class_runs["cat"].to_numpy(), return_counts=True)
        w = self._weights[classes_to_sample]
        self._per_class_sampling_info = pd.DataFrame(
            {
                "prob": w / w.sum(),
                "n_runs": n_runs_per_class.astype(np.int64),
                "first_row_in_runs_of_class": np.concatenate(([0], np.cumsum(n_runs_per_class[:-1]))).astype(np.int64),
            },
            index=pd.Index(classes_to_sample, name="cat"),
        )

    @property
    def weights(self) -> np.ndarray:
        """The weights that the RLE generated based on classes with non-zero weights after masking.

        Returns
        -------
            The weights
        """
        return self._per_class_sampling_info["prob"].to_numpy()

    @property
    def mask(self) -> slice:
        """The currently applied mask"""
        return self._mask

    @mask.setter
    def mask(self, value: slice) -> None:
        # resolve + eagerly rebuild so range errors (run-length, no active class) surface on assignment
        mask = slice(*validate_mask_n_obs_and_resolve(value, len(self._classes)))
        # Try to build the RLE before the mask to surface any other errors
        self._build_rle(mask)
        self._mask = mask

    def slices_from_classes(self, class_of_slice: np.ndarray, rng: np.random.Generator) -> list[slice]:
        """Generate slices for the input classes from the known classes.

        So, to accomplish this, we sample one of the possible run positions within a class i.e.,
        [a: slice(0, 10), b: slice(10, 20), a: slice(20, 30)] would have two possible run positions for a (one of 0 and 2) and one for b (just 1)

        Parameters
        ----------
        class_of_slice
            An array of class labels from which to generate slices to fetch such that each slice contains only that label.

        Returns
        -------
            list of slices
        """
        class_n_runs = self._per_class_sampling_info["n_runs"].to_numpy()
        possible_run_pos_within_a_class = rng.integers(class_n_runs[class_of_slice])
        # Generate a position into the runs table to get the run to fetch within
        first_row_of_class = self._per_class_sampling_info["first_row_in_runs_of_class"].to_numpy()
        chosen = first_row_of_class[class_of_slice] + possible_run_pos_within_a_class
        # Now get that position's slice's star and end
        run_starts = self._class_runs["start"].to_numpy()[chosen]
        run_ends = self._class_runs["end"].to_numpy()[chosen]
        # Finally, sample a valid start position within each chunk so that a chunk slice can fit
        slice_starts = rng.integers(run_starts, run_ends - self._chunk_size + 1)

        return [slice(int(s), int(s + self._chunk_size)) for s in slice_starts]

    @property
    def n_classes(self):
        """The current number of active classes given weights/masking."""
        return len(self._per_class_sampling_info)
