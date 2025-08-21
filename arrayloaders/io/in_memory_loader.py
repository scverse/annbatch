from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data import IterableDataset

from arrayloaders.io.abstract_dataset import AbstractIterableDataset

if TYPE_CHECKING:
    from scipy import sparse as sp


class InMemoryDataset(AbstractIterableDataset, IterableDataset):
    async def _fetch_data(
        self, slices: list[slice], dataset_idx: int
    ) -> sp.csr_matrix | sp.csr_array | np.ndarray:
        dataset = self._dataset_manager.train_datasets[dataset_idx]
        index = np.concat([np.arange(s.start, s.stop) for s in slices])
        return dataset[index]
