# Advanced: Implementing a Custom Sampler

## Overview

To implement a custom sampler, you need to understand two key components:

### 1. `annbatch.abc.Sampler`

This is the abstract base class that all samplers must inherit from. You need to implement:

- **{meth}`annbatch.abc.Sampler._sample`**: The core sampling logic that generates load requests. This method receives the total number of observations and yields {class}`annbatch.types.LoadRequest that specify how data should be loaded and batched.  See below for more information.  The outer {meth}`annbatch.abc.Sampler.sample` that wraps this implemented method calls {meth}`annbatch.abc.Sampler.validate` at runtime.

- **`validate(n_obs: int) -> None`** (optional): Validates the sampler configuration against the given number of observations. Override this method to add custom validation for your sampler parameters. It should raise a `ValueError` if the configuration is invalid.

### 2. `annbatch.types.LoadRequest`

A `TypedDict` that specifies how data should be loaded. Each `LoadRequest` contains:

- **`chunks`**: A list of slices that define which chunks to load from disk. Each slice should have a range up to the chunk_size (except the last one, which may be smaller but not empty). These slices determine which portions of the dataset are read into memory.

  ```
  Example: Loading random chunks from a large array

  Full collection (virtual conncatentation of all on disk files) (e.g., 1000 observations, chunk_size=100):
  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
  │ Chunk 0 │ Chunk 1 │ Chunk 2 │ Chunk 3 │ Chunk 4 │ Chunk 5 │ Chunk 6 │ Chunk 7 │ Chunk 8 │ Chunk 9 │
  │  0-99   │ 100-199 │ 200-299 │ 300-399 │ 400-499 │ 500-599 │ 600-699 │ 700-799 │ 800-899 │ 900-999 │
  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

  LoadRequest with chunks = [slice(200,300), slice(700,800), slice(0,100), slice(500,600)]:
  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
  │    ✓    │         │    ✓    │         │         │    ✓    │         │    ✓    │         │         │
  │ Chunk 0 │         │ Chunk 2 │         │         │ Chunk 5 │         │ Chunk 7 │         │         │
  │  0-99   │         │ 200-299 │         │         │ 500-599 │         │ 700-799 │         │         │
  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
      ↓                   ↓                             ↓                     ↓
      └───────────────────┴─────────────────────────────┴─────────────────────┘
                                          ↓
  Loaded into memory and concatenated (400 observations):
  ┌─────────┬─────────┬─────────┬─────────┐
  │ 200-299 │ 700-799 │  0-99   │ 500-599 │
  └─────────┴─────────┴─────────┴─────────┘
  ```
  **Important:** The number of samples that get loaded into memory at once, should be devisible by the batch size. Otherwise, the reminder will get dropped.

- **`splits`** (optional): A list of numpy arrays that define how the loaded data should be split into batches after being read from disk and concatenated in memory.
  - If not supplied: batches are randomly created based on the loaded chunks.
  - If supplied: you can control how batches are created from the in-memory chunks. Each array contains indices that map into the concatenated in-memory data.
  - The `splits` parameter gives you fine-grained control over how individual batches are created based on the loaded chunks. This is particularly useful when you want to organize batches based on semantic labels, categories, or other metadata.

  ```
  Example: Splitting concatenated data into batches

  Concatenated in-memory data from chunks (400 observations):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  0   1   2   3  ...  99 100 101  ...  199 200  ...  299 300  ...  399   │
  │                                                                         │
  │  [Chunk 200-299]    [Chunk 700-799]   [Chunk 0-99]   [Chunk 500-599]    │
  └─────────────────────────────────────────────────────────────────────────┘

  LoadRequest with splits = [np.array([0,50,150,250]),
                             np.array([1,51,151,251]),
                             np.array([2,52,152])]:

  Batch 1 (4 observations):
  ┌───────────────────────────────────────────────────────────────────┐
  │  indices [0, 50, 150, 250]                                        │
  │     ↓    ↓     ↓     ↓                                            │
  │  ┌───┬────┬────┬────┐                                             │
  │  │ 0 │ 50 │ 150│ 250│  → batch_size = 4                           │
  │  └───┴────┴────┴────┘                                             │
  └───────────────────────────────────────────────────────────────────┘

  Batch 2 (4 observations):
  ┌───────────────────────────────────────────────────────────────────┐
  │  indices [1, 51, 151, 251]                                        │
  │     ↓   ↓    ↓     ↓                                              │
  │  ┌───┬────┬────┬────┐                                             │
  │  │ 1 │ 51 │ 151│ 251│  → batch_size = 4                           │
  │  └───┴────┴────┴────┘                                             │
  └───────────────────────────────────────────────────────────────────┘

  Batch 3 (3 observations):
  ┌───────────────────────────────────────────────────────────────────┐
  │  indices [2, 52, 152]                                             │
  │     ↓   ↓    ↓                                                    │
  │  ┌───┬────┬────┐                                                  │
  │  │ 2 │ 52 │ 152│  → batch_size = 3 (last split can be partial)    │
  │  └───┴────┴────┘                                                  │
  └───────────────────────────────────────────────────────────────────┘
  ```


## Example 1: Implementing a `ChunkedSampler` class

This example demonstrates an efficient sampler that loads contiguous chunks of data from disk:

```python
from annbatch.abc import Sampler
from collections.abc import Iterator
from annbatch.types import LoadRequest
import numpy as np


class ChunkedSampler(Sampler):

    def __init__(self, batch_size: int, chunk_size: int):
        self.batch_size = batch_size
        self.chunk_size = chunk_size

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        """Generate load requests for chunks."""
        # Create all chunk boundaries
        chunk_starts = list(range(0, n_obs, self.chunk_size))
        # Shuffle the chunks
        rng = np.random.default_rng()
        rng.shuffle(chunk_starts)
        # Process chunks in shuffled order
        for start in chunk_starts:
            end = min(start + self.chunk_size, n_obs)
            chunk = slice(start, end)
            # Split the chunk into batches
            chunk_size_actual = end - start
            batch_indices = [
                np.arange(i, min(i + self.batch_size, chunk_size_actual))
                for i in range(0, chunk_size_actual, self.batch_size)
            ]
            yield {"chunks": [chunk], "splits": batch_indices}
```


## Example 2: Implementing a `RandomSampler` class
```python
from annbatch.abc import Sampler
from collections.abc import Iterator
from annbatch.types import LoadRequest
import numpy as np


class RandomSampler(Sampler):

    def __init__(self, n_obs: int, batch_size: int):
        self.n_obs = n_obs
        self.batch_size = batch_size

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        for i in np.array_split(np.random.default_rng().permutation(self.n_obs), self.n_obs // self.batch_size):
            yield {"splits": [np.arange(self.batch_size)], "chunks": [slice(idx, idx + 1) for idx in i]}

```


## Performance Considerations

When implementing a custom sampler, it's crucial to consider **disk access patterns** to ensure efficient data loading. The performance of your sampler heavily depends on how it accesses data from disk-backed arrays.

### Efficient vs. Inefficient Access Patterns

**✅ Example 1 (ChunkedSampler) - Efficient:**
- Loads **contiguous chunks** of data: `[slice(0,100), slice(100,200), slice(200,300)]`
- Minimizes disk seeks by reading sequential blocks
- Takes advantage of chunk-based storage formats
- Optimal for Zarr arrays where data is stored in chunks

```
Disk access pattern (sequential):
Read chunk 0 → Read chunk 1 → Read chunk 2 → ...
└─────────┴─────────┴─────────┘
     Fast sequential reads
```

**❌ Example 2 (RandomSampler) - Inefficient:**
- Loads **individual random observations**: `[slice(42,43), slice(789,790), slice(15,16)]`
- Causes many small disk seeks across the entire dataset
- Each observation may require loading an entire chunk just to extract one element
- Significantly slower for large datasets

```
Disk access pattern (random):
Read obs 42 → Read obs 789 → Read obs 15 → ...
    └────────────┴────────────┴────────┘
    Many random disk seeks (SLOW!)
```

### Best Practices

1. **Load contiguous chunks** whenever possible load data in contiguous chunks and not just indviudal samples
2. **Preshuffle data** during dataset creation if you need random access patterns during training
