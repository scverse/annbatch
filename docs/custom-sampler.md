# Advanced: Implementing a Custom Sampler

To implement a custom sampler, you need to understand two key components:

## {class}`annbatch.abc.Sampler`

This is the abstract base class that all samplers must inherit from. You need to implement:

- **{meth}`~annbatch.abc.Sampler._sample`**: The core sampling logic that generates load requests. This method receives the total number of observations and yields {class}`annbatch.types.LoadRequest` that specify how data should be loaded and batched. See below for more information. The outer {meth}`annbatch.abc.Sampler.sample` that wraps this implemented method calls {meth}`annbatch.abc.Sampler.validate` at runtime.

- **{meth}`~annbatch.abc.Sampler.validate`**: Validates the sampler configuration against the given number of observations. Override this method to add custom validation for your sampler parameters. It should raise a `ValueError` if the configuration is invalid.

## {class}`annbatch.types.LoadRequest`

This `TypedDict` is what {meth}`annbatch.abc.Sampler._sample` yields and specifies how data should be loaded. Each `LoadRequest` contains:

- **{attr}`~annbatch.types.LoadRequest.requests`**: A numpy array containing the indices to load or list of slices that define which contiguous chunks of memory to load from disk. When it is a list of slices, each slice should have a range up to the `chunk_size` (except the last one, which may be smaller but not empty). These slices determine which portions of the dataset are read into memory.

  ```

  Full collection (virtual conncatentation of all on disk files) (e.g., 1000 observations, on-disk chunk_size=100):
  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
  │ Chunk 0 │ Chunk 1 │ Chunk 2 │ Chunk 3 │ Chunk 4 │ Chunk 5 │ Chunk 6 │ Chunk 7 │ Chunk 8 │ Chunk 9 │
  │  0-99   │ 100-199 │ 200-299 │ 300-399 │ 400-499 │ 500-599 │ 600-699 │ 700-799 │ 800-899 │ 900-999 │
  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

  LoadRequest with requests = [slice(200,300), slice(700,800), slice(0,100), slice(500,600)]:
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
  Note: The slices are purely virtual and are defined by the user through the `requests` argument.
  They don't necessarily need to with the underlying zarr chunks.

  **Important:** The number of samples that get loaded into memory at once, must be divisible by the batch size.
  Otherwise, the remainder will yield to a smaller batch size or will be dropped if `drop_last=True`.

- **{attr}`~annbatch.types.LoadRequest`** (optional): A list of numpy arrays that define how the loaded data should be split into batches after being read from disk and concatenated in memory.
  - If not supplied: batches are randomly created based on the loaded requests.
  - If supplied: you can control how batches are created from the in-memory requests. Each array contains indices in **request order** -- position `j` is the `j`-th observation when the chunks are concatenated in the order listed in `requests` (exactly as drawn below).

```{note}
The `splits` parameter gives you fine-grained control over how individual batches are created based on the loaded requests. This capability is particularly useful when you want to organize batches based on semantic labels, categories, or other metadata.
```
```{important}
Split indices are always in **chunk order**. Internally the loader fetches chunks grouped by on-disk dataset for efficiency, so the physical in-memory layout may be reordered, but it remaps your splits back to chunk order before yielding. You therefore never need to account for how chunks map to datasets.

*Changed in 0.2.0:* splits previously had to index into the dataset-grouped physical layout; they now index in chunk order. Custom samplers that compensated for the dataset reordering must drop that compensation.
```

  ```

  Concatenated in-memory data (top row) from slices (bottom row) of 400 observations:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  0   1   2   3  ...  99 100 101  ...  199 200  ...  299 300  ...  399   │
  │                                                                         │
  │  [Slice 200-299]    [Slice 700-799]   [Slice 0-99]   [Slice 500-599]    │
  └─────────────────────────────────────────────────────────────────────────┘

  `LoadRequest` with splits for batch size of 4 = [np.array([0,50,150,250]),
                             np.array([1,51,151,251]),
                             np.array([2,52,152]), ...]:

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


## Example 1: Implementing a `InOrderSampler` class

This example demonstrates creating a simple sampler that only loads sequential, non-random requests of data from disk and yields them in-order:

```python
from annbatch.abc import Sampler
from collections.abc import Iterator
from annbatch.types import LoadRequest
import numpy as np


class InOrderSampler(Sampler):

    def __init__(self, batch_size: int, chunk_size: int):
        self.batch_size = batch_size
        self.chunk_size = chunk_size

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        """Generate load requests."""
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
            yield {"requests": [chunk], "splits": batch_indices}
```


## Example 2: Implementing a `RandomSampler` class

Here we have a sampler that just samples single observations off disk. This is extremely inefficient but instructive (see below for performance considerations):
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
            yield {"splits": [np.arange(self.batch_size)], "requests": [slice(idx, idx + 1) for idx in i]}

```


## Performance Considerations

When implementing a custom sampler, it's crucial to consider **disk access patterns** to ensure efficient data loading.
The performance of your sampler heavily depends on how it accesses data from disk-backed arrays.

### The Core Strategy: Chunked Reads + In-Memory Shuffling

The key to efficient sampling from disk-backed arrays is to **read data from many contiguous chunks** (i.e., more than your batch size generally) and then **shuffle in memory**.
This approach minimizes expensive disk seeks (via sequential reads), improves speed via parallelization (by prefetching many batches), and gives sufficient randomness in your batches (because all that data will be shuffled).

```
Recommended pattern:
┌──────────────────────────────────────────────────────────────────────────────────┐
│  1. Read contiguous chunk(s) from disk       →      2. Shuffle in memory         │
│                                                                                  │
│     Disk (sequential reads per slice)                Memory (shuffled together)  │
│  ┌───────────────┐                                ┌──────────────────────┐       │
│  │ Chunk 0: 0-3  │  ═══════════╗                  │  8  2 11  0  5  9    │       │
│  └───────────────┘             ║                  │ 10  1  4  7  3  6    │       │
│  ┌───────────────┐             ╠═══════════════>  └──────────────────────┘       │
│  │ Chunk 1: 4-7  │  ═══════════╣                             ↓                   │
│  └───────────────┘             ║                       yield batches             │
│  ┌───────────────┐             ║                                                 │
│  │ Chunk 2: 8-11 │  ═══════════╝                  [8 2] [11 0] [5 9] ...         │
│  └───────────────┘                                                               │
└──────────────────────────────────────────────────────────────────────────────────┘
```


This strategy stands in contrast to fully random reads:
```
Anti-pattern (slow):
Read index 42 → Read index 789 → Read index 15 → Read index 456 → ...
    └──────────────┴──────────────┴──────────────┴──────────────┘
    Many random disk seeks across the entire dataset
```

Each random read may:
- Require loading an entire chunk just to extract one element (whereas sequential reads will use most if not all elements from every chunk)
- Cause the disk head to seek to a completely different location orders of magnitude more often (e.g., a factor of `chunk_size` in {class}`~annbatch.samplers.RandomSampler`)

### The Randomness Trade-off

Chunked reading comes with an inherent trade-off: **reduced randomness**.
When you load contiguous blocks and shuffle in memory, samples within a batch are more likely to come from the same block(s).
This means:

- Samples in a batch may be **correlated** (e.g., neighboring cells, adjacent time points)
- You get **local randomness** (within blocks) but not **global randomness** (across the entire dataset)

That is why `annbatch` provides tools to:
1. **Preshuffle your data** during dataset creation to break up correlations via {class}`~annbatch.DatasetCollection`
2. **Load multiple random slices** per batch to increase diversity (see `preload_nchunks` parameter of {class}`~annbatch.Loader` or {class}`~annbatch.samplers.RandomSampler`)
3. **Use larger in-memory buffers** to shuffle across more blocks (accelerated via `preload_to_gpu` argument to {class}`~annbatch.Loader`)
