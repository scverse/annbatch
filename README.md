<!--Links at the top because this document is split for docs home page-->

[uv]: https://github.com/astral-sh/uv

[scverse discourse]: https://discourse.scverse.org/

[issue tracker]: https://github.com/scverse/annbatch/issues

[tests]: https://github.com/scverse/annbatch/actions/workflows/test.yaml

[documentation]: https://annbatch.readthedocs.io

[changelog]: https://annbatch.readthedocs.io/en/latest/changelog.html

[api documentation]: https://annbatch.readthedocs.io/en/latest/api.html

[pypi]: https://pypi.org/project/annbatch

[zarrs-python]: https://zarrs-python.readthedocs.io/

[Lamin Labs]: https://lamin.ai/

[scverse]: https://scverse.org/

[in-depth section of our docs]: https://annbatch.readthedocs.io/en/latest/notebooks/example.html

# annbatch

> [!CAUTION]
> This package does not have a stable API.
> However, we do not anticipate the on-disk format to change in an incompatible manner.

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]
[![PyPI](https://img.shields.io/pypi/v/annbatch.svg)](https://pypi.org/project/annbatch)
[![Downloads](https://static.pepy.tech/badge/annbatch/month)](https://pepy.tech/project/annbatch)
[![Downloads](https://static.pepy.tech/badge/annbatch)](https://pepy.tech/project/annbatch)

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/annbatch/test.yaml?branch=main

[badge-docs]: https://img.shields.io/readthedocs/annbatch

A data loader and io utilities for minibatching on-disk AnnData, co-developed by [Lamin Labs][] and [scverse][]

## Getting started

Please refer to the [documentation][], in particular, the [API documentation][].

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

To install the latest release of `annbatch` from [PyPI][]:

```bash
pip install annbatch
```

We provide extras for `torch`, `cupy-cuda12`, `cupy-cuda13`, and [zarrs-python][].
`cupy` provides accelerated handling of the data via `preload_to_gpu` once it has been read off disk and does not need to be used in conjunction with `torch`.
> [!IMPORTANT]
> [zarrs-python][] gives the necessary performance boost for the sharded data produced by our preprocessing functions to be useful when loading data off a local filesystem.

## Detailed tutorial

For a detailed tutorial, please see the [in-depth section of our docs][]

## Basic usage example

Basic preprocessing:

```python
from annbatch import create_anndata_collection

import zarr
from pathlib import Path

# Using zarrs is necessary for local filesystem performance.
# Ensure you installed it using our `[zarrs]` extra i.e., `pip install annbatch[zarrs]` to get the right version.
zarr.config.set(
    {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
)
# a directory containing `dataset_{i}.zarr`
collection = PreShuffledCollection("path/to/output/collection.zarr")
collection.create_anndata_collection(
    adata_paths=[
        "path/to/your/file1.h5ad",
        "path/to/your/file2.h5ad"
    ],
    shuffle=True,  # shuffling is needed if you want to use chunked access
)
```

Data loading:

```python
from pathlib import Path

from annbatch import Loader
import anndata as ad
import zarr

# Using zarrs is necessary for local filesystem performance.
# Ensure you installed it using our `[zarrs]` extra i.e., `pip install annbatch[zarrs]` to get the right version.
zarr.config.set(
    {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
)

# This settings override ensures that you don't lose/alter your categorical codes when reading the data in!
with ad.settings.override(remove_unused_categories=False):
    ds = Loader(
        batch_size=4096,
        chunk_size=32,
        preload_nchunks=256,
    ).add_anndatas(
        [
            ad.AnnData(
                # note that you can open an AnnData file using any type of zarr store
                X=ad.io.sparse_dataset(zarr.open(p)["X"]),
                obs=ad.io.read_elem(zarr.open(p)["obs"]),
            )
            for p in Path("path/to/output/collection").glob("*.zarr")
        ]
    )

# Iterate over dataloader (plugin replacement for torch.utils.DataLoader)
for batch in ds:
    ...
```

> [!IMPORTANT]
> For usage of our loader inside of `torch`, please see [this note](https://annbatch.readthedocs.io/en/latest/#user-configurable-sampling-strategy) for more info.
> At the minimum, be aware that deadlocking will occur on linux unless you pass `multiprocessing_context="spawn"` to the `torch.utils.data.DataLoader` class.

<!--FOOTER-->

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].
