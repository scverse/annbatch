<picture>
  <source srcset="docs/_static/annbatch-logo-dark.svg" media="(prefers-color-scheme: dark)">
  <img src="docs/_static/annbatch-logo.svg">
</picture>

> [!IMPORTANT]
> This package will now only make breaking changes on the minor version release until its major release.

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]
[![PyPI](https://img.shields.io/pypi/v/annbatch.svg)](https://pypi.org/project/annbatch)
[![Downloads](https://static.pepy.tech/badge/annbatch/month)](https://pepy.tech/project/annbatch)
[![Downloads](https://static.pepy.tech/badge/annbatch)](https://pepy.tech/project/annbatch)
[![CZI's Essential Open Source Software for Science](https://img.shields.io/badge/funded%20by-EOSS-FF414B)](https://czi.co/EOSS)

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/annbatch/test.yaml?branch=main

[badge-docs]: https://img.shields.io/readthedocs/annbatch

A data loader and io utilities for mini-batched data loading of on-disk AnnData files as well as in-memory data, co-developed by [Lamin Labs][] and [scverse][]

## Getting started

Please refer to the [documentation][], in particular, the [API documentation][].

## Installation

```
pip install annbatch
```

Please see our [installation][] page for full documentation about extras, especially [`zarrs-python`][] which is essential for local filesystems but not for remote ones. [`numba`][] is needed for in-memory sparse data.

## Performance

We provide a speed comparison to other comparable dataloaders below:

<img src="https://raw.githubusercontent.com/scverse/annbatch/main/docs/_static/speed_comparision.png" alt="speed_comparison" width="400">

A more in-depth comparison and performance analysis is available in our paper (from which the above figure originates, see [our citation](#Citation)).

## Detailed tutorial

For a detailed tutorial, please see the [in-depth section of our docs][]

## Basic usage example

Basic preprocessing:

```python
from annbatch import DatasetCollection

import zarr
from pathlib import Path

# Using zarrs is necessary for local filesystem performance.
# Ensure you installed it using our `[zarrs]` extra i.e., `pip install "annbatch[zarrs]"` to get the right version.
zarr.config.set(
    {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
)

# Create a collection at the given path. The subgroups will all be anndata stores.
collection = DatasetCollection("path/to/output/collection.zarr")
collection.add_adatas(
    adata_paths=[
        "path/to/your/file1.h5ad",
        "path/to/your/file2.h5ad"
    ],
    shuffle=True,  # shuffling is needed if you want to use chunked access, but is the default
)
```

Data loading:

> [!IMPORTANT]
> Without custom loading via `annbatch.Loader.use_collection` or `load_adata{s}`  or `load_dataset{s}`, *all* columns of the (obs) `pandas.DataFrame` will be loaded and yielded potentially degrading performance.

```python
from pathlib import Path

from annbatch import DatasetCollection, Loader
import anndata as ad
import zarr

# Using zarrs is necessary for local filesystem performance, but should not be used for remote file systems.
# Ensure you installed it using our `[zarrs]` extra i.e., `pip install "annbatch[zarrs]"` to get the right version.
zarr.config.set(
    {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
)


# WARNING: Without custom loading *all* obs columns will be loaded and yielded potentially degrading performance.
def custom_load_func(g: zarr.Group) -> ad.AnnData:
    return ad.AnnData(
        X=ad.io.sparse_dataset(g["layers"]["counts"]),
        obs=ad.io.read_elem(g["obs"])[some_subset_of_columns_useful_for_training]
    )


# A non empty collection
collection = DatasetCollection("path/to/output/collection.zarr")
# This settings override ensures that you don't lose/alter your categorical codes when reading the data in!
with ad.settings.override(remove_unused_categories=False):
    ds = Loader(
        batch_size=4096,
        chunk_size=32,
        preload_nchunks=256,
        to_torch=True
    )
    # `use_collection` automatically uses the on-disk `X` and full `obs` in the `Loader`
    # but the `load_adata` arg can override this behavior
    # (see `custom_load_func` above for an example of customization).
    ds = ds.use_collection(collection, load_adata=custom_load_func)

# Iterate over dataloader (plugin replacement for torch.utils.DataLoader)
for batch in ds:
    x, obs = batch["X"], batch["obs"]
    # Important: For performance reasons convert to dense on GPU
    x = x.cuda().to_dense()

```

> [!IMPORTANT]
> For usage of our loader inside of `torch`, please see [this note](https://annbatch.readthedocs.io/en/latest/detailed-walkthrough.html#user-configurable-sampling-strategy) for more info.
> At the minimum, be aware that deadlocking will occur on linux unless you pass `multiprocessing_context="spawn"` to the `torch.utils.data.DataLoader` class.
> However, we strongly discourage using `torch.utils.data.DataLoader` and if you must, you should not use workers as `annbatch` is already multi-threaded.

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

If you use `annbatch` in your work, please cite the `annbatch` publication as follows:

> **annbatch unlocks terabyte-scale training of biological data in anndata**
>
> Gold, I., Fischer, F., Arnoldt, L., Wolf, F. A., & Theis, F. J. (2026b). annbatch unlocks terabyte-scale training of biological data in anndata. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2604.01949


[uv]: https://github.com/astral-sh/uv

[scverse discourse]: https://discourse.scverse.org/

[issue tracker]: https://github.com/scverse/annbatch/issues

[tests]: https://github.com/scverse/annbatch/actions/workflows/test.yaml

[documentation]: https://annbatch.readthedocs.io

[changelog]: https://annbatch.readthedocs.io/en/latest/changelog.html

[api documentation]: https://annbatch.readthedocs.io/en/latest/api.html

[pypi]: https://pypi.org/project/annbatch

[`zarrs-python`]: https://zarrs-python.readthedocs.io/

[Lamin Labs]: https://lamin.ai/

[scverse]: https://scverse.org/

[in-depth section of our docs]: https://annbatch.readthedocs.io/en/stable/notebooks/example.html

[installation]: https://annbatch.readthedocs.io/en/stable/installation.html

[`numba`]: https://numba.readthedocs.io/en/stable/
