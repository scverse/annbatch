# arrayloaders

> [!CAUTION]
> This pacakge does not have a stable API.  However, we do not anticipate the on-disk format to change as it is simply an anndata file.

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/laminlabs/arrayloaders/test.yaml?branch=main

[badge-docs]: https://img.shields.io/readthedocs/arrayloaders

A minibatch loader for anndata stores

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.11 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install arrayloaders:

<!--
1) Install the latest release of `arrayloaders` from [PyPI][]:

```bash
pip install arrayloaders
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/laminlabs/arrayloaders.git@main
```

We provide extras in the `pyproject.toml` for `torch`, `cupy-cuda12`, `cupy-cuda13`, and `zarrs`.
`cupy` provides accelerated handling of the data once it has been read off disk and does not need to be used in conjunction with `torch`.
> [!IMPORTANT] `zarrs` gives the necessary performance boost for the sharded data produced by {func}`arrayloaders.create_anndata_collection`.

## Basic usage example

First, you'll need to convert your existing `.h5ad` files into a zarr-backed anndata format.
In the process, the data gets shuffled and is distributed across several anndata files.

### Preprocessing

```python
from arrayloaders import create_anndata_collection

create_anndata_collection(
    adata_paths=[
        "path/to/your/file1.h5ad",
        "path/to/your/file2.h5ad"
    ],
    output_path="path/to/output/store", # a directory containing `chunk_{i}.zarr`
    shuffle=True,  # shuffling is needed if you want to use chunked access
)
```

### Data loading

#### Chunked access

The data loader implements a chunked fetching strategy where `preload_nchunks` number of continguous-chunks of size `chunk_size` are loaded.
`chunk_size` corresponds the number of rows of `anndata` store to load sequentially.

Here is a short snippet to get you started:

```python
from pathlib import Path

import anndata as ad
import zarr
import zarrs

# Using zarrs is necessary for local filesystem perforamnce.
zarr.config.set(
    {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
)

from arrayloaders import ZarrSparseDataset

PATH_TO_STORE = Path("path/to/output/store")

ds = ZarrSparseDataset(
    batch_size=4096,
    chunk_size=32,
    preload_nchunks=256,
).add_anndatas(
    [
        ad.AnnData(
            # note that you can open an anndata file using any type of zarr store
            X=ad.io.sparse_dataset(zarr.open(p)["X"]),
            obs=ad.io.read_elem(zarr.open(p)["obs"]),
        )
        for p in PATH_TO_STORE.glob("*.zarr")
    ],
    obs_keys="label_column",
)

# Iterate over dataloader like normal
for batch in ds:
    ...
```

For performance reasons, you should use our dataloader directly without wrapping it into a {class}`torch.utils.data.DataLoader`.
Your code will work the same way as with a {class}`torch.utils.data.DataLoader`, but you will get better performance.

The sharded zarr file format output from {func}`arrayloaders.create_anndata_collection` is meant to reduce the burden on file systems of indexing.
In order to take advantage of this feature to its fullest performance, though, locally, you must set the codec pipeline to use `zarrså`.
We have not tested remote data (i.e., using {func}`zarr.open` with a {class}`zarr.storage.ObjectStore`) but because we use {mod}`zarr`, this data loader should also work over cloud connections via relevant zarr stores.
Note that `zarrs` cannot be used with these sorts of stores.

#### User configurable sampling strategy

At the moment we do not support user-configurable sampling strategies like weighting or sampling.
With a pre-shuffled store and blocked access, your model fit should not be affected by using chunked access.

If you are interested in contributing this feature to the project or leaning more, please get in touch on [zulip](https://scverse.zulipchat.com/) or via the GitHub issues here.

## Speed comparison to other dataloaders

We provide a quickstart notebook that gives both some boilerplate code and provides a speed comparison to other comparable dataloaders:

TODO: figure and notebook

## Why data loading speed matters?

Most models for scRNA-seq data are pretty small in terms of model size compared to models in other domains like computer vision or natural language processing.
This size differential puts significantly more pressure on the data loading pipeline to fully utilize a modern GPU.
Intuitively, if the model is small, doing the actual computation is relatively fast.
Hence, to keep the GPU fully utilized, the data loading needs to be a lot faster.

As an illustrative, example let's train a logistic regression model ([notebook hosted on LaminHub](https://lamin.ai/laminlabs/arrayloader-benchmarks/transform/cV00NQStCAzA?filter%5Band%5D%5B0%5D%5Bor%5D%5B0%5D%5Bbranch.name%5D%5Beq%5D=main&filter%5Band%5D%5B1%5D%5Bor%5D%5B0%5D%5Bis_latest%5D%5Beq%5D=true)).
Our example model has 20.000 input features and 100 output classes. We can now look how the total fit time changes with data loading speed:

<img src="docs/_static/fit_time_vs_loading_speed.png" alt="fit_time_vs_loading_speed" width="400">

From the graph we can see that the fit time can be decreased substantially with faster data loading speeds (several orders of magnitude).
E.g. we are able to reduce the fit time from ~280s for a data loading speed of ~1000 samples/sec to ~1.5s for a data loading speed of ~1.000.000 samples/sec.
This speedup is more than 100x and shows the significant impact data loading has on total training time.

## When would you use this data laoder?

As we just showed, data loading speed matters for small models (e.g., on the order of an scVI model, but perhaps not a "foundation model").
But loading minibatches of bytes off disk will be almost certainly slower than loading them from an in-memory source.
Thus, as a first step to assessing your needs, if your data fits in memory, load it into memory.
However, once you have too much data to fit into memory, for whatever reason, the data loading functionality offered here can provide significant speedups over state of the art out-of-core dataloaders.

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv

[scverse discourse]: https://discourse.scverse.org/

[issue tracker]: https://github.com/laminlabs/arrayloaders/issues

[tests]: https://github.com/laminlabs/arrayloaders/actions/workflows/test.yaml

[documentation]: https://arrayloaders.readthedocs.io

[changelog]: https://arrayloaders.readthedocs.io/en/latest/changelog.html

[api documentation]: https://arrayloaders.readthedocs.io/en/latest/api.html

[pypi]: https://pypi.org/project/arrayloaders
