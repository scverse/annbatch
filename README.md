<!--Links at the top because this document is split for docs home page-->

[uv]: https://github.com/astral-sh/uv

[scverse discourse]: https://discourse.scverse.org/

[issue tracker]: https://github.com/laminlabs/arrayloaders/issues

[tests]: https://github.com/laminlabs/arrayloaders/actions/workflows/test.yaml

[documentation]: https://arrayloaders.readthedocs.io

[changelog]: https://arrayloaders.readthedocs.io/en/latest/changelog.html

[api documentation]: https://arrayloaders.readthedocs.io/en/latest/api.html

[pypi]: https://pypi.org/project/arrayloaders

[zarrs-python]: https://zarrs-python.readthedocs.io/

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

We provide extras in the `pyproject.toml` for `torch`, `cupy-cuda12`, `cupy-cuda13`, and [zarrs-python][].
`cupy` provides accelerated handling of the data once it has been read off disk and does not need to be used in conjunction with `torch`.
> [!IMPORTANT]
> [zarrs-python][] gives the necessary performance boost for the sharded data produced by our preprocessing functions to be useful.

## Basic usage example

Basic preprocessing:
```python
from arrayloaders import create_anndata_collection

import zarr
from pathlib import Path
import zarrs   # noqa: F401

# Using zarrs is necessary for local filesystem perforamnce.
zarr.config.set(
    {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
)

create_anndata_collection(
    adata_paths=[
        "path/to/your/file1.h5ad",
        "path/to/your/file2.h5ad"
    ],
    output_path="path/to/output/collection", # a directory containing `dataset_{i}.zarr`
    shuffle=True,  # shuffling is needed if you want to use chunked access
)
```

Data loading:

```python
from pathlib import Path

from arrayloaders import ZarrSparseDataset
import anndata as ad
import zarr
import zarrs   # noqa: F401

# Using zarrs is necessary for local filesystem perforamnce.
zarr.config.set(
    {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
)

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
        for p in Path("path/to/output/collection").glob("*.zarr")
    ],
    obs_keys="label_column",
)

# Iterate over dataloader (plugin replacement for torch.utils.DataLoader)
for batch in ds:
    ...
```

<!--HEADER-->

For a deeper dive, please see our [LINK BROKEN: docs page](XXXXX)

<!--FOOTER-->
## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a
