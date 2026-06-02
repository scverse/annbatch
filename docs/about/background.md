# About annbatch

`annbatch` is a data loader and io utilities for mini-batched data loading of on-disk
[AnnData](https://anndata.readthedocs.io/) files. It is built to train models on terabyte-scale
collections of `AnnData` that do not fit into memory, while keeping a modern GPU fully utilized
with high-throughput, shuffled mini-batches.

## Why annbatch?

Most models for scRNA-seq data are small compared to models in computer vision or natural language
processing, which shifts the bottleneck from compute onto the data-loading pipeline: to keep the
GPU busy, data loading has to be fast. annbatch combines a chunked, block-shuffled fetching
strategy with sharded, zarr-backed `AnnData` stores — accelerated locally by
[zarrs-python](https://zarrs-python.readthedocs.io/) — to deliver order-of-magnitude faster loading
than other out-of-core dataloaders. See the {doc}`Overview </overview>` for benchmarks and details.

## Ecosystem

annbatch is co-developed by [Lamin Labs](https://lamin.ai/) and [scverse](https://scverse.org/),
and builds directly on [anndata](https://anndata.readthedocs.io/),
[zarr](https://zarr.readthedocs.io/) and [zarrs-python](https://zarrs-python.readthedocs.io/).

## Funding

annbatch is supported by the Chan Zuckerberg Initiative's
[Essential Open Source Software for Science (EOSS)](https://czi.co/EOSS) program.

## scverse

annbatch is part of the scverse® project ([website](https://scverse.org/),
[governance](https://scverse.org/about/roles)), which is fiscally sponsored by
[NumFOCUS](https://numfocus.org/). If you like scverse and want to support our mission, please
consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the
project pay for developer time, professional services, travel, workshops, and a variety of other
needs.

## Citing annbatch

If you use annbatch in your work, please cite it — see {doc}`cite`.
