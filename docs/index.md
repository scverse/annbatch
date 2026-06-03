# annbatch

A data loader and io utilities for mini-batched data loading of on-disk {mod}`anndata` files,
co-developed by [Lamin Labs](https://lamin.ai/) and [scverse](https://scverse.org/).

`annbatch` lets you train models on terabyte-scale collections of `AnnData` files that do not fit
into memory, while keeping your GPU fed with high-throughput, shuffled mini-batches.

```{image} _static/speed_comparision.png
:alt: annbatch data-loading speed compared to other dataloaders
:class: annbatch-hero
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`desktop-download;1.5em;sd-mr-1` Installation
:link: installation
:link-type: doc

New to *annbatch*? Check out the installation guide and pick the right extras.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quickstart
:link: notebooks/example
:link-type: doc

A hands-on notebook: convert your `.h5ad` files and stream shuffled mini-batches.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` User guide
:link: detailed-walkthrough
:link-type: doc

An in-depth tour of preprocessing, chunked loading, sampling and benchmarks.
:::

:::{grid-item-card} {octicon}`code-square;1.5em;sd-mr-1` API reference
:link: api
:link-type: doc

The API reference contains a detailed description of the *annbatch* API.
:::

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-mr-1` Discussion
:link: https://discourse.scverse.org/

Need help? Reach out on the scverse forum to get your questions answered.
:::

:::{grid-item-card} {octicon}`mark-github;1.5em;sd-mr-1` GitHub
:link: https://github.com/scverse/annbatch

Found a bug? Interested in contributing? Check out the source on GitHub.
:::

::::

## Citation

```{eval-rst}
.. include:: about/cite.md
    :start-line: 2
    :parser: myst
```

```{toctree}
:caption: General
:hidden:
:maxdepth: 1

installation
api
changelog
contributing
references
```

```{toctree}
:caption: Tutorials
:hidden:
:maxdepth: 1

notebooks/example
```

```{toctree}
:caption: User guide
:hidden:
:maxdepth: 1

detailed-walkthrough
zarr-configuration
preshuffling
custom-sampler
```

```{toctree}
:caption: About
:hidden:
:maxdepth: 1

about/background
about/cite
GitHub <https://github.com/scverse/annbatch>
Discourse <https://discourse.scverse.org/>
```
