# Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv](https://github.com/astral-sh/uv).

## PyPI

Install the latest release of `annbatch` from [PyPI](https://pypi.org/project/annbatch):

```bash
pip install "annbatch[zarrs]"
```

:::{important}
[zarrs-python](https://zarrs-python.readthedocs.io/) gives the necessary performance boost for the
sharded data produced by our preprocessing functions to be useful when loading data off a local
filesystem, so we recommend always installing the `zarrs` extra.
:::

## Optional dependencies

`annbatch` ships several extras that you can mix and match:

| Extra | What it adds |
| --- | --- |
| `zarrs` | High-performance zarr codec pipeline via [zarrs-python](https://zarrs-python.readthedocs.io/) for local filesystems — strongly recommended. |
| `torch` | Yields batches as {class}`torch.Tensor`s. |
| `cupy-cuda12` | GPU acceleration via `cupy` for CUDA 12. |
| `cupy-cuda13` | GPU acceleration via `cupy` for CUDA 13. |

`cupy` provides accelerated handling of the data via `preload_to_gpu` once it has been read off
disk, and does not need to be used in conjunction with `torch`.

To install several extras at once:

```bash
pip install "annbatch[zarrs,torch,cupy-cuda13]"
```

(Replace `cupy-cuda13` with the extra matching your local CUDA version.)

:::{important}
Always quote the package specifier (`"annbatch[zarrs,torch]"`) and do **not** put spaces between
the extras. Most shells (bash, zsh) treat the square brackets as glob patterns, so an unquoted
`annbatch[zarrs,torch]` — or one written as `annbatch[zarrs, torch]` — will fail to install.
:::
