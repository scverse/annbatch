# Preshuffling Performance Considerations

## Memory usage

While the preshuffler can operate out-of-core in {func}`annbatch.Loader.add_adatas`, each individual shuffled dataset is read into memory before being written to disk.
The amount of memory is configurable with the `dataset_size` parameter.
Upstream of that is the `shuffle_chunk_size` parameter, which controls the (location-randomized) contiguous block size on-disk from your input data that is read into memory before an in-memory shuffle.
This parameter's interaction with `dask` chunks is important - if you dask chunks are very large and `shuffle_chunk_size` considerably smaller, more memory is needed.
For this reason, {func}`anndata.experimental.read_lazy` and {func}`anndata.experimental.read_elem_lazy`'s default chunk size of 1000 is very reasonable.

## Speed

HDF5 files are quite slow as they are single-threaded (controlled by a global lock) and involve repeatedly opening and closing file handles.
In our paper {cite:p}`Gold_2026`, we showed that staring with zarr files gives a nearly 2x speedup for preshuffling.
To accelerate using hdf5 files, though, you can "virtualize" critical parts of your input datasets using [`virtualizarr`](https://virtualizarr.readthedocs.io/en/stable/) to be read through the {mod}`zarr` multihtreaded reader e.g.,

```python
import anndata as ad
from pathlib import Path
import zarr
from virtualizarr.parsers import HDFParser
from obstore.store import LocalStore
from obspec_utils.registry import ObjectStoreRegistry
from concurrent.futures import ProcessPoolExecutor
import h5py

path = Path('path_to_anndatas')

def create_X_store(path: Path):
    parser = HDFParser("X")
    store = LocalStore(prefix=f"{path.parent}/")
    registry = ObjectStoreRegistry({f"file://{str(path.parent)}/": store})
    manifest_store = parser(f"file://{str(path)}/", registry)
    return (path, manifest_store)

with ProcessPoolExecutor(max_workers=64) as executor:
    stores = dict(executor.map(create_X_store, path.glob("*.h5ad")))

def load_adata(path):
    X = ad.experimental.read_elem_lazy(zarr.open(stores[path]))
    with h5py.File(path) as f:
        var=ad.io.read_elem(f["var"])
        obs=ad.io.read_elem(f["obs"])

    return ad.AnnData(X=X, var=var, obs=obs)

collection = annbatch.DatasetCollection("path_to_collection.zarr")
collection.add_adatas(
    adata_paths=path.glob("*.h5ad"),
    load_adata=load_adata,
    n_obs_per_chunk=64,
    dataset_size="64GB",
)
```

If you can open remote hdf5 files, using zarr's internal `async` engine will also likely accelerate i/o as well.
In the future, we hope to offer full anndata object virtualization.