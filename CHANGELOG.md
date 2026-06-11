# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html



## [0.2.0]

### Feature
- Add a `merge` argument to {meth}`annbatch.DatasetCollection.add_adatas` to handle how columns in {attr}`~anndata.AnnData.var` are handled when creating the on-disk dataset.
- Now {attr}`annbatch.types.LoadRequest.requests` (formerly `annbatch.types.LoadRequest.chunks`) can also be a numpy array of integers.
- Support in memory matrices ({class}`scipy.sparse.csr_matrix`, {class}`scipy.sparse.csr_array`, {class}`numpy.ndarray`) requiring `numba` for the sparse cases (new additional dependency group for `numba` included).
- Added {class}`annbatch.samplers.ClassSampler`: a replacement sampler that takes a {class}`pandas.Categorical` (e.g. `adata.obs["cell_type"].astype("category")`). Sampling is with replacement; for each batch, classes are drawn by `class_weights` (uniform by default).

### Breaking
- Removal of deprecated `annbatch.ChunkSampler`
- Deprecated `n_iters` method of {class}`~annbatch.abc.Sampler` in favor of `n_batches` to match the actual semantics of the returned value.
- Added `n_batches` method to {class}`~annbatch.abc.Sampler` and all sampler implementations.
- Remove deprecated `concat_strategy` argument from {class}`~annbatch.Loader`
- {attr}`annbatch.types.LoadRequest.splits` now index in **request order** -- position `j` is the `j`-th observation when the request's `chunks` are concatenated in the order given. Previously, `splits` had to index into the loader's internal dataset-grouped memory layout. The {class}`~annbatch.Loader` now remaps splits to that layout itself, so custom samplers must produce chunk-order splits and stop compensating for the dataset reordering.
- Deprecated `annbatch.types.LoadRequest.chunks` in favor of {attr}`annbatch.types.LoadRequest.requests`.

### Fixed
- Handling of different data types i.e., `float32` vs `float64` in the same {class}`~annbatch.Loader`

## [0.1.6]

### Performance
- New internal use of {class}`numpy.ndarray` for indexing means the small chunk sizes (i.e., perfect random sampling) is much more performant.

### Docs
- New docs including a logo!

## [0.1.5]

### Fixed
- Handle indexers for `indices` and `data` separately because their underlying chunking can differ

## [0.1.4]

### Performance
- Preallocate buffers for in-memory handling.  `concat_strategy` argument no longer has any affect as the new strategy is as memory efficient and as fast as both strategies.

### Features
- Added `groupby` support to {meth}`annbatch.DatasetCollection.add_adatas` to group observations per dataset before writing collections. When appending to an existing on-disk collection, groupby columns must already exist and categorical categories must be identical to those on-disk.

## [0.1.3]

### Features
- Added {class}`annbatch.samplers.RandomSampler` and {class}`annbatch.samplers.SequentialSampler` as replacements for `annbatch.ChunkSampler`.
- Exposed {class}`annbatch.samplers.DistributedSampler` for distributed training.

### Breaking
- Deprecated `annbatch.ChunkSampler` in favor of {class}`annbatch.samplers.RandomSampler` and {class}`annbatch.samplers.SequentialSampler`.

## [0.1.2]

### Fixed

- To handle `torch>=2.11` + `cupy-cuda12x`, because `torch` installs `cuda13` by default from this version onwards, we now install `cupy-cuda12x[ctk]` to ensure the `cuda` version used matches that of `cupy`.  For information on this change [see the cupy docs](https://docs.cupy.dev/en/stable/install.html#installing-cupy).

## [0.1.1]

### Fixed

- Exclude `torch` 2.11 on account of https://github.com/cupy/cupy/issues/9827

## [0.1.0]

### Breaking
- Renamed `annbatch.Loader.add_anndatas` to {meth}`annbatch.Loader.add_adatas`.
- Renamed `annbatch.Loader.add_anndata` to {meth}`annbatch.Loader.add_adata`.
- The ``sparse_chunk_size``, ``sparse_shard_size``, ``dense_chunk_size``, and ``dense_shard_size`` parameters of {func}`annbatch.write_sharded` have been replaced by ``n_obs_per_chunk`` (number of observations per chunk, automatically converted to element counts for sparse arrays) and ``shard_size`` (number of observations per shard or a size string). The corresponding parameters in {meth}`annbatch.DatasetCollection.add_adatas` are ``n_obs_per_chunk`` and ``shard_size``.

### Fixed
- Formatted progress bar descriptions to be more readable.
- {class}`annbatch.DatasetCollection` now accepts a `rng` argument to the {meth}`annbatch.DatasetCollection.add_adatas` method.

### Features
- `shard_size` in {meth}`annbatch.DatasetCollection.add_adatas` and `shard_size` in {func}`annbatch.write_sharded` now accept a human-readable size string (e.g. ``'1GB'``, ``'512MB'``) in addition to an integer number of observations. When a string is provided, the observation count is derived independently for each array element from its uncompressed bytes-per-row so that every shard stays close to the target size.
- ``dataset_size`` in {meth}`annbatch.DatasetCollection.add_adatas` now accepts a human-readable size string (e.g. ``'20GB'``, ``'512MB'``) in addition to an integer number of observations. When a string is provided, the per-row byte size is estimated from the on-disk metadata of the input datasets during validation and used to derive the observation count. The default has changed from ``2_097_152`` to ``'20GB'``.

## [0.0.8]

- {class}`~annbatch.Loader` accepts an `rng` argument now

## [0.0.7]

- Make the in-memory concatenation strategy configurable for {meth}`annbatch.Loader.__iter__` via a `concat_strategy` argument to `__init__` - sparse on-disk will concatenated then shuffled/yielded (faster, higher memory usage) but dense will be shuffled and then concated/yielded (lower memory usage).
- Downcast `indices` of sparse matrices if possible when writing to disk via {attr}`anndata.settings.write_csr_csc_indices_with_min_possible_dtype`

## [0.0.6]

- Don't concatenate all i/o-ed chunks in-memory, instead yielding from individual chunks as though they were concatenated (i.e., not abreaking hcange with the {class}`annbatch.abc.Sampler` API).  Should improve memory performance especially for dense data

## [0.0.5]

- Fix bug with bringing the nullable/categorical columns into memory by default


### Breaking
- Now {class}`annbatch.Loader` expects ``preload_nchunks * chunk_size % batch_size == 0`` for simplification and efficiency.

### Added
- Introduced an {class}`annbatch.abc.Sampler` abstract base class. Users can implement and pass any class instance that is a subclass to the ``batch_sampler`` argument of {class}`annbatch.Loader`.
- Exposed the older default sampling scheme as `annbatch.ChunkSampler`, which is used internally to match older behavior when ``batch_sampler`` isn't provided to {class}`annbatch.Loader`.

## [0.0.4]

- Load into memory nullables/categoricals from `obs` by default when shuffling (i.e., no custom `load_adata` argument to `annbatch.DatasetCollection.add_adatas`)

## [0.0.3]

### Breaking

- Revert `h5ad` shuffling into one big store (i.e., go back to sharding into individual files) and add warning that `h5ad` is not fully supported by `annbatch`. `is_collection_h5ad` argument to initialization of {class}`annbatch.DatasetCollection` must be passed when initializing into to use a preshuffled collection of `h5ad` files, reading or writing.
- Renamed {class}`annbatch.types.LoaderOutput` `["labels"]` and `["data"]` to `["obs"]` and `["X"]` respectively.

## [0.0.2]


### Breaking

- `ZarrSparseDataset` and `ZarrDenseDataset` have been conslidated into {class}`annbatch.Loader`
- `create_anndata_collection` and `add_to_collection` have been moved into the `annbatch.DatasetCollection.add_adatas` method
- Default reading of input data is now fully lazy in `annbatch.DatasetCollection.add_adatas`, and therefore the shuffle process may now be slower although have better memory properties.  Use `load_adata` argument in `annbatch.DatasetCollection.add_adatas` to customize this behavior.
- Files shuffled under the old `create_anndata_collection` will not be recognized by {class}`annbatch.DatasetCollection` and therefore are not usable with the new {class}`annbatch.Loader.use_collection` API.  At the moment, the file metadata we maintain is only for internal purposes - however, if you wish to migrate to be able to use {class}`annbatch.DatasetCollection` in conjunction with {class}`annbatch.Loader.use_collection`, the root folder of the old collection must have attrs `{"encoding-type": "annbatch-preshuffled", "encoding-version": "0.1.0"}` and be a {class}`zarr.Group`. The subfolders (i.e., datasets) must be called `dataset_([0-9]*)`. Otherwise you can use the `annbatch.DatasetCollection.add_adatas` as before.

### Changed

- `preload_to_gpu` now depends on whether `cupy` is installed instead of defaulting to `True`

## [0.0.1]

### Added

- First release
