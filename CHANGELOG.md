# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

# [0.0.3]

- Revert `h5ad` shuffling into one big store (i.e., go back to sharding into individual files) and add warning that `h5ad` is not fully supported by `annbatch`.

## [0.0.2]

### Breaking

- `ZarrSparseDataset` and `ZarrDenseDataset` have been conslidated into {class}`annbatch.Loader`
- `create_anndata_collection` and `add_to_collection` have been moved into the {meth}`annbatch.DatasetCollection.add_adatas` method
- Default reading of input data is now fully lazy in {meth}`annbatch.DatasetCollection.add_adatas`, and therefore the shuffle process may now be slower although have better memory properties.  Use `load_adata` argument in {meth}`annbatch.DatasetCollection.add_adatas` to customize this behavior.
- Files shuffled under the old `create_anndata_collection` will not be recognized by {class}`annbatch.DatasetCollection` and therefore are not usable with the new {class}`annbatch.Loader.use_collection` API.  At the moment, the file metadat we maintain is only for internal purposes - however, if you wish to migrate to be able to use {class}`annbatch.DatasetCollection` in conjunction with {class}`annbatch.Loader.use_collection`, the root folder of the old collection must have attrs `{"encoding-type": "annbatch-preshuffled", "encoding-version": "0.1.0"}` and be a {class}`zarr.Group`. The subfolders (i.e., datasets) must be called `dataset_([0-9]*)`.

### Changed

- `preload_to_gpu` now depends on whether `cupy` is installed instead of defaulting to `True`

## [0.0.1]

### Added

- First release
