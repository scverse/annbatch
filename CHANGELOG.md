# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.0.2]

### Breaking

- `ZarrSparseDataset` and `ZarrDenseDataset` have been conslidated into {class}`annbatch.Loader`
- `create_anndata_collection` and `add_to_collection` have been moved into the {meth}`annbatch.DatasetCollection.add_adatas` method
- Default reading of input data is now fully lazy in {meth}`annbatch.DatasetCollection.add_adatas`, and therefore the shuffle process may now be slower although have better memory properties.  Use `load_adata` argument in {meth}`DatasetCollection.add_adatas` to customize this behavior.

### Changed

- `preload_to_gpu` now depends on whether `cupy` is installed instead of defaulting to `True`

## [0.0.1]

### Added

- First release
