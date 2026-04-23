# Zarr Configuration

If you are using a local file system, use {doc}`zarrs-python <zarrs:index>`:

```python
import zarr
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
```

Otherwise normal use {mod}`zarr` without {doc}`zarrs-python <zarrs:index>` (wich does not support, for example, remote stores).

## `zarrs` Performance

Please look at {doc}`zarrs-python <zarrs:index>`'s docs for more info but there are two important setting to consider:

```python
zarr.config.set({
    "threading.max_workers": None,
    "codec_pipeline": {
        "direct_io": False
    }
})
```

The `threading.max_workers` will control how many threads are used by `zarrs`, and by extension, our data loader.
This parameter is global and controls both the rust parallelism and the Python parallelism.
If you notice thrashing or similar oversubscription behavior of threads, please open an issue.

Some **linux** file systems' [performance may suffer][] from the high level of parallelism combined with many page cache misses/evictions (i.e., when the dataset size on disk far exceeds available RAM).
Concretely, once the page cache is full, if there are a high number of cache misses/evictions relative to cache hits (i.e., the on-disk dataset exceeds that of available RAM by a lot), the data loading freezes.
Our experience matches that of a [paper looking at `mmap` for databases][], that a dataset size about 20X larger than available RAM memory (for the page cache) will trigger this performance degradation.
For example, if you only have 32GB of RAM, a ~600GB dataset on disk will trigger this degradation.
Thus, to bypass the page cache and any potential cache misses, use `direct_io`.
If this setting is set on a system that does not support `direct_io`, file reading will fall back to normal buffered io.

## `zarr-python` performance

In this case, likely the store of interest is in the cloud.
Please see [zarr python's config docs][] for more info but likely of most interest aside from the above mentioned `threading.max_workers` is

```python
zarr.config.set({"async.concurrency": 64})
```

which is 64 by default.
See the [zarr page on concurrency][] for more information. Furthermore, we recommend using [`obstore` over `fsspec`][] for performance reasons.

[performance may suffer]: https://gist.github.com/ilan-gold/705bd36329b0e19542286385b09b421b
[zarr page on concurrency]: https://zarr.readthedocs.io/en/latest/user-guide/consolidated_metadata/#synchronization-and-concurrency
[zarr python's config docs]: https://zarr.readthedocs.io/en/latest/user-guide/config/
[paper looking at `mmap` for databases]: https://db.cs.cmu.edu/papers/2022/cidr2022-p13-crotty.pdf
[`obstore` over `fsspec`]: https://zarr.readthedocs.io/en/latest/user-guide/storage/#object-store
