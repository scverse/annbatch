```{include} ../README.md
:end-before: <!--FOOTER-->
```

## In Depth

Let's go through the above example:

### Preprocessing

```python
colleciton = DatasetCollection("path/to/output/store.zarr").add_adatas(
    adata_paths=[
        "path/to/your/file1.h5ad",
        "path/to/your/file2.h5ad"
    ],
    shuffle=True,  # shuffling is needed if you want to use chunked access
)
```

First, you converted your existing `.h5ad` files into a zarr-backed anndata format.
In the process, the data gets shuffled and is distributed across several anndata files.
Shuffling is important to ensure model convergence, especially because of our contiguous data fetching scheme which is not perfectly random.
The output is a collection of sharded zarr anndata files, meant to reduce the burden on file systems of indexing.
See the [zarr docs on sharding][] for more information.

[zarr docs on sharding]: https://zarr.readthedocs.io/en/stable/user-guide/arrays/#sharding


### Data loading

#### Chunked access

```python
# `use_collection` will automatically get everything in `X` and `obs` and yield it.
ds = Loader(
    batch_size=4096,
    chunk_size=32,
    preload_nchunks=256,
).use_collection(collection)

# Iterate over dataloader (plugin replacement for torch.utils.DataLoader)
for batch in ds:
    x, df, index = batch["X"], batch["obs"], batch["index"]
```

The data loader implements a chunked fetching strategy where `preload_nchunks` number of continguous-chunks of size `chunk_size` are loaded.
`chunk_size` corresponds the number of rows of `anndata` store to load sequentially.

For performance reasons, you should use our dataloader directly without wrapping it into a {class}`torch.utils.data.DataLoader`.
Your code will work the same way as with a {class}`torch.utils.data.DataLoader`, but you will get better performance.

In order to take advantage of the sharded zarr files performance, though, locally, you *must* set the codec pipeline to use {doc}`zarrs-python <zarrs:index>` when reading.
Using {mod}`zarr` on its own will not yield high performance for local filesystems.
We have not tested remote data (i.e., using {func}`zarr.open` with a {class}`zarr.storage.ObjectStore`) but because we use {mod}`zarr`, this data loader should also work over cloud connections via relevant zarr stores.
Note that {doc}`zarrs-python <zarrs:index>` cannot be used with these sorts of non-local stores.

#### User configurable sampling strategy

At the moment we do not support user-configurable sampling strategies like weighting or sampling.
With a pre-shuffled store and blocked access, your model fit should not be affected by using chunked access.

If you are interested in contributing this feature to the project or learning more, please get in touch on [zulip](https://scverse.zulipchat.com/) or via the GitHub issues here.

If you want to use {class}`torch.utils.data.DataLoader` to accelerate perfect random sampling (i.e., wrapping {class}`~annbatch.Loader` with `chunk_size=1`) or begin to experiment with implementing weighted sampling schemes, you will need to pass in `multiprocessing_context="spawn"` to the {class}`torch.utils.data.DataLoader` (see {issue}`google/tensorstore#61`, for example).

### Speed comparison to other dataloaders

We provide a speed comparison to other comparable dataloaders below:

<img src="_static/speed_comparision.png" alt="speed_comparison" width="400">

We've run the above benchmark on an AWS `ml.m5.8xlarge` instance.
The code to reproduce the above results can be found on LaminHub:

* [Benchmark results](https://lamin.ai/laminlabs/arrayloader-benchmarks/transform/e6Ry7noc4Y0d)
* [annbatch code](https://lamin.ai/laminlabs/arrayloader-benchmarks/transform/yl0iTPhJjkqW)
* [MappedCollection code](https://lamin.ai/laminlabs/arrayloader-benchmarks/transform/YfzHfoomTkfu)
* [scDataset code](https://lamin.ai/laminlabs/arrayloader-benchmarks/transform/L6CAf9w0qdQj)

### Why data loading speed matters?

Most models for scRNA-seq data are pretty small in terms of model size compared to models in other domains like computer vision or natural language processing.
This size differential puts significantly more pressure on the data loading pipeline to fully utilize a modern GPU.
Intuitively, if the model is small, doing the actual computation is relatively fast.
Hence, to keep the GPU fully utilized, the data loading needs to be a lot faster.

As an illustrative, example let's train a logistic regression model ([notebook hosted on LaminHub](https://lamin.ai/laminlabs/arrayloader-benchmarks/transform/cV00NQStCAzA?filter%5Band%5D%5B0%5D%5Bor%5D%5B0%5D%5Bbranch.name%5D%5Beq%5D=main&filter%5Band%5D%5B1%5D%5Bor%5D%5B0%5D%5Bis_latest%5D%5Beq%5D=true)).
Our example model has 20.000 input features and 100 output classes.
We can now look how the total fit time changes with data loading speed:

<img src="_static/fit_time_vs_loading_speed.png" alt="fit_time_vs_loading_speed" width="400">

From the graph we can see that the fit time can be decreased substantially with faster data loading speeds (several orders of magnitude).
E.g. we are able to reduce the fit time from ~280s for a data loading speed of ~1000 samples/sec to ~1.5s for a data loading speed of ~1.000.000 samples/sec.
This speedup is more than 100x and shows the significant impact data loading has on total training time.

### When would you use this data laoder?

As we just showed, data loading speed matters for small models (e.g., on the order of an scVI model, but perhaps not a " foundation model").
But loading minibatches of bytes off disk will be almost certainly slower than loading them from an in-memory source.
Thus, as a first step to assessing your needs, if your data fits in memory, load it into memory.
To accelerate reading the data into memory, you may still find {doc}`zarrs-python <zarrs:index>` in conjunction with sharding still helpful in the same way it accelerates io here.
To this end, please have a look at [this gist](https://gist.github.com/ilan-gold/c73383def3798df2724405aa64e40c3d) comparing file loading speeds between {func}`anndata.io.read_zarr` and {func}`anndata.io.read_h5ad`.
It highlights how {doc}`zarrs-python <zarrs:index>` and sharding can help there as well.
However, once you have too much data to fit into memory, for whatever reason, the data loading functionality offered here can provide significant speedups over state of the art out-of-core dataloaders.

```{include} ../README.md
:start-after: <!--FOOTER-->
```

```{toctree}
:hidden: true
:maxdepth: 1

api.md
zarr-configuration.md
changelog.md
contributing.md
references.md
notebooks/index
```
