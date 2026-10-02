# So you want to know...

## Do I need `shuffle=True` if my data is already preshuffled?

Yes. {class}`~annbatch.Loader` defaults to `shuffle=False`, which builds a {class}`~annbatch.samplers.SequentialSampler` and walks your chunks in on-disk order, so every epoch sees the same batches in the same sequence.
Preshuffling fixes what is *inside* each chunk; `shuffle=True` (a {class}`~annbatch.samplers.RandomSampler`) randomizes which chunks are drawn and shuffles observations within the in-memory buffer.
Each call to {meth}`~annbatch.Loader.__iter__` draws fresh randomness, so you do not need to re-run {meth}`~annbatch.DatasetCollection.add_adatas` between epochs.

For reproducibility, note that {func}`torch.manual_seed` has no effect here.
Pass a seeded generator: `Loader(shuffle=True, rng=np.random.default_rng(0))`.

## How do I hold out a validation set?

Two options:
For a random cell-level split, pass a `mask` slice to the sampler: because the collection is preshuffled, a contiguous range of observations is already a random subset.

```python
train = Loader(batch_sampler=RandomSampler(chunk_size=512, preload_nchunks=32, batch_size=4096, mask=slice(0, 900_000)))
val = Loader(batch_sampler=SequentialSampler(chunk_size=512, preload_nchunks=32, batch_size=4096, mask=slice(900_000, None)))
```

That leaks group structure, since the same donor or study lands on both sides.

For a donor- or study-level holdout, split the input files first and build two separate collections.

## How much memory does the loader hold at once?

`chunk_size * preload_nchunks` observations, so the defaults (512 and 32) keep 16,384 rows resident.
For dense `float32` over 20k genes that is roughly 1.3 GB; for sparse, scale by `nnz` per cell instead of `n_vars`.
Raising `preload_nchunks` to keep the GPU busy costs host memory linearly, which is the ceiling on the advice above.

With `preload_to_gpu=True` that buffer sits in device memory alongside your model and activations.
Set it to `False` for dense data under memory pressure.

## How do I do weighted sampling?

Use {class}`annbatch.samplers.WeightedClassSampler` in conjunction with a collection that has been created using the `groupby` parameter (i.e., pre-shuffled datasets within a collection have class-coherent runs dictated by this parameter) from {meth}`annbatch.DatasetCollection.add_adatas`.
{class}`~annbatch.samplers.WeightedClassSampler` will validate its `chunk_size` parameter against the size of consecutive runs created with the `groupby` parameter.

## How do I do pure-class sampling i.e., each batch comes from a single class?

Use {class}`annbatch.samplers.ClassSampler` again in conjunction with a collection that has been created using the `groupby` parameter {meth}`~annbatch.DatasetCollection.add_adatas`.

## How do I know how many chunks to preload for training i.e,. {class}`~annbatch.Loader`'s (or a given {class}`~annbatch.abc.Sampler`'s) `preload_nchunks`?

{cite:p}`xu2022stochasticgradientdescentdata` recommends an in-memory buffer size of ~2% of the training (i.e., `(preload_nchunks * chunk_size) / n_obs`) *when the data is not preshuffled*.
However, since your data *should* be preshuffled, you can get do with much less.

The `preload_nchunks` parameter for {class}`annbatch.Loader` can thus be set fairly small (our default is 32 for `chunk_size` of 512), but the best way to know is to monitor your GPU-usage.
**if your GPU has downtime to 0% usage, this parameter is probably too big as your model** is stalling while waiting for data to load.
GPU utilization is the single most important thing to look at when trying to make model training go faster.
Faster loading will help immensely here (as will bigger batches).

In theory, the `preload_to_gpu` parameter for {class}`~annbatch.Loader` should allow asynchronous transfers of pinned memory and thus alleviate stalling for larger `preload_nchunks` parameters than otherwise would be possible

## How do I know how big my {class}`~annbatch.Loader` (or a given {class}`~annbatch.abc.Sampler`'s) `chunk_size` should be?

Because your data is *preshuffled* you can go fairly large here - `chunk_size` of 512 and `preload_nchunks` of 32 will likely work great (even though sampling only 32 random chunks doesn't sound like a lot).
This setting also happens to be our default.
To see the effect of preshuffling on this parameter see [this notebook presented at the 2025 scverse conference for annbatch][].
In practice, the speed of data loading will likely start to plateau when your `chunk_size` gives chunks of ~`.5MB`.

[this notebook presented at the 2025 scverse conference for annbatch]: https://colab.research.google.com/drive/1yrGHZGgfCOPXc1quan3m7JMfb4k3WJxo

## Can I use a {class}`~annbatch.Loader` with a {class}`~annbatch.samplers.RandomSampler` on a dataset that has been `groupby`-ed?

Model performance may suffer. However, within a given class as specified by `groupby`, the data will be shuffled.
Therefore, it is probably not as bad as having completely unshuffled data, but is likely worse than having completely shuffled data.
We would welcome benchmarks to this end.

## What do I do if I don't see my modality/use-case listed under `tutorials`?

If you modality can fit in an {class}`~anndata.AnnData` object, there's a good chance it will work.

If you can't figure out how to get your modality into an {class}`~anndata.AnnData` object (or need some sort of disk format extension etc.), please file an issue in this repo or in the {mod}`anndata` repo.
