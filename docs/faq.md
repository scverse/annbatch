# So you want to know...

## How do I do weighted sampling?

Use {class}`annbatch.samplers.WeightedClassSampler` in conjunction with a collection that has been created using the `groupby` parameter (i.e., pre-shuffled datasets within a collection have class-coherent runs dictated by this parameter) from {meth}`annbatch.DatasetCollection.add_adatas`. Don't worry, {class}`~annbatch.samplers.WeightedClassSampler` will validate its `chunk_size` parameter against the size of consecutive runs created with the `groupby` parameter.

## How do I do pure-class sampling i.e., each batch comes from a single class?

Use {class}`annbatch.samplers.ClassSampler` again in conjunction with a collection that has been created using the `groupby` parameter {meth}`~annbatch.DatasetCollection.add_adatas`. Same validation will apply as above :)

## How do I know how many chunks to preload for training i.e,. {class}`~annbatch.Loader`'s (or a given {class}`~annbatch.abc.Sampler`'s) `preload_nchunks`?

{cite:p}`xu2022stochasticgradientdescentdata` recommends an in-memory buffer size of ~2% of the training (i.e., `(preload_nchunks * chunk_size) / n_obs`) *when the data is not preshuffled*. However, since your data *should* be preshuffled, you can get do with much less.

The `preload_nchunks` parameter for {class}`annbatch.Loader` can thus be set fairly small (our default is 32 for `chunk_size` of 512), but the best way to know is to monitor your GPU-usage - **if your GPU has downtime to 0% usage, this parameter is probably too big as your model** is stalling while waiting for data to load. This metric is the single most important thing to look at when trying to make model training go faster - how high is your GPU utilization. Faster loading will help immensely here (as will bigger batches).

In theory, the `preload_to_gpu` parameter for {class}`~annbatch.Loader` should allow asynchronous transfers of pinned memory and thus alleviate stalling for larger `preload_nchunks` parameters than otherwise would be possible

## How do I know how big my {class}`~annbatch.Loader` (or a given {class}`~annbatch.abc.Sampler`'s) `chunk_size` should be?

Because your data is *preshuffled* you can go fairly large here - `chunk_size` of 512 and `preload_nchunks` of 32 will likely work great (even though sampling only 32 random chunks doesn't sound like a lot). This setting also happens to be our default. To see the effect of preshuffling on this parameter see [this notebook presented at the 2025 scverse conference for annbatch][]. In practice, the speed of data loading will likely start to plateau when your `chunk_size` gives chunks of ~`.5MB`.

[this notebook presented at the 2025 scverse conference for annbatch]: https://colab.research.google.com/drive/1yrGHZGgfCOPXc1quan3m7JMfb4k3WJxo

## Can I use a {class}`~annbatch.Loader` with a {class}`~annbatch.samplers.RandomSampler` on a dataset that has been `groupby`-ed?

Well, your model performance may suffer but again, within a given class as specificed by `groupby`, the data will be shuffled. So it is probably not as bad as having completely unshuffled data, but is likely worse than having completely shuffled data. We would welcome benchmarks to this end.