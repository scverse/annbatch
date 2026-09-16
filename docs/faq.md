# So you want to know...

## How do I do weighted sampling?

Use {class}`annbatch.samplers.WeightedClassSampler` in conjunction with a collection that has been created using the `groupby` parameter (i.e., pre-shuffled datasets within a collection have class-coherent runs dictated by this parameter) from {meth}`annbatch.DatasetCollection.add_adatas`. Don't worry, {class}`~annbatch.samplers.WeightedClassSampler` will validate its `chunk_size` parameter against the size of consecutive runs created with the `groupby` parameter.

## How do I do pure-class sampling?

Use {class}`annbatch.samplers.ClassSampler` again in conjunction with a collection that has been created using the `groupby` parameter {meth}`~annbatch.DatasetCollection.add_adatas`. Same validation will apply :)

## How do I know how many chunks to preload for training?

The `preload_nchunks` parameter for {class}`annbatch.Loader` can be set fairly large, but the best way to know is to monitor your GPU-usage - if your GPU has downtime to 0% usage, this parameter is probably too big.

## How do I know how big my chunks should be?

Because your data is *preshuffled* you can go fairly large here - `chunk_size` of 512 and `preload_nchunks` of 32 will likely work great (even though sampling only 32 random chunks doesn't sound like a lot). This setting also happens to be our default. To see the effect of preshuffling on this parameter see [this notebook presented at the 2025 scverse conference for annbatch][]

[this notebook presented at the 2025 scverse conference for annbatch]: https://colab.research.google.com/drive/1yrGHZGgfCOPXc1quan3m7JMfb4k3WJxo

## Can I use a {class}`~annbatch.Loader` with a {class}`~annbatch.samplers.RandomSampler` on a dataset that has been `groupby`-ed?

Well, your model performance may suffer but again, within a given class as specificed by `groupby`, the data will be shuffled. So it is probably not as bad as having completely unshuffled data, but is likely worse than having completely shuffled data. I can't say we've benchmarked this specific case, though.