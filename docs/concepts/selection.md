# Survivor selection

Once every candidate of a layer is fitted and scored by the
[criterion](criteria.md), selection decides which ones survive. Candidates
of all families compete together. A candidate whose criterion value is NaN
is dropped first. The survivors become the layer; every other candidate is
deleted.

## How many survive

`model.nbest_neurons` sets the number of survivors per layer, GMDH's
*freedom of choice* (see [GMDH](gmdh.md)). The survivors are the inputs of
the next layer and, in a model with an output head, the inputs of the head.
More survivors give later layers more to combine, at the cost of more
candidates in every later layer: with 16 survivors and 16 features, the next
layer has 32 inputs and 496 pairs.

On California housing (four seeds, test MSE), 32 survivors beat 16 on every
seed, 24 give the same mean as 32 with the smallest spread, and 48 are worse
(California housing README).

## The three methods

`train.neuron_selection_method` chooses how the survivors are picked:

- **`plain`** (default): the `nbest_neurons` candidates with the lowest
  criterion values.
- **`omp_mixed`**: walks the candidates from the lowest criterion value up.
  For each one, it takes the candidate's outputs on the dev split, centred,
  and removes the parts that the survivors kept so far already explain (a
  Gram-Schmidt step). The candidate is kept only if at least
  `train.neuron_selection_orth_threshold` (0.3) of its original norm
  remains. The walk stops at `nbest_neurons` survivors, or earlier if the
  candidates run out.
- **`omp`**: orthogonal matching pursuit on the candidates' outputs. At
  each step it keeps the candidate whose remaining output, after removing
  the survivors' directions, has the largest norm. It ignores the criterion
  values when choosing.

Both OMP methods use the neuron's own output, before any multi-class
projection. In all three methods, the layer's error is computed from the
criterion values of the survivors actually kept.

## Why decorrelate

`plain` keeps the best candidates, which are often near-copies of each
other: the same strong pair of inputs, fitted by slightly different neurons.
On California housing, a linear head over the 16 survivors of a `plain`
layer beats the best survivor alone by only 0.002 in dev error (0.149
against 0.151), so the head has little to combine (header of
`california_housing_legendre_finetune.yaml`). `omp_mixed` keeps survivors
that differ from each other while still preferring good ones. Over four
seeds, `plain` and `omp` cost 0.009 to 0.015 in test MSE against
`omp_mixed`, the largest effect of any setting tested there (California
housing README).

## Diagnostics

With `train.log_layer_diagnostics: true`, the trainer logs for every layer
how correlated the survivors' outputs are on the dev split (the median and
largest absolute correlation between two survivors), their top 10 singular
values, and their effective rank, the number of singular values above 1% of
the largest. It costs one extra pass over the dev split per layer.

## The knobs

`model.nbest_neurons`, `train.neuron_selection_method`,
`train.neuron_selection_orth_threshold`, `train.log_layer_diagnostics`. See
[Configuration keys](../reference/config.md).

<small>Checked against TorchSONN 0.1.5.</small>
