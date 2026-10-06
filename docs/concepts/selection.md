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
A linear head over such survivors has little to combine, since they carry
much the same signal. `omp_mixed` keeps survivors that differ from each
other while still preferring good ones, giving the head more distinct inputs
to work with.

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
